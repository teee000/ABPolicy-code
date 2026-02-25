

from pathlib import Path
import torch
from functools import partial
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf
from torch import nn
from torch.optim import AdamW
import os

from argparse import ArgumentParser
from transformers import AutoModel


from model.diffusion.conditional_unet1d import CondDiT
from model.perceiver import ImgObsPerceiver
from model.policy import CAGE
from model.wrapper import DinoV2EncoderWrapper, ResNetEncoderWrapper

from utils.utils import (count_model_params, 
                         preprocess_5d_tensor, 
                         build_img_preprocess,
                         make_loss_recorder,
                         scale_stats_values
                         )
from utils.hdf5_dataloader import  HDF5Dataset
from utils.data_utils import StateActionNorm

from utils.cal_stats import calculate_hdf5_statistics

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher


from utils.curve_fitter import BSplineFitter

import warnings  
warnings.filterwarnings("ignore")  


def get_parser():
    parser = ArgumentParser()
    # TODO
    parser.add_argument(
        '--config',
        type=str,
        default='cage',
        help='config file used for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='seed for deterministic training'
    )
    
    parser.add_argument(
        '--mixed_precision',
        default=False,
        action='store_true',
        help='whether or not to use mixed precision for training'
    )
    
    return parser

def initialize_model(conf):
    # Load weights from pretrained model  
    i_encoder = AutoModel.from_pretrained( conf.model.image_encoder.name)
    if conf.model.image_encoder.freeze:
        i_encoder.requires_grad_(False)

    print(f"Original number of layers: {len(i_encoder.encoder.layer)}")
    num_layers_to_keep = conf.image_backbone_used_layers
    i_encoder.encoder.layer = i_encoder.encoder.layer[:num_layers_to_keep]
    print(f"Modified number of layers: {len(i_encoder.encoder.layer)}")

    obs_dim = conf.model.obs_dim

    if 'resnet' in conf.model.image_encoder.name:
        Wrapper = ResNetEncoderWrapper
    elif 'dino' in conf.model.image_encoder.name:
        Wrapper = DinoV2EncoderWrapper
        
    Wrapper = partial(
        Wrapper,
        pooled = conf.model.image_encoder.pooled,
        out_dim = obs_dim,
    )
        
    obs_encoders = nn.ModuleDict({
        'cam_high': Wrapper(i_encoder),
    })
            
    img_perceiver = ImgObsPerceiver(
        initial_n=img_patch_num,
        in_channels=1024,
        out_channels=obs_dim,
        mid_channels=obs_dim,
    )
    
    obs_num = conf.dataset.meta_data.fixed_views + conf.dataset.meta_data.in_hand_views
    
    
    obs_horizon = conf.qpos_obs_horizon
    
    backbone = CondDiT(
        input_dim = conf.action_dim,
        input_len = conf.b_spline_ctrl_n,
        obs_dim = obs_dim,
        num_blocks=6,
        conv_kernel_size=3,
        num_norm_groups=8,
        num_attn_heads=8,
        self_attn_masks=None,
    )

    model = CAGE(
        obs_encoders, 
        img_perceiver,
        backbone,
        obs_dim     = obs_dim,
        obs_horizon = obs_horizon,
        obs_num     = obs_num,       # camera num
        obs_image_keys = obs_image_keys,
    )

    return model



def calc_loss(batch, 
              model,
              noise_scheduler, 
              use_proprio=True, 
              weight_dtype=torch.float32
              ):
    
    actions = batch['/observations/action']  # B, Ta, D
    qpos = batch['/observations/qpos'] # 
    
    actions = state_action_normer.normalize(actions)
    qpos = state_action_normer.normalize(qpos)  ## B, To, D
    
    batch['/observations/action'] = actions # B, Ta, D
    batch['/observations/qpos'] = qpos  # 
    
    # Sample gaussian noises
    noise = torch.randn_like(actions)
    
    ### random t
    t_min_value = 0.
    t_max_value = 1.
    ts = (torch.rand(noise.shape[0]).type_as(noise) * (t_max_value - t_min_value)) + t_min_value
    # Input: x0: noise, x1: data, t: timestamps
    # Output: ut: vecter field groundtruth
    t, noisy_actions, target = FM.sample_location_and_conditional_flow(noise, actions, ts)
    
    noisy_actions = noisy_actions.to(dtype=weight_dtype)
    
    
    # Predict the noise residual
    pred = model(
        ts,
        noisy_actions,
        obs_dict=batch
        )  ## B, Ta, D

    # Compute loss and mean over the non-batch dimensions
    loss = F.mse_loss(pred, target )
    return loss


if __name__ == "__main__":
    
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    base_conf = OmegaConf.load(os.path.join('configs', opt.config+'.yaml'))
    cli = OmegaConf.from_dotlist(unknown)
    conf = OmegaConf.merge(base_conf, cli)
    
    # load sub configs
    model_conf_path = os.path.join('configs', 'model', base_conf.model.name+'.yaml')
    if os.path.exists(model_conf_path):
        model_conf = OmegaConf.load(model_conf_path)
        base_conf = OmegaConf.merge(base_conf, model_conf)
    
    conf = OmegaConf.merge(base_conf, cli)
    
    
    DATA_PATH = [
        './record_data/stack_block',
        ]
    
    data_stats_save_path = './assets/dataset_stats.json'
    
    # directory to store the checkpoint
    output_directory = Path(conf.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    
    device = torch.device("cuda")
    log_step_freq = 50
    dataset_fps = 30  
    
    img_size = conf.dataset.preprocess.img_size
    img_patch_size = conf.dataset.preprocess.img_patch_size
    img_patch_num = int(img_size/img_patch_size)**2
    
    
    weight_dtype = torch.bfloat16 if opt.mixed_precision else torch.float32
    
    obs_image_keys = [
        '/observations/images/cam_high_left',
        '/observations/images/cam_high_right',
        ]
    
    
    # TODO
    qpos_indices =  list(range(-(conf.qpos_obs_horizon-1), 1))     # example: range(-3, 1) >>> -3,-2,-1,0
    action_indices = [i for i in range(-conf.action_history_horizon, conf.action_horizon)]
    
    cam_indices = {
        'cam_high_left':  list(range(-(conf.img_obs_horizon-1), 1)),
        'cam_high_right': list(range(-(conf.img_obs_horizon-1), 1))
    }
    
    dataset = HDF5Dataset(
        roots=DATA_PATH,
        qpos_delta_indices=qpos_indices,
        action_delta_indices=action_indices,
        cam_delta_indices=cam_indices
    )
    
    data_stats_dic = calculate_hdf5_statistics(DATA_PATH, data_stats_save_path)
    data_stats_dic = scale_stats_values(data_stats_dic, factor=conf.stats_scale_factor)
    
    data_stats_dic['qpos']
    
    state_action_normer = StateActionNorm(
        min_val=data_stats_dic['action']['min'],   # list, len=7
        max_val=data_stats_dic['action']['max'], 
        device=device,
        target_range='-1_1'       #  0_1 , -1_1
        )
    
    
    model = initialize_model(conf)

    model = model.to(device)
    
    count_model_params(model)
    
    # optimizer
    grouped_params = model.get_optim_groups(conf)
    
    optimizer = AdamW(grouped_params)
    
    
    FM = ConditionalFlowMatcher(sigma=0.0)
    
    # Create dataloader for offline training
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )
    
    image_preprocess_func = build_img_preprocess(
        crop_size=(480,480),
        crop_jitter_max=(0, 0),
        resize_size=(img_size, img_size),
        brightness_contrast=(0.1, 0.1),  
        noise_std=2.0,              
        train_flag=True
        )
    
    loss_recorder = make_loss_recorder(log_step_freq, save_dir="./log/loss") 
    
    total_steps = conf.epoch * len(dataloader)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer           = optimizer,
        num_warmup_steps    = conf.warmup_steps,
        num_training_steps  = total_steps,
        num_cycles          = 0.5      
        )
    
    bspline_fit_length = conf.action_history_horizon + conf.action_horizon
    action_fitter = BSplineFitter(T=bspline_fit_length,
                                  k=conf.b_spline_k,
                                  n_ctrl=conf.b_spline_ctrl_n
                                  )
    
    # Run training loop.
    step = 0
    for epoch in range(conf.epoch):
        for batch in dataloader:
            
            fitter_input = batch['/observations/action'].numpy()
            
            action_ctrl_y = action_fitter.fit_batch(fitter_input)
            
            batch['/observations/action'] = torch.from_numpy(action_ctrl_y).float()
            
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            '''
            '/observations/qpos',                      # B, Th, D
            '/observations/action', 
            '/observations/images/cam_high_left',      # B, Th, C, H, W
            '/observations/images/cam_high_right', 
            'qpos_is_pad', 
            'action_is_pad', 
            'images_is_pad'
            '''
            
            # qpos add noise
            batch['/observations/qpos'] += torch.normal(
                mean=0, 
                std=conf.qpos_noise_std, 
                size=batch['/observations/qpos'].shape, 
                device=device
                )
            
            
            for k in obs_image_keys:
                batch[k] = preprocess_5d_tensor(batch[k], image_preprocess_func)    # B, Th, C, H, W
            
            # Compute loss
            loss = calc_loss(batch, model, conf.model.use_proprio, weight_dtype)
    
            # Backpropagate
            optimizer.zero_grad()
            
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   
            optimizer.step()
            
            lr_scheduler.step()    
            
            # ---------- record loss ----------
            loss_recorder(loss.item(), epoch, step)
            
            step += 1

        # save model
        if epoch % conf.model_save_epoch == 0:
            
            model_save_path = os.path.join(output_directory, 'final_model_{}.bin'.format(epoch))
            print('save model to:', model_save_path)
            torch.save(model.state_dict(), model_save_path)






