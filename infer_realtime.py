import time
import torch
from collections import deque
import threading
import os
from torch import nn
import cv2
import numpy as np
from pynput import keyboard
from argparse import ArgumentParser
from omegaconf import OmegaConf

from functools import partial
import json
from einops import rearrange

from piper_sdk import C_PiperInterface
from utils.camera_async_utils import CameraRecorderBlock
from utils.arm_utils import torque_enable, set_rest_position

from model.wrapper import ResNetEncoderWrapper
from transformers import AutoModel
from model.diffusion.conditional_unet1d import CondDiT
from model.perceiver import ImgObsPerceiver
from model.policy import CAGE
from model.wrapper import DinoV2EncoderWrapper

from utils.data_utils import StateActionNorm
from utils.utils import (
                         preprocess_5d_tensor,
                         build_img_preprocess, 
                         ActionSmoother,
                         scale_stats_values
                         )

from utils.curve_fitter import BSplineFitter


def sample_actions(batch,
                   model, 
                   weight_dtype=torch.float32, 
                   num_inference_steps=10
                   ):

    qpos = batch['/observations/qpos']  
    qpos = state_action_normer.normalize(qpos)  ## B, To, D
    batch['/observations/qpos'] = qpos 
    
    if conf.use_bspline_flag:
        z = torch.randn((1, conf.b_spline_ctrl_n, ACTION_DIM), device=device)  
    else:
        z = torch.randn((1, conf.action_horizon, ACTION_DIM), device=device)  
        
    obs_emb, qpos_emb = model.preprocess_obs(batch)
    
    t_vals = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)
    for i in range(num_inference_steps):
        t = torch.full((z.size(0),), t_vals[i], device=device)
        r = torch.full((z.size(0),), t_vals[i + 1], device=device)

        t_ = rearrange(t, "b -> b 1 1").detach().clone()
        r_ = rearrange(r, "b -> b 1 1").detach().clone()

        v = model(t, z, obs_emb=obs_emb, qpos_emb=qpos_emb)
        z = z - (t_-r_) * v
    pred_actions = state_action_normer.denormalize(z)
    
    return pred_actions



class ModelInfer:
    
    def __init__(self,):  
        
        self.arm_speed = 50
        
        # arm joint angle factor 1000*180/3.14
        self.joint_factor = 57324.840764  
        self.gripper_factor = 70000  # arm gripper range (0-70000); controller trigger range (0-1) 
        
        # set control freq
        self.control_freq = 30
        
        self.loop_interval = 1 / self.control_freq
        
        # TODO
        # camera names
        self.camera_name_id_map = {  
                'cam_high_left':    '134322070579',   
                'cam_high_right' :  '152122074559',     
            }  
        self.camera_id_name_map = {v: k for k, v in self.camera_name_id_map.items()}  
        self.camera_names = [k for k, v in self.camera_name_id_map.items()]
        
        self.qpos_indices = qpos_indices
        
        
        # init arm
        self.arm = C_PiperInterface("can0")
        self.arm.ConnectPort()
        
        
        self.lastest_actions_queue = deque(maxlen=20)
        self.history_obs_queue = deque(maxlen=15)  
        self.ctrl_y_queue = deque(maxlen=1)
        self.stop_thread = False  
        
        bspline_fit_length = conf.action_history_horizon + conf.action_horizon
        self.action_fitter = BSplineFitter(
            T=bspline_fit_length,
            k=conf.b_spline_k,
            n_ctrl=conf.b_spline_ctrl_n
            )

    def start_data_acquisition_thread(self):
        self.stop_thread = False
        self.data_acquisition_thread = threading.Thread(target=self.data_acquisition_loop)
        self.data_acquisition_thread.start()
        
        # used to record the episode
        self.data_dict = {}
        for cam_name in self.camera_names:
            self.data_dict[f'/observations/images/{cam_name}'] = []
        
        self.data_dict['action'] = []
        
    def stop_data_acquisition_thread(self):
        self.stop_thread = True
        self.data_acquisition_thread.join() 



    def start_model_infer_thread(self):
        self.stop_model_infer_flag = False
        self.model_infer_thread = threading.Thread(target=self.model_infer_loop)
        self.model_infer_thread.start()
        
    def stop_model_infer_thread(self):
        self.stop_model_infer_flag = True
        self.model_infer_thread.join()  

    def model_infer_loop(self):
        while not self.stop_thread: 
            if len(self.history_obs_queue) != 0:  
                timestamps, observations = zip(*self.history_obs_queue) 
                obs_data_list = list(observations)

                closest_obs_data = [obs_data_list[int(idx) - 1] for idx in self.qpos_indices]
                
                # qpos
                input_batch = {}
                input_batch['/observations/qpos'] = torch.cat([obs['/observations/qpos'].unsqueeze(1) 
                                                              for obs in closest_obs_data],
                                                             dim=1) # B, T ,D
                
                # image
                for cam_name in obs_image_keys:  
                    input_batch[cam_name] =  closest_obs_data[-1][cam_name].unsqueeze(1)  # B, N, C, H, W
            
                input_batch = {k: v.to(device, non_blocking=True) for k, v in input_batch.items()}
                
                # crop resize norm
                for k in obs_image_keys:
                    input_batch[k] = preprocess_5d_tensor(input_batch[k], image_preprocess_func) 
                
                ss = time.time()
                
                # model infernce 
                with torch.no_grad():
                    ctrl_y_pred = sample_actions(
                        input_batch,
                        model, 
                        num_inference_steps=conf.denoise_infer_steps
                        )
                
                ctrl_y_pred = ctrl_y_pred.cpu().numpy().squeeze()
                
                t_obs = timestamps[-1]
                self.ctrl_y_queue.append((t_obs, ctrl_y_pred))
                
                print('model infer time delay:', time.time() - ss)
                
            else:
                print('history_obs_queue is null')
                        

    def data_acquisition_loop(self, ):
        while not self.stop_thread:
            start_time = time.perf_counter()
            
            # blocked
            latest_frames = self.camera_recoders.get_frames(self.camera_ids)
            
            observation = {}
            for cam_name, cam_id in self.camera_name_id_map.items():
                rgb_frame = latest_frames[cam_id]['color_image']
                
                # save images
                self.data_dict[f'/observations/images/{cam_name}'].append(rgb_frame)
                
                # TODO
                # Realsense: BGR   Opencv: BGR   PyTorch: RGB
                rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)   # (H, W, C)
                rgb_frame_np = torch.from_numpy(rgb_frame).type(torch.float32)
                observation[f'/observations/images/{cam_name}'] = rgb_frame_np
            
            arm_qpos = self.get_observation()  
            
            observation['/observations/qpos'] = torch.from_numpy(arm_qpos).type(torch.float32)
            
            
            for name in observation:
                observation[name] = observation[name].permute(2, 0, 1).contiguous() if "image" in name else observation[name]
                observation[name] = observation[name].unsqueeze(0)

            timestamp = time.perf_counter()
            self.history_obs_queue.append((timestamp, observation))
            
            # control fps
            elapsed_time = time.perf_counter() - start_time
            time.sleep(max((1/30) - elapsed_time, 0))
    
    def infer_loop(self):
        self._arm_init(self.arm, self.arm_speed)
        
        self.camera_recoders = CameraRecorderBlock(  
                fps=30,  
                width=640,  
                height=480,  
                exposure_value=None,  
                gain_value=None
            )  
        
        self.camera_ids = self.camera_recoders.serial_numbers
        
        time.sleep(1.5)  # wait camera init
        self.start_data_acquisition_thread()  
        
        all_action_data_lst = []
        
        self.stop_thread = False
        time.sleep(0.0333*16)
        
        self.start_model_infer_thread()
        time.sleep(0.5) 
        
        for i in range(20):
            self.lastest_actions_queue.append(np.zeros(7))  # action dim
        
        action_queue = deque(maxlen=16)
        
        try:
            for loop_idx in range(900):
                print('loop_idx:', loop_idx)
                
                start_time = time.perf_counter()
                
                if len(self.ctrl_y_queue) != 0:
                    t_obs, ctrl_y_pred = self.ctrl_y_queue.popleft()
                    print('ctrl_y_pred:', ctrl_y_pred.shape) # torch.Size([1, 8, 7])
                    
                    # TODO
                    t_exec = time.perf_counter()
                    delay_steps = int(round((t_exec - t_obs) * self.control_freq))  # 30Hz
                    print('delay_steps:', delay_steps)
                    
   
                    # use history action predict
                    y_prefix = np.array(list(self.lastest_actions_queue)[-conf.action_history_horizon-delay_steps:])
                    n_prefix = conf.action_history_horizon+delay_steps
                    
                    
                    # TODO
                    
                    # pred_actions, _ = self.action_fitter.rebuild(ctrl_y_pred)  # (40, D)
                    
                    
                    # ctrl_y_pred = self.action_fitter.refit_prefix(y_prefix,
                    #                                               ctrl_y_pred,
                    #                                               n_prefix=n_prefix,
                    #                                               n_free=4,
                    #                                               )   
                    
                    
                    # The larger the `last_pt_weight`, the more the "last free control point" is anchored at its current value; 0 indicates no constraint
                    ctrl_y_pred_refit = self.action_fitter.refit_prefix_w(y_prefix,
                                                                  ctrl_y_pred,
                                                                  n_prefix=n_prefix,
                                                                  n_free=4,
                                                                  last_pt_weight=0.05  , 
                                                                  )   # Only adjust the first n_free control points           
                    
                    
                    
                    pred_actions, _ = self.action_fitter.rebuild(ctrl_y_pred_refit)    # (T, D)
                    
                    pred_actions_grapper, _ = self.action_fitter.rebuild(ctrl_y_pred[:,-1:])
                    
                    # TODO
                    pred_actions = pred_actions[n_prefix:]  # remove history actions
                    pred_actions_grapper = pred_actions_grapper[n_prefix:]
                    
                    
                    action_queue.clear()
                    for i in range(conf.execute_action_horizon):
                        action_queue.append(pred_actions[i])
                    
                action = action_queue.popleft()
                
                all_action_data_lst.append(action)
                self.lastest_actions_queue.append(action)
                self.data_dict['action'].append(action )
                
                joints_action = action[:6]
                gripper_action = action[6]

                self.send_arm_cmd(self.arm, joints_action, gripper_action)

                looptime = time.perf_counter() - start_time
                time.sleep(max(self.loop_interval - looptime, 0))
                
                # TODO
                if stop_event.is_set():
                    print("\n [Spacebar] detected. Stop...")
                    break  

            
        finally:
            # first stop thread
            self.stop_data_acquisition_thread()  
            
            self.ctrl_y_queue.clear()
            self.history_obs_queue.clear()
            self.lastest_actions_queue.clear()
            
            self.camera_recoders.clear()
            torch.cuda.empty_cache()
    
        
    
    def _arm_init(self, piper, arm_speed=50):
        # torque on
        torque_enable_flag = torque_enable(piper=piper)
        print('\ntorque enable flag:', torque_enable_flag)
        
        # reset arm 
        if not torque_enable_flag:
            input('Enter to reset arm:')
            # reset,  disable torque
            piper.MotionCtrl_1(0x02,0,0)
            
            torque_enable_flag = torque_enable(piper=piper)
        
        
        if torque_enable_flag:
            
            # set rest/home position, use joint control mode
            set_rest_position(piper, execut_time=2)
            
            # set control mode
            piper.MotionCtrl_2(ctrl_mode=0x01, 
                                move_mode=0x01,                ### joint control 
                                move_spd_rate_ctrl=arm_speed,  ### range:0~100 
                                is_mit_mode=0x00
                                )
    
    def send_arm_cmd(self, piper, joint_positions, gripper_pose):
        # joints
        joint_0 = round(joint_positions[0]*self.joint_factor)
        joint_1 = round(joint_positions[1]*self.joint_factor)
        joint_2 = round(joint_positions[2]*self.joint_factor)
        joint_3 = round(joint_positions[3]*self.joint_factor)
        joint_4 = round(joint_positions[4]*self.joint_factor)
        joint_5 = round(joint_positions[5]*self.joint_factor)
        # gripper
        joint_6 = round(gripper_pose * self.gripper_factor)
        ### not blocked
        piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)

        # position(0-70,000)  effort(1000=1N/m)   mode 
        piper.GripperCtrl(joint_6, 600, 0x01, 0)
        
        

    def get_observation(self):
        # arm joints 
        self.pre_action_joints = self.get_arm_joints(self.arm)
        # arm gripper
        gripper_angle = self.get_arm_gripper(self.arm)
        
        arm_qpos = np.zeros(7)
        arm_qpos[0:6] = self.pre_action_joints
        arm_qpos[6] = gripper_angle

        return arm_qpos

    def get_arm_joints(self, piper):
        ### get arm eef pose
        arm_joints_dic = piper.GetArmJointsDic()
        joints = []
        for i in range(1,7):
            joints.append(np.radians(arm_joints_dic[str(i)]))
        return joints

    def get_arm_gripper(self, piper):
        arm_gripper_dic = piper.GetArmGripperDic()
        gripper_angle = arm_gripper_dic['gripper_angle'] / self.gripper_factor
        gripper_angle = np.clip(gripper_angle, 0, 1)
        return gripper_angle

def get_parser():
    parser = ArgumentParser()
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


if __name__ == "__main__":
    
    FPS = 30
    load_checkpoint_epoch = 140  
    
    inference_time_s = 30
    device = torch.device("cuda")  
    
    
    
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
    
    ACTION_HORIZON = conf.action_horizon
    QPOS_HISTORY = conf.qpos_obs_horizon
    qpos_indices =  list(range(-(QPOS_HISTORY-1), 1))     # example: range(-3, 1) >>> -3,-2,-1,0
    
    cam_indices = {
        'cam_high_left':  list(range(-(conf.img_obs_horizon-1), 1)),
        'cam_high_right': list(range(-(conf.img_obs_horizon-1), 1))
    }
    
    
    ACTION_DIM = conf.action_dim
    
    img_size = conf.dataset.preprocess.img_size
    img_patch_size = conf.dataset.preprocess.img_patch_size
    img_patch_num = int(img_size/img_patch_size)**2
    
    weight_dtype = torch.bfloat16 if opt.mixed_precision else torch.float32
    
    checkpoint_directory = conf.out_dir
    load_model_path = os.path.join(checkpoint_directory, 'final_model_{}.bin'.format(load_checkpoint_epoch))
    
    
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
        
    obs_image_keys = [
        '/observations/images/cam_high_left',
        '/observations/images/cam_high_right',
        ]
    
    
    data_stats_save_path = './assets/dataset_stats.json'
    with open(data_stats_save_path, 'r', encoding='utf-8') as f:
        data_stats_dic = json.load(f)
    
    data_stats_dic = scale_stats_values(data_stats_dic, factor=conf.stats_scale_factor)
    
    
    action_smoother = ActionSmoother(window_size=3, mode='gauss', sigma=2)
    
    image_preprocess_func = build_img_preprocess(
        crop_size=(480, 480),
        crop_jitter_max=(0, 0),
        resize_size=(img_size, img_size),
        train_flag=False
        )
    
    
    state_action_normer = StateActionNorm(
        min_val=data_stats_dic['action']['min'],   # list, len=7
        max_val=data_stats_dic['action']['max'], 
        device=device,
        target_range='-1_1'       #  0_1 , -1_1
        )
    
    generator = torch.Generator(device=device).manual_seed(opt.seed)
    
    model = initialize_model(conf)

    model.load_state_dict(torch.load(load_model_path))
    model.eval()
    model = model.to(device)
    
    
    stop_event = threading.Event()
    
    def on_press(key):
        if key == keyboard.Key.space:
            print("\n Stop...")

            stop_event.set()


    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    print("Press the [Spacebar] to stop in the loop...")
    
    policy_infer = ModelInfer()
    
    for i in range(25):
        stop_event.clear()
        
        policy_infer.infer_loop()
        
        input('Enter  Next:')
        
    listener.stop()
    listener.join()
    
    
    
    