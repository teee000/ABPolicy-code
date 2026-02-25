import torch
from diffusers.models.embeddings import get_timestep_embedding
from torch import nn


class JointMLP(nn.Module):
    def __init__(self,
                 dim_in: int,          # D_in
                 dim_hidden: int = 256,
                 dim_out: int = 256,
                 norm_type: str = "layer"   # 'layer' | 'batch' | 'group'
                 ):
        super().__init__()

        # 根据 norm_type 选择不同的归一化层
        if norm_type == "batch":
            Norm = lambda dim: nn.BatchNorm1d(dim)   
        elif norm_type == "group":

            Norm = lambda dim: nn.GroupNorm(num_groups=8, num_channels=dim)
        elif norm_type == "layer":
            Norm = lambda dim: nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            Norm(dim_hidden),
            nn.ReLU(inplace=True),

            nn.Linear(dim_hidden, dim_out),
            Norm(dim_out)
        )

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.norm_type = norm_type

    def forward(self, x):
        B, T, D = x.shape
        assert D == self.dim_in, "dim_in not align"

        x = x.view(B * T, D)         # (B*T, D_in)
        x = self.mlp(x)              # (B*T, D_out)
        x = x.view(B, T, self.dim_out)
        return x







class CAGE(nn.Module):
    def __init__(self, 
                 obs_encoders, 
                 perceiver, 
                 backbone,
                 obs_dim=512, 
                 obs_horizon=1, 
                 obs_num=1,
                 obs_image_keys=[],
                 ):
        super().__init__()

        
        
        self.qpos_time_emb = nn.Parameter(
            get_timestep_embedding(torch.arange(obs_horizon), obs_dim).reshape(1, obs_horizon, obs_dim)
            )
        self.obs_encoders = obs_encoders
        self.perceiver = perceiver
        self.backbone = backbone
        
        self.qpos_encoder = JointMLP(
            dim_in=7,
            dim_hidden=256,
            dim_out=obs_dim,
            norm_type="layer"
            )
        
        self.obs_image_keys = obs_image_keys

    def preprocess_obs(self, obs_dict, obs_type = 'cam_high'):
        all_obs_lst = []
        img_encoder = self.obs_encoders[obs_type]
        
        qpos = obs_dict['/observations/qpos']
        
        for img_key in self.obs_image_keys:
            obs = obs_dict[img_key]
            B, N = obs.shape[:2]
            for i in range(N):   # N : camera history num 
                emb = img_encoder(obs[:, i])   # B,N,C   

            all_obs_lst.append(emb)  
            
        all_obs_emb = torch.cat(all_obs_lst, dim=-1)        
        obs_emb = self.perceiver(all_obs_emb)   
        
        qpos_emb = self.qpos_encoder(qpos)  
        qpos_emb = qpos_emb + self.qpos_time_emb
        
        return obs_emb, qpos_emb

    def forward(self,timesteps, noisy_actions, obs_emb=None, qpos_emb=None, obs_dict=None):
        
        if obs_emb is None:
            obs_emb, qpos_emb = self.preprocess_obs(obs_dict)
            
        return self.backbone(noisy_actions, timesteps, cond=obs_emb, qpos_cond=qpos_emb)
    
    
    def get_optim_groups(self, conf):
        params = []
        '''
        obs_encoders
        perceiver
        backbone
        qpos_encoder
        '''
        
        perceiver_params = {
            "params": [p for n, p in self.named_parameters() if n.startswith(("qpos_encoder", 
                                                                              "perceiver"))],
            "weight_decay": conf.weight_decay,
            'lr': conf.learning_rate_perceiver,
            'betas':conf.betas_perceiver,
        }
        
        backbone_params = {
            "params": [p for n, p in self.named_parameters() if n.startswith('backbone')],
            "weight_decay": conf.weight_decay,
            'lr': conf.learning_rate,
            'betas':conf.betas,
        }
        
        # LoRA params of pretrained obs encoder
        obs_params = {
            'params': [p for p in self.obs_encoders.parameters() if p.requires_grad],
            'weight_decay': conf.weight_decay,
            'lr': conf.learning_rate,
            'betas':conf.betas,
        }
        
        params.append(obs_params)
        params.append(perceiver_params)
        params.append(backbone_params)
        
        return params

