import numpy as np
import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps,get_timestep_embedding
import math
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Mlp


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 8):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # shape: [max_len, 1]
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0., **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.norm0 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn0 = Attention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=dropout,
        )
        
        self.attn1 = Attention(
            hidden_size,
            cross_attention_dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=dropout,
        )
        
        self.attn2 = Attention(
            hidden_size,
            cross_attention_dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=dropout,
        )
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        # TODO 
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 5 * hidden_size, bias=True)  # 5 OR 7
        )
    
    
    # ### NOT use gate 
    def forward(self, x, emb_t, emb_c=None, emb_q=None, mask=None, cond_mask=None):
        shift_msa, scale_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(emb_t).chunk(5, dim=1)
        
        # self attn
        norm_x = self.norm0(x)
        x = self.attn0(norm_x, attention_mask=mask) + x
        
        # cross attn
        xq = modulate(self.norm1(x), shift_msa, scale_msa)
        
        x1 = self.attn1(xq,
                        emb_c,
                        attention_mask=cond_mask
                        )
        
        x2 = self.attn2(xq,
                        emb_q,
                        attention_mask=cond_mask
                        )
        x = x + x1 + x2
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class CondDiT(nn.Module):
    def __init__(self, 
                 input_dim,
                 input_len,
                 obs_dim,
                 num_blocks=6,
                 conv_kernel_size=3,
                 num_norm_groups=8,
                 num_attn_heads=8,
                 self_attn_masks=None,
                 ):
        super().__init__()

        self.timestep_emb = Timesteps(obs_dim, flip_sin_to_cos=False, downscale_freq_shift=1)
        self.timestep_proj = nn.Sequential(
            nn.Linear(obs_dim, obs_dim * 4),
            nn.SiLU(),
            nn.Linear(obs_dim * 4, obs_dim),
        )
        
        self.action_time_emb = nn.Parameter(
            get_timestep_embedding(torch.arange(input_len), input_dim).reshape(1, input_len, input_dim)
            )
        

        self.cond_norm = nn.LayerNorm(obs_dim)
        
        self.qpos_cond_norm = nn.LayerNorm(obs_dim)

        self.conv_in = nn.Conv1d(input_dim, 
                                 obs_dim, 
                                 kernel_size=conv_kernel_size,
                                 padding=conv_kernel_size // 2
                                 )
        
        attn_modules = []
        for i in range(num_blocks):
            
            attn_block = DiTBlock(
                obs_dim,
                num_heads=num_attn_heads,
            )
            
            attn_modules.append(attn_block)
        
        self.attn_modules = nn.ModuleList(attn_modules)
        
        self.layer_norm_out = nn.LayerNorm(obs_dim)
        self.activation_out = nn.Mish()
        self.conv_out = nn.Conv1d(obs_dim, input_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2)


        # initialize weights
        self.apply(self._init_weights)

    def forward(self, sample, timesteps, cond, qpos_cond):
        
        sample = sample + self.action_time_emb
        
        sample = sample.permute(0, 2, 1)
        
        if not torch.is_tensor(timesteps):
            # this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0]) 

        t_emb = self.timestep_emb(timesteps)  
        t_emb = t_emb.to(dtype=sample.dtype)
        t_emb = self.timestep_proj(t_emb)  

        cond = cond.flatten(1, -2)  # B, L, D 

        cond = self.cond_norm(cond)
        
        qpos_cond = qpos_cond.flatten(1, -2)  # B, L, D 
        qpos_cond = self.qpos_cond_norm(qpos_cond)
        
        x = self.conv_in(sample)  
        
        x = x.permute(0, 2, 1)              # channel last (B, C, L) -> (B, L, D)
        
        # TODO
        for block in self.attn_modules:
            x = block(x, t_emb, cond, qpos_cond) 
            
        x = self.layer_norm_out(x)
        
        # [128, 8, 512] -> [128, 512, 8]
        x = x.permute(0, 2, 1)
        
        x = self.activation_out(x)
        x = self.conv_out(x)

        
        return x.permute(0, 2, 1)    # B, Ta, Da

    def _init_weights(self, module):
        init_std = 0.02
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.trunc_normal_(module.weight, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            