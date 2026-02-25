import torch
import torch.nn as nn
import math
from einops import rearrange
# from model.diffusion.blocks import TransformerBlock


# class CausalObservationPerceiver(nn.Module):
#     def __init__(self, dim, obs_dim, obs_horizon, layers=1, dropout=0.):
#         super().__init__()

#         self.latents = nn.Parameter(torch.randn(1, obs_horizon, dim))

#         self.obs_norm = nn.LayerNorm(obs_dim)

#         self.x_attn = TransformerBlock(dim, cond_dim=obs_dim, dropout=dropout, cross_attn_only=True)
#         self.blocks = nn.ModuleList([
#             TransformerBlock(dim, dropout=dropout) for _ in range(layers)
#         ])

#     def forward(self, obs_emb):
#         B, N, T, L, D = obs_emb.shape

#         obs_emb = obs_emb.flatten(1, -2)
#         obs_emb = self.obs_norm(obs_emb)

#         mask = torch.ones(T, T, dtype=torch.bool).tril()
#         mask = mask.unsqueeze(0).expand(B, T, T).to(device=obs_emb.device)
#         cond_mask = mask.reshape(B, T, T, 1).repeat(1, 1, N, L).reshape(B, T, N*T*L)

#         latents = self.latents.expand(B, T, D)
#         latents = self.x_attn(latents, obs_emb, cond_mask=cond_mask)
        
#         for block in self.blocks:
#             latents = block(latents, mask=mask)

#         return latents


class ImgObsPerceiver(nn.Module):
    """
    使用 einops.rearrange 处理 (B, N, C) 特征的模块。
    1. 将输入重塑为 (B, C, H, W) 的图像格式。
    2. 使用三层 3x3 卷积进行处理。
    3. 将输出重塑回 (B, N', C') 的序列格式。
    """
    def __init__(self, initial_n: int, in_channels: int, out_channels: int, mid_channels: int = None):
        """
        初始化模块。

        参数:
            initial_n (int): 输入特征的序列长度 N，例如 324。
            in_channels (int): 输入特征的通道数 C，例如 512。
            out_channels (int): 最终输出特征的通道数 C'。
            mid_channels (int): 中间卷积层的通道数。如果为 None，则默认为 in_channels。
        """
        super().__init__()

        # --- 合法性检查与尺寸计算 ---
        h = int(math.sqrt(initial_n))
        if h * h != initial_n:
            raise ValueError(f"输入序列长度 N ({initial_n}) 必须是一个完全平方数，以便转换为 HxW 网格。")
        
        # 将 H 和 W 存储起来，rearrange 会用到
        self.h = h
        self.w = h
        
        if mid_channels is None:
            mid_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # nn.Conv2d(mid_channels, out_channels, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x_reshaped = rearrange(x, 'b (h w) c -> b c h w', h=self.h, w=self.w)

        conv_output = self.conv_layers(x_reshaped)
        
        final_output = rearrange(conv_output, 'b c h w -> b (h w) c')
        
        return final_output

# --- 示例用法 ---
if __name__ == '__main__':
    # 定义输入参数
    BATCH_SIZE = 16
    SEQUENCE_LENGTH_N = 324  # 18 * 18
    CHANNELS_C = 512
    
    # 创建模型实例
    processor = ImgObsPerceiver(
        initial_n=SEQUENCE_LENGTH_N,
        in_channels=CHANNELS_C,
        out_channels=CHANNELS_C,
        mid_channels=CHANNELS_C,
    )
    
    # 创建一个虚拟的输入张量
    dummy_input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH_N, CHANNELS_C)
    print(f"输入 Tensor 形状: {dummy_input.shape}")
    
    # 将输入传递给模型
    output = processor(dummy_input)
    
    # 打印输出形状进行验证 (预期输出形状: (16, 25, 512))
    print(f"输出 Tensor 形状: {output.shape}")
