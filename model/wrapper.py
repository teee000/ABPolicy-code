import torch
import torch.nn as nn
from diffusers.models.embeddings import get_2d_sincos_pos_embed
from einops import rearrange
import torchvision

from model.resnet import wrap_resnet


class ResNetEncoderWrapper(nn.Module):
    def __init__(self, resnet, pooled=True, out_dim=512):
        super().__init__()
        
        ## center crop camera images
        image_crop_shape = (224, 224)   ### fixed 224
        self.center_crop = torchvision.transforms.CenterCrop(image_crop_shape)

        self.model = wrap_resnet(resnet, out_dim=out_dim)
        self.pooled = pooled
        self.linear = nn.Linear(resnet.config.hidden_sizes[-1], out_dim) if not pooled else nn.Identity()

    def forward(self, x):
        ##  腕部和第三视角相机分别调用 encoder
        ## print('ResNetEncoderWrapper input x:', x.shape)  ## torch.Size([64, 3, 480, 640])  64 = B*N*T    
        
        x = self.center_crop(x)
        # print('ResNetEncoderWrapper center_crop x:', x.shape)  ## torch.Size([64, 3, 224, 224])
        
        output = self.model(x)

        if self.pooled:
            output = output.pooler_output.unsqueeze(1) # B, 1, D
        else:
            B, C, H, W = output.last_hidden_state.shape
            pos_emb = get_2d_sincos_pos_embed(embed_dim=C, grid_size=(H, W))
            pos_emb = torch.tensor(pos_emb, device=x.device, dtype=x.dtype).unsqueeze(0)
            output = rearrange(output.last_hidden_state, 'B C H W -> B (H W) C')
            output = output + pos_emb

        return self.linear(output)


class DinoV2EncoderWrapper(nn.Module):
    def __init__(self, dino, pooled=False, out_dim=512):
        super().__init__()

        self.model = dino
        self.pooled = pooled
        if pooled:
            self.linear = nn.Linear(dino.config.hidden_size*2, out_dim)
        else:
            self.linear = nn.Linear(dino.config.hidden_size, out_dim)

    def forward(self, x):
        output = self.model(x)

        if self.pooled:
            cls_token = output.last_hidden_state[:, :1]
            avg_token = output.last_hidden_state[:, 1:].mean(dim=1, keepdim=True)

            output = torch.cat([cls_token, avg_token], dim=-1)  # B, 1, 2*D
        else:
            output = output.last_hidden_state[:, 1:]    # B, L, D
        
        return self.linear(output)
