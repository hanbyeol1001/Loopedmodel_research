import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTBlock(nn.Module):
    def __init__(self, hidden_dim, nhead=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        return x

class MazeViTUTModel(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, hidden_dim=128, max_steps=4, nhead=4, mlp_dim=512, act=False, out_channels=2):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        
        self.img_size = img_size 
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim 
        
        num_patches = (img_size // patch_size) ** 2 
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim))
        
        self.blocks = nn.ModuleList([ViTBlock(hidden_dim, nhead, mlp_dim)])
        self.max_steps = max_steps
        self.act = act
        if act:
            self.halt_layer = nn.Linear(hidden_dim, 1)

        self.out_channels = out_channels
        self.decoder_upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 4, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x, return_all_steps=False):
        B, C, H_orig, W_orig = x.shape 

        x = self.patch_embed(x)  # (B, hidden_dim, H_orig/P, W_orig/P)
        H_patch, W_patch = x.shape[2], x.shape[3] # 패치 임베딩 후의 해상도
        x = x.flatten(2).transpose(1, 2)  # (B, N, hidden_dim) (N = H_patch * W_patch)
        if x.size(1) > self.pos_embed.size(1):
            raise ValueError(f"Number of patches ({x.size(1)}) exceeds pre-allocated positional embeddings ({self.pos_embed.size(1)}). "
                             f"Consider increasing img_size or patch_size in __init__, or implementing positional embedding interpolation.")
        x = x + self.pos_embed[:, :x.size(1), :]

        thoughts = []
        halt_prob = torch.zeros(B, device=x.device)
        total_steps = torch.zeros(B, device=x.device)
        accumulated_x = torch.zeros_like(x)

        for step in range(self.max_steps):
            for block in self.blocks:
                x = block(x)
            
            # (B, N, hidden_dim) -> (B, hidden_dim, H_patch, W_patch)
            decoded_latent = x.transpose(1, 2).reshape(B, self.hidden_dim, H_patch, W_patch) 
            
            # ConvTranspose2d 스택을 통해 업샘플링
            decoded_intermediate = self.decoder_upsample(decoded_latent)
            
            final_output = F.interpolate(decoded_intermediate, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
            
            thoughts.append(final_output)

            final_decoded = thoughts[-1] # 마지막 스텝의 최종 업샘플링된 결과

        if return_all_steps:
            # 모든 스텝의 예측을 쌓아 반환 (steps, B, out_channels, H_orig, W_orig)
            return torch.stack(thoughts)  
        else:
            return final_decoded # 마지막 스텝의 예측 반환
