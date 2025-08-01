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

class MazeViTUTModelACT(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=3, hidden_dim=128, max_steps=100,
                 nhead=4, mlp_dim=512, ponder_epsilon=0.01, out_channels=2):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim))

        self.block = ViTBlock(hidden_dim, nhead, mlp_dim)
        self.halt_layer = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.halt_layer.bias, 0)  # 초기값

        self.max_steps = max_steps
        self.ponder_epsilon = ponder_epsilon
        self.out_channels = out_channels

        self.decoder_upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 4, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        B, C, H_orig, W_orig = x.shape
        x = self.patch_embed(x)  # (B, hidden_dim, H, W)
        H_patch, W_patch = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, hidden_dim)
        x = x + self.pos_embed[:, :x.size(1), :]

        accumulated_halting = torch.zeros(B, device=x.device)
        accumulated_x = torch.zeros_like(x)
        ponder_cost = torch.zeros(B, device=x.device)
        stopped_at_step = torch.full((B,), -1, dtype=torch.long, device=x.device)

        for step in range(self.max_steps):
            x = self.block(x)

           # p = torch.sigmoid(self.halt_layer(x[:, 0])).squeeze(-1)
            p = torch.sigmoid(self.halt_layer(x).mean(dim=1)).squeeze(-1)


            still_running = (accumulated_halting < 1.0 - self.ponder_epsilon)

            stopping_this_step = still_running & ((accumulated_halting + p) >= (1.0 - self.ponder_epsilon))
            remainder = torch.where(stopping_this_step, 1.0 - accumulated_halting, p)
            effective_weight = torch.where(still_running, remainder, torch.zeros_like(p))

            accumulated_x += x * effective_weight[:, None, None]
            
            ponder_cost += torch.where(still_running, 1.0 - p, torch.zeros_like(p))
            accumulated_halting += torch.where(still_running, p, torch.zeros_like(p))

            for b in range(B):
                if stopping_this_step[b] and stopped_at_step[b] == -1:
                    stopped_at_step[b] = step

            if (accumulated_halting >= 1.0 - self.ponder_epsilon).all():
                break

        remaining = (accumulated_halting < 1.0 - self.ponder_epsilon)
        if remaining.any():
            final_remainder = 1.0 - accumulated_halting
            accumulated_x += x * final_remainder[:, None, None]
            for b in range(B):
                if remaining[b] and stopped_at_step[b] == -1:
                    stopped_at_step[b] = self.max_steps - 1

        avg_ponder_cost = torch.mean(ponder_cost)
        x_final = accumulated_x / (accumulated_halting[:, None, None] + 1e-6)

        decoded_latent = x_final.transpose(1, 2).reshape(B, -1, H_patch, W_patch)
        
        decoded_intermediate = self.decoder_upsample(decoded_latent)
        final_output = F.interpolate(decoded_intermediate, size=(H_orig, W_orig), mode='bilinear', align_corners=False)

        self.last_num_steps = stopped_at_step.float().mean().item()
        self.stopped_at_step = stopped_at_step.detach()
        return final_output, avg_ponder_cost