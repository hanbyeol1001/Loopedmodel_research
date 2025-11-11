import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== BasicBlock for ResNet-style CNN Encoder ======
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return self.dropout(out)

# ====== Unified CNN Encoder with Dropout ======
class UnifiedEncoder(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=128):
        super().__init__()
        width = hidden_dim // 64
        self.conv1 = nn.Conv2d(input_channels, 64 * width, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout1 = nn.Dropout2d(p=0.1)
        self.block1 = BasicBlock(64 * width, 64 * width)
        self.block2 = BasicBlock(64 * width, 64 * width)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.block1(x)
        return self.block2(x)

# ====== Shared Transformer Block (repeated steps) ======
class SharedTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, nhead=4):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True, dropout=0.1
        )

    def forward(self, x):
        return self.block(x)

# ====== MazeUTModel: CNN Encoder + Transformer + Linear Decoder ======
class MazeUTModel(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=128, max_steps=4, nhead=4, height=24, width=24):
        super().__init__()
        self.encoder = UnifiedEncoder(input_channels, hidden_dim)
        self.transformer = SharedTransformerBlock(hidden_dim, nhead)  # Transformer Block (shared)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, 2, kernel_size=1)
        )
        self.hidden_dim = hidden_dim
        self.max_iters=max_steps

        # Positional Encoding (Learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, height * width, hidden_dim))
        self.base_h, self.base_w = height, width  # 기준 크기(예: 24, 24) 저장
    
    def _resize_pos_embed(self, H, W):
        """
        (1, HW0, D) 형태의 learnable pos_embed를 (1, H*W, D)로 2D 보간하여 리턴
        """
        # (1, HW0, D) -> (1, D, base_h, base_w)
        pos2d = self.pos_embed.view(1, self.base_h * self.base_w, self.hidden_dim)
        pos2d = pos2d.view(1, self.base_h, self.base_w, self.hidden_dim).permute(0, 3, 1, 2)
        # 2D 보간 (bicubic 권장)
#         pos2d = F.interpolate(pos2d, size=(H, W), mode="bicubic", align_corners=False)
        pos2d = F.interpolate(pos2d, size=(H, W), mode="nearest")
        # (1, D, H, W) -> (1, H*W, D)
        pos_new = pos2d.permute(0, 2, 3, 1).contiguous().view(1, H * W, self.hidden_dim)
        return pos_new

    def forward(self, x, return_all_steps=False):
        B = x.size(0)

        # 1) CNN Encoder: (B, hidden_dim, H, W)
        x = self.encoder(x)
        H, W = x.shape[2], x.shape[3]

        # 2) Flatten and Add Positional Encoding: (B, H*W, hidden_dim)
        x = x.flatten(2).permute(0, 2, 1)     # (B, C, H*W) → (B, H*W, C) = (B, N, D)
        # 현재 H×W에 맞춘 pos_embed 생성
        pos = self._resize_pos_embed(H, W).to(x.dtype).to(x.device)
#         pos = self.pos_embed[:, :H*W, :].to(x.device)
        x = x + pos  # pos 추가 (길이 맞춰 슬라이스)

        thoughts = []

        # 3) Iterative Transformer Steps (UT)
        for _ in range(self.max_iters):
            x = self.transformer(x)  # shared block 1회 적용
            # decoded = self.decoder(x)  # (B, H*W, 2)
            # decoded = decoded.view(B, H, W, 2).permute(0, 3, 1, 2)  # (B, 2, H, W)

            # 오류 수정
            # x: (B, N, D) = (B, H*W, hidden_dim)
            x_map = x.permute(0, 2, 1).view(B, self.hidden_dim, H, W)  # (B, D, H, W)
            decoded = self.decoder(x_map)        # (B, 2, H, W)

            thoughts.append(decoded)

        all_thoughts = torch.stack(thoughts)  # (steps, B, 2, H, W)

        return all_thoughts if return_all_steps else all_thoughts[-1]