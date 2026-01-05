import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== BasicBlock for ResNet-style CNN Encoder ======
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1):
        super().__init__()
        padding = dilation  # 3x3 + stride=1일 때 output size 유지용
        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, 
            padding=padding, dilation=dilation, bias=False
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, 
            padding=padding, dilation=dilation, bias=False)
        
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
    def __init__(self, input_channels=3, hidden_dim=128, dilation=1):
        super().__init__()
        padding = dilation 
        width = hidden_dim // 64
        self.conv1 = nn.Conv2d(
            input_channels, 64 * width, kernel_size=3, 
            stride=1, padding=padding, dilation=dilation, bias=False
        )
        self.dropout1 = nn.Dropout2d(p=0.1)
        self.block1 = BasicBlock(64 * width, 64 * width, dilation)
        self.block2 = BasicBlock(64 * width, 64 * width, dilation)

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
    def __init__(self, input_channels=3, hidden_dim=128, max_steps=4, nhead=4, height=24, width=24, dilation=1):
        super().__init__()
        padding = dilation
        
        self.encoder = UnifiedEncoder(input_channels, hidden_dim, dilation)
        self.transformer = SharedTransformerBlock(hidden_dim, nhead)  # Transformer Block (shared)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=1, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, stride=1, padding=padding, dilation=dilation),
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
    
    def _make_pos_embed(self, H: int, W: int, *, device, dtype) -> torch.Tensor:
        """
        정규화된 2D sin-cos 위치임베딩 생성.
        Returns: (1, H*W, hidden_dim)
        """
        pos = get_2d_normalized_sincos_pos_embed(self.hidden_dim, H, W)  # [H*W, D]
        pos = pos.to(device=device, dtype=dtype).unsqueeze(0)            # [1, H*W, D]
        return pos
    
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
#         pos = self._make_pos_embed(H, W, device=x.device, dtype=x.dtype)  # (1, H*W, D)
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


def get_1d_sincos_pos_embed_from_u(embed_dim: int,
                                   u: torch.Tensor,
                                   temperature: float = 10000.0) -> torch.Tensor:
    """
    1D sin-cos positional embedding from a vector of normalized coords u in [0,1].

    Args:
        embed_dim: dimension of the embedding (must be even). 최종 임베딩 차원
        u: [N] or [N,] tensor of coordinates in [0,1].
        temperature: base for frequency scaling (like in Vaswani).

    Returns:
        pos_embed: [N, embed_dim]
    """
    # sin과 cos를 반반 이어 붙이므로, 최종 차원 embed_dim은 짝수여야 한다는 검사.
    assert embed_dim % 2 == 0, "embed_dim must be even"
    u = u.float().unsqueeze(-1)  # [N, 1]: 이후 브로드캐스팅을 쉽게 하려는 준비

    dim_half = embed_dim // 2   # sin용 절반, cos용 절반
    # frequencies like in standard sin/cos PE
    freq = torch.arange(dim_half, dtype=torch.float32, device=u.device)
    freq = 1.0 / (temperature ** (freq / dim_half))  # [dim_half]

    # angles: [N, dim_half]
    angles = u * freq  # broadcast

    sin = torch.sin(angles)
    cos = torch.cos(angles)
    pos_embed = torch.cat([sin, cos], dim=-1)  # [N, dim_half*2 = embed_dim]
    return pos_embed


def get_2d_normalized_sincos_pos_embed(embed_dim: int,
                                       H: int,
                                       W: int,
                                       temperature: float = 10000.0) -> torch.Tensor:
    """
    2D sin-cos positional embedding for a HxW maze, using normalized coords.

    We split embed_dim into half for the x-axis (row) and half for the y-axis (col),
    like ViT-style 2D encoding.

    Args:
        embed_dim: total embedding dim (must be even).
        H, W: maze height and width.
        temperature: frequency base.

    Returns:
        pos_embed: [H*W, embed_dim] (can be reshaped to [H, W, C] or [1, H*W, C])
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    dim_each = embed_dim // 2

    # ---- normalized coordinates in [0,1] ----
    if H > 1:
        u = torch.linspace(0., 1., steps=H)  # rows
    else:
        u = torch.tensor([0.0])
    if W > 1:
        v = torch.linspace(0., 1., steps=W)  # cols
    else:
        v = torch.tensor([0.0])

    # 1D embeddings for u and v
    pos_u = get_1d_sincos_pos_embed_from_u(dim_each, u, temperature)  # [H, dim_each]
    pos_v = get_1d_sincos_pos_embed_from_u(dim_each, v, temperature)  # [W, dim_each]

    # combine into 2D grid:
    #  - For each (x,y), concat pos_u[x] and pos_v[y]
    #  - Result: [H, W, embed_dim]
    pos_u = pos_u[:, None, :].expand(H, W, dim_each)   # [H, W, dim_each]
    pos_v = pos_v[None, :, :].expand(H, W, dim_each)   # [H, W, dim_each]
    pos = torch.cat([pos_u, pos_v], dim=-1)            # [H, W, embed_dim]

    pos = pos.view(H * W, embed_dim)
    return pos