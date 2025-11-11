import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return self.dropout(out)

    
class UnifiedEncoder(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=128):
        super().__init__()
        width = hidden_dim // 64
        self.conv1 = nn.Conv2d(input_channels, 64 * width, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout1 = nn.Dropout2d(p=0.1)
        self.block1 = BasicBlock(64 * width, 64 * width)
        self.block2 = BasicBlock(64 * width, 64 * width)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # not inplace
        x = self.dropout1(x)
        x = self.block1(x)
        x = self.block2(x)
        return x

    
class SharedTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, nhead=4):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True,
                                               dropout=0.1)

    def forward(self, x):
        return self.block(x)

    
class MazeUTModelACT(nn.Module):
    def __init__(
        self, input_channels=3, hidden_dim=128, max_steps=10, 
        nhead=4, height=32, width=32, out_channels=2, ponder_epsilon=0.01, 
        time_penalty=0.01
    ):
        super().__init__()
        self.encoder = UnifiedEncoder(input_channels, hidden_dim)
        self.transformer = SharedTransformerBlock(hidden_dim, nhead)
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, 2, 1)
        )

        self.hidden_dim = hidden_dim
        self.max_iters = max_steps
        self.ponder_epsilon = ponder_epsilon
        self.time_penalty = time_penalty

        self.sigmoid = nn.Sigmoid()
        self.halt_fc = nn.Linear(hidden_dim, 1)

        self.last_num_steps = 0
        self.stopped_at_step = None
        self.weighted_output_history = None
        
        # Positional Encoding (Learnable) 
        # self.pos_embed = nn.Parameter(torch.randn(1, height * width, hidden_dim))
        # self.base_h, self.base_w = height, width  # 기준 크기(예: 24, 24) 저장

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

    def forward(self, x):
        B, _, H_img, W_img = x.size()
        device = x.device

        # 1) CNN Encoder: 특징 추출 (출력 크기: B, hidden_dim, H, W)
        # UnifiedEncoder 통해 convolutional feature map 생성
        x = self.encoder(x)
        # 업데이트된 feature map의 height, width 가져오기
        H, W = x.shape[2], x.shape[3]

        # 2) Flatten + Positional Encoding 추가
        # (B, hidden_dim, H, W) → (B, H*W, hidden_dim)
        x = x.flatten(2).permute(0, 2, 1)
        # 위치 임베딩 pos_embed를 더해 Transformer 입력 준비
        # pos = self._resize_pos_embed(H, W).to(x.dtype).to(x.device)  # 현재 H×W에 맞춘 pos_embed 생성
#         pos = self.pos_embed[:, :H*W, :].to(x.device)
        pos = self._make_pos_embed(H, W, device=x.device, dtype=x.dtype)  # (1, H*W, D)
        x = x + pos  # pos 추가 (길이 맞춰 슬라이스)

        # 3) ACT(Adaptive Computation Time) 초기 변수 설정
        halting_prob = torch.zeros(B, H * W, device=device)        # 누적 halting 확률
        remainders   = torch.zeros(B, H * W, device=device)        # 남은 확률(remainder)
        n_updates    = torch.zeros(B, H * W, device=device)        # 각 위치별 업데이트 횟수
        weighted_output = torch.zeros(B, 2, H, W, device=device)   # 최종 누적 출력 (decoder 결과 가중합)
        still_running  = torch.ones(B, H * W, device=device, dtype=torch.bool)  # 아직 멈추지 않은 위치 표시

        self.weighted_output_history = []  # 각 step별 출력을 저장하는 리스트 (디버깅/분석용)

        # 4) 반복적으로 Transformer + halting mechanism 실행
        for step in range(self.max_iters):  # 최대 max_iters(예: 10)번 반복
            x = self.transformer(x)  # Transformer Encoder Block 통과 → (B, H*W, hidden_dim)
            p = self.sigmoid(self.halt_fc(x)).squeeze(-1)  # 각 위치별 halting 확률 p_t 계산 (B, H*W)
            p = torch.where(still_running, p, torch.zeros_like(p))  # 이미 멈춘 위치는 확률 0으로 처리

            # 이번 step에서 새로 멈출지 결정 (누적 halting_prob + p > 1 - epsilon 이면 멈춤)
            new_halted = (halting_prob + p * still_running.float() > 1 - self.ponder_epsilon) & still_running
            still_running = still_running & ~new_halted  # 멈춘 위치는 still_running에서 제거

            # 업데이트 가중치 계산
            update_weights = torch.where(
                new_halted,
                (1 - halting_prob) / (p + 1e-8),  # 마지막 스텝에서는 남은 확률을 맞춰줌
                torch.ones_like(p)                 # 그 외에는 그냥 p 그대로 사용
            )
            update_weights = update_weights * p  # 최종 update 가중치 w_t

            # 누적 halting 확률 업데이트
            halting_prob = halting_prob + update_weights
            # remainder 업데이트 (halted일 경우 남은 확률 저장)
            remainders = torch.where(new_halted, 1 - halting_prob, remainders)
            # 업데이트 횟수 증가 (아직 running이거나 새로 멈춘 경우)
            n_updates = n_updates + still_running.float() + new_halted.float()

            # 5) Decoder CNN을 거쳐 pixel-wise 출력 생성
            x_reshaped = x.permute(0, 2, 1).reshape(B, self.hidden_dim, H, W)  # (B, hidden_dim, H, W) 복원
            out = self.decoder_conv(x_reshaped)                                # (B, 2, H, W) → 2-class 예측
            self.weighted_output_history.append(out)                           # 히스토리에 저장

            # 6) 가중합 업데이트
            weighted_output += out * update_weights.view(B, 1, H, W)

            # 모든 위치가 멈췄으면 조기 종료
            if still_running.sum() == 0:
                break

        # 7) 로그/모니터링용 변수 저장
        self.last_num_steps = n_updates.mean().item()                   # 평균 업데이트 횟수
        self.stopped_at_step = n_updates.view(B, H, W).mean(dim=(1, 2)) # 배치별 평균 스텝
        ponder_cost = n_updates + remainders
        avg_ponder_cost = ponder_cost.mean()                            # 전체 평균 ponder cost

        return weighted_output, avg_ponder_cost


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
    u = u.float().unsqueeze(-1)  # [N, 1]

    dim_half = embed_dim // 2
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