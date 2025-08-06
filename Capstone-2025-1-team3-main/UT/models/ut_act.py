import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        return F.relu(out)

class UnifiedEncoder(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=128):
        super().__init__()
        width = hidden_dim // 64
        self.conv1 = nn.Conv2d(input_channels, 64 * width, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = BasicBlock(64 * width, 64 * width)
        self.block2 = BasicBlock(64 * width, 64 * width)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # not inplace
        x = self.block1(x)
        x = self.block2(x)
        return x

class SharedTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, nhead=4):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)

    def forward(self, x):
        return self.block(x)

class MazeUTModelACT(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=128, max_steps=10, nhead=4, height=32, width=32, out_channels=2, ponder_epsilon=0.01, time_penalty=0.01):
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
        self.height = height
        self.width = width
        self.max_iters = max_steps
        self.ponder_epsilon = ponder_epsilon
        self.time_penalty = time_penalty

        self.sigmoid = nn.Sigmoid()
        self.halt_fc = nn.Linear(hidden_dim, 1)

        self.last_num_steps = 0
        self.stopped_at_step = None
        self.weighted_output_history = None

    def forward(self, x):
        B, _, H, W = x.size()
        device = x.device
        x = self.encoder(x)  # (B, C, H, W)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        pos_embed = torch.randn(1, H * W, self.hidden_dim, device=device)
        x = x + pos_embed

        halting_prob = torch.zeros(B, H * W, device=device)
        remainders = torch.zeros(B, H * W, device=device)
        n_updates = torch.zeros(B, H * W, device=device)
        weighted_output = torch.zeros(B, 2, H, W, device=device)
        still_running = torch.ones(B, H * W, device=device, dtype=torch.bool)

        self.weighted_output_history = []

        for step in range(self.max_iters):
            x = self.transformer(x)  # (B, H*W, hidden_dim)
            p = self.sigmoid(self.halt_fc(x)).squeeze(-1)  # (B, H*W)
            p = torch.where(still_running, p, torch.zeros_like(p))

            new_halted = (halting_prob + p * still_running.float() > 1 - self.ponder_epsilon) & still_running
            still_running = still_running & ~new_halted

            update_weights = torch.where(
                new_halted,
                (1 - halting_prob) / (p + 1e-8),
                torch.ones_like(p)
            )
            update_weights = update_weights * p
            halting_prob = halting_prob + update_weights
            remainders = torch.where(new_halted, 1 - halting_prob, remainders)
            n_updates = n_updates + still_running.float() + new_halted.float()

            x_reshaped = x.permute(0, 2, 1).reshape(B, self.hidden_dim, H, W)
            out = self.decoder_conv(x_reshaped)
            self.weighted_output_history.append(out)

            weighted_output += out * update_weights.view(B, 1, H, W)

            if still_running.sum() == 0:
                break

        self.last_num_steps = n_updates.mean().item()
        self.stopped_at_step = n_updates.view(B, H, W).mean(dim=(1, 2))
        avg_ponder_cost = n_updates.mean()

        return weighted_output, avg_ponder_cost
