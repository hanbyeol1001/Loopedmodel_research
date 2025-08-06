import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm # Assuming tqdm is imported

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class BasicBlock(nn.Module):
    """Basic residual block class"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return self.dropout(out)


class RecurResNetACT(nn.Module):
    """ACT가 적용된 수정된 ResNet 모델 클래스"""

    def __init__(self, block=BasicBlock, num_blocks=[2], depth=44, width=2, ponder_epsilon=0.01, time_penalty=0.01, max_iters=50):
        super(RecurResNetACT, self).__init__()
        self.max_iters = max_iters
        self.ponder_epsilon = ponder_epsilon
        self.time_penalty = time_penalty

        self.in_planes = int(width*64)
        self.conv1 = nn.Conv2d(3, int(width * 64), kernel_size=3,
                               stride=1, padding=1, bias=False)

        # 순환 블록 생성
        layers = []
        for i in range(len(num_blocks)):
            layers.append(self._make_layer(block, int(width*64), num_blocks[i], stride=1))
        self.recur_block = nn.Sequential(*layers)

        # 출력 레이어들
        self.conv2 = nn.Conv2d(int(width*64), 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(8, 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

        # # halting unit 기존
        # self.halting_unit = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),  # Global average pooling
        #     nn.Flatten(),
        #     nn.Linear(int(width*64), 1),
        #     nn.Sigmoid()
        # )
        # 더 복잡하게 수정
        self.halting_unit = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(self.in_planes),
            nn.Linear(self.in_planes, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.weighted_output_history = [] 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        
        # 초기화
        self.weighted_output_history = [] 
        out = F.relu(self.conv1(x))

        # ACT 관련 변수들
        accumulated_halting = torch.zeros(batch_size, device=device)
        weighted_output = torch.zeros(batch_size, 2, x.shape[2], x.shape[3], device=device)
        ponder_cost = torch.zeros(batch_size, device=device)

        # 각 iteration에서의 halting probabilities 저장 (디버깅용)
        halting_probs = []
        # 각 샘플이 멈춘 스텝 인덱스 저장
        stopped_at_step = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        for i in range(self.max_iters):
            # 순환 블록 통과
            out = self.recur_block(out)

            # halting probability 계산
            halting_prob = self.halting_unit(out).squeeze(-1)  # [batch_size]
            halting_probs.append(halting_prob)

            # 현재 iteration의 출력 계산 (thought)
            thought = F.relu(self.conv2(out))
            thought = F.relu(self.conv3(thought))
            current_output = self.conv4(thought)  # [batch_size, 2, H, W]

            # 아직 멈추지 않은 샘플들 식별
            still_computing = (accumulated_halting < 1.0 - self.ponder_epsilon)

            # 이번에 멈출 샘플들 처리
            stopping_this_step = still_computing & (
                (accumulated_halting + halting_prob) >= (1.0 - self.ponder_epsilon)
            )

            # 멈추는 샘플들의 remainder 계산
            remainder = torch.where(
                stopping_this_step,
                1.0 - accumulated_halting,
                halting_prob
            )

            # 계속 계산하는 샘플들은 halting_prob 사용, 멈추는 샘플들은 remainder 사용
            effective_weight = torch.where(still_computing, remainder, 0.0)

            # weighted output 누적
            for b in range(batch_size):
                if effective_weight[b] > 0:
                    weighted_output[b] += effective_weight[b] * current_output[b]
                # 멈추는 샘플의 스텝 기록
                if stopping_this_step[b] and stopped_at_step[b] == -1: # 처음 멈추는 시점만 기록
                    stopped_at_step[b] = i

            self.weighted_output_history.append(weighted_output.detach().clone())
            
            # ponder cost 누적 (계속 계산하는 샘플들에 대해서만)
            ponder_cost += torch.where(still_computing, 1.0 - halting_prob, 0.0)

            # accumulated halting 업데이트
            accumulated_halting += torch.where(still_computing, halting_prob, 0.0)

            # 모든 샘플이 멈췄는지 확인
            if torch.all(accumulated_halting >= 1.0 - self.ponder_epsilon):
                break

        # 마지막으로 멈추지 않은 샘플들 처리 (안전장치)
        remaining_samples = (accumulated_halting < 1.0 - self.ponder_epsilon)
        if torch.any(remaining_samples):
            final_remainder = 1.0 - accumulated_halting
            for b in range(batch_size):
                if remaining_samples[b]:
                    weighted_output[b] += final_remainder[b] * current_output[b]
                    # max_iters까지 멈추지 않은 경우 마지막 스텝으로 기록
                    if stopped_at_step[b] == -1:
                        stopped_at_step[b] = self.max_iters - 1 

        # 최종 ponder cost 계산 (평균)
        avg_ponder_cost = torch.mean(ponder_cost)

        # 디버깅 정보 저장
        self.last_halting_probs = torch.stack(halting_probs, dim=0)  # [num_steps, batch_size]
        self.last_ponder_cost = avg_ponder_cost
        self.last_num_steps = len(halting_probs) # 실제로 진행된 iteration 횟수
        self.stopped_at_step = stopped_at_step # <<<<<<< 추가된 부분

        return weighted_output, avg_ponder_cost

def recur_resnet_act(depth, width, ponder_epsilon=0.01, time_penalty=0.01, max_iters=50):
    """ACT가 적용된 순환 ResNet 생성 함수"""
    return RecurResNetACT(BasicBlock, [2], depth=depth, width=width,
                          ponder_epsilon=ponder_epsilon, time_penalty=time_penalty,
                          max_iters=max_iters)