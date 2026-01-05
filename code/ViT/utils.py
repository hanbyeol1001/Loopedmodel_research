import torch
import torch.nn.functional as F
from models.maze_vitut import MazeViTUTModel
from models.maze_vitutact import MazeViTUTModelACT
from easy_to_hard_data import MazeDataset
from torch.optim import SGD, Adam, AdamW
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
import datetime
import json
import os
import sys

import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader
import torch.utils.data as data

# def get_dataloaders(train_batch_size, test_batch_size, train_maze_size=9, test_maze_size=9, shuffle=True):
#     train_data = MazeDataset("./data", size=train_maze_size, train=True)
#     test_data = MazeDataset("./data", size=test_maze_size, train=False)
#     trainloader = data.DataLoader(train_data, num_workers=4, batch_size=train_batch_size,
#                                    shuffle=shuffle, drop_last=True)
#     testloader = data.DataLoader(test_data, num_workers=4, batch_size=test_batch_size,
#                                    shuffle=False, drop_last=False)
#     return trainloader, testloader


class NumpyMazeDataset(data.Dataset):
    def __init__(self, inputs_path, solutions_path):
        self.inputs = np.load(inputs_path)       # (N, 3, H, W)
        self.solutions = np.load(solutions_path) # (N, H, W)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.solutions[idx], dtype=torch.float32)
        return x, y


def get_dataloaders(train_batch_size,test_batch_size,train_maze_size=9,
    test_maze_size=9,shuffle=True,data_root="../data"):
    # 학습 데이터셋 경로
    train_inputs_path = f"{data_root}/maze_data_train_{train_maze_size}/inputs.npy"
    train_solutions_path = f"{data_root}/maze_data_train_{train_maze_size}/solutions.npy"

    # 테스트 데이터셋 경로
    test_inputs_path = f"{data_root}/maze_data_test_{test_maze_size}/inputs.npy"
    test_solutions_path = f"{data_root}/maze_data_test_{test_maze_size}/solutions.npy"

    # Dataset 객체
    train_data = NumpyMazeDataset(train_inputs_path, train_solutions_path)
    test_data = NumpyMazeDataset(test_inputs_path, test_solutions_path)

    # DataLoader
    trainloader = data.DataLoader(train_data, batch_size=train_batch_size,
                                  shuffle=shuffle, drop_last=True, num_workers=0)
    testloader = data.DataLoader(test_data, batch_size=test_batch_size,
                                 shuffle=False, drop_last=False, num_workers=0)

    return trainloader, testloader

# 모델 선택
def get_model(model, width, depth):
    model = model.lower()
    if model == "maze_vitut":
        return MazeViTUTModel(img_size=64, patch_size=8, in_channels=3, hidden_dim=width*32, max_steps=depth)
    elif model == "maze_vitutact":
        #return MazeViTUTModel(img_size=64, patch_size=8, in_channels=3, act=True, hidden_dim=width*32, max_steps=depth)
        return MazeViTUTModelACT(img_size=64, patch_size=8, in_channels=3, hidden_dim=width*32, max_steps=depth)
    elif model == "maze_ut":
        return MazeUTModel(input_channels=3, hidden_dim=width * 32, max_steps=depth)
    elif model == "recur_resnet":
        return recur_resnet(width, depth)
    elif model == "ff_resnet":
        return ff_resnet(width, depth)
    else:
        raise ValueError(f"Unknown model: {model}")

# 옵티마이저
def get_optimizer(optimizer_name, model, net, lr):
    param_groups = [{'params': [p for n, p in net.named_parameters()]}]
    if optimizer_name == "adam":
        return Adam(param_groups, lr=lr, weight_decay=2e-4)
    elif optimizer_name == "adamw":
        return AdamW(param_groups, lr=lr, weight_decay=1e-2)
    elif optimizer_name == "sgd":
        return SGD(param_groups, lr=lr, momentum=0.9, weight_decay=2e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

# 옵티마이저+스케줄러 클래스
@dataclass
class OptimizerWithSched:
    optimizer: object
    scheduler: object
    warmup: object

# def train_baseline(net, trainloader, optimizer_obj, device): # only ViT
#     net.train()
#     optimizer = optimizer_obj.optimizer
#     lr_scheduler = optimizer_obj.scheduler
#     warmup_scheduler = optimizer_obj.warmup
#     criterion = torch.nn.CrossEntropyLoss(reduction="mean")
#     train_loss, correct, total = 0, 0, 0
    
#     for inputs, targets in tqdm(trainloader, leave=False):
#         inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1).long()

#         optimizer.zero_grad()
        
#         outputs = net(inputs)

#         # ViT or UT structure: use last step's output
#         if isinstance(net, (MazeViTUTModel)) and outputs.dim() == 5:
#             outputs = outputs[-1]

#         B_out, C_out, H_out, W_out = outputs.size()
#         targets_resized = F.interpolate(targets.float(), size=(H_out, W_out), mode='nearest').long().squeeze(1)
#         reshaped_outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, C_out)
#         reshaped_targets = targets_resized.view(-1)
        
#         loss = criterion(reshaped_outputs, reshaped_targets)
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item() * reshaped_targets.size(0)
#         total += reshaped_targets.size(0)
#         predicted = outputs.argmax(1)
#         correct += (predicted == targets_resized).sum().item()
    
#     train_loss /= total
#     acc = 100.0 * correct / total
#     lr_scheduler.step()
#     warmup_scheduler.dampen()

#     return train_loss, acc

def train_baseline(net, trainloader, optimizer_obj, device):
    net.train()
    optimizer = optimizer_obj.optimizer
    lr_scheduler = optimizer_obj.scheduler
    warmup_scheduler = optimizer_obj.warmup
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    
    train_loss, correct, total = 0, 0, 0
    
    for inputs, targets in tqdm(trainloader, leave=False):
        # targets shape: [B, 1, H, W]
        inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1).long()

        optimizer.zero_grad()
        outputs = net(inputs)

        # ViT or UT structure: use last step's output
        if isinstance(net, (MazeViTUTModel)) and outputs.dim() == 5:
            outputs = outputs[-1]

        B_out, C_out, H_out, W_out = outputs.size()
        
        # 1. 타겟 리사이즈 및 준비
        targets_resized = F.interpolate(targets.float(), size=(H_out, W_out), mode='nearest').long().squeeze(1)
        
        # 2. Loss 계산용 (픽셀 단위)
        reshaped_outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, C_out)
        reshaped_targets = targets_resized.view(-1)
        
        loss = criterion(reshaped_outputs, reshaped_targets)
        loss.backward()
        optimizer.step()
        
        # 3. 통계 누적
        # Loss는 기존처럼 전체 픽셀 평균을 위해 픽셀 수만큼 가중치를 주어 누적
        train_loss += loss.item() * reshaped_targets.size(0)
        
        # ======= 정확도 계산 (Sample-wise Exact Match) =======
        predicted = outputs.argmax(1) # [B, H_out, W_out]
        
        # 각 샘플별로 모든 픽셀이 일치하는지 확인 (결과: [B] 크기의 True/False 벡터)
        # .view(B_out, -1)로 픽셀을 펼친 후 모든(all) 픽셀이 같은지 체크합니다.
        sample_match = (predicted == targets_resized).view(B_out, -1).all(dim=1)
        
        correct += sample_match.sum().item() # 완전히 맞춘 이미지 개수 누적
        total += B_out                       # 이미지 장수(Batch Size) 누적
    
    # 4. 최종 결과 산출
    # train_loss는 픽셀 단위 총합을 전체 픽셀 수로 나눕니다.
    # (원래 코드의 total이 픽셀 수였으므로 그 논리를 유지하기 위해 별도 변수 관리도 가능하지만, 
    # 가독성을 위해 직관적으로 계산합니다.)
    
    pixel_total = total * H_out * W_out
    train_loss /= pixel_total
    
    # acc는 (맞춘 이미지 수 / 전체 이미지 수) * 100
    acc = 100.0 * correct / total
    
    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc


# def train_default(net, trainloader, optimizer_obj, device):
#     net.train()
#     optimizer = optimizer_obj.optimizer
#     lr_scheduler = optimizer_obj.scheduler
#     warmup_scheduler = optimizer_obj.warmup
#     criterion = torch.nn.CrossEntropyLoss(reduction="mean")
#     train_loss, correct, total = 0, 0, 0
    
#     for inputs, targets in tqdm(trainloader, leave=False):
#         inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1).long()

#         optimizer.zero_grad()
        
#         net_output = net(inputs)
#         lambda_ponder = 0.0  
#         if isinstance(net_output, tuple):
#             outputs, ponder_cost = net_output
#             lambda_ponder = 0.01
#         else:
#             outputs = net_output
#             ponder_cost = 0.0
        
#         if isinstance(net, (MazeViTUTModel)) and outputs.dim() == 5:
#             outputs = outputs[-1]

#         B_out, C_out, H_out, W_out = outputs.size()
#         targets_resized = F.interpolate(targets.float(), size=(H_out, W_out), mode='nearest').long().squeeze(1)
#         reshaped_outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, C_out)
#         reshaped_targets = targets_resized.view(-1)
        
#         ce_loss = criterion(reshaped_outputs, reshaped_targets)
#         loss = ce_loss + lambda_ponder * ponder_cost
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item() * reshaped_targets.size(0)
#         total += reshaped_targets.size(0)
#         predicted = outputs.argmax(1)
#         correct += (predicted == targets_resized).sum().item()
    
#     train_loss /= total
#     acc = 100.0 * correct / total
#     lr_scheduler.step()
#     warmup_scheduler.dampen()

#     if hasattr(net, "last_num_steps"):
#         print(f"[Train Epoch] Avg halting steps this epoch: {net.last_num_steps:.2f}")
#     if hasattr(net, "stopped_at_step"):
#         print(f"[Train Epoch] Sample stopped_at_step[:10]: {net.stopped_at_step[:10].cpu().tolist()}")

#     return train_loss, acc


def train_default(net, trainloader, optimizer_obj, device):
    net.train()
    optimizer = optimizer_obj.optimizer
    lr_scheduler = optimizer_obj.scheduler
    warmup_scheduler = optimizer_obj.warmup
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    
    # 통계 계산을 위한 변수 분리
    train_loss = 0
    total_pixels = 0  # Loss 정규화용 (전체 픽셀 수)
    correct_samples = 0 # 샘플 단위 정답 수
    total_samples = 0   # 전체 샘플 수 (이미지 장수)
    
    torch.set_printoptions(profile="full")

    for inputs, targets in tqdm(trainloader, leave=False):
        # targets: [B, 1, H, W]
        inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1).long()

        optimizer.zero_grad()
        
        net_output = net(inputs)
        lambda_ponder = 0.0  
        if isinstance(net_output, tuple):
            outputs, ponder_cost = net_output
            lambda_ponder = 0.01
        else:
            outputs = net_output
            ponder_cost = 0.0
        
        if isinstance(net, (MazeViTUTModel)) and outputs.dim() == 5:
            outputs = outputs[-1]

        B_out, C_out, H_out, W_out = outputs.size()
        
        # 1. 타겟 리사이즈
        targets_resized = F.interpolate(targets.float(), size=(H_out, W_out), mode='nearest').long().squeeze(1)
        
        # 2. Loss 계산용 (픽셀 단위 평탄화)
        reshaped_outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, C_out)
        reshaped_targets = targets_resized.view(-1)
        
        ce_loss = criterion(reshaped_outputs, reshaped_targets)
        loss = ce_loss + lambda_ponder * ponder_cost
        loss.backward()
        optimizer.step()
        
        # 3. Loss 통계 누적 (픽셀 가중 평균을 위해 픽셀 수 곱함)
        train_loss += loss.item() * reshaped_targets.size(0)
        total_pixels += reshaped_targets.size(0)
        
        # 4. 정확도 계산 (Sample-wise Exact Match)
        predicted = outputs.argmax(1) # [B, H, W]
        
        # 해당 이미지의 모든 픽셀이 일치해야 True
        sample_match = (predicted == targets_resized).view(B_out, -1).all(dim=1)
        
        correct_samples += sample_match.sum().item()
        total_samples += B_out
    
    # 5. 최종 결과 산출
    # Loss는 전체 픽셀 수로 나누어 픽셀당 평균 손실 계산
    train_loss /= max(1, total_pixels)
    # Acc는 (완전 일치 이미지 수 / 전체 이미지 수) * 100
    acc = 100.0 * correct_samples / max(1, total_samples)
    
    lr_scheduler.step()
    warmup_scheduler.dampen()

    if hasattr(net, "last_num_steps"):
        print(f"[Train Epoch] Avg halting steps this epoch: {net.last_num_steps:.2f}")
    if hasattr(net, "stopped_at_step"):
        print(f"[Train Epoch] Sample stopped_at_step[:10]: {net.stopped_at_step[:10].cpu().tolist()}")

    return train_loss, acc


def train(net, trainloader, mode, optimizer_obj, device):
    train_func_dict = {
        "default": train_default,
        "baseline": train_baseline
    }

    if mode not in train_func_dict:
        print(f"train_{mode}() not implemented. Exiting.")
        exit()

    return train_func_dict[mode](net, trainloader, optimizer_obj, device)

# def test_default(net, testloader, device):
#     net.eval()
#     net.to(device)
#     correct, total = 0, 0
    
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1).long()

#             # 모델 forward
#             net_output = net(inputs)
#             lambda_ponder = 0.0  # ACT가 없는 경우에도 에러 안 나게 기본값 설정
#             # ViT-ACT의 경우 (output, ponder_cost) 형태
#             if isinstance(net_output, tuple):
#                 outputs = net_output[0]
#             else:
#                 outputs = net_output

#             # UT, ViT 등에서 step별 출력이 쌓여 있을 경우 → 마지막 step 출력 사용
#             if outputs.dim() == 5:  # [step, B, C, H, W]
#                 outputs = outputs[-1]

#             B_out, C_out, H_out, W_out = outputs.size()
#             targets_resized = targets.squeeze(1)  # [B, H, W]
#             predicted = outputs.argmax(1)         # [B, H, W]

#             correct += (predicted == targets_resized).sum().item()
#             total += targets_resized.numel()

#     acc = 100.0 * correct / total
#     return acc


def test_default(net, testloader, device):
    net.eval()
    net.to(device)
    correct, total = 0, 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1).long()

            # 모델 forward
            net_output = net(inputs)
            
            # ViT-ACT의 경우 (output, ponder_cost) 형태 처리
            if isinstance(net_output, tuple):
                outputs = net_output[0]
            else:
                outputs = net_output

            # UT, ViT 등에서 step별 출력이 쌓여 있을 경우 마지막 step 사용
            if outputs.dim() == 5:  # [step, B, C, H, W]
                outputs = outputs[-1]

            B_out, C_out, H_out, W_out = outputs.size()
            
            # targets_resized: [B, H, W]
            targets_resized = targets.squeeze(1) 
            # predicted: [B, H, W]
            predicted = outputs.argmax(1)         

            # ======= 정확도 계산 (Sample-wise Exact Match) =======
            # 각 샘플(B)에 대해 모든 픽셀(H*W)이 일치하는지 확인
            # .view(B_out, -1)을 통해 픽셀들을 펼친 후 한 이미지 내의 모든(all) 값이 True인지 체크
            sample_match = (predicted == targets_resized).view(B_out, -1).all(dim=1)

            correct += sample_match.sum().item() # 완전히 일치한 이미지 개수 누적
            total += B_out                       # 전체 이미지 장수 누적

    # acc는 (맞춘 이미지 수 / 전체 이미지 수) * 100
    acc = 100.0 * correct / max(1, total)
    return acc


def test(net, testloader, mode, device):
    test_func_dict = {
        "default": test_default
    }

    if mode not in test_func_dict:
        print(f"test_{mode}() not implemented. Exiting.")
        exit()

    return test_func_dict[mode](net, testloader, device)

    # try:
    #     acc = eval(f"test_{mode}")(net, testloader, device)
    # except NameError:
    #     print(f"test_{mode}() not implemented. Exiting.")
    #     exit()
    # return acc

def visualize_single_sample(weighted_output, input_img, weighted_output_history, target, batch_idx, sample_type, cmap="magma", max_cols = 5, save_path = './iter_img'):
    """
    input_img: [C, H, W] (torch.Tensor, C=3 for RGB or 1 for grayscale)
    target: [H, W] (torch.Tensor)
    weighted_output_history: list of [2, H, W] tensors (for a single sample)
    """
    os.makedirs(save_path, exist_ok = True)
    
    # 현재 시각을 문자열로 생성
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 이미지와 타겟을 numpy 배열로 변환
    input_np = input_img.permute(1, 2, 0).cpu().numpy()
    target_np = target.cpu().numpy()
    
    # weighted_output (최종 결과) 시각화: 경로 채널 (1번 인덱스) 사용
    weighted_output_visual = torch.nn.functional.softmax(weighted_output, dim=0)[1] * input_img.max(0)[0]
    weighted_output_visual_np = weighted_output_visual.cpu().numpy()

    num_iters = len(weighted_output_history)
    total_plots = num_iters + 3 # Input, Final W-output, Target + num_iters (for history)

    cols = min(max_cols, total_plots)
    rows = math.ceil(total_plots / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axs = axs.flatten()
    
    im = None # colorbar를 위해 imshow 객체를 저장할 변수

    # 0: Input 이미지
    axs[0].imshow(input_np)
    axs[0].set_title("Input")
    axs[0].axis('off')

    # 1: 최종 Weighted Output (시각화 목적)
    axs[1].imshow(weighted_output_visual_np, cmap=cmap, interpolation='nearest')
    axs[1].set_title("Final W-output")
    axs[1].axis('off')

    # 2부터: 각 iteration의 Weighted Output History 시각화
    for i in range(num_iters):
        current_iter_w_output_visual = torch.nn.functional.softmax(weighted_output_history[i], dim=0)[1] * input_img.max(0)[0]
        im = axs[i + 2].imshow(current_iter_w_output_visual.cpu().numpy(), cmap=cmap, interpolation='nearest')
        axs[i + 2].set_title(f'W-output iter {i + 1}')
        axs[i + 2].axis('off')

    # 마지막: Target 이미지
    axs[len(axs) - 1].imshow(target_np, cmap='gray')
    axs[len(axs) - 1].set_title("Target")
    axs[len(axs) - 1].axis('off')

    # 남은 서브플롯 끄기
    for j in range(total_plots, len(axs)):
        axs[j].axis('off')

    # colorbar 추가 (im이 마지막에 그린 w_output_history 이미지여야 함)
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 위치와 크기 조정
        fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # 파일명에 timestamp 추가 (함수 내에서 생성된 timestamp 사용)
    fig.savefig(save_path + f'/img_{current_timestamp}_batch_{batch_idx}_{sample_type}_sample0.png')
    plt.close(fig)


# 모델 체크포인트 로드
def load_model_from_checkpoint(model_name, model_path, width, depth):
    net = get_model(model_name, width, depth)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict["net"])
    net = net.to(device)
    return net, state_dict["epoch"], state_dict["optimizer"]

# JSON 저장
def to_json(stats, out_dir, log_name="test_stats.json"):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)
    if os.path.isfile(fname):
        with open(fname, 'r') as fp:
            data_from_json = json.load(fp)
            num_entries = data_from_json['num entries']
        data_from_json[num_entries] = stats
        data_from_json["num entries"] += 1
        with open(fname, 'w') as fp:
            json.dump(data_from_json, fp)
    else:
        data_from_json = {0: stats, "num entries": 1}
        with open(fname, 'w') as fp:
            json.dump(data_from_json, fp)

# 로그 기록
def to_log_file(out_dict, out_dir, log_name="log.txt"):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)
    with open(fname, "a") as fh:
        fh.write(str(now()) + " " + str(out_dict) + "\n" + "\n")
    print("logging done in " + out_dir + ".")

# 현재 시각
def now():
    return datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")