import datetime 
import json
import os
import sys
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F

from easy_to_hard_data import MazeDataset
import torch
import torch.utils.data as data
from icecream import ic
from torch.optim import SGD, Adam, AdamW
from tqdm import tqdm

from models.recur_resnet_segment import recur_resnet 
from models.recur_resnet_act import recur_resnet_act


class NumpyMazeDataset(data.Dataset):
    def __init__(self, inputs_path, solutions_path):
        self.inputs = np.load(inputs_path)       # (N, 3, H, W)
        self.solutions = np.load(solutions_path) # (N, H, W)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.solutions[idx], dtype=torch.float32)
        
        # 32 x 32가 되도록 padding
        x = F.pad(x, (4, 4, 4, 4))
        y = F.pad(y, (4, 4, 4, 4))
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


def get_model(model, width, depth, dilation):
    """Function to load the model object
    input:
        model:      str, Name of the model
        width:      int, Width of network
        depth:      int, Depth of network
    return:
        net:        Pytorch Network Object
    """
    model = model.lower()
    net = eval(model)(depth=depth, width=width, dilation=dilation)
    return net


def get_optimizer(optimizer_name, model, net, lr):
    optimizer_name = optimizer_name.lower()
    model = model.lower()

    if "recur" in model:
        # 모델 파라미터에서 recur_block에 해당하는 파라미터만 분리
        base_params = [p for n, p in net.named_parameters() if "recur_block" not in n]
        recur_params = [p for n, p in net.named_parameters() if "recur_block" in n]
        iters = net.iters
    else:
        base_params = [p for n, p in net.named_parameters()]
        recur_params = []
        iters = 1

    all_params = [{'params': base_params}, {'params': recur_params, 'lr': lr / iters}]

    if optimizer_name == "sgd":
        optimizer = SGD(all_params, lr=lr, weight_decay=2e-4, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = Adam(all_params, lr=lr, weight_decay=2e-4)
    elif optimizer_name == "adamw":
        optimizer = AdamW(all_params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                          amsgrad=False)
    else:
        print(f"{ic.format()}: Optimizer choise of {optimizer_name} not yet implmented. Exiting.")
        sys.exit()

    return optimizer


def load_model_from_checkpoint(model, model_path, width, depth, dilation):
    net = get_model(model, width, depth, dilation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict["net"])
    net = net.to(device)
    return net, state_dict["epoch"], state_dict["optimizer"]


def now():
    return datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")


@dataclass
class OptimizerWithSched:
    optimizer: "typing.Any"
    scheduler: "typing.Any"
    warmup: "typing.Any"


def test(net, testloader, mode, device):
    try:
        accuracy = eval(f"test_{mode}")(net, testloader, device)
    except NameError:
        print(f"{ic.format()}: test_{mode}() not implemented. Exiting.")
        sys.exit()
    return accuracy


def set_dynamic_ponder_epsilon(model, inputs, min_eps=0.002, max_eps=0.01):
    entropies = compute_image_entropy(inputs)  # shape: [B]
    avg_entropy = entropies.mean().item()

    normalized = np.clip((avg_entropy - 1.5) / (3.5 - 1.5), 0.0, 1.0)

    ponder_epsilon = max_eps - normalized * (max_eps - min_eps)
    model.ponder_epsilon = ponder_epsilon
    return ponder_epsilon


def test_default(net, testloader, device):
    net.eval()
    net.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1).long()
            outputs = net(inputs)
            pred = outputs.argmax(1)                  # (N,H,W)
            t2   = targets.squeeze(1)                 # (N,H,W)
            mask = (inputs.max(1)[0] > 0) & (t2 >= 0) # 유효 픽셀만 평가

            eq = (pred == t2) | (~mask)               # 마스크 밖은 자동 True
            per_sample_exact = eq.reshape(eq.size(0), -1).all(dim=1)        # (N,)
            per_sample_valid = mask.reshape(mask.size(0), -1).any(dim=1)    # (N,) 유효픽셀 존재?

            correct += per_sample_exact[per_sample_valid].sum().item()      # 샘플 개수
            total   += per_sample_valid.sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def test_max_conf(net, testloader, device):

    net.eval()
    net.to(device)
    correct = 0
    confidence = torch.zeros(net.iters)
    total = 0
    total_pixels = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1).long()
            net(inputs)
            confidence_array = torch.zeros(net.iters, inputs.size(0))
            for i, thought in enumerate(net.thoughts):
                conf = torch.nn.functional.softmax(thought.detach(), dim=1).max(1)[0] \
                       * inputs.max(1)[0]
                confidence[i] += conf.sum().item()
                confidence_array[i] = conf.sum([1, 2]) / inputs.max(1)[0].sum([1, 2])

            exit_iter = confidence_array.argmax(0)

            best_thoughts = net.thoughts[exit_iter, torch.arange(net.thoughts.size(1))].squeeze()
            if best_thoughts.shape[0] != inputs.shape[0]:
                best_thoughts = best_thoughts.unsqueeze(0)
            pred = best_thoughts.argmax(1)                # (N,H,W)
            t2   = targets.squeeze(1)                 # (N,H,W)
            mask = (inputs.max(1)[0] > 0) & (t2 >= 0) # 유효 픽셀만 평가

            eq = (pred == t2) | (~mask)               # 마스크 밖은 자동 True
            per_sample_exact = eq.reshape(eq.size(0), -1).all(dim=1)        # (N,)
            per_sample_valid = mask.reshape(mask.size(0), -1).any(dim=1)    # (N,) 유효픽셀 존재?

            correct += per_sample_exact[per_sample_valid].sum().item()      # 샘플 개수
            total   += per_sample_valid.sum().item()   

    accuracy = 100.0 * correct / total
    return accuracy


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


def to_log_file(out_dict, out_dir, log_name="log.txt"):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, log_name)

    with open(fname, "a") as fh:
        fh.write(str(now()) + " " + str(out_dict) + "\n" + "\n")

    print("logging done in " + out_dir + ".")


def train(net, trainloader, mode, optimizer_obj, device):
    try:
        train_loss, acc, net = eval(f"train_{mode}")(net, trainloader, optimizer_obj, device)
    except NameError:
        print(f"{ic.format()}: train_{mode}() not implemented. Exiting.")
        sys.exit()
    return train_loss, acc, net

                
def train_default(net, trainloader, optimizer_obj, device):
    # 초기 설정
    net.train()  # 모델을 학습 모드로 전환
    net = net.to(device)  # GPU 또는 CPU로 모델 이동
    # 옵티마이저와 학습률 스케줄러를 받아옴.
    optimizer = optimizer_obj.optimizer
    lr_scheduler = optimizer_obj.scheduler
    warmup_scheduler = optimizer_obj.warmup

    criterion = torch.nn.CrossEntropyLoss(reduction="none")  # 픽셀 단위 손실 계산 가능.
#     time_penalty = net.time_penalty  # 조정 가능한 시간 패널티 계수(수식에서 람다 역할)
    
    # 손실, 정확도, 픽셀 수, ponder cost 누적을 위한 변수 초기화
    train_loss = 0
    correct = 0
    total = 0
    total_pixels = 0
#     total_ponder_cost = 0  # Ponder cost 추적 추가
    
    # 미니배치 루프
    torch.set_printoptions(profile="full")

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1).long()
        optimizer.zero_grad()
        outputs = net(inputs)

        n, c, h, w = outputs.size()
        reshaped_outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous()
        reshaped_outputs = reshaped_outputs[targets.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        reshaped_outputs = reshaped_outputs.view(-1, c)

        reshaped_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
        reshaped_inputs = reshaped_inputs.mean(3).unsqueeze(-1)
        reshaped_inputs = reshaped_inputs[targets.view(n, h, w, 1).repeat(1, 1, 1, 1) >= 0]
        reshaped_inputs = reshaped_inputs.view(-1, 1)
        path_mask = (reshaped_inputs > 0).squeeze()

        mask = targets >= 0.0
        reshaped_targets = targets[mask]

        loss = criterion(reshaped_outputs, reshaped_targets)
        loss = loss[path_mask].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()

        train_loss += loss.item() * path_mask.size(0)
        total_pixels += path_mask.size(0)
        
        # ======= 정확도 계산 ========
        pred = outputs.argmax(1)                  # (N,H,W)
        t2   = targets.squeeze(1)                 # (N,H,W)
        mask = (inputs.max(1)[0] > 0) & (t2 >= 0) # 유효 픽셀만 평가

        eq = (pred == t2) | (~mask)               # 마스크 밖은 자동 True
        per_sample_exact = eq.reshape(eq.size(0), -1).all(dim=1)        # (N,)
        per_sample_valid = mask.reshape(mask.size(0), -1).any(dim=1)    # (N,) 유효픽셀 존재?

        correct += per_sample_exact[per_sample_valid].sum().item()      # 샘플 개수
        total   += per_sample_valid.sum().item()                        # 샘플 개수

    train_loss = train_loss / max(1, total_pixels)
    acc = 100.0 * correct / max(1, total)
    
    lr_scheduler.step()
    warmup_scheduler.dampen()

#     print(f"[Train Epoch] Avg halting steps this epoch: {net.last_num_steps:.2f}")
#     print(f"[Train Epoch] Sample stopped_at_step[:10]: {net.stopped_at_step[:10].cpu().tolist()}")

    return train_loss, acc, net


def train_act(net, trainloader, optimizer_obj, device):

    net.train()
    net = net.to(device)
    optimizer = optimizer_obj.optimizer
    lr_scheduler = optimizer_obj.scheduler
    warmup_scheduler = optimizer_obj.warmup

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    
    # --- 누적용 카운터 ---
    train_loss_sum = 0.0       # 마스크 적용된 픽셀 손실의 총합
    train_pix_count = 0        # 손실에 사용한 '유효 픽셀' 수

    pixel_correct = 0          # (1) 경로-마스크 픽셀 정확도 분자
    pixel_total   = 0          # (1) 경로-마스크 픽셀 정확도 분모

    sample_solved = 0          # (2) 샘플 완벽일치(참고용) 분자
    num_samples   = 0          # (2) 샘플 완벽일치(참고용) 분모

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs = inputs.to(device)                             # [N,C,H,W]
        targets = targets.to(device).long().squeeze(1)         # [N,H,W]
        optimizer.zero_grad()
        
        outputs = net(inputs)                                 # [N,C,H,W]
        
        lambda_ponder = 0.0
        if isinstance(outputs, tuple):
            outputs, ponder_cost = outputs
            lambda_ponder = net.time_penalty
        else:
            ponder_cost = 0.0
        
        outputs = _ensure_logits(outputs) 
        
         # --- 경로(유효) 픽셀 마스크: 입력의 '하얀 경로'만 True ---
        path_mask = (inputs.max(1)[0] > 0)                    # [N,H,W], bool

        # --- 손실: 경로 픽셀만 CrossEntropy ---
        logits_flat  = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.size(1))  # [N*H*W, C]
        targets_flat = targets.reshape(-1)                                      # [N*H*W]
        mask_flat    = path_mask.reshape(-1)                                    # [N*H*W]

        loss_per_pixel = criterion(logits_flat, targets_flat)   # [N*H*W]
        
        # ponder cost 추가
        loss = loss_per_pixel[mask_flat].mean() + lambda_ponder * ponder_cost
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        # --- 손실 누적(마스크 픽셀 기준) ---
        used_pixels = mask_flat.sum().item()
        train_loss_sum += loss.item() * used_pixels
        train_pix_count += used_pixels

        # --- 정확도 계산 ---
        with torch.no_grad():
            pred = outputs.argmax(1)  # [N,H,W]

            # (1) 경로-마스크 픽셀 정확도
            pixel_correct += (pred[path_mask] == targets[path_mask]).sum().item()
            pixel_total   += path_mask.sum().item()   

            # (2) test와 동일 정의: 마스크 곱 후 '샘플 완벽일치' (참고용 출력용)
            imask   = inputs.max(1)[0]              # [N,H,W], test와 동일 처리
            pred_m  = pred * imask
            sample_solved += torch.amin(pred_m == targets, dim=[1, 2]).sum().item()
            num_samples   += targets.size(0)

    # --- 에폭 집계 ---
    train_loss = train_loss_sum / max(1, train_pix_count)
    acc_pixel  = 100.0 * pixel_correct / max(1, pixel_total)       # 반환용
    acc_exact  = 100.0 * sample_solved / max(1, num_samples)       # 참고용

    lr_scheduler.step()
    warmup_scheduler.dampen()

    print(f"[INFO] train pixel-acc(masked): {acc_pixel:.2f}% | train exact-acc: {acc_exact:.2f}%")
    return train_loss, acc_exact, net


class AllLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()