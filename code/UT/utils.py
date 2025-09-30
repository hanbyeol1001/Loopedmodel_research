""" utils.py
    utility functions and classes
    Developed as part of DeepThinking2 project
    April 2021
"""
import datetime
import json
import os
import sys
from dataclasses import dataclass
import math
import numpy as np
import random

# If running headless or MPLBACKEND is set to a Jupyter-only backend, force Agg
if not os.environ.get("DISPLAY") or os.environ.get("MPLBACKEND", "").startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib  # safe now: reads MPLBACKEND=Agg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from easy_to_hard_data import MazeDataset
import torch
import torch.utils.data as data
from torch.optim import SGD, Adam, AdamW
from tqdm import tqdm
from icecream import ic


from models.maze_ut import MazeUTModel
from models.ut_act import MazeUTModelACT
from torch.utils.data import DataLoader, random_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611


# def get_dataloaders(train_batch_size, test_batch_size, train_maze_size=9, test_maze_size=9, shuffle=True):

#     train_data = MazeDataset("./data", size=train_maze_size, train=True)
#     test_data = MazeDataset("./data", size=test_maze_size, train=False)

#     trainloader = data.DataLoader(train_data, num_workers=0, batch_size=train_batch_size,
#                                   shuffle=shuffle, drop_last=True)
#     testloader = data.DataLoader(test_data, num_workers=0, batch_size=test_batch_size,
#                                  shuffle=False, drop_last=False)
#     return trainloader, testloader


# def get_dataloaders(train_batch_size,test_batch_size,train_maze_size=9,
#     test_maze_size=9,shuffle=True,data_root="../data"):
#     # 학습 데이터셋 경로
#     train_inputs_path = f"{data_root}/maze_data_train_{train_maze_size}/inputs.npy"
#     train_solutions_path = f"{data_root}/maze_data_train_{train_maze_size}/solutions.npy"

#     # 테스트 데이터셋 경로
#     test_inputs_path = f"{data_root}/maze_data_test_{test_maze_size}/inputs.npy"
#     test_solutions_path = f"{data_root}/maze_data_test_{test_maze_size}/solutions.npy"

#     # Dataset 객체
#     train_data = NumpyMazeDataset(train_inputs_path, train_solutions_path)
#     test_data = NumpyMazeDataset(test_inputs_path, test_solutions_path)

#     # DataLoader
#     trainloader = data.DataLoader(train_data, batch_size=train_batch_size,
#                                   shuffle=shuffle, drop_last=True, num_workers=0)
#     testloader = data.DataLoader(test_data, batch_size=test_batch_size,
#                                  shuffle=False, drop_last=False, num_workers=0)

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

    
def get_dataloaders(train_batch_size,
                    test_batch_size,
                    train_maze_size=9,
                    test_maze_size=9,
                    shuffle=True,
                    data_root="../data",
                    val_ratio=0.2,
                    seed=42):
    """
    train/val/test DataLoader를 반환합니다.
    val_ratio : train dataset 중 validation으로 쓸 비율 (0~1)
    """

    # 학습 데이터셋 경로
    train_inputs_path = f"{data_root}/maze_data_train_{train_maze_size}/inputs.npy"
    train_solutions_path = f"{data_root}/maze_data_train_{train_maze_size}/solutions.npy"

    # 테스트 데이터셋 경로
    test_inputs_path = f"{data_root}/maze_data_test_{test_maze_size}/inputs.npy"
    test_solutions_path = f"{data_root}/maze_data_test_{test_maze_size}/solutions.npy"

    # Dataset 객체
    train_data = NumpyMazeDataset(train_inputs_path, train_solutions_path)
    test_data = NumpyMazeDataset(test_inputs_path, test_solutions_path)

    # === train/val split ===
    val_size = int(len(train_data) * val_ratio)
    train_size = len(train_data) - val_size
    train_subset, val_subset = random_split(train_data, 
                                            [train_size, val_size], 
                                            generator=torch.Generator().manual_seed(seed))

    # DataLoader
    trainloader = DataLoader(train_subset, batch_size=train_batch_size,
                             shuffle=shuffle, drop_last=True, num_workers=0)
    valloader = DataLoader(val_subset, batch_size=test_batch_size,
                           shuffle=False, drop_last=False, num_workers=0)
    testloader = DataLoader(test_data, batch_size=test_batch_size,
                            shuffle=False, drop_last=False, num_workers=0)

    return trainloader, valloader, testloader


def get_model(model, width, depth):
    """Function to load the Universal Transformer model"""
    model = model.lower()
    if model== "maze_ut":
        return MazeUTModel(input_channels=3, hidden_dim=128, max_steps=4, nhead=4, height=24, width=24)
    elif model=="ut_act":
        return MazeUTModelACT(input_channels=3, hidden_dim=128, max_steps=10, nhead=4, height=24, width=24, out_channels=2, ponder_epsilon=0.01, time_penalty=0.01)
    else:
        raise ValueError(f"Unknown model: {model}")


def get_optimizer(optimizer_name, model, net, lr):
    optimizer_name = optimizer_name.lower()
    model = model.lower()

    if "recur" in model:
        base_params = [p for n, p in net.named_parameters() if "recur" not in n]
        recur_params = [p for n, p in net.named_parameters() if "recur" in n]
        iters = getattr(net, 'iters', 1)
        param_groups = [
            {'params': base_params},
            {'params': recur_params, 'lr': lr / iters}
        ] if recur_params else [{'params': base_params}]
    else:
        param_groups = [{'params': [p for n, p in net.named_parameters()]}]

    if optimizer_name == "sgd":
        optimizer = SGD(param_groups, lr=lr, weight_decay=2e-4, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = Adam(param_groups, lr=lr, weight_decay=2e-4)
    elif optimizer_name == "adamw":
        optimizer = AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    else:
        print(f"Optimizer choice of {optimizer_name} not yet implemented. Exiting.")
        sys.exit()

    return optimizer


def load_model_from_checkpoint(model, model_path, width, depth):
    net = get_model(model, width, depth)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict["net"])
    net = net.to(device)
    return net, state_dict["epoch"], state_dict["optimizer"]


def now():
    return datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")


@dataclass
class OptimizerWithSched:
    """Attributes for optimizer, lr schedule, and lr warmup"""
    optimizer: "typing.Any"
    scheduler: "typing.Any"
    warmup: "typing.Any"


def _ensure_logits(outputs: torch.Tensor) -> torch.Tensor:
    """
    다양한 반환 케이스를 받아 항상 (B, C, H, W) 로짓 텐서로 정규화한다.
    - (logits, aux...) 형태면 첫 번째만 사용
    - (steps, B, C, H, W) 형태면 마지막 스텝만 사용
    """
    # (logits, aux) / [logits, aux] 등
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]

    # all-steps 쌓인 텐서: (steps, B, C, H, W) -> 마지막 스텝
    if isinstance(outputs, torch.Tensor) and outputs.dim() == 5:
        outputs = outputs[-1]

    if not isinstance(outputs, torch.Tensor):
        raise TypeError(f"Model must return a Tensor; got {type(outputs)}")

    if outputs.dim() != 4:
        raise ValueError(f"Expected logits with shape (B,C,H,W); got shape {tuple(outputs.shape)}")

    return outputs


def test(net, testloader, mode, device):
    try:
        accuracy = eval(f"test_{mode}")(net, testloader, device)
    except NameError:
        print(f"{ic.format()}: test_{mode}() not implemented. Exiting.")
        sys.exit()
    return accuracy


def test_default(net, testloader, device):
    net.eval()
    net.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1).long()
            outputs = net(inputs)

            outputs = _ensure_logits(outputs) 

            targets = targets.squeeze(1)          # [N, H, W]
            # 예측 클래스 맵 [N, H, W]
            predicted = outputs.argmax(1) * inputs.max(1)[0]
            # 샘플 단위 '완벽 일치' 여부. amin은 모든 픽셀 True여야 True가 됨
            correct += torch.amin(predicted == targets, dim=[1, 2]).sum().item()
            total += targets.size(0)

    accuracy = 100.0 * correct / total    # # = 완벽히 푼 샘플 비율(%)
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


def visualize_single_sample(input_img, confidences, target, batch_idx, cmap="magma", max_cols = 5, save_path = './iter_img'):
    """
    input_img: [1, H, W] (torch.Tensor)
    target: [H, W] (torch.Tensor)
    confidence_array: [25, 64] [iter, batch_size]
    """
    os.makedirs(save_path, exist_ok = True)

    # print(len(confidences), confidences[0].shape)
    sample_idx = 0

    input_np = input_img.permute(1, 2, 0).cpu().numpy()
    target_np = target.cpu().numpy()

    num_iters = len(confidences)
    total = num_iters + 2

    max_cols=5
    cols = min(max_cols, total)
    rows = math.ceil(total / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axs = axs.flatten()

    axs[0].imshow(input_np)
    axs[0].set_title("Input")
    axs[0].axis('off')

    for i in range(num_iters):
        conf_map = confidences[i][sample_idx]
        im = axs[i + 1].imshow(conf_map, cmap=cmap, interpolation='nearest')
        axs[i + 1].set_title(f'iter {i + 1}')
        axs[i + 1].axis('off')

    axs[len(axs) - 1].imshow(target_np, cmap='gray')
    axs[len(axs) - 1].set_title("Target")
    axs[len(axs) - 1].axis('off')

    for j in range(total - 1, len(axs)):
        axs[j].axis('off')

    # colorbar 추가 (im이 마지막에 그린 conf_map이어야 함)
    if im is not None:
        # tight_layout 전에 colorbar 추가
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 위치와 크기 조정
        fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    fig.savefig(save_path + f'/img_batch_{batch_idx}_sample0.png')
    plt.close(fig)
    print(f'Saving img : {save_path}/img_batch_{batch_idx}_sample0.png')
    

# 이부분 코드 이해 필요
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

            conf_visual_array = []
            confidence_array = torch.zeros(net.iters, inputs.size(0))
            for i, thought in enumerate(net.thoughts):
                conf = torch.nn.functional.softmax(thought.detach(), dim=1).max(1)[0] \
                       * inputs.max(1)[0]
                confidence[i] += conf.sum().item()
                confidence_array[i] = conf.sum([1, 2]) / inputs.max(1)[0].sum([1, 2])

                softmax_map = torch.nn.functional.softmax(thought.detach(), dim=1)
                visual_conf = softmax_map[:, 1] * inputs.max(1)[0]
                conf_visual_array.append(visual_conf.cpu().numpy())
            
            exit_iter = confidence_array.argmax(0)

            best_thoughts = net.thoughts[exit_iter, torch.arange(net.thoughts.size(1))].squeeze()
            if best_thoughts.shape[0] != inputs.shape[0]:
                best_thoughts = best_thoughts.unsqueeze(0)
            predicted = best_thoughts.argmax(1) * inputs.max(1)[0]

            targets = targets.squeeze(1)
            correct += torch.amin(predicted == targets, dim=[1, 2]).sum().item()

            total_pixels += inputs.max(1)[0].sum().item()
            total += targets.size(0)

            if batch_idx % 20 == 0:
                # 시각화
                sample_input = inputs[0]
                target_img = targets[0].squeeze(0)               
                visualize_single_sample(sample_input, conf_visual_array, target_img, batch_idx)

    accuracy = 100.0 * correct / total
    return accuracy


