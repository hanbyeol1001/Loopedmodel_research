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
# from models.resnet_segment import ff_resnet
from models.recur_resnet_act import RecurResNetACT, BasicBlock 

def get_dataloaders(train_batch_size, test_batch_size, train_maze_size=9, test_maze_size=9, shuffle=True):

    train_data = MazeDataset("./data", size=train_maze_size, train=True)
    test_data = MazeDataset("./data", size=test_maze_size, train=False)

    trainloader = data.DataLoader(train_data, num_workers=4, batch_size=train_batch_size,
                                   shuffle=shuffle, drop_last=True)
    testloader = data.DataLoader(test_data, num_workers=4, batch_size=test_batch_size,
                                   shuffle=False, drop_last=False)
    return trainloader, testloader


def get_model(model, width, depth):
    model = model.lower()
    # RecurResNetACT 모델을 직접 반환하도록 수정 (만약 recur_resnet_segment에 RecurResNetACT가 없다면)
    if model == "recur_resnet_act":
        net = RecurResNetACT(BasicBlock, [2], depth=depth, width=width) 
    else:
        net = eval(model)(depth=depth, width=width)
    return net


def get_optimizer(optimizer_name, model, net, lr):
    optimizer_name = optimizer_name.lower()
    model = model.lower()

    if "recur" in model:
        # 모델 파라미터에서 recur_block에 해당하는 파라미터만 분리
        base_params = [p for n, p in net.named_parameters() if "recur_block" not in n]
        recur_params = [p for n, p in net.named_parameters() if "recur_block" in n]
        iters = net.max_iters
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


def load_model_from_checkpoint(model, model_path, width, depth):
    net = model
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
    accuracy = eval(f"test_{mode}")(net, testloader, device)
    return accuracy

def compute_image_entropy(x):
    x_gray = x.mean(dim=1)  # (B, H, W)
    b, h, w = x_gray.shape
    x_flat = x_gray.view(b, -1)

    # 정규화 후 histogram 기반 entropy 계산
    x_norm = (x_flat - x_flat.min(dim=1, keepdim=True)[0]) / \
             (x_flat.max(dim=1, keepdim=True)[0] - x_flat.min(dim=1, keepdim=True)[0] + 1e-8)

    entropies = []
    for i in range(b):
        hist = torch.histc(x_norm[i], bins=32, min=0.0, max=1.0)
        hist /= hist.sum() + 1e-8
        entropy = -torch.sum(hist * torch.log(hist + 1e-8))
        entropies.append(entropy)

    return torch.tensor(entropies, device=x.device)  # shape: [B]
def set_dynamic_ponder_epsilon(model, inputs, min_eps=0.002, max_eps=0.01):
    entropies = compute_image_entropy(inputs)  # shape: [B]
    avg_entropy = entropies.mean().item()

    normalized = np.clip((avg_entropy - 1.5) / (3.5 - 1.5), 0.0, 1.0)

    ponder_epsilon = max_eps - normalized * (max_eps - min_eps)
    model.ponder_epsilon = ponder_epsilon
    return ponder_epsilon


def test_default(net, testloader, device, tta_steps=5, lr=1e-2, title="Halting Step Distribution (Test)", save_dir="./hist_output"):
    net.eval()
    net.to(device)

    correct = 0
    total = 0
    all_stopped_at_steps = []

    
    # 여기 주석 풀면 TTA
    # for param in net.parameters():
    #     param.requires_grad = False
    # for param in net.halting_unit.parameters():
    #     param.requires_grad = True

    # optimizer = torch.optim.Adam(net.halting_unit.parameters(), lr=lr)
    
    # for inputs, targets in tqdm(testloader, leave=False):
    #     inputs = inputs.to(device)
    #     targets = targets.to(device).unsqueeze(1).long()  # ONLY for evaluation

    #     net.train()  # halting_unit 업데이트를 위해 train mode

    #     for _ in range(tta_steps):
    #         weighted_output, ponder_cost = net(inputs)
    #         probs = F.softmax(weighted_output, dim=1)

    #         # pixel-wise entropy
    #         entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

    #         loss = entropy_loss #+ net.time_penalty * ponder_cost.mean()

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         net.eval()
    #         with torch.no_grad():
    #             weighted_output, _ = net(inputs)
    #             predicted = weighted_output.argmax(1) * inputs.max(1)[0]
    #             correct += torch.amin(predicted == targets.squeeze(1), dim=[1, 2]).sum().item()
    #             total += targets.size(0)

    #             all_stopped_at_steps.extend(net.stopped_at_step.cpu().tolist())

    # print(f"[Train Epoch] Avg halting steps this epoch: {net.last_num_steps:.2f}")

    
    # 여기만 주석 풀면 일반 돌리기

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader, leave=False)):
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1).long()

            # eps = set_dynamic_ponder_epsilon(net, inputs) # 이건 동적 epsilon
            # print(f"[Dynamic ε] adjusted to {eps:.5f} based on input complexity")

            weighted_output, _ = net(inputs)

            predicted = weighted_output.argmax(1) * inputs.max(1)[0]
            correct += torch.amin(predicted == targets.squeeze(1), dim=[1, 2]).sum().item()
            total += targets.size(0)

            all_stopped_at_steps.extend(net.stopped_at_step.cpu().tolist())

    os.makedirs(save_dir, exist_ok=True)
    
    # NumPy 배열로 변환
    steps_array = np.array(all_stopped_at_steps)
    
    # 저장 (npy + txt)
    np.save(os.path.join(save_dir, "halting_steps.npy"), steps_array) # 급하게 돌려본거라 돌리고나서 size랑 모델 이름에 넣어서 바꿔줘야함!!!!
    print(f"[Train Epoch] Avg halting steps this epoch: {net.last_num_steps:.2f}")
    accuracy = 100.0 * correct / total
    print(f"[Test] Accuracy: {accuracy:.2f}%")
    

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


def test_max_conf(net, testloader, device):
    net.eval()
    net.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader, leave=False)):
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1).long()
            
            weighted_output, _ = net(inputs) 
            
            weighted_output_history_full_batch = net.weighted_output_history
            
            predicted = weighted_output.argmax(1) * inputs.max(1)[0]
            
            is_correct = torch.amin(predicted == targets.squeeze(1), dim=[1,2])
            
            correct += is_correct.sum().item()
            total += targets.size(0)
            
            # 60 배치마다 시각화
            if batch_idx % 60 == 0:
                # 맞춘 샘플 찾기
                correct_indices = torch.where(is_correct == True)[0]
                if len(correct_indices) > 0:
                    sample_idx_correct = correct_indices[0].item()
                    sample_history_correct = [
                        hist_iter_batch[sample_idx_correct] for hist_iter_batch in weighted_output_history_full_batch
                    ]
                    visualize_single_sample(
                        weighted_output[sample_idx_correct],
                        inputs[sample_idx_correct],
                        sample_history_correct,
                        targets[sample_idx_correct].squeeze(0),
                        batch_idx,
                        sample_type="correct" # timestamp 인자 제거
                    )
                else:
                    print(f"Batch {batch_idx}: No correct samples to visualize.")

                # 틀린 샘플 찾기
                incorrect_indices = torch.where(is_correct == False)[0]
                if len(incorrect_indices) > 0:
                    sample_idx_incorrect = incorrect_indices[0].item()
                    sample_history_incorrect = [
                        hist_iter_batch[sample_idx_incorrect] for hist_iter_batch in weighted_output_history_full_batch
                    ]
                    visualize_single_sample(
                        weighted_output[sample_idx_incorrect],
                        inputs[sample_idx_incorrect],
                        sample_history_incorrect,
                        targets[sample_idx_incorrect].squeeze(0),
                        batch_idx,
                        sample_type="incorrect" # timestamp 인자 제거
                    )
                else:
                    print(f"Batch {batch_idx}: No incorrect samples to visualize.")
                    
    print(f"[Train Epoch] Avg halting steps this epoch: {net.last_num_steps:.2f}")
    print(f"[Train Epoch] Sample stopped_at_step[:10]: {net.stopped_at_step[:10].cpu().tolist()}")


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
    train_loss, acc, net = eval(f"train_{mode}")(net, trainloader, optimizer_obj, device)
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
    time_penalty = net.time_penalty  # 조정 가능한 시간 패널티 계수(수식에서 람다 역할)
    
    # 손실, 정확도, 픽셀 수, ponder cost 누적을 위한 변수 초기화
    train_loss = 0
    correct = 0
    total = 0
    total_pixels = 0
    total_ponder_cost = 0  # Ponder cost 추적 추가
    
    # 미니배치 루프
    torch.set_printoptions(profile="full")

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        # unsqueeze(1) → Segmentation처럼 (B,1,H,W) 형태로 만듦
        inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1).long()
        optimizer.zero_grad()
        
        # 모델 forward
        weighted_output, avg_ponder_cost = net(inputs)

        # 출력 형태 재구성
        n, c, h, w = weighted_output.size()
        # (B, C, H, W) → (B*H*W, C)
        reshaped_outputs = weighted_output.permute(0, 2, 3, 1).contiguous().view(-1, c)
        
        # 유효 픽셀 마스킹 
        mask = (targets >= 0).squeeze(1)
        reshaped_targets = targets.squeeze(1)[mask].view(-1)
        
        # 경로 마스크 계산
        reshaped_inputs = inputs.permute(0, 2, 3, 1).contiguous()
        reshaped_inputs = reshaped_inputs.mean(dim=3, keepdim=True)
        reshaped_inputs = reshaped_inputs[mask].view(-1, 1)
        path_mask = (reshaped_inputs > 0).squeeze()

        # Task loss 계산
        task_loss = criterion(reshaped_outputs, reshaped_targets)
        task_loss = task_loss[path_mask].mean()
        
        # 전체 손실 계산
        total_loss = task_loss + time_penalty * avg_ponder_cost
        total_loss.backward()
        
        # 그래디언트 클리핑 추가: 폭발 방지 (max norm = 1.0)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        # loss 계산은 픽셀 단위로
        train_loss += task_loss.item() * path_mask.size(0)
        total_ponder_cost += avg_ponder_cost.item() * inputs.size(0)
        total_pixels += path_mask.size(0)

        # 정확도 계산
        targets = targets.squeeze(1)
        predicted = weighted_output.argmax(1) * inputs.max(1)[0]
        correct += torch.amin(predicted == targets, dim=[1, 2]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / total_pixels
    acc = 100.0 * correct / total
    lr_scheduler.step()
    warmup_scheduler.dampen()

    print(f"[Train Epoch] Avg halting steps this epoch: {net.last_num_steps:.2f}")
    print(f"[Train Epoch] Sample stopped_at_step[:10]: {net.stopped_at_step[:10].cpu().tolist()}")

    return train_loss, acc, net

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
