import sys
import torch
from tqdm import tqdm
from icecream import ic
from utils import _ensure_logits


def train(net, trainloader, mode, optimizer_obj, device):
    try:
        train_loss, acc = eval(f"train_{mode}")(net, trainloader, optimizer_obj, device)
    except NameError:
        print(f"{ic.format()}: train_{mode}() not implemented. Exiting.")
        sys.exit()
    return train_loss, acc


def train_default(net, trainloader, optimizer_obj, device):  # without act

    net.train()
    net = net.to(device)
    optimizer = optimizer_obj.optimizer
    lr_scheduler = optimizer_obj.scheduler
    warmup_scheduler = optimizer_obj.warmup

    criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
    
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
        outputs = _ensure_logits(outputs) 
        
         # --- 경로(유효) 픽셀 마스크: 입력의 '하얀 경로'만 True ---
        path_mask = (inputs.max(1)[0] > 0)                    # [N,H,W], bool

        # --- 손실: 경로 픽셀만 CrossEntropy ---
        logits_flat  = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.size(1))  # [N*H*W, C]
        targets_flat = targets.reshape(-1)                                      # [N*H*W]
        mask_flat    = path_mask.reshape(-1)                                    # [N*H*W]

        loss_per_pixel = criterion(logits_flat, targets_flat)   # [N*H*W]
        
        loss = loss_per_pixel[mask_flat].mean()
        
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
    return train_loss, acc_pixel


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
    return train_loss, acc_pixel