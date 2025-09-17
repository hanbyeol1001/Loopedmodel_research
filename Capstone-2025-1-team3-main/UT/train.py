import torch
from tqdm import tqdm


def train_default(net, trainloader, optimizer_obj, device):

    net.train()
    net = net.to(device)
    optimizer = optimizer_obj.optimizer
    lr_scheduler = optimizer_obj.scheduler
    warmup_scheduler = optimizer_obj.warmup

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0
    total_pixels = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        if batch_idx == 0:
            print(f"Inputs tensor shape: {inputs.shape}")
            print(f"Targets tensor shape: {targets.shape}")
        inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1).long()
        if batch_idx == 0:
            print(f"After unsqueeze Inputs tensor shape: {inputs.shape}")
            print(f"After unsqueeze Targets tensor shape: {targets.shape}")
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        if batch_idx == 0:
            print(f"Outputs tensor shape: {outputs.shape}")

        outputs = _ensure_logits(outputs) 
        if batch_idx == 0:
            print(f"After ensure_logits Outputs tensor shape: {outputs.shape}")

        # 출력(outputs) 및 정답(targets) 텐서 재구성
        n, c, h, w = outputs.size()
        # (n, c, h, w) -> (n, h, w, c)
        reshaped_outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous()
        # 마스킹을 위한 준비: (n, h, w, 1)로 변환 후, 채널 차원으로 복제 -> 손실계산에 포함할 픽셀들만
        reshaped_outputs = reshaped_outputs[targets.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        # (총 픽셀 수, c) 형태로 재구성: CrossEntropyLoss 함수가 요구하는 일반적인 형태
        reshaped_outputs = reshaped_outputs.view(-1, c)

        # (n, c, h, w) -> (n, h, w, c)
        reshaped_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
        # 미로의 특정 픽셀이 경로 or 벽인지 구분하기 위해 평균 취함.
        reshaped_inputs = reshaped_inputs.mean(3).unsqueeze(-1)
        reshaped_inputs = reshaped_inputs[targets.view(n, h, w, 1).repeat(1, 1, 1, 1) >= 0]
        reshaped_inputs = reshaped_inputs.view(-1, 1)
        # 손실을 계산할 픽셀을 최종적으로 선별.(경로에 해당하는 픽셀만)
        path_mask = (reshaped_inputs > 0).squeeze()

        mask = targets >= 0.0
        reshaped_targets = targets[mask]
        
        loss = criterion(reshaped_outputs, reshaped_targets)
        loss = loss[path_mask].mean()
        loss.backward()
        # 기울기의 크기를 1.0로 제한.(역전파 과정에서 기울기가 너무 커져 모델이 불안정해지는 것을 방지).
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0) 
        # 모델의 가중치 업데이트
        optimizer.step()

        # loss 누적
        train_loss += loss.item() * path_mask.size(0)
        # 나중에 평균 손실을 계산할 때 사용할 손실 계산에 사용된 총 픽셀 수
        total_pixels += path_mask.size(0)

        # acc는 결과 이미지가 같냐: 해당 Maze input의 경로를 제대로 맞췄냐 아니냐
        targets = targets.squeeze(1)
        
        # 각 픽셀에 대한 클래스별 확률값에서 가장 높은 값의 인덱스를 반환
        predicted = outputs.argmax(1)
        
        # 한 배치에 대한 픽셀 정확도
        batch_pix_acc = (predicted == targets).float().mean().item()
        # 배치 정확도 누적
        correct += batch_pix_acc
        total += targets.size(0)

    # 최종 결과 반환
    train_loss = train_loss / total_pixels
    # 최종 정확도
    acc = 100.0 * correct / total
    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc


def train_act(net, trainloader, optimizer_obj, device):
    """
    UT 모델에 ACT(Adaptive Computation Time)를 적용하여 학습하는 함수입니다.
    Ponder Cost를 손실에 추가하여 모델이 효율적으로 연산을 중단하도록 유도합니다.
    """
    net.train()
    net = net.to(device)
    optimizer = optimizer_obj.optimizer
    lr_scheduler = optimizer_obj.scheduler
    warmup_scheduler = optimizer_obj.warmup

    # Ponder Cost를 포함한 손실 계산을 위해 CrossEntropyLoss와 Ponder Cost를 정의합니다.
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    time_penalty = net.time_penalty  # 모델 클래스에서 정의된 시간 페널티 하이퍼파라미터를 가져옵니다.

    train_loss = 0
    correct = 0
    total = 0
    total_pixels = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1).long()
        optimizer.zero_grad()
        
        # UT 모델의 forward 함수는 outputs와 avg_ponder_cost를 함께 반환합니다.
        outputs, avg_ponder_cost = net(inputs)

        # 모델 출력은 튜플이 아닌, 텐서여야 하므로 _ensure_logits 함수에 outputs를 바로 전달합니다.
        outputs = _ensure_logits(outputs)
        
        # 출력(outputs) 및 정답(targets) 텐서 재구성
        n, c, h, w = outputs.size()
        reshaped_outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous()
        reshaped_outputs = reshaped_outputs[targets.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        reshaped_outputs = reshaped_outputs.view(-1, c)

        # 입력(inputs) 텐서 재구성 및 경로 마스크 생성
        reshaped_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
        reshaped_inputs = reshaped_inputs.mean(3).unsqueeze(-1)
        reshaped_inputs = reshaped_inputs[targets.view(n, h, w, 1).repeat(1, 1, 1, 1) >= 0]
        reshaped_inputs = reshaped_inputs.view(-1, 1)
        path_mask = (reshaped_inputs > 0).squeeze()

        mask = targets >= 0.0
        reshaped_targets = targets[mask]
        
        # 손실(Loss) 계산
        # 1. 크로스 엔트로피 손실(Cross-Entropy Loss) 계산
        data_loss = criterion(reshaped_outputs, reshaped_targets)
        data_loss = data_loss[path_mask].mean()
        
        # 2. ACT 논문에 언급된 Ponder Cost를 최종 손실에 추가합니다.
        # Ponder Cost는 avg_ponder_cost * time_penalty로 계산됩니다.
        loss = data_loss + time_penalty * avg_ponder_cost

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        # 학습 결과 누적
        train_loss += loss.item() * path_mask.size(0)
        total_pixels += path_mask.size(0)

        targets = targets.squeeze(1)
        predicted = outputs.argmax(1)

        batch_pix_acc = (predicted == targets).float().mean().item()
        correct += batch_pix_acc
        total += targets.size(0)

    # 최종 결과 반환
    train_loss = train_loss / total_pixels
    acc = 100.0 * correct / total
    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc
