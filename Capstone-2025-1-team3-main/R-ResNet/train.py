import copy
import argparse
import os
import sys
from collections import OrderedDict

from icecream import ic
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from models.recur_resnet_act import recur_resnet_act

import warmup
from utils import train, test, OptimizerWithSched, load_model_from_checkpoint, \
    get_dataloaders, to_json, get_optimizer, to_log_file, now, get_model, AllLogger

def main():
    print("\n_________________________________________________\n")
    print(now(), "train.py main() running.")

    # 다양한 하이퍼파라미터와 옵션을 커맨드라인에서 설정할 수 있도록
    parser = argparse.ArgumentParser(description="Deep Thinking")
    parser.add_argument("--checkpoint", default="check_default", type=str,
                        help="where to save the network")
    parser.add_argument("--data_path", default="../data", type=str, help="path to data files")
    parser.add_argument("--depth", default=1, type=int, help="depth of the network")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs for training")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--lr_factor", default=0.1, type=float, help="learning rate decay factor")  # 학습률 감소 비율
    parser.add_argument("--lr_schedule", nargs="+", default=[100, 150], type=int,
                        help="how often to decrease lr")  # 학습률 감소 시점들. nargs="+"는 하나 이상 받겠다는 뜻
    parser.add_argument("--model", default="recur_resnet", type=str, help="model for training")
    parser.add_argument("--model_path", default=None, type=str, help="where is the model saved?")  # 체크포인트 로드 경로
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle",
                        help="shuffle training data?")  # --no_shuffle을 넣으면 shuffle <- false
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer")
    parser.add_argument("--output", default="output_default", type=str, help="output subdirectory")
    parser.add_argument("--quick_test", action="store_true", help="only test on eval data")  # --quick_test를 넣으면 true
    parser.add_argument("--save_json", action="store_true", help="save json")
    parser.add_argument("--save_period", default=None, type=int, help="how often to save")   # 몇 에폭마다 저장할지
    parser.add_argument("--test_batch_size", default=500, type=int, help="batch size for testing")
    parser.add_argument("--test_iterations", default=None, type=int,
                        help="how many, if testing with a different number iterations")
    parser.add_argument("--test_mode", default="default", type=str, help="testing mode")  # 테스트 모드: 왜 있는 지 모르겠음
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="batch size for training")
    parser.add_argument("--train_log", default="train_log.txt", type=str,
                        help="name of the log file")
    parser.add_argument("--train_mode", default="default", type=str, help="training mode")  # 훈련 모드: 왜 있는 지 모르겠음
    parser.add_argument("--val_period", default=20, type=int, help="how often to validate")  # 검증 주기 (에폭 단위)
    parser.add_argument("--warmup_period", default=5, type=int, help="warmup period")  # warmup 적용 에폭 수: 뭔지 잘 모르겠음
    parser.add_argument("--width", default=4, type=int, help="width of the network")  # 모델 너비
    parser.add_argument("--test_maze_size", default=13, type=int, help="test_maze_size")  # 13x13 사이즈 미로
    parser.add_argument("--train_maze_size", default=9, type=int, help="train_maze_size")  # 9x9 사이즈 미로


    args = parser.parse_args()  # CLI 인자 파싱
    print(args.shuffle)
    args.train_mode, args.test_mode = args.train_mode.lower(), args.test_mode.lower()  # 모드 소문자로 정리
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 여부 설정

    if args.save_period is None:
        args.save_period = args.epochs  # 저장 주기 기본값 설정 (학습 마지막 시점)

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")  # 모든 인자 출력
        
    train_log = args.train_log   # 로그 파일 이름에서 task ID 추출 (근데 뒤에서 사용되지 않음.)
    try:
        array_task_id = train_log[:-4].split("_")[-1]  # 예: "train_log_23.txt" -> "23"
    except:
        array_task_id = 1
    
    # 로그 파일 저장 및 텐서보드 로그 디렉토리 설정
    to_log_file(args, args.output, train_log)  # args 전체를 로그 파일로 기록
    writer = SummaryWriter(log_dir=f"{args.output}/runs/{train_log[:-4]}")  # TensorBoard 로그 디렉토리 생성

    ####################################################
    #         Dataset and Network and Optimizer'
    ## 데이터 로더
    trainloader, testloader = get_dataloaders(
        args.train_batch_size, args.test_batch_size, args.train_maze_size, 
        args.test_maze_size, shuffle=args.shuffle
        )

    ## 모델 로드 or 생성
    if args.model_path is not None:
        net = recur_resnet_act(
            args.depth, args.width, ponder_epsilon=0.01, time_penalty=0.01, 
            max_iters=int((args.depth - 4) // 4)
            )
        print(f"Loading model from checkpoint {args.model_path}...")
        net, start_epoch, optimizer_state_dict = load_model_from_checkpoint(net,
                                                                            args.model_path,
                                                                            args.width,
                                                                            args.depth)
        start_epoch += 1

    else:
        net = recur_resnet_act(args.depth, args.width, ponder_epsilon=0.01, time_penalty=0.005, max_iters=int((args.depth - 4) // 4))
        # net = get_model(args.model, args.depth, args.width)
        print(f'max_iters: {int((args.depth - 4) // 4)}')
        start_epoch = 0
        optimizer_state_dict = None

    net = net.to(device)
    pytorch_total_params = sum(p.numel() for p in net.parameters())

    ## 옵티마이저 생성
    optimizer = get_optimizer(args.optimizer, args.model, net, args.lr)

    # print(net)
    print(f"This {args.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    print(f"Training will start at epoch {start_epoch}.")

    ## 옵티마이저 로드 + warmup scheduler 설정
    if optimizer_state_dict is not None:
        print(f"Loading optimizer from checkpoint {args.model_path}...")
        optimizer.load_state_dict(optimizer_state_dict)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=0)
    else:
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=args.warmup_period)

    ## 학습률 스케줄러 설정
    lr_scheduler = MultiStepLR(optimizer, milestones=args.lr_schedule, gamma=args.lr_factor,
                               last_epoch=start_epoch-1)
    
    ## 옵티마이저 통합 객체 생성
    optimizer_obj = OptimizerWithSched(optimizer, lr_scheduler, warmup_scheduler)
    torch.backends.cudnn.benchmark = True  # cudnn 성능 최적화

    ####################################################
    #         Train
    print(f"==> Starting training for {args.epochs - start_epoch} epochs...")
    train_losses = []
    train_accuracies = []
    # best_acc = 0
    # best_epoch = 0
    # best_model_state = None
    check = 0

    for epoch in range(start_epoch, args.epochs):
        loss, acc, net = train(net, trainloader, args.train_mode, optimizer_obj, device)

        train_losses.append(loss)    
        train_accuracies.append(acc)  
        
        print(f"epoch {epoch}")
        print(f"{now()} Training loss at epoch {epoch}: {loss}")
        print(f"{now()} Training accuracy at epoch {epoch}: {acc}")

      
        if np.isnan(float(loss)):
            print(f"{ic.format()} Loss is nan, exiting...")
            sys.exit()

        # 텐서보드 기록
        writer.add_scalar("Loss/train_loss", loss, epoch)
        writer.add_scalar("Accuracy/train_acc", acc, epoch)

        for i in range(len(optimizer.param_groups)):
            writer.add_scalar(f"Learning_rate/group{i}", optimizer.param_groups[i]["lr"], epoch)

        # validation
        if (epoch + 1) % args.val_period == 0:
            train_acc = test(net, trainloader, args.test_mode, device)
            # test_acc = test(net, testloader, args.test_mode, device)

            print(f"Val_period")
            print(f"{now()} Validation accuracy: {train_acc}")
            # print(f"{now()} Testing accuracy: {test_acc}")
            # writer.add_scalar("Accuracy/test_acc_in_training", test_acc, epoch)

            # stats = [train_acc, test_acc]
            # stat_names = ["train_acc", "test_acc"]
            # for stat_idx, stat in enumerate(stats):
            #     stat_name = os.path.join("val", stat_names[stat_idx])
            #     writer.add_scalar(stat_name, stat, epoch)

        # 모델 저장
        if (epoch + 1) % args.save_period == 0 or (epoch + 1) == args.epochs or check >= 10:
            print(f"Save best model weight at epoch {epoch}")
            state = {
                # "net": best_model_state,
                "net": net.state_dict(),
                "epoch": epoch,
                "optimizer": optimizer.state_dict()
            }
            
            out_str = os.path.join(args.checkpoint,
                                   f"recur_resnet_act_{args.optimizer}"
                                   f"_depth={args.depth}"
                                   f"_width={args.width}"
                                   f"_lr={args.lr}"
                                   f"_batchsize={args.train_batch_size}"
                                   f"_at{epoch}"
                                   f"_epoch={args.epochs-1}"
                                   f"_{array_task_id}.pth")

            print(f"{now()} Saving model to: ", args.checkpoint, " out_str: ", out_str)
            if not os.path.isdir(args.checkpoint):
                os.makedirs(args.checkpoint)
            torch.save(state, out_str)

    writer.flush()
    writer.close()

    epochs = list(range(start_epoch + 1, args.epochs + 1))
    plt.figure(figsize=(12, 5))

    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o', color='blue', label='Train Loss')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, marker='o', color='green', label='Train Accuracy')
    plt.title("Training Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    os.makedirs(args.output, exist_ok=True)
    save_filename = (
        f"train_metrics"
        f"_model={args.model}"
        f"_depth={args.depth}"
        f"_width={args.width}"
        f"_lr={args.lr}"
        f"_batch={args.train_batch_size}"
        f"_epoch={args.epochs}"
        f"_testiter={args.test_iterations if args.test_iterations is not None else -1}.png"
    )
        
    save_path = os.path.join(args.output, save_filename)
    plt.savefig(save_path)  # 그래프 사진 저장
    print(f"✅ Training loss & accuracy plot saved to: {save_path}")
    plt.close()

    txt_log_path = os.path.join(args.output, save_filename + ".txt")
    sys.stdout = AllLogger(txt_log_path)
    sys.stderr = sys.stdout

    ####################################################
    #         Test
    print("==> Starting testing...")
    # net.load_state_dict(best_model_state)
    
    if args.test_iterations is not None:
        net.max_iters = args.test_iterations
    else:
        args.test_iterations = -1

    test_acc = test(net, testloader, args.test_mode, device)
    train_acc = -1 if args.quick_test else test(net, trainloader, args.test_mode, device)    

    print(f"{now()} Training accuracy: {train_acc}")
    print(f"{now()} Testing accuracy: {test_acc}")

    model_name_str = f"{args.model}_depth={args.depth}_width={args.width}"
    stats = OrderedDict([("epochs", args.epochs),
                          ("learning rate", args.lr),
                          ("lr", args.lr),
                          ("lr_factor", args.lr_factor),
                          ("model", model_name_str),
                          ("num_params", pytorch_total_params),
                          ("optimizer", args.optimizer),
                          ("test_acc", test_acc),
                          ("test_iter", args.test_iterations),
                          ("test_mode", args.test_mode),
                          ("train_acc", train_acc),
                          ("train_batch_size", args.train_batch_size),
                          ("train_mode", args.train_mode)])

    if args.save_json:
        to_json(stats, args.output)
    ####################################################


if __name__ == "__main__":
    main()
