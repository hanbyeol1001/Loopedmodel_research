
"""train.py
   Train, test, and save models
   Developed as part of DeepThinking2 project
   April 2021
"""

import argparse
import os
import sys
from collections import OrderedDict
import copy

from icecream import ic
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import warmup
from utils import train, test, OptimizerWithSched, load_model_from_checkpoint, \
    get_dataloaders, to_json, get_optimizer, to_log_file, now, get_model


# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


def main():

    print("\n_________________________________________________\n")
    print(now(), "train.py main() running.")

    parser = argparse.ArgumentParser(description="Deep Thinking")
    parser.add_argument("--checkpoint", default="check_default", type=str,
                        help="where to save the network")
    parser.add_argument("--data_path", default="../data", type=str, help="path to data files")
    parser.add_argument("--depth", default=1, type=int, help="depth of the network")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs for training")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--lr_factor", default=0.1, type=float, help="learning rate decay factor")
    parser.add_argument("--lr_schedule", nargs="+", default=[100, 150], type=int,
                        help="how often to decrease lr")
    parser.add_argument("--model", default="recur_resnet", type=str, help="model for training")
    parser.add_argument("--model_path", default=None, type=str, help="where is the model saved?")
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle",
                        help="shuffle training data?")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer")
    parser.add_argument("--output", default="output_default", type=str, help="output subdirectory")
    parser.add_argument("--quick_test", action="store_true", help="only test on eval data")
    parser.add_argument("--save_json", action="store_true", help="save json")
    parser.add_argument("--save_period", default=None, type=int, help="how often to save")
    parser.add_argument("--test_batch_size", default=500, type=int, help="batch size for testing")
    parser.add_argument("--test_iterations", default=None, type=int,
                        help="how many, if testing with a different number iterations")
    parser.add_argument("--test_mode", default="default", type=str, help="testing mode")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="batch size for training")
    parser.add_argument("--train_log", default="train_log.txt", type=str,
                        help="name of the log file")
    parser.add_argument("--train_mode", default="default", type=str, help="training mode")
    parser.add_argument("--val_period", default=20, type=int, help="how often to validate")
    parser.add_argument("--warmup_period", default=5, type=int, help="warmup period")
    parser.add_argument("--width", default=2, type=int, help="width of the network")

    args = parser.parse_args()
    print(args.shuffle)
    args.train_mode, args.test_mode = args.train_mode.lower(), args.test_mode.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.save_period is None:
        args.save_period = args.epochs

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # TensorBoard
    train_log = args.train_log
    try:
        array_task_id = train_log[:-4].split("_")[-1]
    except:
        array_task_id = 1
    to_log_file(args, args.output, train_log)
    writer = SummaryWriter(log_dir=f"{args.output}/runs/{train_log[:-4]}")

    ####################################################
    #               Dataset and Network and Optimizer
    trainloader, testloader = get_dataloaders(args.train_batch_size, args.test_batch_size, shuffle=args.shuffle)

    if args.model_path is not None:
        print(f"Loading model from checkpoint {args.model_path}...")
        net, start_epoch, optimizer_state_dict = load_model_from_checkpoint(args.model,
                                                                            args.model_path,
                                                                            args.width,
                                                                            args.depth)
        start_epoch += 1

    else:
        net = get_model(args.model, args.width, args.depth)
        start_epoch = 0
        optimizer_state_dict = None

    net = net.to(device)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    optimizer = get_optimizer(args.optimizer, args.model, net, args.lr)

    # print(net)
    print(f"This {args.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    print(f"Training will start at epoch {start_epoch}.")

    if optimizer_state_dict is not None:
        print(f"Loading optimizer from checkpoint {args.model_path}...")
        optimizer.load_state_dict(optimizer_state_dict)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=0)
    else:
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=args.warmup_period)

    lr_scheduler = MultiStepLR(optimizer, milestones=args.lr_schedule, gamma=args.lr_factor,
                               last_epoch=start_epoch-1)

    optimizer_obj = OptimizerWithSched(optimizer, lr_scheduler, warmup_scheduler)
    torch.backends.cudnn.benchmark = True
    ####################################################

    ####################################################
    #        Train
    print(f"==> Starting training for {args.epochs - start_epoch} epochs...")

    best_acc = 0
    best_epoch = 0
    best_model_state = None
    check = 0

    if best_model_state is not None:
        net.load_state_dict(best_model_state)
    elif args.model_path is not None:
        print("best_model_state가 None이므로, 체크포인트에서 모델을 불러옵니다.")
        net, _, _ = load_model_from_checkpoint(args.model, args.model_path, args.width, args.depth)
        best_model_state = copy.deepcopy(net.state_dict())  # ✅ 이 줄을 꼭 추가하세요
    else:
        print("Warning: best_model_state와 model_path 모두 없습니다. 랜덤 파라미터로 테스트합니다.")
    

    for epoch in range(start_epoch, args.epochs):
        loss, acc = train(net, trainloader, args.train_mode, optimizer_obj, device)
        for name, param in net.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"[Epoch {epoch}] {name} has NaN or Inf!")
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"[Epoch {epoch}] {name} gradient has NaN or Inf!")
                if torch.max(torch.abs(param.grad)) > 1e3:  # 예: 폭발하는 gradient 감지
                    print(f"[Epoch {epoch}] {name} gradient may be exploding! Max: {torch.max(torch.abs(param.grad))}")
            #일단 best model 저장만 해두고 test 시에 best model로 하는 방식으로
        if best_acc < acc:
            best_acc = acc
            best_model_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch
            check = 0
        else:
            if check == 30:
                break
            else:
                check += 1
        
        print(f"{now()} Training loss at epoch {epoch}: {loss}")
        print(f"{now()} Training accuracy at epoch {epoch}: {acc}")
        print(f"{now()} Training accuracy at best epoch {best_epoch}:{best_acc}")
        print(f"Early Stopping Check Point : {check}")

        # if the loss is nan, then stop the training
        if np.isnan(float(loss)):
            print(f"{ic.format()} Loss is nan, exiting...")
            sys.exit()

        # TensorBoard loss writing
        writer.add_scalar("Loss/loss", loss, epoch)
        writer.add_scalar("Accuracy/acc", acc, epoch)

        for i in range(len(optimizer.param_groups)):
            writer.add_scalar(f"Learning_rate/group{i}", optimizer.param_groups[i]["lr"], epoch)

        # 이 부분 잘 이해 못함 : 선택적으로 수행하는거 같아서 일단 skip
        if (epoch + 1) % args.val_period == 0:
            train_acc = test(net, trainloader, args.test_mode, device)
            test_acc = test(net, testloader, args.test_mode, device)

            print(f"{now()} Training accuracy: {train_acc}")
            print(f"{now()} Testing accuracy: {test_acc}")

            stats = [train_acc, test_acc]
            stat_names = ["train_acc", "test_acc"]
            for stat_idx, stat in enumerate(stats):
                stat_name = os.path.join("val", stat_names[stat_idx])
                writer.add_scalar(stat_name, stat, epoch)

        # 학습 끝나고
        if (epoch + 1) % args.save_period == 0 or (epoch + 1) == args.epochs:
            print(f"Save best model weight at epoch {best_epoch}")
            state = {
                "net": best_model_state,
                "epoch": best_epoch,
                "optimizer": optimizer.state_dict()
            }
            out_str = os.path.join(args.checkpoint,
                                   f"{args.model}_{args.optimizer}"
                                   f"_depth={args.depth}"
                                   f"_width={args.width}"
                                   f"_lr={args.lr}"
                                   f"_batchsize={args.train_batch_size}"
                                   f"_at{best_epoch}"
                                   f"_epoch={args.epochs-1}"
                                   f"_{array_task_id}.pth")

            print(f"{now()} Saving model to: ", args.checkpoint, " out_str: ", out_str)
            if not os.path.isdir(args.checkpoint):
                os.makedirs(args.checkpoint)
            torch.save(state, out_str)
        

    writer.flush()
    writer.close()
    ####################################################

    ####################################################
    #        Test
    print("==> Starting testing...")
    net.load_state_dict(best_model_state)
    

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
