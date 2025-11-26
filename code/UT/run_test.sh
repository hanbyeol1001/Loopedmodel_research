#!/bin/bash


# UT with ACT 9x9 train -> 13x13 test
python main.py \
--model ut_act \
--epochs 0 \
--train_batch_size 64 \
--test_batch_size 64 \
--lr 0.001 \
--train_maze_size 9 \
--test_maze_size 13 \
--quick_test \
--save_json \
--train_mode act \
--model_path check_default/ut_act_adam_lr=0.001_batchsize=64_at20_epoch=29_start=2025-11-13_01-33_log.pth

# UT with ACT 9x9 train -> 15x15 test
python main.py \
--model ut_act \
--epochs 0 \
--train_batch_size 64 \
--test_batch_size 64 \
--lr 0.001 \
--train_maze_size 9 \
--test_maze_size 15 \
--quick_test \
--save_json \
--train_mode act \
--model_path check_default/ut_act_adam_lr=0.001_batchsize=64_at20_epoch=29_start=2025-11-13_01-33_log.pth


# UT with ACT 13x13 train -> 9x9 test
python main.py \
--model ut_act \
--epochs 0 \
--train_batch_size 64 \
--test_batch_size 64 \
--lr 0.001 \
--train_maze_size 13 \
--test_maze_size 9 \
--quick_test \
--save_json \
--train_mode act \
--model_path check_default/ut_act_adam_lr=0.001_batchsize=64_at28_epoch=29_start=2025-11-13_02-27_log.pth

# UT with ACT 13x13 train -> 15x15 test
python main.py \
--model ut_act \
--epochs 0 \
--train_batch_size 64 \
--test_batch_size 64 \
--lr 0.001 \
--train_maze_size 13 \
--test_maze_size 15 \
--quick_test \
--save_json \
--train_mode act \
--model_path check_default/ut_act_adam_lr=0.001_batchsize=64_at28_epoch=29_start=2025-11-13_02-27_log.pth


# UT with ACT 15x15 train -> 9x9 test
python main.py \
--model ut_act \
--epochs 0 \
--train_batch_size 64 \
--test_batch_size 64 \
--lr 0.001 \
--train_maze_size 15 \
--test_maze_size 9 \
--quick_test \
--save_json \
--train_mode act \
--model_path check_default/ut_act_adam_lr=0.001_batchsize=64_at27_epoch=29_start=2025-11-13_04-27_log.pth

# UT with ACT 15x15 train -> 13x13 test
python main.py \
--model ut_act \
--epochs 0 \
--train_batch_size 64 \
--test_batch_size 64 \
--lr 0.001 \
--train_maze_size 15 \
--test_maze_size 13 \
--quick_test \
--save_json \
--train_mode act \
--model_path check_default/ut_act_adam_lr=0.001_batchsize=64_at27_epoch=29_start=2025-11-13_04-27_log.pth




# # UT without ACT 9x9 train -> 13x13 test
# python main.py \
# --model maze_ut \
# --epochs 0 \
# --train_batch_size 64 \
# --test_batch_size 64 \
# --lr 0.001 \
# --train_maze_size 9 \
# --test_maze_size 13 \
# --quick_test \
# --save_json \
# --model_path check_default/maze_ut_adam_lr=0.001_batchsize=64_at26_epoch=29_start=2025-11-12_17-31_log.pth

# # UT without ACT 9x9 train -> 15x15 test
# python main.py \
# --model maze_ut \
# --epochs 0 \
# --train_batch_size 64 \
# --test_batch_size 64 \
# --lr 0.001 \
# --train_maze_size 9 \
# --test_maze_size 15 \
# --quick_test \
# --save_json \
# --model_path check_default/maze_ut_adam_lr=0.001_batchsize=64_at26_epoch=29_start=2025-11-12_17-31_log.pth


# # UT without ACT 13x13 train -> 9x9 test
# python main.py \
# --model maze_ut \
# --epochs 0 \
# --train_batch_size 64 \
# --test_batch_size 64 \
# --lr 0.001 \
# --train_maze_size 13 \
# --test_maze_size 9 \
# --quick_test \
# --save_json \
# --model_path check_default/maze_ut_adam_lr=0.001_batchsize=64_at28_epoch=29_start=2025-11-12_17-55_log.pth

# # UT without ACT 13x13 train -> 15x15 test
# python main.py \
# --model maze_ut \
# --epochs 0 \
# --train_batch_size 64 \
# --test_batch_size 64 \
# --lr 0.001 \
# --train_maze_size 13 \
# --test_maze_size 15 \
# --quick_test \
# --save_json \
# --model_path check_default/maze_ut_adam_lr=0.001_batchsize=64_at28_epoch=29_start=2025-11-12_17-55_log.pth


# # UT without ACT 15x15 train -> 9x9 test
# python main.py \
# --model maze_ut \
# --epochs 0 \
# --train_batch_size 64 \
# --test_batch_size 64 \
# --lr 0.001 \
# --train_maze_size 15 \
# --test_maze_size 9 \
# --quick_test \
# --save_json \
# --model_path check_default/maze_ut_adam_lr=0.001_batchsize=64_at29_epoch=29_start=2025-11-12_18-43_log.pth

# # UT without ACT 15x15 train -> 13x13 test
# python main.py \
# --model maze_ut \
# --epochs 0 \
# --train_batch_size 64 \
# --test_batch_size 64 \
# --lr 0.001 \
# --train_maze_size 15 \
# --test_maze_size 13 \
# --quick_test \
# --save_json \
# --model_path check_default/maze_ut_adam_lr=0.001_batchsize=64_at29_epoch=29_start=2025-11-12_18-43_log.pth