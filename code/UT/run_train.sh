#!/bin/bash


# UT with ACT 9x9 train -> 9x9 test
python main.py \
--model ut_act \
--epochs 30 \
--train_batch_size 64 \
--test_batch_size 64 \
--lr 0.001 \
--train_maze_size 9 \
--test_maze_size 9 \
--quick_test \
--save_json \
--val_ratio 0.2 \
--val_period 10 \
--train_mode act

# UT with ACT 13x13 train -> 13x13 test
python main.py \
--model ut_act \
--epochs 30 \
--train_batch_size 64 \
--test_batch_size 64 \
--lr 0.001 \
--train_maze_size 13 \
--test_maze_size 13 \
--quick_test \
--save_json \
--val_ratio 0.2 \
--val_period 10 \
--train_mode act


# UT with ACT 15x15 train -> 15x15 test
python main.py \
--model ut_act \
--epochs 30 \
--train_batch_size 64 \
--test_batch_size 64 \
--lr 0.001 \
--train_maze_size 15 \
--test_maze_size 15 \
--quick_test \
--save_json \
--val_ratio 0.2 \
--val_period 10 \
--train_mode act




# # UT without ACT 9x9 train -> 9x9 test
# python main.py \
# --model maze_ut \
# --epochs 30 \
# --train_batch_size 64 \
# --test_batch_size 64 \
# --lr 0.001 \
# --train_maze_size 9 \
# --test_maze_size 9 \
# --quick_test \
# --save_json \
# --val_ratio 0.2 \
# --val_period 10 


# # UT without ACT 13x13 train -> 13x13 test
# python main.py \
# --model maze_ut \
# --epochs 30 \
# --train_batch_size 64 \
# --test_batch_size 64 \
# --lr 0.001 \
# --train_maze_size 13 \
# --test_maze_size 13 \
# --quick_test \
# --save_json \
# --val_ratio 0.2 \
# --val_period 10 


# # UT without ACT 15x15 train -> 15x15 test
# python main.py \
# --model maze_ut \
# --epochs 30 \
# --train_batch_size 64 \
# --test_batch_size 64 \
# --lr 0.001 \
# --train_maze_size 15 \
# --test_maze_size 15 \
# --quick_test \
# --save_json \
# --val_ratio 0.2 \
# --val_period 10