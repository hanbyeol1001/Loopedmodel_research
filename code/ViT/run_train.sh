#!/bin/bash

# ViT without ACT 9x9 train -> 13x13 test
python train.py \
--model maze_vitut \
--epochs 200 \
--train_batch_size 64 \
--test_batch_size 64 \
--lr 0.001 \
--warmup_period 5 \
--train_maze_size 9 \
--test_maze_size 13 \
--val_period 20 \
--save_period 50 \
--quick_test \
--save_json


# ViT with ACT 9x9 train -> 13x13 test
python train.py \
--model maze_vitutact \
--epochs 200 \
--train_batch_size 64 \
--test_batch_size 64 \
--lr 0.001 \
--warmup_period 5 \
--train_maze_size 9 \
--test_maze_size 13 \
--val_period 20 \
--save_period 50 \
--quick_test \
--train_mode default \
--save_json