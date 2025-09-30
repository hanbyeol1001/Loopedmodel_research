#!/bin/bash

# 첫 번째 실험
python main.py --model ut_act --depth 4 --width 4 --epochs 20 --train_batch_size 128 --test_batch_size 128 --lr 0.001 \
	        --val_period 10 --train_maze_size 9 --test_maze_size 9 --quick_test --val_ratio 0.2 --save_json

# 두 번째 실험
python main.py --model maze_ut --depth 4 --width 4 --epochs 20 --train_batch_size 128 --test_batch_size 128 --lr 0.001 \
	--val_period 10 --train_maze_size 9 --test_maze_size 9 --quick_test --val_ratio 0.2 --save_json

# 세 번째 실험 (ponder cost 적용한 ut_act)
python main.py --model ut_act --depth 4 --width 4 --epochs 20 --train_batch_size 128 --test_batch_size 128 --lr 0.001 \
	--val_period 10 --train_maze_size 9 --test_maze_size 9 --quick_test --val_ratio 0.2 --save_json --train_mode act

# 네 번째 실험 (ponder cost 추가하여 수정된 코드로 maze_ut: 두 번째 실험과 동일한 결과 나와야 함.)
python main.py --model maze_ut --depth 4 --width 4 --epochs 20 --train_batch_size 128 --test_batch_size 128 --lr 0.001 \                                                                              
	        --val_period 10 --train_maze_size 9 --test_maze_size 9 --quick_test --val_ratio 0.2 --save_json  --train_mode act
