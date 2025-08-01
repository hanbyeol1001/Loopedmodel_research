train.py 에 argument 설명 존재



> example : UT
```!python train.py \
--model maze_ut \
--depth 4 \
--width 4 \
--epochs 20 \
--train_batch_size 128 \
--lr 0.001 \
--train_maze_size 9 \
--test_maze_size 9\
--model_path ./file_name.pth
```

> example : UT+ACT
```!python train.py \
--model ut_act \
--depth 4 \
--width 4 \
--epochs 20 \
--train_batch_size 128 \
--lr 0.001 \
--train_maze_size 9 \
--test_maze_size 9\
--model_path ./file_name.pth
```
