train.py 에 argument 설명 존재



> example : Only ViT
```!python train.py \
--model maze_vitut \
--depth 4 \
--width 4 \
--epochs 20 \
--train_batch_size 128 \
--lr 0.001 \
--train_maze_size 9 \
--test_maze_size 9
```

> example : ViT+ACT
```!python train.py \
--model maze_vitut_act \
--depth 4 \
--width 4 \
--epochs 20 \
--train_batch_size 128 \
--lr 0.001 \
--train_maze_size 9 \
--test_maze_size 9
```
