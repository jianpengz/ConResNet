#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=$RANDOM train_conresnet.py \
--data_dir='path-to-your-dataset/' \
--train_list='list/train_list.txt' \
--val_list='list/val_list.txt' \
--snapshot_dir='path-to-save-checkpoint/' \
--input_size='80,160,160' \
--batch_size=2 \
--num_gpus=2 \
--num_steps=40000 \
--val_pred_every=2000 \
--learning_rate=1e-4 \
--num_classes=3 \
--num_workers=4 \
--random_mirror=True \
--random_scale=True \
> path-to-save-log-file/log.file 2>&1 &
