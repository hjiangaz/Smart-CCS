#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train.py \
  --backbone_model CCS \
  --checkpoint path/to/CCS_vitL_100M.pth \
  --train_set splits/train.txt \
  --val_set splits/val.txt \
  --test_set splits/test.txt \
  --output_dir outputs \
  --batch_size 32 \
  --epoch 20 \
  --lr 1e-4 \
  --n_classes 7 \
  --in_dim 1024
