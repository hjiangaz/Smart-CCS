#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python test.py \
  --backbone_model CCS \
  --checkpoint path/to/CCS_vitL_100M.pth \
  --test_set splits/test.txt \
  --classifier_ckpt outputs/best_model.pth \
  --batch_size 32 \
  --n_classes 7 \
  --in_dim 1024
