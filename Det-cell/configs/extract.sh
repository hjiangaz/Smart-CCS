#!/bin/sh
OUTPUT_DIR=/path/to/output
DATA_ROOT=/new/data/path

CUDA_VISIBLE_DEVICES=3 python -u \
extract.py \
--eval_batch_size 20 \
--num_classes 7 \
--data_root ${DATA_ROOT} \
--num_queries 300 \
--num_worker 12 \
--dataset CCS \
--resume /path/to/det_ckpt \
--output_dir ${OUTPUT_DIR} 
