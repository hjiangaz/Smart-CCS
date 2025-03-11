#!/bin/sh
N_GPUS=4
DATA_ROOT=/new/data/path
OUTPUT_DIR=/new/output/path

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
test.py \
--num_queries 300 \
--num_classes 7 \
--eval_batch_size 120 \
--num_worker 12 \
--data_root ${DATA_ROOT} \
--dataset CCS \
--resume /path/to/ckpt \
--output_dir ${OUTPUT_DIR} 
