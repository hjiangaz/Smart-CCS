#!/usr/bin/env bash
LOG_DIR=path/to/output
CONFIG=configs/tta_config.yaml

CUDA_VISIBLE_DEVICES=0 nohup python -u test_tta.py \
    --config ${CONFIG} \
    > ${LOG_DIR}/results_TTA.log 2>&1 &
