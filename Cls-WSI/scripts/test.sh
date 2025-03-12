CLASSIFIER=MeanMIL
LOG_DIR=/path/to/log

CUDA_VISIBLE_DEVICES=1 nohup python -u test.py \
--backbone_model CCS \
--classifier ${CLASSIFIER} \
--n_classes 7 \
--checkpoint /path/to/backbone_ckpt \
--classifier_ckpt /path/to/classifier_ckpt \
--selection_K 100 \
--dataset CCS_JSON_TOP \
--infer_set /path/to/infer_set \
--output_folder /path/to/output \
> ${LOG_DIR}/results.log 2>&1 &