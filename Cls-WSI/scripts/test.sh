CLASSIFIER=MeanMIL
LOG_DIR=path/to/output

CUDA_VISIBLE_DEVICES=0 nohup python -u test.py \
--backbone_model CCS \
--classifier ${CLASSIFIER} \
--n_classes 7 \
--checkpoint path/to/CCS_vitL_100M.pth \
--classifier_ckpt path/to/MeanMIL.pth \
--selection_K 100 \
--dataset CCS_JSON_TOP \
--infer_set path/to/test_set.txt \
--output_folder path/to/output \
> ${LOG_DIR}/results.log 2>&1 &
