CLASSIFIER=MeanMIL
LOG_DIR=/path/to/train_log

CUDA_VISIBLE_DEVICES=1 nohup python -u train.py \
--backbone_model CCS \
--classifier ${CLASSIFIER} \
--n_classes 7 \
--act relu \
--lr 1e-4 \
--batch_size 1 \
--epoch 50 \
--train_mode feat \
--selection_K 100 \
--dataset CCS_feat_TOP \
--train_set /path/to/train \
--test_set /path/to/test \
--val_set /path/to/val \
--output_folder /path/to/output/train_${CLASSIFIER} \
> ${LOG_DIR}/train_${CLASSIFIER}.log 2>&1 &
