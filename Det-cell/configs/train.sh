N_GPUS=2
BATCH_SIZE=8
DATA_ROOT=/new/data/path
OUTPUT_DIR=/new/output/path

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26500 \
--nproc_per_node=${N_GPUS} \
train.py \
--num_queries 300 \
--num_classes 7 \
--data_root ${DATA_ROOT} \
--dataset CCS \
--batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--epoch 100 \
--epoch_lr_drop 40 \
--output_dir ${OUTPUT_DIR} 
