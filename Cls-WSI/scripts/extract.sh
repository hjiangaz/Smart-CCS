
CUDA_VISIBLE_DEVICES=0 nohup python -u extract.py \
--batch_size 8 \
--backbone_model CCS \
--train_mode feat \
--output_folder /path/to/output \
--dataset CCS_JSON_TOP \
--infer_set /path/to/cell_candidate.txt \
--checkpoint /path/to/CCS_ckpt.pth \
> /path/to/extract_log.log 2>&1 &
