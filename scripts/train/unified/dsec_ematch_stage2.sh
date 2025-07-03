#!/bin/sh
GPU=0
MODEL="models/configs_model/ematch/unified.yaml"
DATASET_FLOW="datasets/configs_dataset/ematch/unified/dsec_train/stage2_flow.yaml"
DATASET_DISPARITY="datasets/configs_dataset/ematch/unified/dsec_train/stage2_disparity.yaml"
CHECKPOINT_DIR="checkpoints/ematch/unified/dsec/default/stage2"
RESUME="checkpoints/ematch/unified/dsec/default/stage1/final.pth"

mkdir -p ${CHECKPOINT_DIR} && CUDA_VISIBLE_DEVICES=${GPU} python train_dsec_ematch_unified.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--configs_model ${MODEL} \
--configs_dataset_flow ${DATASET_FLOW} \
--configs_dataset_disparity ${DATASET_DISPARITY} \
--resume ${RESUME} \
--batch_size 1 \
--num_steps 150000 \
--lr 0.00001 \
--pct_start 0 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
