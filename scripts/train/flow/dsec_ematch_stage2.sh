#!/bin/sh
GPU=0
MODEL="models/configs_model/ematch/flow.yaml"
DATASET="datasets/configs_dataset/ematch/flow/dsec_train/stage2.yaml"

# -----------------------------------------------------------------------------------------------------
CHECKPOINT_DIR="checkpoints/ematch/flow/dsec/default/stage2"
RESUME="checkpoints/ematch/flow/dsec/default/stage1/final.pth"

# CHECKPOINT_DIR="checkpoints/ematch/flow/dsec/dsecFlowBase/stage2"
# RESUME="checkpoints/ematch/flow/dsec/dsecFlowBase/stage1/final.pth"
# -----------------------------------------------------------------------------------------------------

mkdir -p ${CHECKPOINT_DIR} && CUDA_VISIBLE_DEVICES=${GPU} python train_dsec_ematch_flow.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--configs_model ${MODEL} \
--configs_dataset ${DATASET} \
--resume ${RESUME} \
--batch_size 1 \
--num_steps 150000 \
--lr 0.00001 \
--pct_start 0 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
