#!/bin/sh
GPU=0
MODEL="models/configs_model/ematch/flow.yaml"
DATASET="datasets/configs_dataset/ematch/flow/dsec_train/stage1.yaml"

# -----------------------------------------------------------------------------------------------------
CHECKPOINT_DIR="checkpoints/ematch/flow/dsec/default/stage1"
RESUME=""

# CHECKPOINT_DIR="checkpoints/ematch/flow/dsec/dsecFlowBase/stage1"
# RESUME="checkpoints/ematch/disparity/dsec/default/stage1/final.pth"
# -----------------------------------------------------------------------------------------------------

mkdir -p ${CHECKPOINT_DIR} && CUDA_VISIBLE_DEVICES=${GPU} python train_dsec_ematch_flow.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--configs_model ${MODEL} \
--configs_dataset ${DATASET} \
--resume ${RESUME} \
--batch_size 4 \
--num_steps 400000 \
--lr 0.0003 \
--pct_start 0.01 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
