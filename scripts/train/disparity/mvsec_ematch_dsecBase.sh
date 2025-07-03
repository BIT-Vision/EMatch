#!/bin/sh
GPU=0
MODEL="models/configs_model/ematch/disparity.yaml"

# -----------------------------------------------------------------------------------------------------
DATASET="datasets/configs_dataset/ematch/disparity/mvsec_train/split1.yaml"
CHECKPOINT_DIR="checkpoints/ematch/disparity/mvsec_split1/dsecUnifiedBase"
RESUME="checkpoints/ematch/unified/dsec/default/stage1/final.pth"

# DATASET="datasets/configs_dataset/ematch/disparity/mvsec_train/split3.yaml"
# CHECKPOINT_DIR="checkpoints/ematch/disparity/mvsec_split3/dsecUnifiedBase"
# RESUME="checkpoints/ematch/unified/dsec/default/stage1/final.pth"
# -----------------------------------------------------------------------------------------------------

mkdir -p ${CHECKPOINT_DIR} && CUDA_VISIBLE_DEVICES=${GPU} python train_mvsec_ematch_disparity.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--configs_model ${MODEL} \
--configs_dataset ${DATASET} \
--resume ${RESUME} \
--batch_size 4 \
--num_steps 200000 \
--configs_dataset ./datasets/configs_dataset/ematch/disparity/mvsec_train_split3.yaml \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
