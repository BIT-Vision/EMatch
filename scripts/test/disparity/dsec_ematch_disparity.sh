#!/bin/sh
GPU=0
MODEL="models/configs_model/ematch/disparity.yaml"
DATASET="datasets/configs_dataset/ematch/disparity/dsec_test/eval.yaml"

# -----------------------------------------------------------------------------------------------------
# CHECKPOINT="checkpoints/ematch/disparity/dsec/dsecFlowBase/stage2/final.pth"
# SAVE_DIR="outcomes/ematch/disparity/dsec/dsecFlowBase/stage2"

CHECKPOINT="checkpoints/ematch/disparity/dsec/default/stage2/final.pth"
SAVE_DIR="outcomes/ematch/disparity/dsec/default/stage2"
# -----------------------------------------------------------------------------------------------------

mkdir -p ${SAVE_DIR} && CUDA_VISIBLE_DEVICES=${GPU} python submission_dsec_ematch_disparity.py \
--configs_model ${MODEL} \
--configs_dataset ${DATASET} \
--checkpoint ${CHECKPOINT} \
--save_dir ${SAVE_DIR} \
2>&1 | tee -a ${SAVE_DIR}/test.log
