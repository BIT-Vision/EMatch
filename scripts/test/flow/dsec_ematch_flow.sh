#!/bin/sh
GPU=0
MODEL="models/configs_model/ematch/flow.yaml"
DATASET="datasets/configs_dataset/ematch/flow/dsec_test/eval.yaml"

# -----------------------------------------------------------------------------------------------------
# CHECKPOINT="checkpoints/ematch/flow/dsec/dsecDisparityBase/stage2/final.pth"
# SAVE_DIR="outcomes/ematch/flow/dsec/dsecDisparityBase/stage2/flow"

CHECKPOINT="checkpoints/ematch/flow/dsec/default/stage2/final.pth"
SAVE_DIR="outcomes/ematch/flow/dsec/default/stage2/flow"
# -----------------------------------------------------------------------------------------------------

mkdir -p ${SAVE_DIR} && CUDA_VISIBLE_DEVICES=${GPU} python submission_dsec_ematch_flow.py \
--configs_model ${MODEL} \
--configs_dataset ${DATASET} \
--checkpoint ${CHECKPOINT} \
--save_dir ${SAVE_DIR} \
2>&1 | tee -a ${SAVE_DIR}/test.log
