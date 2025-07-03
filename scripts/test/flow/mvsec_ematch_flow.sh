#!/bin/sh
GPU=0
MODEL="models/configs_model/ematch/flow.yaml"

# -----------------------------------------------------------------------------------------------------
# CHECKPOINT="checkpoints/ematch/flow/mvsec_out2_dt1/default/final.pth"
# DATASET="datasets/configs_dataset/ematch/flow/mvsec_test/dt1.yaml"
# SAVE_DIR="outcomes/ematch/flow/mvsec_out2_dt1/default"

# CHECKPOINT="checkpoints/ematch/flow/mvsec_out2_dt1/dsecUnifiedBase/final.pth"
# DATASET="datasets/configs_dataset/ematch/flow/mvsec_test/dt1.yaml"
# SAVE_DIR="outcomes/ematch/flow/mvsec_out2_dt1/dsecUnifiedBase"
# -----------------------------------------------------------------------------------------------------
# CHECKPOINT="checkpoints/ematch/flow/mvsec_out2_dt4/default/final.pth"
# DATASET="datasets/configs_dataset/ematch/flow/mvsec_test/dt4.yaml"
# SAVE_DIR="outcomes/ematch/flow/mvsec_out2_dt4/default"

# CHECKPOINT="checkpoints/ematch/flow/mvsec_out2_dt4/dsecUnifiedBase/final.pth"
# DATASET="datasets/configs_dataset/ematch/flow/mvsec_test/dt4.yaml"
# SAVE_DIR="outcomes/ematch/flow/mvsec_out2_dt4/dsecUnifiedBase"
# -----------------------------------------------------------------------------------------------------
# CHECKPOINT="checkpoints/ematch/flow/mvsec_out2in3_dt1/default/final.pth"
# DATASET="datasets/configs_dataset/ematch/flow/mvsec_test/dt1.yaml"
# SAVE_DIR="outcomes/ematch/flow/mvsec_out2in3_dt1/default"

# CHECKPOINT="checkpoints/ematch/flow/mvsec_out2in3_dt1/dsecUnifiedBase/final.pth"
# DATASET="datasets/configs_dataset/ematch/flow/mvsec_test/dt1.yaml"
# SAVE_DIR="outcomes/ematch/flow/mvsec_out2in3_dt1/dsecUnifiedBase"
# -----------------------------------------------------------------------------------------------------
# CHECKPOINT="checkpoints/ematch/flow/mvsec_out2in3_dt4/default/final.pth"
# DATASET="datasets/configs_dataset/ematch/flow/mvsec_test/dt4.yaml"
# SAVE_DIR="outcomes/ematch/flow/mvsec_out2in3_dt4/default"

CHECKPOINT="checkpoints/ematch/flow/mvsec_out2in3_dt4/dsecUnifiedBase/final.pth"
DATASET="datasets/configs_dataset/ematch/flow/mvsec_test/dt4.yaml"
SAVE_DIR="outcomes/ematch/flow/mvsec_out2in3_dt4/dsecUnifiedBase"
# -----------------------------------------------------------------------------------------------------

mkdir -p ${SAVE_DIR} && CUDA_VISIBLE_DEVICES=${GPU} python test_mvsec_ematch_flow.py \
--configs_model ${MODEL} \
--checkpoint ${CHECKPOINT} \
--configs_dataset ${DATASET} \
--save_dir ${SAVE_DIR} \
2>&1 | tee -a ${SAVE_DIR}/test.log
