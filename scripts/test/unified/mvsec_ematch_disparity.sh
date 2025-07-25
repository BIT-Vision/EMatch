#!/bin/sh
GPU=0
MODEL="models/configs_model/ematch/disparity.yaml"

# -----------------------------------------------------------------------------------------------------
CHECKPOINT="checkpoints/ematch/unified/mvsec_split1/default/final.pth"
DATASET="datasets/configs_dataset/ematch/disparity/mvsec_test/split1.yaml"
SAVE_DIR="outcomes/ematch/unified/mvsec_split1/default"

# CHECKPOINT="checkpoints/ematch/unified/mvsec_split3/default/final.pth"
# DATASET="datasets/configs_dataset/ematch/disparity/mvsec_test/split3.yaml"
# SAVE_DIR="outcomes/ematch/unified/mvsec_split3/default"
# -----------------------------------------------------------------------------------------------------

mkdir -p ${SAVE_DIR} && CUDA_VISIBLE_DEVICES=${GPU} python test_mvsec_ematch_disparity.py \
--configs_model ${MODEL} \
--checkpoint ${CHECKPOINT} \
--configs_dataset ${DATASET} \
--save_dir ${SAVE_DIR} \
2>&1 | tee -a ${SAVE_DIR}/test.log
