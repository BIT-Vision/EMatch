#!/bin/sh
GPU=0
MODEL="models/configs_model/ematch/unified.yaml"

# -----------------------------------------------------------------------------------------------------
DATASET_FLOW="datasets/configs_dataset/ematch/unified/mvsec_train/flow_setting/out2_dt1_flow.yaml"
DATASET_DISPARITY="datasets/configs_dataset/ematch/unified/mvsec_train/flow_setting/out2_dt1_disparity.yaml"
CHECKPOINT_DIR=checkpoints/ematch/unified/mvsec_out2_dt1/default

# DATASET_FLOW="datasets/configs_dataset/ematch/unified/mvsec_train/flow_setting/out2_dt4_flow.yaml"
# DATASET_DISPARITY="datasets/configs_dataset/ematch/unified/mvsec_train/flow_setting/out2_dt4_disparity.yaml"
# CHECKPOINT_DIR=checkpoints/ematch/unified/mvsec_out2_dt4/default
# -----------------------------------------------------------------------------------------------------
# DATASET_FLOW="datasets/configs_dataset/ematch/unified/mvsec_train/disparity_setting/split1_flow.yaml"
# DATASET_DISPARITY="datasets/configs_dataset/ematch/unified/mvsec_train/disparity_setting/split1_disparity.yaml"
# CHECKPOINT_DIR=checkpoints/ematch/unified/mvsec_split1/default

# DATASET_FLOW="datasets/configs_dataset/ematch/unified/mvsec_train/disparity_setting/split3_flow.yaml"
# DATASET_DISPARITY="datasets/configs_dataset/ematch/unified/mvsec_train/disparity_setting/split3_disparity.yaml"
# CHECKPOINT_DIR=checkpoints/ematch/unified/mvsec_split3/default
# -----------------------------------------------------------------------------------------------------

mkdir -p ${CHECKPOINT_DIR} && CUDA_VISIBLE_DEVICES=${GPU} python train_mvsec_ematch_unified.py --checkpoint_dir ${CHECKPOINT_DIR} \
--checkpoint_dir ${CHECKPOINT_DIR} \
--configs_model ${MODEL} \
--configs_dataset_flow ${DATASET_FLOW} \
--configs_dataset_disparity ${DATASET_DISPARITY} \
--batch_size 6 \
--num_steps 200000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
