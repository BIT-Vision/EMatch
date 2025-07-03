#!/bin/sh
GPU=0
MODEL="models/configs_model/ematch/flow.yaml"
CHECKPOINT="checkpoints/ematch/flow/mvsec/default/dt1/out2"
DATA_PATH="datasets/configs_dataset/ematch/flow/mvsec_train/dt1/out2.yaml"

mkdir -p ${CHECKPOINT} && CUDA_VISIBLE_DEVICES=${GPU} python train_mvsec_ematch_flow.py \
--checkpoint_dir ${CHECKPOINT} \
--batch_size 8 \
--num_steps 300000 \
2>&1 | tee -a ${CHECKPOINT}/train.log
