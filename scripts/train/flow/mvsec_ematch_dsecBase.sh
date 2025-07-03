#!/bin/sh
GPU=0
CHECKPOINT=checkpoints/ematch/flow/mvsec/dsecUnifiedBase_dt1_addin3

mkdir -p ${CHECKPOINT} && CUDA_VISIBLE_DEVICES=3 python train_mvsec_ematch_flow.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume ./checkpoints/ematch/unified/dsec/default/stage1_base/step_400000.pth \
--batch_size 8 \
--num_steps 300000 \
2>&1 | tee -a ${CHECKPOINT}/train.log
