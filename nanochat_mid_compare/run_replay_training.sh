#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/driver/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=1
export HCCL_CONNECT_TIMEOUT=1800
export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_FUSION_ENABLE=1
export ASCEND_ENABLE_TRANSFORMER_FUSION=1

BATCH_DIR="/home/ma-user/work/nanochat_model_dir/mid_checkpoints/replay_batches"
START_STEP=320
END_STEP=330

torchrun --standalone --nproc_per_node=8 -m replay_training \
    --batch-dir="$BATCH_DIR" \
    --start-step=$START_STEP \
    --end-step=$END_STEP \
    --model-tag=base \
    --model-step=0 \
    --device-batch-size=8 \
    --max-seq-len=2048 \
    --total-batch-size=524288 \
    --embedding-lr=0.3 \
    --unembedding-lr=0.008 \
    --matrix-lr=0.02 \
    --weight-decay=0.28
