#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NANOCHAT_REPO="${NANOCHAT_REPO:-/home/ma-user/work/nanochat_midtrain_326}"
NANOCHAT_MODEL_DIR="${NANOCHAT_MODEL_DIR:-/home/ma-user/work/nanochat_model_dir}"
NUM_NPU=8

# ══════ CONFIGURATION ══════
BATCH_DIR="${BATCH_DIR:-$NANOCHAT_MODEL_DIR/mid_checkpoints/replay_batches}"
START_STEP="${START_STEP:-320}"
END_STEP="${END_STEP:-330}"
BASE_MODEL_TAG="${BASE_MODEL_TAG:-d24_0320}"
BASE_MODEL_STEP="${BASE_MODEL_STEP:-6612}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-}"
NUM_ITERATIONS="${NUM_ITERATIONS:-11}"

MID_CKPT_DIR="$NANOCHAT_MODEL_DIR/mid_checkpoints/$BASE_MODEL_TAG"
if [ -z "$TOTAL_BATCH_SIZE" ]; then
    META_JSON=$(ls "$MID_CKPT_DIR"/meta_*.json 2>/dev/null | sort | tail -1)
    if [ -n "$META_JSON" ]; then
        TOTAL_BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$META_JSON'))['total_batch_size'])")
    else
        TOTAL_BATCH_SIZE=524288
        echo "WARNING: No meta JSON found in $MID_CKPT_DIR, falling back to 524288"
    fi
fi

# ══════ NPU ENVIRONMENT (identical to run_experiment.sh) ══════
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export NANOCHAT_BASE_DIR="$NANOCHAT_MODEL_DIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_HCCL_PATH=/usr/local/Ascend/ascend-toolkit/latest/hccl
export LD_LIBRARY_PATH=${ASCEND_HCCL_PATH}/lib64:${LD_LIBRARY_PATH:-}
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_WHITELIST_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

export PYTORCH_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_LOG_LEVEL=3

ASCEND_DEVICE_LIST=$(seq -s, 0 $((NUM_NPU - 1)))
export ASCEND_VISIBLE_DEVICES="$ASCEND_DEVICE_LIST"
export RANK_SIZE=$NUM_NPU
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export HCCL_EXEC_TIMEOUT=1200
export ASCEND_DISABLE_MEM_SWAP=1
export ASCEND_LAUNCH_BLOCKING=0
export NPU_DISABLE_RECORD=1
export PYTHONUNBUFFERED=1
export ASCEND_COMPILE_OPT_LEVEL=O3
export TORCH_NPU_LAZY_COMPILE=1
export PYTHONPRELOAD=torch_npu
export TORCH_NPU_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,memory_pool:True"
export PYTORCH_NPU_ALLOC_MAX_SIZE=60G
export ASCEND_ENABLE_CACHE=1
export ASCEND_CACHE_POLICY=2
export ASCEND_FUSION_ENABLE=1
export ASCEND_GEMM_DTiling=1
export TORCH_NPU_ENABLE_NUMA=1
export ASCEND_MEMORY_COPY_MODE=1
export ASCEND_HBM_ALLOC_TYPE=1
export ASCEND_OPP_LEVEL=O3
export ASCEND_FUSION_PASS_ENABLE=1
export ASCEND_GEMM_BTiling=1
export ASCEND_GEMM_ATiling=1
export ASCEND_CONV_ALGO_SELECTION=1
export ASCEND_ENABLE_TRANSFORMER_FUSION=1
export ASCEND_MEMORY_REUSE_MODE=2
export ASCEND_ENABLE_PREFETCH=1
export ASCEND_NPU_ENABLE_UNIFIED_MEMORY=1
export ASCEND_OPTIMIZER_AGGRESSIVE_MODE=1
export ASCEND_SYNCHRONIZATION_MODE=0
export PYTORCH_NPU_ENABLE_LARGE_CONCAT=1
export PYTORCH_NPU_ENABLE_TORCHscript=1
export NPU_PERF_MODE=high_performance
export NANOCHAT_DTYPE=bfloat16

# ══════ REPLAY ENV VARS (consumed by replay_mid_train.py) ══════
export REPLAY_DIR="$BATCH_DIR"
export REPLAY_START="$START_STEP"
export REPLAY_END="$END_STEP"

export PYTHONPATH="$SCRIPT_DIR:$NANOCHAT_REPO:${PYTHONPATH:-}"

# ══════ STEP 1: Generate replay_mid_train.py from mid_train.py ══════
echo "Generating replay_mid_train.py from mid_train.py..."
python3 "$SCRIPT_DIR/generate_replay_script.py" "$NANOCHAT_REPO"

# ══════ STEP 2: Run replay training ══════
echo ""
echo "Replay training: steps $START_STEP-$END_STEP from $BATCH_DIR"
echo "  Base model: $BASE_MODEL_TAG (step $BASE_MODEL_STEP)"
echo "  Device batch size: $DEVICE_BATCH_SIZE"
echo ""

pushd "$NANOCHAT_REPO" > /dev/null
python3 -m torch.distributed.run --standalone --nproc_per_node="$NUM_NPU" -m scripts.replay_mid_train -- \
    --run="replay_mid" \
    --model-tag="$BASE_MODEL_TAG" \
    --model-step="$BASE_MODEL_STEP" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --num-iterations="$NUM_ITERATIONS" \
    --core-metric-every=-1 \
    --eval-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --data-dir="$BATCH_DIR"
popd > /dev/null
