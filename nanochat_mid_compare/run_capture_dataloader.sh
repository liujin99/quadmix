#!/bin/bash
# Capture dataloader batches at a specific step for analysis
#
# Usage:
#   ./run_capture_dataloader.sh \
#       --data-dir=/path/to/quality_data_fineweb_edu \
#       --target-step=329 \
#       --output-dir=/home/ma-user/work/nanochat_model_dir/mid_checkpoints/step329_capture

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NANOCHAT_REPO="${NANOCHAT_REPO:-/home/ma-user/work/nanochat_midtrain_326}"
NANOCHAT_MODEL_DIR="${NANOCHAT_MODEL_DIR:-/home/ma-user/work/nanochat_model_dir}"
NUM_NPU="${NUM_NPU:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

DATA_DIR=""
TARGET_STEP=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
        --target-step=*) TARGET_STEP="${1#*=}"; shift ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        --nanochat-repo=*) NANOCHAT_REPO="${1#*=}"; shift ;;
        --num-npu=*) NUM_NPU="${1#*=}"; shift ;;
        --device-batch-size=*) DEVICE_BATCH_SIZE="${1#*=}"; shift ;;
        --max-seq-len=*) MAX_SEQ_LEN="${1#*=}"; shift ;;
        --grad-accum-steps=*) GRAD_ACCUM_STEPS="${1#*=}"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$DATA_DIR" ] || [ -z "$TARGET_STEP" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 --data-dir=<path> --target-step=<N> --output-dir=<path>"
    echo ""
    echo "Required:"
    echo "  --data-dir          Training data directory (e.g., quality_data_fineweb_edu)"
    echo "  --target-step       Step number to capture (0-indexed)"
    echo "  --output-dir        Output directory for captured batches"
    echo ""
    echo "Optional:"
    echo "  --nanochat-repo     Nanochat repo path (default: $NANOCHAT_REPO)"
    echo "  --num-npu           Number of NPUs (default: $NUM_NPU)"
    echo "  --device-batch-size Per-device batch size (default: $DEVICE_BATCH_SIZE)"
    echo "  --max-seq-len       Max sequence length (default: $MAX_SEQ_LEN)"
    echo "  --grad-accum-steps  Gradient accumulation steps (default: $GRAD_ACCUM_STEPS)"
    exit 1
fi

echo "========================================"
echo "  Capture Dataloader Batches"
echo "========================================"
echo "  Data dir:         $DATA_DIR"
echo "  Target step:      $TARGET_STEP"
echo "  Output dir:       $OUTPUT_DIR"
echo "  Num NPU:          $NUM_NPU"
echo "  Device batch:     $DEVICE_BATCH_SIZE"
echo "  Max seq len:      $MAX_SEQ_LEN"
echo "  Grad accum:       $GRAD_ACCUM_STEPS"
echo "========================================"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$NANOCHAT_MODEL_DIR"
mkdir -p "$NANOCHAT_MODEL_DIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_HCCL_PATH=/usr/local/Ascend/ascend-toolkit/latest/hccl
export LD_LIBRARY_PATH=${ASCEND_HCCL_PATH}/lib64:${LD_LIBRARY_PATH:-}
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_WHITELIST_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

export PYTORCH_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_LOG_LEVEL=3

if [ -z "${ASCEND_DEVICE_ID:-}" ] && [ -n "${LOCAL_RANK:-}" ]; then
    export ASCEND_DEVICE_ID=$LOCAL_RANK
elif [ -z "${ASCEND_DEVICE_ID:-}" ]; then
    export ASCEND_DEVICE_ID=0
fi

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

mkdir -p "$OUTPUT_DIR"

pushd "$NANOCHAT_REPO" > /dev/null

python3 -m torch.distributed.run --standalone --nproc_per_node="$NUM_NPU" \
    "$SCRIPT_DIR/capture_dataloader_only.py" \
    --data-dir="$DATA_DIR" \
    --target-step="$TARGET_STEP" \
    --output-dir="$OUTPUT_DIR" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --grad-accum-steps="$GRAD_ACCUM_STEPS"

popd > /dev/null

echo ""
echo "========================================"
echo "  Capture complete!"
echo "  Output: $OUTPUT_DIR"
echo "========================================"
