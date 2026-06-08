#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Evaluate Base Model CORE Metric
# ──────────────────────────────────────────────────────────────
#
# This script evaluates the base model's CORE metric for comparison
# with mid-trained models (QuadMix and Random).
#
# Usage:
#   bash nanochat_mid_compare/eval_base_model.sh
#
# Or with custom config:
#   BASE_MODEL_TAG=d24_0320 \
#   OUTPUT_DIR=/path/to/output \
#   bash nanochat_mid_compare/eval_base_model.sh
#
# ──────────────────────────────────────────────────────────────

set -euo pipefail

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

# Nanochat base directory (contains tokenizer/, base_checkpoints/)
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

# Base model tag (pretrained model in $NANOCHAT_BASE_DIR/base_checkpoints/<tag>/)
BASE_MODEL_TAG="${BASE_MODEL_TAG:-d24_0320}"

# Nanochat repo root
NANOCHAT_ROOT="${NANOCHAT_ROOT:-$HOME/nanochat-npu}"

# Output directory for evaluation log
OUTPUT_DIR="${OUTPUT_DIR:-$(cd "$(dirname "$0")" && pwd)/results/base_eval}"

# Number of NPU cards
NUM_NPU="${NUM_NPU:-8}"

# ══════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════

if [ ! -d "$NANOCHAT_ROOT" ]; then
    echo "ERROR: Nanochat repo not found: $NANOCHAT_ROOT"
    echo "  Set via: NANOCHAT_ROOT=/path/to/nanochat-npu"
    exit 1
fi

BASE_CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$BASE_MODEL_TAG"
if [ ! -d "$BASE_CKPT_DIR" ]; then
    echo "ERROR: Base model checkpoint not found: $BASE_CKPT_DIR"
    exit 1
fi

TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ]; then
    echo "ERROR: Tokenizer not found: $TOKENIZER_DIR/tokenizer.pkl"
    exit 1
fi

# ══════════════════════════════════════════════════════════════
#  NPU ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR
mkdir -p "$NANOCHAT_BASE_DIR"

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

# ══════════════════════════════════════════════════════════════
#  PRINT CONFIG
# ══════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Base Model CORE Metric Evaluation"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Nanochat base dir:   $NANOCHAT_BASE_DIR"
echo "  Nanochat repo:       $NANOCHAT_ROOT"
echo "  Base model tag:      $BASE_MODEL_TAG"
echo "  Output directory:    $OUTPUT_DIR"
echo "  NPU cards:           $NUM_NPU"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

mkdir -p "$OUTPUT_DIR"

# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

EVAL_LOG="$OUTPUT_DIR/eval_base_${BASE_MODEL_TAG}.log"

echo "Evaluating base model: $BASE_MODEL_TAG"
echo "Log file: $EVAL_LOG"
echo ""

cd "$NANOCHAT_ROOT"
torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.base_eval -- \
    --eval=core \
    --device-batch-size=32 \
    --model-tag="$BASE_MODEL_TAG" \
    --model-type="base" \
    2>&1 | tee "$EVAL_LOG"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Evaluation Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Log file: $EVAL_LOG"
echo ""
echo "  To compare with mid-trained models, check:"
echo "    - QuadMix: results/<timestamp>/eval_quadmix.log"
echo "    - Random:  results/<timestamp>/eval_random.log"
echo ""
echo "════════════════════════════════════════════════════════════"
