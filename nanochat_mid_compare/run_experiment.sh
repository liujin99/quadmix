#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# QuadMix vs Random: Nanochat Mid-Training Comparison Experiment
# ──────────────────────────────────────────────────────────────
#
# This script runs nanochat mid-training on two datasets:
#   1. QuadMix-selected subset from essential-web
#   2. Random subset from essential-web (token-count aligned)
#
# Usage:
#   bash nanochat_mid_compare/run_experiment.sh
#
# Or override config via environment variables:
#   QUADMIX_DATASET=/path/to/sampled_dataset.parquet \
#   ESSENTIAL_WEB_DIR=/path/to/essential-web-v1 \
#   bash nanochat_mid_compare/run_experiment.sh
#
# ──────────────────────────────────────────────────────────────

set -euo pipefail

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these or set via environment variables
# ══════════════════════════════════════════════════════════════

# QuadMix output: sampled_dataset.parquet from QuadMix pipeline
QUADMIX_DATASET="${QUADMIX_DATASET:-}"

# Essential-web raw parquet shards directory (shard_XXXXX.parquet)
ESSENTIAL_WEB_DIR="${ESSENTIAL_WEB_DIR:-}"

# Nanochat base directory (contains tokenizer/, base_checkpoints/)
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

# Base model tag (pretrained model in $NANOCHAT_BASE_DIR/base_checkpoints/<tag>/)
BASE_MODEL_TAG="${BASE_MODEL_TAG:-d24_0320}"

# Nanochat repo root
NANOCHAT_ROOT="${NANOCHAT_ROOT:-$HOME/nanochat-npu}"

# Experiment output directory
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$(cd "$(dirname "$0")/.." && pwd)/nanochat_mid_compare/results/$(date +%Y%m%d_%H%M%S)}"

# Mid-training hyperparameters
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-0.1}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
NUM_NPU="${NUM_NPU:-8}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:-500}"
EVAL_EVERY="${EVAL_EVERY:-100}"

# Data preparation
SHARD_SIZE="${SHARD_SIZE:-10000}"
VAL_RATIO="${VAL_RATIO:-0.05}"
SEED="${SEED:-42}"
MAX_RANDOM_SCAN="${MAX_RANDOM_SCAN:-500}"

# Mid-training model tags (auto-generated if empty)
QUADMIX_MODEL_TAG="${QUADMIX_MODEL_TAG:-}"
RANDOM_MODEL_TAG="${RANDOM_MODEL_TAG:-}"

# ══════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════

if [ -z "$QUADMIX_DATASET" ]; then
    echo "ERROR: QUADMIX_DATASET not set."
    echo "  Set via: QUADMIX_DATASET=/path/to/sampled_dataset.parquet"
    exit 1
fi

if [ -z "$ESSENTIAL_WEB_DIR" ]; then
    echo "ERROR: ESSENTIAL_WEB_DIR not set."
    echo "  Set via: ESSENTIAL_WEB_DIR=/path/to/essential-web-v1"
    exit 1
fi

if [ ! -f "$QUADMIX_DATASET" ]; then
    echo "ERROR: QuadMix dataset not found: $QUADMIX_DATASET"
    exit 1
fi

if [ ! -d "$ESSENTIAL_WEB_DIR" ]; then
    echo "ERROR: Essential-web directory not found: $ESSENTIAL_WEB_DIR"
    exit 1
fi

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

# Auto-generate model tags
TIMESTAMP=$(date +%m%d_%H%M)
if [ -z "$QUADMIX_MODEL_TAG" ]; then
    QUADMIX_MODEL_TAG="${BASE_MODEL_TAG}_quadmix_${TIMESTAMP}"
fi
if [ -z "$RANDOM_MODEL_TAG" ]; then
    RANDOM_MODEL_TAG="${BASE_MODEL_TAG}_random_${TIMESTAMP}"
fi

# ══════════════════════════════════════════════════════════════
#  NPU ENVIRONMENT SETUP (from nanochat speedrun.sh)
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

export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
echo "  QuadMix vs Random — Nanochat Mid-Training Comparison"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  QuadMix dataset:     $QUADMIX_DATASET"
echo "  Essential-web dir:   $ESSENTIAL_WEB_DIR"
echo "  Nanochat base dir:   $NANOCHAT_BASE_DIR"
echo "  Nanochat repo:       $NANOCHAT_ROOT"
echo "  Base model tag:      $BASE_MODEL_TAG"
echo "  Experiment output:   $EXPERIMENT_DIR"
echo ""
echo "  Mid-training config:"
echo "    target-param-data-ratio: $TARGET_PARAM_DATA_RATIO"
echo "    device-batch-size:       $DEVICE_BATCH_SIZE"
echo "    NPU cards:               $NUM_NPU"
echo ""
echo "  Model tags:"
echo "    QuadMix: $QUADMIX_MODEL_TAG"
echo "    Random:  $RANDOM_MODEL_TAG"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

mkdir -p "$EXPERIMENT_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$EXPERIMENT_DIR/data"

# ══════════════════════════════════════════════════════════════
#  STEP 1: DATA PREPARATION
# ══════════════════════════════════════════════════════════════

echo ""
echo "╔══ Step 1: Prepare datasets ══╗"
echo ""

if [ -f "$DATA_DIR/dataset_stats.json" ]; then
    echo "  Datasets already exist. Skipping preparation."
    echo "  (Delete $DATA_DIR to regenerate)"
else
    python3 "$SCRIPT_DIR/prepare_data.py" \
        --quadmix-dataset "$QUADMIX_DATASET" \
        --essential-web-dir "$ESSENTIAL_WEB_DIR" \
        --output-dir "$DATA_DIR" \
        --shard-size "$SHARD_SIZE" \
        --val-ratio "$VAL_RATIO" \
        --seed "$SEED" \
        --max-random-scan "$MAX_RANDOM_SCAN"
fi

echo ""
echo "  Dataset stats:"
python3 -c "
import json
stats = json.load(open('$DATA_DIR/dataset_stats.json'))
q, r = stats['quadmix'], stats['random']
print(f'    QuadMix: {q[\"train_docs\"]:,} train docs, ~{q[\"estimated_tokens\"]:,} tokens, {q[\"shards\"]} shards')
print(f'    Random:  {r[\"train_docs\"]:,} train docs, ~{r[\"estimated_tokens\"]:,} tokens, {r[\"shards\"]} shards')
print(f'    Shared val: {q[\"val_docs\"]:,} docs')
"
echo ""
echo "╚════════════════════════════════╗"
echo ""

# ══════════════════════════════════════════════════════════════
#  STEP 2: COPY BASE CHECKPOINT FOR BOTH RUNS
# ══════════════════════════════════════════════════════════════

echo ""
echo "╔══ Step 2: Prepare base checkpoints ══╗"
echo ""

BASE_CKPTS="$NANOCHAT_BASE_DIR/base_checkpoints"

for TAG in "$QUADMIX_MODEL_TAG" "$RANDOM_MODEL_TAG"; do
    TARGET="$BASE_CKPTS/$TAG"
    if [ -d "$TARGET" ]; then
        echo "  Checkpoint already exists: $TAG (skipping copy)"
    else
        echo "  Copying base checkpoint -> $TAG"
        cp -r "$BASE_CKPT_DIR" "$TARGET"
    fi
done

echo ""
echo "╚══════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════
#  STEP 3: MID-TRAINING
# ══════════════════════════════════════════════════════════════

QUADMIX_DATA="$DATA_DIR/quadmix_data"
RANDOM_DATA="$DATA_DIR/random_data"

run_mid_training() {
    local DATA_PATH="$1"
    local MODEL_TAG="$2"
    local RUN_NAME="$3"
    local LOG_FILE="$4"

    echo "  Starting mid-training: $RUN_NAME"
    echo "    Data:  $DATA_PATH"
    echo "    Model: $MODEL_TAG"
    echo "    Log:   $LOG_FILE"

    cd "$NANOCHAT_ROOT"
    torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.mid_train -- \
        --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --run="$RUN_NAME" \
        --model-tag="$MODEL_TAG" \
        --core-metric-every="$CORE_METRIC_EVERY" \
        --eval-every="$EVAL_EVERY" \
        --data-dir="$DATA_PATH" \
        2>&1 | tee "$LOG_FILE"
}

echo ""
echo "╔══ Step 3a: Mid-training on QuadMix data ══╗"
echo ""

QUADMIX_LOG="$EXPERIMENT_DIR/mid_train_quadmix.log"
run_mid_training "$QUADMIX_DATA" "$QUADMIX_MODEL_TAG" "quadmix_mid" "$QUADMIX_LOG"

echo ""
echo "╚════════════════════════════════════════════╝"
echo ""

echo ""
echo "╔══ Step 3b: Mid-training on Random data ══╗"
echo ""

RANDOM_LOG="$EXPERIMENT_DIR/mid_train_random.log"
run_mid_training "$RANDOM_DATA" "$RANDOM_MODEL_TAG" "random_mid" "$RANDOM_LOG"

echo ""
echo "╚═══════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════
#  STEP 4: EVALUATION
# ══════════════════════════════════════════════════════════════

run_eval() {
    local MODEL_TAG="$1"
    local MODEL_TYPE="$2"
    local LOG_FILE="$3"

    echo "  Evaluating: $MODEL_TAG ($MODEL_TYPE)"

    cd "$NANOCHAT_ROOT"
    torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.base_eval -- \
        --device-batch-size=32 \
        --model-tag="$MODEL_TAG" \
        --model-type="$MODEL_TYPE" \
        2>&1 | tee "$LOG_FILE"
}

echo ""
echo "╔══ Step 4: Evaluation ══╗"
echo ""

QUADMIX_EVAL_LOG="$EXPERIMENT_DIR/eval_quadmix.log"
run_eval "$QUADMIX_MODEL_TAG" "mid" "$QUADMIX_EVAL_LOG"

RANDOM_EVAL_LOG="$EXPERIMENT_DIR/eval_random.log"
run_eval "$RANDOM_MODEL_TAG" "mid" "$RANDOM_EVAL_LOG"

echo ""
echo "╚════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════
#  STEP 5: SUMMARY
# ══════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Experiment Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Output directory: $EXPERIMENT_DIR"
echo ""
echo "  Files:"
echo "    ├── data/                    # Training datasets"
echo "    │   ├── quadmix_data/        # QuadMix-selected shards"
echo "    │   ├── random_data/         # Random baseline shards"
echo "    │   └── dataset_stats.json   # Dataset statistics"
echo "    ├── mid_train_quadmix.log    # QuadMix mid-training log"
echo "    ├── mid_train_random.log     # Random mid-training log"
echo "    ├── eval_quadmix.log         # QuadMix evaluation log"
echo "    └── eval_random.log          # Random evaluation log"
echo ""
echo "  Checkpoints:"
echo "    $NANOCHAT_BASE_DIR/mid_checkpoints/$QUADMIX_MODEL_TAG/"
echo "    $NANOCHAT_BASE_DIR/mid_checkpoints/$RANDOM_MODEL_TAG/"
echo ""
echo "════════════════════════════════════════════════════════════"
