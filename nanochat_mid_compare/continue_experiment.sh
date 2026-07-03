#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Continue Experiment: fineweb_edu mid-training + all evals + report
#
# Resumes from an existing experiment where QuadMix, Random, and
# dclm mid-training have already completed, but fineweb_edu crashed.
#
# Steps:
#   1. Run fineweb_edu mid-training
#   2. Evaluate all 4 models (QuadMix, Random, dclm, fineweb_edu)
#   3. Generate experiment report
#
# Usage:
#   bash nanochat_mid_compare/continue_experiment.sh
#
# Override via environment variables:
#   RESULT_DIR=/path/to/results/20260623_220114 \
#   bash nanochat_mid_compare/continue_experiment.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QUADMIX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ══════ CONFIGURATION ══════
RESULT_DIR="${RESULT_DIR:-$SCRIPT_DIR/results/20260623_220114}"
NANOCHAT_MODEL_DIR="${NANOCHAT_MODEL_DIR:-/home/ma-user/work/nanochat_model_dir}"
NANOCHAT_REPO="${NANOCHAT_REPO:-/home/ma-user/work/nanochat_midtrain_326}"
BASE_MODEL_TAG="${BASE_MODEL_TAG:-d24_0320}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
NUM_NPU="${NUM_NPU:-8}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
EVAL_EVERY="${EVAL_EVERY:--1}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-0.5}"
NUM_SCALING_PARAMS="${NUM_SCALING_PARAMS:-1300000000}"
MID_CHECKPOINTS_OUTPUT_DIR="${MID_CHECKPOINTS_OUTPUT_DIR:-$HOME/.cache/nanochat_mid_compare/mid_checkpoints}"

# Extract timestamp from result directory name
TIMESTAMP="$(basename "$RESULT_DIR")"

# Model tags (must match original run_experiment.sh naming)
QUADMIX_MODEL_TAG="${QUADMIX_MODEL_TAG:-${BASE_MODEL_TAG}_quadmix_${TIMESTAMP}}"
RANDOM_MODEL_TAG="${RANDOM_MODEL_TAG:-${BASE_MODEL_TAG}_random_${TIMESTAMP}}"
QUALITY_METHODS="${QUALITY_METHODS:-dclm,fineweb_edu}"
IFS=',' read -ra QUALITY_METHOD_ARRAY <<< "$QUALITY_METHODS"

DATA_DIR="$RESULT_DIR/data"
STATS_FILE="$DATA_DIR/dataset_stats.json"

# ══════ VALIDATION ══════

if [ ! -d "$RESULT_DIR" ]; then
    echo "ERROR: Result directory not found: $RESULT_DIR"
    exit 1
fi

if [ ! -f "$STATS_FILE" ]; then
    echo "ERROR: dataset_stats.json not found: $STATS_FILE"
    exit 1
fi

BASE_CKPT_DIR="$NANOCHAT_MODEL_DIR/base_checkpoints/$BASE_MODEL_TAG"
if [ ! -d "$BASE_CKPT_DIR" ]; then
    echo "ERROR: Base model checkpoint not found: $BASE_CKPT_DIR"
    exit 1
fi

if [ ! -d "$NANOCHAT_REPO" ]; then
    echo "ERROR: Nanochat repo not found: $NANOCHAT_REPO"
    exit 1
fi

# ══════ NPU ENVIRONMENT ══════

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

# ══════ PRINT CONFIG ══════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Continue Experiment: fineweb_edu + evals + report"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Result dir:        $RESULT_DIR"
echo "  Data dir:          $DATA_DIR"
echo "  Base model:        $BASE_MODEL_TAG"
echo "  Timestamp:         $TIMESTAMP"
echo ""
echo "  Model tags:"
echo "    QuadMix:         $QUADMIX_MODEL_TAG"
echo "    Random:          $RANDOM_MODEL_TAG"
for method in "${QUALITY_METHOD_ARRAY[@]}"; do
    echo "    Quality ($method): ${BASE_MODEL_TAG}_quality_${method}_${TIMESTAMP}"
done
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# ══════ SETUP MID_CHECKPOINTS DIRECTORY ══════

if [ -n "$MID_CHECKPOINTS_OUTPUT_DIR" ]; then
    mkdir -p "$MID_CHECKPOINTS_OUTPUT_DIR"
    LINK_PATH="$NANOCHAT_MODEL_DIR/mid_checkpoints"

    if [ -L "$LINK_PATH" ]; then
        echo "  Symlink already exists: $LINK_PATH -> $(readlink "$LINK_PATH")"
    elif [ -d "$LINK_PATH" ]; then
        echo "  WARNING: $LINK_PATH exists as a directory (not a symlink)."
    else
        echo "  Creating symlink: $LINK_PATH -> $MID_CHECKPOINTS_OUTPUT_DIR"
        ln -s "$MID_CHECKPOINTS_OUTPUT_DIR" "$LINK_PATH"
    fi
fi

# ══════ HELPER FUNCTIONS ══════

run_mid_training() {
    local DATA_PATH="$1"
    local MODEL_TAG="$2"
    local RUN_NAME="$3"
    local LOG_FILE="$4"
    local DATASET_TOKENS="$5"

    local META_JSON=$(ls "$BASE_CKPT_DIR"/meta_*.json 2>/dev/null | sort | tail -1)
    if [ -n "$META_JSON" ]; then
        local TOTAL_BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$META_JSON'))['total_batch_size'])")
        echo "    Read total_batch_size=$TOTAL_BATCH_SIZE from checkpoint"
    else
        local TOTAL_BATCH_SIZE=524288
        echo "    WARNING: No meta JSON found in $BASE_CKPT_DIR, falling back to 524288"
    fi
    local TARGET_TOKENS=$(python3 -c "print(int($TARGET_PARAM_DATA_RATIO * $NUM_SCALING_PARAMS))")
    local ACTUAL_TOKENS=$(python3 -c "print(min($TARGET_TOKENS, $DATASET_TOKENS))")
    local NUM_ITERATIONS=$((ACTUAL_TOKENS / TOTAL_BATCH_SIZE))
    local ACTUAL_RATIO=$(python3 -c "print(f'{$ACTUAL_TOKENS / $NUM_SCALING_PARAMS:.4f}')")

    echo "  Starting mid-training: $RUN_NAME"
    echo "    Data:       $DATA_PATH"
    echo "    Source:     $BASE_MODEL_TAG (base)"
    echo "    Save as:    $MODEL_TAG (mid)"
    echo "    Dataset:    $DATASET_TOKENS tokens"
    echo "    Target:     $TARGET_TOKENS tokens (ratio=$TARGET_PARAM_DATA_RATIO)"
    echo "    Actual:     $ACTUAL_TOKENS tokens (ratio=$ACTUAL_RATIO)"
    echo "    Steps:      $NUM_ITERATIONS"
    echo "    Log:        $LOG_FILE"

    local LINK_DIR="$NANOCHAT_MODEL_DIR/base_checkpoints/$MODEL_TAG"

    if [ ! -e "$LINK_DIR" ]; then
        echo "    Creating symlink: $LINK_DIR -> $BASE_CKPT_DIR"
        ln -s "$BASE_CKPT_DIR" "$LINK_DIR"
    fi

    pushd "$NANOCHAT_REPO" > /dev/null
    python3 -m torch.distributed.run --standalone --nproc_per_node="$NUM_NPU" -m scripts.mid_train -- \
        --num-iterations="$NUM_ITERATIONS" \
        --target-param-data-ratio="$ACTUAL_RATIO" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --total-batch-size="$TOTAL_BATCH_SIZE" \
        --run="$RUN_NAME" \
        --model-tag="$MODEL_TAG" \
        --core-metric-every="$CORE_METRIC_EVERY" \
        --eval-every="$EVAL_EVERY" \
        --data-dir="$DATA_PATH" \
        2>&1 | tee "$LOG_FILE"
    popd > /dev/null

    if [ -L "$LINK_DIR" ]; then
        rm "$LINK_DIR"
    fi
}

run_eval() {
    local MODEL_TAG="$1"
    local MODEL_TYPE="$2"
    local LOG_FILE="$3"

    echo "  Evaluating: $MODEL_TAG ($MODEL_TYPE)"

    pushd "$NANOCHAT_REPO" > /dev/null
    python3 -m torch.distributed.run --standalone --nproc_per_node="$NUM_NPU" -m scripts.base_eval -- \
        --eval=core \
        --device-batch-size=32 \
        --model-tag="$MODEL_TAG" \
        --model-type="$MODEL_TYPE" \
        2>&1 | tee "$LOG_FILE"
    popd > /dev/null
}

# ══════════════════════════════════════════════════════════════
#  STEP 1: fineweb_edu MID-TRAINING
# ══════════════════════════════════════════════════════════════

FINWEB_DATA="$DATA_DIR/quality_data_fineweb_edu"
FINWEB_MODEL_TAG="${BASE_MODEL_TAG}_quality_fineweb_edu_${TIMESTAMP}"
FINWEB_LOG="$RESULT_DIR/mid_train_quality_fineweb_edu.log"

if [ ! -d "$FINWEB_DATA" ]; then
    echo "ERROR: fineweb_edu data not found: $FINWEB_DATA"
    exit 1
fi

FINWEB_TOKENS=$(STATS_FILE="$STATS_FILE" python3 -c "
import os, json
s = json.load(open(os.environ['STATS_FILE']))
print(s['quality_fineweb_edu']['tokens'])
")

echo ""
echo "╔══ Step 1: Mid-training on Quality Top-K (fineweb_edu) data ══╗"
echo ""

run_mid_training "$FINWEB_DATA" "$FINWEB_MODEL_TAG" "quality_fineweb_edu_mid" "$FINWEB_LOG" "$FINWEB_TOKENS"

echo ""
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════
#  STEP 2: EVALUATION (all 4 models)
# ══════════════════════════════════════════════════════════════

echo ""
echo "╔══ Step 2: Evaluation ══╗"
echo ""

QUADMIX_EVAL_LOG="$RESULT_DIR/eval_quadmix.log"
if [ -f "$QUADMIX_EVAL_LOG" ]; then
    echo "  Skipping QuadMix eval (log already exists: $QUADMIX_EVAL_LOG)"
else
    run_eval "$QUADMIX_MODEL_TAG" "mid" "$QUADMIX_EVAL_LOG"
fi

RANDOM_EVAL_LOG="$RESULT_DIR/eval_random.log"
if [ -f "$RANDOM_EVAL_LOG" ]; then
    echo "  Skipping Random eval (log already exists: $RANDOM_EVAL_LOG)"
else
    run_eval "$RANDOM_MODEL_TAG" "mid" "$RANDOM_EVAL_LOG"
fi

for method in "${QUALITY_METHOD_ARRAY[@]}"; do
    QUALITY_MODEL_TAG="${BASE_MODEL_TAG}_quality_${method}_${TIMESTAMP}"
    QUALITY_EVAL_LOG="$RESULT_DIR/eval_quality_${method}.log"
    if [ -f "$QUALITY_EVAL_LOG" ]; then
        echo "  Skipping Quality ($method) eval (log already exists: $QUALITY_EVAL_LOG)"
    else
        run_eval "$QUALITY_MODEL_TAG" "mid" "$QUALITY_EVAL_LOG"
    fi
done

echo ""
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════
#  STEP 3: GENERATE REPORT
# ══════════════════════════════════════════════════════════════

echo ""
echo "╔══ Step 3: Generate experiment report ══╗"
echo ""

QUADMIX_LOG="$RESULT_DIR/mid_train_quadmix.log"
RANDOM_LOG="$RESULT_DIR/mid_train_random.log"

REPORT_ARGS=(
    --result-dir "$RESULT_DIR"
    --dataset-stats "$STATS_FILE"
    --quadmix-train-log "$QUADMIX_LOG"
    --random-train-log "$RANDOM_LOG"
    --quadmix-eval-log "$QUADMIX_EVAL_LOG"
    --random-eval-log "$RANDOM_EVAL_LOG"
)

QUALITY_TRAIN_LOGS=()
QUALITY_EVAL_LOGS=()
for method in "${QUALITY_METHOD_ARRAY[@]}"; do
    QUALITY_TRAIN_LOGS+=("$RESULT_DIR/mid_train_quality_${method}.log")
    QUALITY_EVAL_LOGS+=("$RESULT_DIR/eval_quality_${method}.log")
done
REPORT_ARGS+=(--quality-train-log "${QUALITY_TRAIN_LOGS[@]}")
REPORT_ARGS+=(--quality-eval-log "${QUALITY_EVAL_LOGS[@]}")

python3 "$SCRIPT_DIR/generate_report.py" "${REPORT_ARGS[@]}"

echo ""
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Experiment Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Output directory: $RESULT_DIR"
echo ""
echo "  Files:"
echo "    ├── mid_train_quality_fineweb_edu.log  # fineweb_edu training log"
echo "    ├── eval_quadmix.log                   # QuadMix eval"
echo "    ├── eval_random.log                    # Random eval"
for method in "${QUALITY_METHOD_ARRAY[@]}"; do
    echo "    ├── eval_quality_${method}.log         # Quality ($method) eval"
done
echo "    └── experiment_report.md               # Comparison report"
echo ""
echo "════════════════════════════════════════════════════════════"
