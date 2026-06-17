#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# QuadMix vs Random: Nanochat Mid-Training Comparison Experiment
# ──────────────────────────────────────────────────────────────
#
# This script runs nanochat mid-training on two datasets:
#   1. QuadMix-selected subset from essential-web
#   2. Random subset from essential-web (token-count aligned)
#
# Usage (after running QuaDMix pipeline, e.g. bash scripts/demo_run_full.sh):
#   bash nanochat_mid_compare/run_experiment.sh
#
# QUADMIX_SAMPLED_DATA auto-detects the latest result/*/sampled_dataset.parquet.
# PREPROCESSED_DATA_DIR defaults to $HOME/.cache/QuaDMix/temp/preprocessed.
#
# Override any config via environment variables:
#   QUADMIX_SAMPLED_DATA=/path/to/sampled_dataset.parquet \
#   PREPROCESSED_DATA_DIR=/path/to/preprocessed \
#   NANOCHAT_MODEL_DIR=/path/to/.cache/nanochat \
#   bash nanochat_mid_compare/run_experiment.sh
#
# ──────────────────────────────────────────────────────────────

set -euo pipefail

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these or set via environment variables
# ══════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QUADMIX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# QuadMix output: sampled_dataset.parquet from QuadMix pipeline
# Auto-detects the latest result directory if not set
if [ -z "${QUADMIX_SAMPLED_DATA:-}" ]; then
    QUADMIX_SAMPLED_DATA="$(ls -t "$QUADMIX_DIR"/result/*/sampled_dataset.parquet 2>/dev/null | head -1)"
fi
QUADMIX_SAMPLED_DATA="${QUADMIX_SAMPLED_DATA:-}"

# Preprocessed shards directory (preprocessed_*.parquet with quality scores)
# Used for both random baseline and quality top-k baseline
PREPROCESSED_DATA_DIR="${PREPROCESSED_DATA_DIR:-$HOME/.cache/QuaDMix/temp/preprocessed}"

# Quality score methods for top-k selection (comma-separated)
# Options: dclm, fineweb_edu, english, math_general, math_openweb
QUALITY_METHODS="${QUALITY_METHODS:-dclm,fineweb_edu}"

# Nanochat model directory (contains tokenizer/, base_checkpoints/)
NANOCHAT_MODEL_DIR="${NANOCHAT_MODEL_DIR:-/home/ma-user/work/nanochat_model_dir}"

# Base model tag (pretrained model in $NANOCHAT_MODEL_DIR/base_checkpoints/<tag>/)
BASE_MODEL_TAG="${BASE_MODEL_TAG:-d24_0320}"

# Nanochat repo root
NANOCHAT_REPO="${NANOCHAT_REPO:-/home/ma-user/work/nanochat_midtrain_326}"

# Mid-training checkpoint output directory (where trained models are saved)
# If set, $NANOCHAT_MODEL_DIR/mid_checkpoints will be symlinked here
# to avoid filling up EVS storage with large checkpoint files
MID_CHECKPOINTS_OUTPUT_DIR="${MID_CHECKPOINTS_OUTPUT_DIR:-$HOME/.cache/nanochat_mid_compare/mid_checkpoints}"

# Experiment output directory (logs, data, reports)
RESULT_DIR="${RESULT_DIR:-$SCRIPT_DIR/results/$TIMESTAMP}"

# ── Mid-training hyperparameters ──
# Training token budget: min(target_ratio * num_scaling_params, dataset_tokens)
# - If data < ratio * params: use all data (1 epoch, no overfitting)
# - If data > ratio * params: cap at ratio * params (no over-training)
# d24 model: num_scaling_params (total) ≈ 1.3B
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-0.5}"
NUM_SCALING_PARAMS="${NUM_SCALING_PARAMS:-1300000000}"  # d24 ≈ 1.3B
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
NUM_NPU="${NUM_NPU:-8}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
# Val BPB disabled by default (-1) because QuadMix and Random use different data,
# so their val sets are not comparable. Use CORE metric for comparison instead.
EVAL_EVERY="${EVAL_EVERY:--1}"

# Data preparation
SHARD_SIZE="${SHARD_SIZE:-10000}"
VAL_RATIO="${VAL_RATIO:-0}"
SEED="${SEED:-42}"
MAX_RANDOM_SCAN="${MAX_RANDOM_SCAN:-500}"

# Mid-training model tags (auto-generated if empty)
# The script creates a symlink from BASE_MODEL_TAG to MODEL_TAG in base_checkpoints/
# so mid_train.py loads the base model and saves to mid_checkpoints/<MODEL_TAG>/
QUADMIX_MODEL_TAG="${QUADMIX_MODEL_TAG:-}"
RANDOM_MODEL_TAG="${RANDOM_MODEL_TAG:-}"

# ══════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════

if [ -z "$QUADMIX_SAMPLED_DATA" ]; then
    echo "ERROR: QUADMIX_SAMPLED_DATA not set and no sampled_dataset.parquet found under $QUADMIX_DIR/result/"
    echo "  Run the QuaDMix pipeline first (e.g. bash scripts/demo_run_full.sh),"
    echo "  or set via: QUADMIX_SAMPLED_DATA=/path/to/sampled_dataset.parquet"
    exit 1
fi

if [ ! -f "$QUADMIX_SAMPLED_DATA" ]; then
    echo "ERROR: QuadMix dataset not found: $QUADMIX_SAMPLED_DATA"
    exit 1
fi

if [ ! -d "$PREPROCESSED_DATA_DIR" ]; then
    echo "ERROR: Preprocessed directory not found: $PREPROCESSED_DATA_DIR"
    echo "  Set via: PREPROCESSED_DATA_DIR=/path/to/preprocessed"
    exit 1
fi

if [ ! -d "$NANOCHAT_REPO" ]; then
    echo "ERROR: Nanochat repo not found: $NANOCHAT_REPO"
    echo "  Set via: NANOCHAT_REPO=/path/to/nanochat-npu"
    exit 1
fi

BASE_CKPT_DIR="$NANOCHAT_MODEL_DIR/base_checkpoints/$BASE_MODEL_TAG"
if [ ! -d "$BASE_CKPT_DIR" ]; then
    echo "ERROR: Base model checkpoint not found: $BASE_CKPT_DIR"
    exit 1
fi

TOKENIZER_DIR="$NANOCHAT_MODEL_DIR/tokenizer"
if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ]; then
    echo "ERROR: Tokenizer not found: $TOKENIZER_DIR/tokenizer.pkl"
    exit 1
fi

# Auto-generate model tags
if [ -z "$QUADMIX_MODEL_TAG" ]; then
    QUADMIX_MODEL_TAG="${BASE_MODEL_TAG}_quadmix_${TIMESTAMP}"
fi
if [ -z "$RANDOM_MODEL_TAG" ]; then
    RANDOM_MODEL_TAG="${BASE_MODEL_TAG}_random_${TIMESTAMP}"
fi

IFS=',' read -ra QUALITY_METHOD_ARRAY <<< "$QUALITY_METHODS"

DO_QUALITY=0
if [ -n "$QUALITY_METHODS" ]; then
    DO_QUALITY=1
fi

# ══════════════════════════════════════════════════════════════
#  NPU ENVIRONMENT SETUP (from nanochat speedrun.sh)
# ══════════════════════════════════════════════════════════════

export OMP_NUM_THREADS=1
export WANDB_MODE=offline
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

# ══════════════════════════════════════════════════════════════
#  PRINT CONFIG
# ══════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  QuadMix vs Random — Nanochat Mid-Training Comparison"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  QuadMix dataset:     $QUADMIX_SAMPLED_DATA"
echo "  Preprocessed dir:    $PREPROCESSED_DATA_DIR"
echo "  Nanochat model dir:  $NANOCHAT_MODEL_DIR"
echo "  Nanochat repo:       $NANOCHAT_REPO"
echo "  Base model tag:      $BASE_MODEL_TAG (source for all runs)"
echo "  Experiment output:   $RESULT_DIR"
if [ -n "$MID_CHECKPOINTS_OUTPUT_DIR" ]; then
    echo "  Mid checkpoint output: $MID_CHECKPOINTS_OUTPUT_DIR"
fi
if [ "$DO_QUALITY" -eq 1 ]; then
    echo "  Quality methods:     ${QUALITY_METHOD_ARRAY[*]}"
else
    echo "  Quality baseline:    DISABLED (set QUALITY_METHODS to enable)"
fi
echo ""
echo "  Mid-training config:"
echo "    target-param-data-ratio: $TARGET_PARAM_DATA_RATIO"
echo "    num-scaling-params:      $NUM_SCALING_PARAMS"
echo "    device-batch-size:       $DEVICE_BATCH_SIZE"
echo "    NPU cards:               $NUM_NPU"
echo ""
echo "  Model tags (save):"
echo "    QuadMix: $QUADMIX_MODEL_TAG"
echo "    Random:  $RANDOM_MODEL_TAG"
if [ "$DO_QUALITY" -eq 1 ]; then
    for method in "${QUALITY_METHOD_ARRAY[@]}"; do
        echo "    Quality ($method): ${BASE_MODEL_TAG}_quality_${method}_${TIMESTAMP}"
    done
fi
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

mkdir -p "$RESULT_DIR"

DATA_DIR="$RESULT_DIR/data"

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
    PREP_ARGS=(
        --quadmix-sampled-data "$QUADMIX_SAMPLED_DATA"
        --preprocessed-data-dir "$PREPROCESSED_DATA_DIR"
        --output-dir "$DATA_DIR"
        --tokenizer-pkl "$TOKENIZER_DIR/tokenizer.pkl"
        --shard-size "$SHARD_SIZE"
        --val-ratio "$VAL_RATIO"
        --seed "$SEED"
        --max-random-scan "$MAX_RANDOM_SCAN"
        --num-npu "$NUM_NPU"
    )
    if [ "$DO_QUALITY" -eq 1 ]; then
        PREP_ARGS+=(--quality-method "$QUALITY_METHODS")
    fi
    python3 "$SCRIPT_DIR/prepare_data.py" "${PREP_ARGS[@]}"
fi

DATA_DIR="$DATA_DIR" BASE_MODEL_TAG="$BASE_MODEL_TAG" \
TARGET_PARAM_DATA_RATIO="$TARGET_PARAM_DATA_RATIO" \
NUM_SCALING_PARAMS="$NUM_SCALING_PARAMS" \
DEVICE_BATCH_SIZE="$DEVICE_BATCH_SIZE" \
NUM_NPU="$NUM_NPU" python3 -c "
import os, json
stats_path = os.path.join(os.environ['DATA_DIR'], 'dataset_stats.json')
stats = json.load(open(stats_path))
stats['config'].update({
    'base_model_tag': os.environ['BASE_MODEL_TAG'],
    'target_param_data_ratio': float(os.environ['TARGET_PARAM_DATA_RATIO']),
    'num_scaling_params': int(os.environ['NUM_SCALING_PARAMS']),
    'device_batch_size': int(os.environ['DEVICE_BATCH_SIZE']),
    'num_npu': int(os.environ['NUM_NPU']),
})
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)
"

echo ""
echo "  Dataset stats:"
DATA_DIR="$DATA_DIR" python3 -c "
import os, json
stats = json.load(open(os.path.join(os.environ['DATA_DIR'], 'dataset_stats.json')))
q, r = stats['quadmix'], stats['random']
method = stats['config'].get('token_method', 'unknown')
print(f'    Token method: {method}')
print(f'    QuadMix: {q[\"train_docs\"]:,} train docs, {q[\"tokens\"]:,} tokens, {q[\"shards\"]} shards')
print(f'    Random:  {r[\"train_docs\"]:,} train docs, {r[\"tokens\"]:,} tokens, {r[\"shards\"]} shards')
for key in stats:
    if key.startswith('quality_'):
        method = key[len('quality_'):]
        ql = stats[key]
        print(f'    Quality ({method}): {ql[\"train_docs\"]:,} train docs, {ql[\"tokens\"]:,} tokens, {ql[\"shards\"]} shards')
print(f'    Shared val: {q[\"val_docs\"]:,} docs')
"
echo ""
echo "╚════════════════════════════════╗"
echo ""

# ══════════════════════════════════════════════════════════════
#  STEP 2: SETUP MID_CHECKPOINTS DIRECTORY
# ══════════════════════════════════════════════════════════════

if [ -n "$MID_CHECKPOINTS_OUTPUT_DIR" ]; then
    echo ""
    echo "╔══ Step 2: Setup mid-training checkpoint output directory ══╗"
    echo ""

    mkdir -p "$MID_CHECKPOINTS_OUTPUT_DIR"
    LINK_PATH="$NANOCHAT_MODEL_DIR/mid_checkpoints"

    if [ -L "$LINK_PATH" ]; then
        echo "  Symlink already exists: $LINK_PATH -> $(readlink "$LINK_PATH")"
    elif [ -d "$LINK_PATH" ]; then
        echo "  WARNING: $LINK_PATH exists as a directory (not a symlink)."
        echo "  Skipping symlink creation. Mid checkpoints will be saved here."
    else
        echo "  Mid-training checkpoints will be saved to: $MID_CHECKPOINTS_OUTPUT_DIR"
        echo "  Creating symlink: $LINK_PATH -> $MID_CHECKPOINTS_OUTPUT_DIR"
        ln -s "$MID_CHECKPOINTS_OUTPUT_DIR" "$LINK_PATH"
    fi

    echo ""
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
fi

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
    local DATASET_TOKENS="$5"

    local TOTAL_BATCH_SIZE=524288
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

    local BASE_CKPT_DIR="$NANOCHAT_MODEL_DIR/base_checkpoints/$BASE_MODEL_TAG"
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
        --run="$RUN_NAME" \
        --model-tag="$MODEL_TAG" \
        --core-metric-every="$CORE_METRIC_EVERY" \
        --eval-every="$EVAL_EVERY" \
        --data-dir="$DATA_PATH" \
        2>&1 | tee "$LOG_FILE"
    popd > /dev/null

    if [ -L "$LINK_DIR" ]; then
        echo "    Cleaning up symlink: $LINK_DIR"
        rm "$LINK_DIR"
    fi
}

STATS_FILE="$DATA_DIR/dataset_stats.json"
read -r QUADMIX_TOKENS RANDOM_TOKENS < <(
    STATS_FILE="$STATS_FILE" python3 -c "
import os, json
s = json.load(open(os.environ['STATS_FILE']))
print(s['quadmix']['tokens'], s['random']['tokens'])
"
)

echo ""
echo "╔══ Step 3a: Mid-training on QuadMix data ══╗"
echo ""

QUADMIX_LOG="$RESULT_DIR/mid_train_quadmix.log"
run_mid_training "$QUADMIX_DATA" "$QUADMIX_MODEL_TAG" "quadmix_mid" "$QUADMIX_LOG" "$QUADMIX_TOKENS"

echo ""
echo "╚════════════════════════════════════════════╝"
echo ""

echo ""
echo "╔══ Step 3b: Mid-training on Random data ══╗"
echo ""

RANDOM_LOG="$RESULT_DIR/mid_train_random.log"
run_mid_training "$RANDOM_DATA" "$RANDOM_MODEL_TAG" "random_mid" "$RANDOM_LOG" "$RANDOM_TOKENS"

echo ""
echo "╚═══════════════════════════════════════════╝"
echo ""

if [ "$DO_QUALITY" -eq 1 ]; then
    for method in "${QUALITY_METHOD_ARRAY[@]}"; do
        echo ""
        echo "╔══ Step 3c: Mid-training on Quality Top-K ($method) data ══╗"
        echo ""

        QUALITY_DATA="$DATA_DIR/quality_data_${method}"
        QUALITY_MODEL_TAG="${BASE_MODEL_TAG}_quality_${method}_${TIMESTAMP}"
        QUALITY_LOG="$RESULT_DIR/mid_train_quality_${method}.log"
        QUALITY_TOKENS=$(STATS_FILE="$STATS_FILE" METHOD="$method" python3 -c "
import os, json
s = json.load(open(os.environ['STATS_FILE']))
print(s[f'quality_{os.environ[\"METHOD\"]}']['tokens'])
")
        run_mid_training "$QUALITY_DATA" "$QUALITY_MODEL_TAG" "quality_${method}_mid" "$QUALITY_LOG" "$QUALITY_TOKENS"

        echo ""
        echo "╚════════════════════════════════════════════════╝"
        echo ""
    done
fi

# ══════════════════════════════════════════════════════════════
#  STEP 4: EVALUATION
# ══════════════════════════════════════════════════════════════

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

echo ""
echo "╔══ Step 4: Evaluation ══╗"
echo ""

QUADMIX_EVAL_LOG="$RESULT_DIR/eval_quadmix.log"
run_eval "$QUADMIX_MODEL_TAG" "mid" "$QUADMIX_EVAL_LOG"

RANDOM_EVAL_LOG="$RESULT_DIR/eval_random.log"
run_eval "$RANDOM_MODEL_TAG" "mid" "$RANDOM_EVAL_LOG"

if [ "$DO_QUALITY" -eq 1 ]; then
    for method in "${QUALITY_METHOD_ARRAY[@]}"; do
        QUALITY_MODEL_TAG="${BASE_MODEL_TAG}_quality_${method}_${TIMESTAMP}"
        QUALITY_EVAL_LOG="$RESULT_DIR/eval_quality_${method}.log"
        run_eval "$QUALITY_MODEL_TAG" "mid" "$QUALITY_EVAL_LOG"
    done
fi

echo ""
echo "╚══════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════
#  STEP 5: GENERATE REPORT
# ══════════════════════════════════════════════════════════════

echo ""
echo "╔══ Step 5: Generate experiment report ══╗"
echo ""

REPORT_ARGS=(
    --result-dir "$RESULT_DIR"
    --dataset-stats "$DATA_DIR/dataset_stats.json"
    --quadmix-train-log "$QUADMIX_LOG"
    --random-train-log "$RANDOM_LOG"
    --quadmix-eval-log "$QUADMIX_EVAL_LOG"
    --random-eval-log "$RANDOM_EVAL_LOG"
)
if [ "$DO_QUALITY" -eq 1 ]; then
    QUALITY_TRAIN_LOGS=()
    QUALITY_EVAL_LOGS=()
    for method in "${QUALITY_METHOD_ARRAY[@]}"; do
        QUALITY_TRAIN_LOGS+=("$RESULT_DIR/mid_train_quality_${method}.log")
        QUALITY_EVAL_LOGS+=("$RESULT_DIR/eval_quality_${method}.log")
    done
    REPORT_ARGS+=(--quality-train-log "${QUALITY_TRAIN_LOGS[@]}")
    REPORT_ARGS+=(--quality-eval-log "${QUALITY_EVAL_LOGS[@]}")
fi
python3 "$SCRIPT_DIR/generate_report.py" "${REPORT_ARGS[@]}"

echo ""
echo "╚════════════════════════════════════════╝"
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
echo "    ├── data/                    # Training datasets"
echo "    │   ├── quadmix_data/        # QuadMix-selected shards"
echo "    │   ├── random_data/         # Random baseline shards"
if [ "$DO_QUALITY" -eq 1 ]; then
    for method in "${QUALITY_METHOD_ARRAY[@]}"; do
        echo "    │   ├── quality_data_${method}/  # Quality ($method) shards"
    done
fi
echo "    │   └── dataset_stats.json   # Dataset statistics"
echo "    ├── mid_train_quadmix.log    # QuadMix mid-training log"
echo "    ├── mid_train_random.log     # Random mid-training log"
if [ "$DO_QUALITY" -eq 1 ]; then
    for method in "${QUALITY_METHOD_ARRAY[@]}"; do
        echo "    ├── mid_train_quality_${method}.log  # Quality ($method) training log"
    done
fi
echo "    ├── eval_quadmix.log         # QuadMix evaluation log"
echo "    ├── eval_random.log          # Random evaluation log"
if [ "$DO_QUALITY" -eq 1 ]; then
    for method in "${QUALITY_METHOD_ARRAY[@]}"; do
        echo "    ├── eval_quality_${method}.log     # Quality ($method) evaluation log"
    done
fi
echo "    └── experiment_report.md     # Comparison report"
echo ""
echo "  Mid-training checkpoints:"
echo "    $MID_CHECKPOINTS_OUTPUT_DIR/$QUADMIX_MODEL_TAG/"
echo "    $MID_CHECKPOINTS_OUTPUT_DIR/$RANDOM_MODEL_TAG/"
if [ "$DO_QUALITY" -eq 1 ]; then
    for method in "${QUALITY_METHOD_ARRAY[@]}"; do
        echo "    $MID_CHECKPOINTS_OUTPUT_DIR/${BASE_MODEL_TAG}_quality_${method}_${TIMESTAMP}/"
    done
fi
echo ""
echo "════════════════════════════════════════════════════════════"
