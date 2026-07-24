#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# QuadMix vs Random vs Manual Ratio: STEM Mid-Training Comparison
# ──────────────────────────────────────────────────────────────
#
# This script runs nanochat mid-training on three datasets:
#   1. QuadMix-selected subset from STEM data
#   2. Random subset from STEM data (token-count aligned)
#   3. Manual Ratio subset from STEM data (domain-proportional, token-count aligned)
#
# Usage (after running QuaDMix pipeline on STEM data):
#   bash nanochat_mid_compare/run_stem_experiment.sh
#
# Override any config via environment variables:
#   QUADMIX_SAMPLED_DATA=/path/to/stem_sampled_dataset.parquet \
#   STEM_DATA_DIR=/path/to/100B_stem_parquet_filtered \
#   MANUAL_RATIO="数学=60:物理=15:化学=12.5:生物学=12.5" \
#   bash nanochat_mid_compare/run_stem_experiment.sh
#
# ──────────────────────────────────────────────────────────────

set -euo pipefail

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QUADMIX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# QuadMix output: sampled_dataset.parquet from STEM pipeline
QUADMIX_SAMPLED_DATA="${QUADMIX_SAMPLED_DATA:-/home/ma-user/work/quadmix/result/reoptimize_20260724_095141/sampled_dataset.parquet}"

# STEM data directory (raw parquets with category_name, char_count_col, etc.)
STEM_DATA_DIR="${STEM_DATA_DIR:-/home/ma-user/work/100B_stem_parquet_filtered}"

# STEM schema config (relative to QUADMIX_DIR)
SCHEMA_CONFIG="${SCHEMA_CONFIG:-configs/schema_stem.yaml}"

# File pattern for STEM parquets (not preprocessed_*.parquet)
FILE_PATTERN="${FILE_PATTERN:-*.parquet}"

# Manual Ratio: domain=ratio format
MANUAL_RATIO="${MANUAL_RATIO:-数学=60:物理=15:化学=12.5:生物学=12.5}"

# Nanochat model directory
NANOCHAT_MODEL_DIR="${NANOCHAT_MODEL_DIR:-/home/ma-user/work/nanochat_model_dir}"

# Base model tag
BASE_MODEL_TAG="${BASE_MODEL_TAG:-d24}"

# Nanochat repo root
NANOCHAT_REPO="${NANOCHAT_REPO:-/home/ma-user/work/nanochat-npu}"

# Mid-training checkpoint output directory
MID_CHECKPOINTS_OUTPUT_DIR="${MID_CHECKPOINTS_OUTPUT_DIR:-$HOME/.cache/nanochat_mid_compare_stem/mid_checkpoints}"

# Experiment output directory
RESULT_DIR="${RESULT_DIR:-$SCRIPT_DIR/results_stem/$TIMESTAMP}"

# ── Mid-training hyperparameters ──
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-0.5}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
NUM_NPU="${NUM_NPU:-8}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
EVAL_EVERY="${EVAL_EVERY:--1}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-stem}"

# Data preparation
SHARD_SIZE="${SHARD_SIZE:-10000}"
VAL_RATIO="${VAL_RATIO:-0}"
SEED="${SEED:-42}"
MAX_SHARDS="${MAX_SHARDS:-0}"

# Model tags
QUADMIX_MODEL_TAG="${QUADMIX_MODEL_TAG:-}"
RANDOM_MODEL_TAG="${RANDOM_MODEL_TAG:-}"
MANUAL_RATIO_MODEL_TAG="${MANUAL_RATIO_MODEL_TAG:-}"

# ══════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════

if [ -z "$QUADMIX_SAMPLED_DATA" ]; then
    echo "ERROR: QUADMIX_SAMPLED_DATA not set"
    echo "  Set via: QUADMIX_SAMPLED_DATA=/path/to/stem_sampled_dataset.parquet"
    exit 1
fi

if [ ! -f "$QUADMIX_SAMPLED_DATA" ]; then
    echo "ERROR: QuadMix dataset not found: $QUADMIX_SAMPLED_DATA"
    exit 1
fi

if [ ! -d "$STEM_DATA_DIR" ]; then
    echo "ERROR: STEM data directory not found: $STEM_DATA_DIR"
    echo "  Set via: STEM_DATA_DIR=/path/to/100B_stem_parquet_filtered"
    exit 1
fi

SCHEMA_PATH="$QUADMIX_DIR/$SCHEMA_CONFIG"
if [ ! -f "$SCHEMA_PATH" ]; then
    echo "ERROR: Schema config not found: $SCHEMA_PATH"
    exit 1
fi

if [ ! -d "$NANOCHAT_REPO" ]; then
    echo "ERROR: Nanochat repo not found: $NANOCHAT_REPO"
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

# Auto-detect num_scaling_params from checkpoint meta JSON
MODEL_INFO=$(python3 "$SCRIPT_DIR/get_model_info.py" \
    --ckpt-dir "$BASE_CKPT_DIR" \
    --nanochat-repo "$NANOCHAT_REPO")
NUM_SCALING_PARAMS=$(echo "$MODEL_INFO" | grep NUM_SCALING_PARAMS | cut -d= -f2)
CKPT_TOTAL_BATCH_SIZE=$(echo "$MODEL_INFO" | grep TOTAL_BATCH_SIZE | cut -d= -f2)

echo "  Auto-detected: NUM_SCALING_PARAMS=$NUM_SCALING_PARAMS, TOTAL_BATCH_SIZE=$CKPT_TOTAL_BATCH_SIZE"

# Auto-generate model tags
if [ -z "$QUADMIX_MODEL_TAG" ]; then
    QUADMIX_MODEL_TAG="${BASE_MODEL_TAG}_stem_quadmix_${TIMESTAMP}"
fi
if [ -z "$RANDOM_MODEL_TAG" ]; then
    RANDOM_MODEL_TAG="${BASE_MODEL_TAG}_stem_random_${TIMESTAMP}"
fi
if [ -z "$MANUAL_RATIO_MODEL_TAG" ]; then
    MANUAL_RATIO_MODEL_TAG="${BASE_MODEL_TAG}_stem_manual_ratio_${TIMESTAMP}"
fi

# ══════════════════════════════════════════════════════════════
#  NPU ENVIRONMENT SETUP
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
echo "  QuadMix vs Random vs Manual Ratio — STEM Mid-Training"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  QuadMix dataset:     $QUADMIX_SAMPLED_DATA"
echo "  STEM data dir:       $STEM_DATA_DIR"
echo "  Schema config:       $SCHEMA_PATH"
echo "  Nanochat model dir:  $NANOCHAT_MODEL_DIR"
echo "  Nanochat repo:       $NANOCHAT_REPO"
echo "  Base model tag:      $BASE_MODEL_TAG"
echo "  Experiment output:   $RESULT_DIR"
if [ -n "$MID_CHECKPOINTS_OUTPUT_DIR" ]; then
    echo "  Mid checkpoint dir:  $MID_CHECKPOINTS_OUTPUT_DIR"
fi
echo ""
echo "  Manual Ratio:        $MANUAL_RATIO"
echo ""
echo "  Mid-training config:"
echo "    target-param-data-ratio: $TARGET_PARAM_DATA_RATIO"
echo "    num-scaling-params:      $NUM_SCALING_PARAMS (auto-detected)"
echo "    device-batch-size:       $DEVICE_BATCH_SIZE"
echo "    NPU cards:               $NUM_NPU"
echo "    eval-benchmarks:         $EVAL_BENCHMARKS"
echo ""
echo "  Model tags (save):"
echo "    QuadMix:       $QUADMIX_MODEL_TAG"
echo "    Random:        $RANDOM_MODEL_TAG"
echo "    Manual Ratio:  $MANUAL_RATIO_MODEL_TAG"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

mkdir -p "$RESULT_DIR"

DATA_DIR="$RESULT_DIR/data"

# ══════════════════════════════════════════════════════════════
#  STEP 1: DATA PREPARATION
# ══════════════════════════════════════════════════════════════

echo ""
echo "╔══ Step 1: Prepare STEM datasets ══╗"
echo ""

_DATA_READY=true
if [ ! -f "$DATA_DIR/dataset_stats.json" ]; then
    _DATA_READY=false
fi
for _subdir in quadmix random manual_ratio; do
    if [ ! -d "$DATA_DIR/$_subdir" ]; then
        _DATA_READY=false
        break
    fi
done
if $_DATA_READY; then
    echo "  Datasets already exist. Skipping preparation."
    echo "  (Delete $DATA_DIR to regenerate)"
else
    PREP_ARGS=(
        --quadmix-sampled-data "$QUADMIX_SAMPLED_DATA"
        --preprocessed-data-dir "$STEM_DATA_DIR"
        --output-dir "$DATA_DIR"
        --tokenizer-pkl "$TOKENIZER_DIR/tokenizer.pkl"
        --schema "$SCHEMA_PATH"
        --file-pattern "$FILE_PATTERN"
        --manual-ratio "$MANUAL_RATIO"
        --data-ratio "$TARGET_PARAM_DATA_RATIO"
        --num-scaling-params "$NUM_SCALING_PARAMS"
        --shard-size "$SHARD_SIZE"
        --val-ratio "$VAL_RATIO"
        --seed "$SEED"
        --max-shards "$MAX_SHARDS"
        --num-npu "$NUM_NPU"
    )
    python3 "$SCRIPT_DIR/prepare_data.py" "${PREP_ARGS[@]}"
fi

# Update dataset_stats.json with experiment metadata
DATA_DIR="$DATA_DIR" BASE_MODEL_TAG="$BASE_MODEL_TAG" \
TARGET_PARAM_DATA_RATIO="$TARGET_PARAM_DATA_RATIO" \
NUM_SCALING_PARAMS="$NUM_SCALING_PARAMS" \
DEVICE_BATCH_SIZE="$DEVICE_BATCH_SIZE" \
NUM_NPU="$NUM_NPU" \
TOTAL_BATCH_SIZE="${CKPT_TOTAL_BATCH_SIZE:-524288}" \
TOKENIZER_PKL="$TOKENIZER_DIR/tokenizer.pkl" \
NANOCHAT_REPO="$NANOCHAT_REPO" \
NANOCHAT_MODEL_DIR="$NANOCHAT_MODEL_DIR" \
MID_CHECKPOINTS_OUTPUT_DIR="$MID_CHECKPOINTS_OUTPUT_DIR" \
QUADMIX_GIT_HASH="$(git -C "$QUADMIX_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
NANOCHAT_GIT_HASH="$(git -C "$NANOCHAT_REPO" rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
MANUAL_RATIO="$MANUAL_RATIO" \
EVAL_BENCHMARKS="$EVAL_BENCHMARKS" \
python3 -c "
import os, json
stats_path = os.path.join(os.environ['DATA_DIR'], 'dataset_stats.json')
stats = json.load(open(stats_path))
stats['config'].update({
    'base_model_tag': os.environ['BASE_MODEL_TAG'],
    'target_param_data_ratio': float(os.environ['TARGET_PARAM_DATA_RATIO']),
    'num_scaling_params': int(os.environ['NUM_SCALING_PARAMS']),
    'device_batch_size': int(os.environ['DEVICE_BATCH_SIZE']),
    'total_batch_size': int(os.environ['TOTAL_BATCH_SIZE']),
    'num_npu': int(os.environ['NUM_NPU']),
    'tokenizer_pkl': os.environ.get('TOKENIZER_PKL', ''),
    'nanochat_repo': os.environ.get('NANOCHAT_REPO', ''),
    'nanochat_model_dir': os.environ.get('NANOCHAT_MODEL_DIR', ''),
    'mid_checkpoints_output_dir': os.environ.get('MID_CHECKPOINTS_OUTPUT_DIR', ''),
    'quadmix_git_hash': os.environ.get('QUADMIX_GIT_HASH', ''),
    'nanochat_git_hash': os.environ.get('NANOCHAT_GIT_HASH', ''),
    'manual_ratio': os.environ.get('MANUAL_RATIO', ''),
    'eval_benchmarks': os.environ.get('EVAL_BENCHMARKS', ''),
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
if 'manual_ratio' in stats:
    mr = stats['manual_ratio']
    print(f'    Manual Ratio: {mr[\"train_docs\"]:,} train docs, {mr[\"tokens\"]:,} tokens, {mr[\"shards\"]} shards')
for key in stats:
    if key.startswith('quality_'):
        method = key[len('quality_'):]
        ql = stats[key]
        print(f'    Quality ({method}): {ql[\"train_docs\"]:,} train docs, {ql[\"tokens\"]:,} tokens, {ql[\"shards\"]} shards')
budget = stats['config'].get('budget_cap', 'N/A')
target = stats['config'].get('target_tokens', 'N/A')
print(f'    Budget cap: {budget}')
print(f'    Target tokens: {target}')
"
echo ""
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════
#  STEP 2: SETUP MID_CHECKPOINTS DIRECTORY
# ══════════════════════════════════════════════════════════════

if [ -n "$MID_CHECKPOINTS_OUTPUT_DIR" ]; then
    echo ""
    echo "╔══ Step 2: Setup mid-training checkpoint output ══╗"
    echo ""

    mkdir -p "$MID_CHECKPOINTS_OUTPUT_DIR"
    LINK_PATH="$NANOCHAT_MODEL_DIR/mid_checkpoints"

    if [ -L "$LINK_PATH" ]; then
        echo "  Symlink already exists: $LINK_PATH -> $(readlink "$LINK_PATH")"
    elif [ -d "$LINK_PATH" ]; then
        echo "  WARNING: $LINK_PATH exists as a directory (not a symlink)."
        echo "  Skipping symlink creation."
    else
        echo "  Creating symlink: $LINK_PATH -> $MID_CHECKPOINTS_OUTPUT_DIR"
        ln -s "$MID_CHECKPOINTS_OUTPUT_DIR" "$LINK_PATH"
    fi

    echo ""
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
fi

# ══════════════════════════════════════════════════════════════
#  STEP 3: MID-TRAINING
# ══════════════════════════════════════════════════════════════

QUADMIX_DATA="$DATA_DIR/quadmix_data"
RANDOM_DATA="$DATA_DIR/random_data"
MANUAL_RATIO_DATA="$DATA_DIR/manual_ratio_data"

run_mid_training() {
    local DATA_PATH="$1"
    local MODEL_TAG="$2"
    local RUN_NAME="$3"
    local LOG_FILE="$4"
    local DATASET_TOKENS="$5"
    local TRAIN_TOKENS="$6"

    local BASE_CKPT_DIR="$NANOCHAT_MODEL_DIR/base_checkpoints/$BASE_MODEL_TAG"
    local META_JSON=$(ls "$BASE_CKPT_DIR"/meta_*.json 2>/dev/null | sort | tail -1)
    if [ -n "$META_JSON" ]; then
        local TOTAL_BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$META_JSON'))['total_batch_size'])")
        echo "    Read total_batch_size=$TOTAL_BATCH_SIZE from checkpoint"
    else
        echo "    ERROR: No meta JSON found in $BASE_CKPT_DIR" >&2
        exit 1
    fi
    local NUM_ITERATIONS=$(( (TRAIN_TOKENS + TOTAL_BATCH_SIZE - 1) / TOTAL_BATCH_SIZE ))

    echo "  Starting mid-training: $RUN_NAME"
    echo "    Data:       $DATA_PATH"
    echo "    Source:     $BASE_MODEL_TAG (base)"
    echo "    Save as:    $MODEL_TAG (mid)"
    echo "    Dataset:    $DATASET_TOKENS tokens"
    echo "    Train:      $TRAIN_TOKENS tokens (budget_cap, ratio=$TARGET_PARAM_DATA_RATIO)"
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
        --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
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
        echo "    Cleaning up symlink: $LINK_DIR"
        rm "$LINK_DIR"
    fi
}

STATS_FILE="$DATA_DIR/dataset_stats.json"
BUDGET_CAP=$(STATS_FILE="$STATS_FILE" python3 -c "
import os, json
s = json.load(open(os.environ['STATS_FILE']))
print(s['config'].get('budget_cap', '0'))
")
if [ "$BUDGET_CAP" -le 0 ] 2>/dev/null || [ -z "$BUDGET_CAP" ]; then
    echo "ERROR: budget_cap is missing or zero in dataset_stats.json" >&2
    echo "       Delete $DATA_DIR and re-run prepare_data to regenerate." >&2
    exit 1
fi
read -r QUADMIX_TOKENS RANDOM_TOKENS < <(
    STATS_FILE="$STATS_FILE" python3 -c "
import os, json
s = json.load(open(os.environ['STATS_FILE']))
print(s['quadmix']['tokens'], s['random']['tokens'])
"
)
MANUAL_RATIO_TOKENS=$(STATS_FILE="$STATS_FILE" python3 -c "
import os, json
s = json.load(open(os.environ['STATS_FILE']))
if 'manual_ratio' in s:
    print(s['manual_ratio']['tokens'])
else:
    print('0')
")
echo "  Common training budget: $BUDGET_CAP tokens (from budget_cap)"

echo ""
echo "╔══ Step 3a: Mid-training on QuadMix data ══╗"
echo ""

QUADMIX_LOG="$RESULT_DIR/mid_train_quadmix.log"
run_mid_training "$QUADMIX_DATA" "$QUADMIX_MODEL_TAG" "stem_quadmix_mid" "$QUADMIX_LOG" "$QUADMIX_TOKENS" "$BUDGET_CAP"

echo ""
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo ""
echo "╔══ Step 3b: Mid-training on Random data ══╗"
echo ""

RANDOM_LOG="$RESULT_DIR/mid_train_random.log"
run_mid_training "$RANDOM_DATA" "$RANDOM_MODEL_TAG" "stem_random_mid" "$RANDOM_LOG" "$RANDOM_TOKENS" "$BUDGET_CAP"

echo ""
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo ""
echo "╔══ Step 3c: Mid-training on Manual Ratio data ══╗"
echo ""

MANUAL_RATIO_LOG="$RESULT_DIR/mid_train_manual_ratio.log"
run_mid_training "$MANUAL_RATIO_DATA" "$MANUAL_RATIO_MODEL_TAG" "stem_manual_ratio_mid" "$MANUAL_RATIO_LOG" "$MANUAL_RATIO_TOKENS" "$BUDGET_CAP"

echo ""
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

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
        --eval-benchmarks="$EVAL_BENCHMARKS" \
        --device-batch-size=16 \
        --model-tag="$MODEL_TAG" \
        --model-type="$MODEL_TYPE" \
        2>&1 | tee "$LOG_FILE"
    popd > /dev/null
}

echo ""
echo "╔══ Step 4: Evaluation (benchmarks=$EVAL_BENCHMARKS) ══╗"
echo ""

QUADMIX_EVAL_LOG="$RESULT_DIR/eval_quadmix.log"
run_eval "$QUADMIX_MODEL_TAG" "mid" "$QUADMIX_EVAL_LOG"

RANDOM_EVAL_LOG="$RESULT_DIR/eval_random.log"
run_eval "$RANDOM_MODEL_TAG" "mid" "$RANDOM_EVAL_LOG"

MANUAL_RATIO_EVAL_LOG="$RESULT_DIR/eval_manual_ratio.log"
run_eval "$MANUAL_RATIO_MODEL_TAG" "mid" "$MANUAL_RATIO_EVAL_LOG"

echo ""
echo "╚════════════════════════════════════════════════════════════╝"
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
    --manual-ratio-train-log "$MANUAL_RATIO_LOG"
    --quadmix-eval-log "$QUADMIX_EVAL_LOG"
    --random-eval-log "$RANDOM_EVAL_LOG"
    --manual-ratio-eval-log "$MANUAL_RATIO_EVAL_LOG"
)
python3 "$SCRIPT_DIR/generate_report.py" "${REPORT_ARGS[@]}"

echo ""
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  STEM Experiment Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Output directory: $RESULT_DIR"
echo ""
echo "  Files:"
echo "    ├── data/                        # Training datasets"
echo "    │   ├── quadmix_data/            # QuadMix shards"
echo "    │   ├── random_data/             # Random baseline shards"
echo "    │   ├── manual_ratio_data/       # Manual Ratio shards"
echo "    │   └── dataset_stats.json       # Statistics"
echo "    ├── mid_train_quadmix.log"
echo "    ├── mid_train_random.log"
echo "    ├── mid_train_manual_ratio.log"
echo "    ├── eval_quadmix.log"
echo "    ├── eval_random.log"
echo "    ├── eval_manual_ratio.log"
echo "    └── experiment_report.md"
echo ""
echo "  Mid-training checkpoints:"
echo "    $MID_CHECKPOINTS_OUTPUT_DIR/$QUADMIX_MODEL_TAG/"
echo "    $MID_CHECKPOINTS_OUTPUT_DIR/$RANDOM_MODEL_TAG/"
echo "    $MID_CHECKPOINTS_OUTPUT_DIR/$MANUAL_RATIO_MODEL_TAG/"
echo ""
echo "════════════════════════════════════════════════════════════"