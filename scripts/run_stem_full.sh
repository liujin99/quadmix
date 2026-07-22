#!/usr/bin/env bash
# QuaDMix full run with direct parquet_filter input (no preprocessing output).
#
# Server layout:
#   /home/ma-user/work/data-mixing/parquet_filter/*.parquet
#   /home/ma-user/work/data-mixing/quadmix/
#   /home/ma-user/work/data-mixing/quadmix/data/tokenizers/gpt-neox-20b/
#   /home/ma-user/work/data-mixing/quadmix_result/stem_full_01/
#
# Full run:
#   bash scripts/run_stem_full.sh
#
# Two-shard CPU smoke test:
#   STEM_SHARD_LIMIT=2 NUM_EXPERIMENTS=8 NUM_SEARCH=100 TRAIN_STEPS=3 \
#   BLOCK_SIZE=64 MICRO_BATCH_SIZE=2 GLOBAL_BATCH_SIZE=8 ALLOW_CPU_FULL=1 \
#   OUTPUT_DIR=result/stem_smoke bash scripts/run_stem_full.sh

set -euo pipefail

NANO_ENV_PATH="${CONDA_ENV_PATH:-/home/ma-user/miniforge3/envs/nano}"
if [[ "${CONDA_PREFIX:-}" == "$NANO_ENV_PATH" ]]; then
    echo "  Using active Conda environment: $NANO_ENV_PATH"
else
    if [[ ! -f /home/ma-user/miniforge3/etc/profile.d/conda.sh ]]; then
        echo "ERROR: Miniforge initialization script not found." >&2
        exit 2
    fi
    if [[ ! -d "$NANO_ENV_PATH" ]]; then
        echo "ERROR: Conda environment not found: $NANO_ENV_PATH" >&2
        exit 2
    fi
    source /home/ma-user/miniforge3/etc/profile.d/conda.sh
    conda activate "$NANO_ENV_PATH"
fi

PYTHON_BIN="$NANO_ENV_PATH/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python not found: $PYTHON_BIN" >&2
    exit 2
fi
export LD_LIBRARY_PATH="$NANO_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"

QUADMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_MIXING_DIR="${DATA_MIXING_DIR:-$(cd "$QUADMIX_DIR/.." && pwd)}"
PARQUET_FILTER_DIR="${PARQUET_FILTER_DIR:-/data/l00916525/parquet_filter——v3}"
export STEM_METADATA_CACHE_DIR="${STEM_METADATA_CACHE_DIR:-$PARQUET_FILTER_DIR/.quadmix_metadata_cache}"

export PYTHONPATH="${QUADMIX_DIR}/src:${PYTHONPATH:-}"
export PATH="$HOME/.local/bin:$PATH"
QUADMIX_BASE_TEMP_DIR="${QUADMIX_TEMP_DIR:-$QUADMIX_DIR/temp}"
export QUADMIX_TEMP_DIR="$QUADMIX_BASE_TEMP_DIR/stem_raw"

TOKEN_CACHE_DIR="$QUADMIX_TEMP_DIR/token_cache"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ma-user/work/data-mixing/quadmix_result/stem_full_01}"
export QUADMIX_TOKENIZER_PATH="${QUADMIX_TOKENIZER_PATH:-$QUADMIX_DIR/data/tokenizers/gpt-neox-20b}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

NUM_EXPERIMENTS="${NUM_EXPERIMENTS:-500}"
NUM_SEARCH="${NUM_SEARCH:-5000}"
TOP_K="${TOP_K:-5}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-64}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
RANK_REF_SIZE="${RANK_REF_SIZE:-10000}"
VAL_SET="${VAL_SET:-cap_v1}"

# A command-line --val-set takes precedence over VAL_SET.
previous=""
for argument in "$@"; do
    if [[ "$previous" == "--val-set" ]]; then
        VAL_SET="$argument"
        break
    fi
    previous="$argument"
done

if [[ ! -d "$PARQUET_FILTER_DIR" ]]; then
    echo "ERROR: parquet_filter directory not found: $PARQUET_FILTER_DIR" >&2
    exit 2
fi
if ! find "$PARQUET_FILTER_DIR" -maxdepth 1 -type f -name '*.parquet' -print -quit | grep -q .; then
    echo "ERROR: no parquet files found in $PARQUET_FILTER_DIR" >&2
    exit 2
fi
for tokenizer_file in tokenizer.json tokenizer_config.json config.json; do
    if [[ ! -f "$QUADMIX_TOKENIZER_PATH/$tokenizer_file" ]]; then
        echo "ERROR: local tokenizer file not found:" >&2
        echo "       $QUADMIX_TOKENIZER_PATH/$tokenizer_file" >&2
        exit 2
    fi
done

SHARD_ARGS=()
if [[ -n "${STEM_SHARD_LIMIT:-}" ]]; then
    SHARD_ARGS=(--shard-limit "$STEM_SHARD_LIMIT")
fi

echo "═══════════════════════════════════════════"
echo "  QuaDMix STEM Direct Input"
echo "  No preprocessed parquet will be generated"
echo "  Source: $PARQUET_FILTER_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Tokenizer: $QUADMIX_TOKENIZER_PATH (offline)"
echo "  Domains: 数学 / 化学 / 生物学 / 物理"
echo "  Quality direction: higher is better for all five columns"
if [[ -n "${STEM_SHARD_LIMIT:-}" ]]; then
    echo "  Shard limit: $STEM_SHARD_LIMIT"
fi
echo "═══════════════════════════════════════════"

echo "[0/3] Validating nano Python and torch_npu..."
"$PYTHON_BIN" - <<'PY'
import sqlite3
import torch
import torch_npu

if not torch.npu.is_available():
    raise SystemExit("torch_npu imported, but torch.npu.is_available() is False")
print(f"  Python: {__import__('sys').executable}")
print(f"  torch: {torch.__version__}")
print("  NPU runtime OK")
PY

echo "[1/3] Validating source parquet schemas..."
"$PYTHON_BIN" - "$PARQUET_FILTER_DIR" "${STEM_SHARD_LIMIT:-}" <<'PY'
import glob
import os
import sys
import pyarrow.parquet as pq

data_dir = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
files = sorted(
    path for path in glob.glob(os.path.join(data_dir, "*.parquet"))
    if not os.path.basename(path).startswith("_")
)
if limit is not None:
    files = files[:limit]
required = {
    "text", "char_count_col", "category_name", "category_score", "stem_relevance",
    "knowledge_value", "notation_fidelity", "rigor_coherence", "noise_level",
}
for path in files:
    missing = required - set(pq.ParquetFile(path).schema_arrow.names)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
print(f"  schema OK: {len(files)} shards")
PY

echo "[2/3] Ensuring validation set: $VAL_SET"
source "$QUADMIX_DIR/scripts/ensure_val_data.sh"
ensure_val_set "$VAL_SET" "$QUADMIX_DIR/data"

if [[ "${CLEAN_TOKEN_CACHE:-0}" == "1" && -d "$TOKEN_CACHE_DIR" ]]; then
    echo "  Cleaning token cache: $TOKEN_CACHE_DIR"
    rm -rf "$TOKEN_CACHE_DIR"
fi

DEVICE_ARGS=()
if command -v npu-smi &>/dev/null; then
    NPU_DEVICES="${NPU_DEVICES:-8}"
    DEVICE_ARGS=(--device-type npu --npu-devices "$NPU_DEVICES")
    echo "  Device: Ascend NPU ($NPU_DEVICES devices)"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    DEVICE_ARGS=(--device-type cuda --npu-devices 1)
    echo "  Device: CUDA ($CUDA_VISIBLE_DEVICES)"
elif [[ "${ALLOW_CPU_FULL:-0}" == "1" ]]; then
    DEVICE_ARGS=(--device-type cpu --npu-devices 1)
    echo "  WARNING: CPU mode is intended only for a reduced smoke test"
else
    echo "ERROR: no NPU/CUDA device detected. Set ALLOW_CPU_FULL=1 only for a reduced smoke test." >&2
    exit 2
fi

export STEM_METADATA_WORKERS="${STEM_METADATA_WORKERS:-32}"
export PRESAMPLE_MAX_WORKERS="${PRESAMPLE_MAX_WORKERS:-64}"
export TOKENIZE_WORKERS="${TOKENIZE_WORKERS:-16}"
export TOKENIZE_THREADS_PER_WORKER="${TOKENIZE_THREADS_PER_WORKER:-1}"
export QUADMIX_PERF_TIMER="${QUADMIX_PERF_TIMER:-1}"

echo "[3/3] Running QuaDMix directly on parquet_filter..."
"$PYTHON_BIN" "$QUADMIX_DIR/scripts/runners/run_essential_web_v1.py" \
    --preprocessed-dir "$PARQUET_FILTER_DIR" \
    --schema "$QUADMIX_DIR/configs/schema_stem.yaml" \
    "${SHARD_ARGS[@]}" \
    --num-experiments "$NUM_EXPERIMENTS" \
    --num-search "$NUM_SEARCH" \
    --top-k "$TOP_K" \
    --block-size "$BLOCK_SIZE" \
    --tiny-steps "$TRAIN_STEPS" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --rank-ref-size "$RANK_REF_SIZE" \
    --checkpoint-interval 0 \
    --val-set "$VAL_SET" \
    --search-mode r2_sigma_weighted \
    --output "$OUTPUT_DIR" \
    "${DEVICE_ARGS[@]}" \
    "$@"

echo "═══════════════════════════════════════════"
echo "  QuaDMix STEM direct run complete"
echo "  Output: $OUTPUT_DIR"
echo "    optimal_parameters.json"
echo "    pipeline_summary.json"
echo "    sampled_dataset.parquet"
echo "    quadmix_report.md"
echo "    proxy_experiments/"
echo "═══════════════════════════════════════════"
