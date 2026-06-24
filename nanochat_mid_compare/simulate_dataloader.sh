#!/usr/bin/env bash
# Simulate the nanochat dataloader for ALL DDP ranks to find tokenizer hangs
#
# Matches mid_train.py config exactly:
#   tokenizer_threads=16, tokenizer_batch_size=256, buffer_size=2000
#   device_batch_size=8, seq_len=2048, grad_accum=4
#   total_batch_size = 8 * 2048 * 8 * 4 = 524288
#
# Usage:
#   bash nanochat_mid_compare/simulate_dataloader.sh --data-dir /path/to/quality_data_fineweb_edu

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATA_DIR=""
TOKENIZER_DIR="/home/ma-user/work/nanochat_model_dir/tokenizer"
TARGET_STEP=350
TIMEOUT_PER_BATCH=120
NUM_NPU=8
DEVICE_BATCH_SIZE=8
SEQ_LEN=2048
GRAD_ACCUM=4
BUFFER_SIZE=2000
TOKENIZER_BATCH_SIZE=256
TOKENIZER_THREADS=16

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)            DATA_DIR="$2"; shift 2 ;;
        --tokenizer-dir)       TOKENIZER_DIR="$2"; shift 2 ;;
        --target-step)         TARGET_STEP="$2"; shift 2 ;;
        --timeout)             TIMEOUT_PER_BATCH="$2"; shift 2 ;;
        --num-npu)             NUM_NPU="$2"; shift 2 ;;
        --device-batch-size)   DEVICE_BATCH_SIZE="$2"; shift 2 ;;
        --seq-len)             SEQ_LEN="$2"; shift 2 ;;
        --grad-accum)          GRAD_ACCUM="$2"; shift 2 ;;
        --buffer-size)         BUFFER_SIZE="$2"; shift 2 ;;
        --tokenizer-batch-size) TOKENIZER_BATCH_SIZE="$2"; shift 2 ;;
        --tokenizer-threads)   TOKENIZER_THREADS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$DATA_DIR" ]; then
    echo "Usage: bash $0 --data-dir /path/to/quality_data_fineweb_edu"
    echo ""
    echo "Options:"
    echo "  --data-dir DIR             (required) training data directory"
    echo "  --tokenizer-dir DIR        tokenizer directory (default: /home/ma-user/work/nanochat_model_dir/tokenizer)"
    echo "  --target-step N            simulate up to step N (default: 350)"
    echo "  --timeout SECS             per-batch timeout (default: 120)"
    echo "  --num-npu N                world size (default: 8)"
    echo "  --device-batch-size N      (default: 8)"
    echo "  --seq-len N                (default: 2048)"
    echo "  --grad-accum N             (default: 4)"
    echo "  --buffer-size N            (default: 2000)"
    echo "  --tokenizer-batch-size N   (default: 256)"
    echo "  --tokenizer-threads N      (default: 16)"
    exit 1
fi

python3 "$SCRIPT_DIR/simulate_dataloader.py" \
    --data-dir "$DATA_DIR" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --target-step "$TARGET_STEP" \
    --timeout "$TIMEOUT_PER_BATCH" \
    --num-npu "$NUM_NPU" \
    --device-batch-size "$DEVICE_BATCH_SIZE" \
    --seq-len "$SEQ_LEN" \
    --grad-accum "$GRAD_ACCUM" \
    --buffer-size "$BUFFER_SIZE" \
    --tokenizer-batch-size "$TOKENIZER_BATCH_SIZE" \
    --tokenizer-threads "$TOKENIZER_THREADS"
