#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Demo: QuaDMix Re-evaluation — 切换验证集快速重跑
# ──────────────────────────────────────────────────────────────
# 加载已保存的代理模型权重，在新验证集上重新评估，
# 然后重新拟合 LightGBM、搜索最优参数、抽样、生成报告。
#
# 无需重新训练代理模型（节省大量时间）。
#
# Usage:
#   bash scripts/demo_reval.sh --result-dir result/quadmix_20260609_120000
#
# 切换验证集：
#   bash scripts/demo_reval.sh --result-dir result/xxx --val-set openhermes
#   bash scripts/demo_reval.sh --result-dir result/xxx --val-set core
#   bash scripts/demo_reval.sh --result-dir result/xxx --val-path /path/to/custom.pt
#
# 指定设备：
#   bash scripts/demo_reval.sh --result-dir result/xxx --device-type npu
#
# 自定义输出目录：
#   bash scripts/demo_reval.sh --result-dir result/xxx --output result/my_reval
#
# 调整搜索参数：
#   bash scripts/demo_reval.sh --result-dir result/xxx --num-search 50000 --top-k 5
#
# 指定目标数据量（单位 B tokens）：
#   bash scripts/demo_reval.sh --result-dir result/xxx --target-tokens 10
#
# HF 镜像加速（中国用户）：
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/demo_reval.sh --result-dir result/xxx
# ──────────────────────────────────────────────────────────────

set -euo pipefail

if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate nano
fi

QUADMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${QUADMIX_DIR}/src:${PYTHONPATH:-}"
export PATH="$HOME/.local/bin:$PATH"
export QUADMIX_TEMP_DIR="${QUADMIX_TEMP_DIR:-$HOME/.cache/QuaDMix/temp}"

PREPROCESSED_DIR="$QUADMIX_TEMP_DIR/preprocessed"

RESULT_DIR=""
VAL_SET="core"
VAL_PATH=""
OUTPUT=""
DEVICE_TYPE="npu"
NUM_SEARCH="100000"
TOP_K="10"
TARGET_TOKENS="0"
BLOCK_SIZE="2048"
MODEL_VARIANT="tinyllama_1M"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --result-dir)    RESULT_DIR="$2"; shift 2 ;;
        --val-set)       VAL_SET="$2"; shift 2 ;;
        --val-path)      VAL_PATH="$2"; shift 2 ;;
        --output|-o)     OUTPUT="$2"; shift 2 ;;
        --device-type)   DEVICE_TYPE="$2"; shift 2 ;;
        --num-search)    NUM_SEARCH="$2"; shift 2 ;;
        --top-k)         TOP_K="$2"; shift 2 ;;
        --target-tokens) TARGET_TOKENS="$2"; shift 2 ;;
        --block-size)    BLOCK_SIZE="$2"; shift 2 ;;
        --model-variant) MODEL_VARIANT="$2"; shift 2 ;;
        --preprocessed-dir) PREPROCESSED_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/demo_reval.sh --result-dir <path> [options]"
            echo ""
            echo "Required:"
            echo "  --result-dir PATH        Original pipeline result directory"
            echo ""
            echo "Options:"
            echo "  --val-set {core,openhermes}  New validation set (default: core)"
            echo "  --val-path PATH              Custom .pt file (overrides --val-set)"
            echo "  --output PATH                Output directory (default: auto)"
            echo "  --device-type {cpu,cuda,npu} Device (default: cpu)"
            echo "  --num-search N               Search points (default: 100000)"
            echo "  --top-k N                    Top-K average (default: 10)"
            echo "  --target-tokens N            Target in billions (default: 0)"
            echo "  --block-size N               Block size (default: 2048)"
            echo "  --model-variant NAME         Model variant (default: tinyllama_1M)"
            echo "  --preprocessed-dir PATH      Preprocessed shards dir"
            exit 0
            ;;
        *)
            echo "[Error] Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$RESULT_DIR" ]]; then
    echo "[Error] --result-dir is required"
    echo "Usage: bash scripts/demo_reval.sh --result-dir result/quadmix_20260609_120000"
    exit 1
fi

if [[ ! -d "$RESULT_DIR/proxy_experiments" ]]; then
    echo "[Error] proxy_experiments not found in: $RESULT_DIR"
    exit 1
fi

if [[ ! -d "$PREPROCESSED_DIR" ]]; then
    echo "[Error] Preprocessed dir not found: $PREPROCESSED_DIR"
    echo "  Set --preprocessed-dir or run a pipeline first"
    exit 1
fi

MODEL_COUNT=$(find "$RESULT_DIR/proxy_experiments" -name "model.pt" 2>/dev/null | wc -l)
if [[ "$MODEL_COUNT" -eq 0 ]]; then
    echo "[Error] No model.pt found in $RESULT_DIR/proxy_experiments/"
    echo "  The original run must save model weights (model.pt in each exp dir)"
    exit 1
fi

echo "╔══ QuaDMix Re-evaluation ══╗"
echo ""
echo "  Source:        $RESULT_DIR"
echo "  Models found:  $MODEL_COUNT"
echo "  Val set:       $VAL_SET"
[[ -n "$VAL_PATH" ]] && echo "  Val path:      $VAL_PATH"
echo "  Device:        $DEVICE_TYPE"
echo "  Preprocessed:  $PREPROCESSED_DIR"
echo "  Search points: $NUM_SEARCH"
echo "  Top-K:         $TOP_K"
[[ "$TARGET_TOKENS" != "0" ]] && echo "  Target tokens: ${TARGET_TOKENS}B"
echo ""
echo "╚════════════════════════════╝"
echo ""

ARGS=(
    --result-dir "$RESULT_DIR"
    --preprocessed-dir "$PREPROCESSED_DIR"
    --val-set "$VAL_SET"
    --device-type "$DEVICE_TYPE"
    --num-search "$NUM_SEARCH"
    --top-k "$TOP_K"
    --target-tokens "$TARGET_TOKENS"
    --block-size "$BLOCK_SIZE"
    --model-variant "$MODEL_VARIANT"
)

[[ -n "$VAL_PATH" ]] && ARGS+=(--val-path "$VAL_PATH")
[[ -n "$OUTPUT" ]] && ARGS+=(--output "$OUTPUT")

python3 "$QUADMIX_DIR/scripts/reval_with_new_valset.py" "${ARGS[@]}"
