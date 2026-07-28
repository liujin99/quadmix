#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Demo: QuaDMix Re-validate — 换新验证集重新评估
# ──────────────────────────────────────────────────────────────
# 从 Stage 4 开始：加载已保存的代理模型权重，用新验证集重新
# 计算 loss → 拟合 LightGBM → 搜索最优参数 → 抽样 → 生成报告。
#
# 适用场景：
#   - 切换验证集（如从 core 换到 openhermes）
#   - 验证集更新后需要重新评估所有代理模型
#
# 无需重新训练代理模型（节省大量时间）。
#
# 如果 loss 已经算好，只想重跑 LightGBM + search，请用：
#   bash scripts/demo_reoptimize.sh
#
# Usage:
#   bash scripts/demo_revalidate.sh --source-dir result/quadmix_20260609_120000
#
# 切换验证集：
#   bash scripts/demo_revalidate.sh --source-dir result/xxx --val-set stem_v2
#   bash scripts/demo_revalidate.sh --source-dir result/xxx --val-set core
#   bash scripts/demo_revalidate.sh --source-dir result/xxx --val-path /path/to/custom.pt
#
# 指定 schema（默认根据 val-set 自动推断）：
#   bash scripts/demo_revalidate.sh --source-dir result/xxx --schema configs/schema_stem.yaml
#
# 指定设备：
#   bash scripts/demo_revalidate.sh --source-dir result/xxx --device-type npu
#
# 自定义输出目录：
#   bash scripts/demo_revalidate.sh --source-dir result/xxx --output result/my_revalidate
#
# 调整搜索参数：
#   bash scripts/demo_revalidate.sh --source-dir result/xxx --num-search 50000 --top-k 5
#
# 指定目标数据量（单位 B tokens）：
#   bash scripts/demo_revalidate.sh --source-dir result/xxx --target-tokens 10
#
# HF 镜像加速（中国用户）：
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/demo_revalidate.sh --source-dir result/xxx
# ──────────────────────────────────────────────────────────────

set -euo pipefail

export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"

if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate nano
fi

QUADMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${QUADMIX_DIR}/src:${PYTHONPATH:-}"
export PATH="$HOME/.local/bin:$PATH"
export QUADMIX_TEMP_DIR="${QUADMIX_TEMP_DIR:-$HOME/.cache/QuaDMix/temp}"

PREPROCESSED_DIR="$QUADMIX_TEMP_DIR/preprocessed"

SOURCE_DIR="${SOURCE_DIR:-}"
VAL_SET="stem_v2"
VAL_PATH=""
OUTPUT=""
DEVICE_TYPE="npu"
NUM_SEARCH="100000"
TOP_K="10"
TARGET_TOKENS="0"
BLOCK_SIZE="2048"
MODEL_VARIANT="tinyllama_1M"
SEARCH_MODE="r2_weighted"
SCHEMA=""
SCHEMA_SET=0
PREPROCESSED_DIR_SET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-dir)    SOURCE_DIR="$2"; shift 2 ;;
        --val-set)       VAL_SET="$2"; shift 2 ;;
        --val-path)      VAL_PATH="$2"; shift 2 ;;
        --output|-o)     OUTPUT="$2"; shift 2 ;;
        --device-type)   DEVICE_TYPE="$2"; shift 2 ;;
        --num-search)    NUM_SEARCH="$2"; shift 2 ;;
        --top-k)         TOP_K="$2"; shift 2 ;;
        --target-tokens) TARGET_TOKENS="$2"; shift 2 ;;
        --block-size)    BLOCK_SIZE="$2"; shift 2 ;;
        --model-variant) MODEL_VARIANT="$2"; shift 2 ;;
        --preprocessed-dir) PREPROCESSED_DIR="$2"; PREPROCESSED_DIR_SET=1; shift 2 ;;
        --search-mode)     SEARCH_MODE="$2"; shift 2 ;;
        --schema)          SCHEMA="$2"; SCHEMA_SET=1; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/demo_revalidate.sh --source-dir <path> [options]"
            echo ""
            echo "Required:"
            echo "  --source-dir PATH        Original pipeline result directory"
            echo ""
            echo "Options:"
            echo "  --val-set {core,openhermes,core_bmk_v6,cap_v1,stem_v1,stem_v2}  New validation set (default: stem_v2)"
            echo "  --val-path PATH              Custom .pt file (overrides --val-set)"
            echo "  --schema PATH                Dataset schema YAML (default: auto from val-set)"
            echo "  --output PATH                Output directory (default: auto)"
            echo "  --device-type {cpu,cuda,npu} Device (default: npu)"
            echo "  --num-search N               Search points (default: 100000)"
            echo "  --top-k N                    Top-K average (default: 10)"
            echo "  --target-tokens N            Target in billions (default: 0)"
            echo "  --block-size N               Block size (default: 2048)"
            echo "  --model-variant NAME         Model variant (default: tinyllama_1M)"
            echo "  --preprocessed-dir PATH      Preprocessed shards dir"
            echo "  --search-mode {r2_weighted,equal_weight,r2_sigma_weighted}  Search mode (default: r2_weighted)"
            exit 0
            ;;
        *)
            echo "[Error] Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$SOURCE_DIR" ]]; then
    echo "[Error] --source-dir is required"
    echo "Usage: bash scripts/demo_revalidate.sh --source-dir result/quadmix_20260609_120000"
    exit 1
fi

if [[ ! -d "$SOURCE_DIR/proxy_experiments" ]]; then
    echo "[Error] proxy_experiments not found in: $SOURCE_DIR"
    exit 1
fi

# ── Auto-detect schema and preprocessed-dir from pipeline_summary.json ──
if [[ -f "$SOURCE_DIR/pipeline_summary.json" ]]; then
    _detected=$(python3 -c "
import json, os
try:
    with open('$SOURCE_DIR/pipeline_summary.json') as f:
        s = json.load(f)
    p = s.get('input_file', '')
    if p and os.path.isdir(p):
        print(f'PPD:{p}')
    vs = s.get('reval', {}).get('new_val_set') or s.get('config', {}).get('val_set', '')
    if vs and vs != 'unknown':
        print(f'VSET:{vs}')
except: pass
" 2>/dev/null)
    while IFS= read -r _line; do
        case "$_line" in
            PPD:*)
                if [[ "$PREPROCESSED_DIR_SET" -eq 0 ]]; then
                    PREPROCESSED_DIR="${_line#PPD:}"
                    echo "  [auto] preprocessed-dir: $PREPROCESSED_DIR (from pipeline_summary.json)"
                fi
                ;;
            VSET:*)
                if [[ "$SCHEMA_SET" -eq 0 ]]; then
                    _vs="${_line#VSET:}"
                    case "$_vs" in
                        stem_v1|stem_v2)
                            SCHEMA="$QUADMIX_DIR/configs/schema_stem.yaml"
                            ;;
                        *)
                            SCHEMA="$QUADMIX_DIR/configs/schema_essential_web.yaml"
                            ;;
                    esac
                    echo "  [auto] schema: $SCHEMA (from val_set=$_vs in pipeline_summary.json)"
                fi
                ;;
        esac
    done <<< "$_detected"
fi

# Auto-detect schema from val-set if still not set
if [[ -z "$SCHEMA" ]]; then
    case "$VAL_SET" in
        stem_v1|stem_v2)
            SCHEMA="$QUADMIX_DIR/configs/schema_stem.yaml"
            ;;
        *)
            SCHEMA="$QUADMIX_DIR/configs/schema_essential_web.yaml"
            ;;
    esac
fi

if [[ ! -f "$SCHEMA" ]]; then
    echo "[Error] Schema file not found: $SCHEMA"
    echo "  Specify with --schema /path/to/schema.yaml"
    exit 1
fi

if [[ -z "$VAL_PATH" ]]; then
    source "$QUADMIX_DIR/scripts/ensure_val_data.sh"
    ensure_val_set "$VAL_SET" "$QUADMIX_DIR/data" || exit 1
fi

if [[ ! -d "$PREPROCESSED_DIR" ]]; then
    echo "[Error] Preprocessed dir not found: $PREPROCESSED_DIR"
    echo "  Set --preprocessed-dir or run a pipeline first"
    exit 1
fi

MODEL_COUNT=$(find "$SOURCE_DIR/proxy_experiments" -name "model.pt" 2>/dev/null | wc -l)
if [[ "$MODEL_COUNT" -eq 0 ]]; then
    echo "[Error] No model.pt found in $SOURCE_DIR/proxy_experiments/"
    echo "  The original run must save model weights (model.pt in each exp dir)"
    exit 1
fi

echo "╔══ QuaDMix Re-evaluation ══╗"
echo ""
echo "  Source:        $SOURCE_DIR"
echo "  Models found:  $MODEL_COUNT"
echo "  Val set:       $VAL_SET"
[[ -n "$VAL_PATH" ]] && echo "  Val path:      $VAL_PATH"
echo "  Schema:        $SCHEMA"
echo "  Device:        $DEVICE_TYPE"
echo "  Preprocessed:  $PREPROCESSED_DIR"
echo "  Search points: $NUM_SEARCH"
echo "  Top-K:         $TOP_K"
[[ "$TARGET_TOKENS" != "0" ]] && echo "  Target tokens: ${TARGET_TOKENS}B"
echo ""
echo "╚════════════════════════════╝"
echo ""

ARGS=(
    --source-dir "$SOURCE_DIR"
    --preprocessed-dir "$PREPROCESSED_DIR"
    --val-set "$VAL_SET"
    --schema "$SCHEMA"
    --device-type "$DEVICE_TYPE"
    --num-search "$NUM_SEARCH"
    --top-k "$TOP_K"
    --target-tokens "$TARGET_TOKENS"
    --block-size "$BLOCK_SIZE"
    --model-variant "$MODEL_VARIANT"
    --search-mode "$SEARCH_MODE"
)

[[ -n "$VAL_PATH" ]] && ARGS+=(--val-path "$VAL_PATH")
[[ -n "$OUTPUT" ]] && ARGS+=(--output "$OUTPUT")

python3 "$QUADMIX_DIR/scripts/runners/reval_with_new_valset.py" "${ARGS[@]}"
