#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Demo: QuaDMix Re-optimize — 用已有 loss 重新优化
# ──────────────────────────────────────────────────────────────
# 从 Stage 5 开始：加载已计算好的 proxy loss → 拟合 LightGBM →
# 搜索最优参数 → 抽样 → 生成报告。
#
# 适用场景：
#   - 调整 LightGBM 参数或搜索策略
#   - 刚跑完 demo_revalidate.sh，想换搜索参数重跑
#   - loss 数据已有，只想重做优化部分
#
# 如果需要换新验证集重新算 loss，请用：
#   bash scripts/demo_revalidate.sh
#
# Usage:
#   bash scripts/demo_reoptimize.sh --source-dir result/quadmix_20260609_120000
#
# 自定义输出目录：
#   bash scripts/demo_reoptimize.sh --source-dir result/xxx --output result/my_reoptimize
#
# 调整搜索参数：
#   bash scripts/demo_reoptimize.sh --source-dir result/xxx --num-search 50000 --top-k 5
#
# 指定目标数据量（单位 B tokens）：
#   bash scripts/demo_reoptimize.sh --source-dir result/xxx --target-tokens 10
#
# 切换搜索模式（等权 vs R²加权）：
#   bash scripts/demo_reoptimize.sh --search-mode equal_weight
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
SCHEMA=""
SCHEMA_SET=0
OUTPUT=""
NUM_SEARCH="100000"
TOP_K="10"
TARGET_TOKENS="0"
SEARCH_MODE="r2_weighted"
PREPROCESSED_DIR_SET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-dir)    SOURCE_DIR="$2"; shift 2 ;;
        --schema)        SCHEMA="$2"; SCHEMA_SET=1; shift 2 ;;
        --output|-o)     OUTPUT="$2"; shift 2 ;;
        --num-search)    NUM_SEARCH="$2"; shift 2 ;;
        --top-k)         TOP_K="$2"; shift 2 ;;
        --target-tokens) TARGET_TOKENS="$2"; shift 2 ;;
        --preprocessed-dir) PREPROCESSED_DIR="$2"; PREPROCESSED_DIR_SET=1; shift 2 ;;
        --search-mode)     SEARCH_MODE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/demo_reoptimize.sh --source-dir <path> [options]"
            echo ""
            echo "Required:"
            echo "  --source-dir PATH        Original pipeline result directory"
            echo ""
            echo "Options:"
            echo "  --schema PATH            Dataset schema YAML (default: auto from pipeline_summary.json)"
            echo "  --output PATH            Output directory (default: auto)"
            echo "  --num-search N           Search points (default: 100000)"
            echo "  --top-k N                Top-K average (default: 10)"
            echo "  --target-tokens N        Target in billions (default: 0)"
            echo "  --preprocessed-dir PATH  Preprocessed shards dir"
            echo "  --search-mode MODE       r2_weighted (default), equal_weight, or r2_sigma_weighted"
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
    echo "Usage: bash scripts/demo_reoptimize.sh --source-dir result/quadmix_20260609_120000"
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

if [[ -z "$SCHEMA" ]]; then
    echo "[Error] Schema could not be auto-detected"
    echo "  Specify with --schema /path/to/schema.yaml"
    echo "  e.g. --schema configs/schema_stem.yaml"
    exit 1
fi

PROXY_DIR="$SOURCE_DIR/proxy_experiments"
if [[ ! -d "$PROXY_DIR" ]]; then
    echo "[Error] proxy_experiments not found in: $SOURCE_DIR"
    exit 1
fi

if [[ ! -d "$PREPROCESSED_DIR" ]]; then
    echo "[Error] Preprocessed dir not found: $PREPROCESSED_DIR"
    echo "  Set --preprocessed-dir or run a pipeline first"
    exit 1
fi

EXP_COUNT=$(find "$PROXY_DIR" -name "meta.json" 2>/dev/null | wc -l)
if [[ "$EXP_COUNT" -eq 0 ]]; then
    echo "[Error] No meta.json found in $PROXY_DIR/"
    echo "  Run demo_revalidate.sh first to compute losses"
    exit 1
fi

echo "╔══ QuaDMix Re-optimize (from Stage 5) ══╗"
echo ""
echo "  Source:        $SOURCE_DIR"
echo "  Experiments:   $EXP_COUNT"
echo "  Schema:        $SCHEMA"
echo "  Preprocessed:  $PREPROCESSED_DIR"
echo "  Search points: $NUM_SEARCH"
echo "  Top-K:         $TOP_K"
echo "  Search mode:   $SEARCH_MODE"
[[ "$TARGET_TOKENS" != "0" ]] && echo "  Target tokens: ${TARGET_TOKENS}B"
echo ""
echo "╚═════════════════════════════════════════╝"
echo ""

ARGS=(
    --proxy-dir "$PROXY_DIR"
    --preprocessed-dir "$PREPROCESSED_DIR"
    --schema "$SCHEMA"
    --num-search "$NUM_SEARCH"
    --top-k "$TOP_K"
    --target-tokens "$TARGET_TOKENS"
    --search-mode "$SEARCH_MODE"
)

[[ -n "$OUTPUT" ]] && ARGS+=(--output "$OUTPUT")

python3 "$QUADMIX_DIR/scripts/runners/resume_from_stage5.py" "${ARGS[@]}"
