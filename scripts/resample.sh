#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Resample: 用已有最优参数对扩容后的数据池重新采样
# ──────────────────────────────────────────────────────────────
# 场景：用户已跑完 QuaDMix pipeline 得到 optimal_parameters.json，
#       之后数据池扩容（同分布），用同一组 θ* 对新数据重新采样。
#
# Usage:
#   DATA_DIR=/path/to/essential-web-v1 \
#   PARAMS_FILE=/path/to/optimal_parameters.json \
#   bash scripts/resample.sh
#
# 可选参数：
#   TARGET_TOKENS=5.0       — 限制输出 token 数（单位：billions，默认不限制）
#   SEED=42                 — 随机种子（默认 42）
#   OUTPUT=/path/to/output  — 输出目录（默认 result/resample_<timestamp>）
#   FORCE_PREPROCESS=1      — 强制重新预处理（默认 0，增量预处理）
#   WORKERS=64              — 预处理并行 workers（默认 64）
#   PREPROCESSED_DIR=/path  — 自定义预处理缓存目录（默认 ~/.cache/QuaDMix/resample/preprocessed）
# ──────────────────────────────────────────────────────────────

set -euo pipefail

if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate nano
fi

QUADMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${QUADMIX_DIR}/src:${PYTHONPATH:-}"
export PATH="$HOME/.local/bin:$PATH"

# ── 参数校验 ──────────────────────────────────────────────────
if [ -z "${DATA_DIR:-}" ]; then
    echo "Error: DATA_DIR is required"
    echo "Usage: DATA_DIR=/path/to/data PARAMS_FILE=/path/to/params.json bash scripts/resample.sh"
    exit 1
fi

if [ -z "${PARAMS_FILE:-}" ]; then
    echo "Error: PARAMS_FILE is required"
    echo "Usage: DATA_DIR=/path/to/data PARAMS_FILE=/path/to/params.json bash scripts/resample.sh"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: DATA_DIR does not exist: $DATA_DIR"
    exit 1
fi

if [ ! -f "$PARAMS_FILE" ]; then
    echo "Error: PARAMS_FILE does not exist: $PARAMS_FILE"
    exit 1
fi

# ── 参数默认值 ────────────────────────────────────────────────
TARGET_TOKENS="${TARGET_TOKENS:-0}"
SEED="${SEED:-42}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
WORKERS="${WORKERS:-64}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-$HOME/.cache/QuaDMix/resample/preprocessed}"

RAW_SHARD_COUNT=$(find "$DATA_DIR" -maxdepth 1 -name "*.parquet" 2>/dev/null | wc -l)

echo "╔══ 配置摘要 ═══╗"
echo ""
echo "  Data dir:       $DATA_DIR"
echo "  Params file:    $PARAMS_FILE"
echo "  Raw shards:     $RAW_SHARD_COUNT"
echo "  Cache dir:      $PREPROCESSED_DIR"
echo "  Target tokens:  ${TARGET_TOKENS}B (0=unlimited)"
echo "  Seed:           $SEED"
echo "  Workers:        $WORKERS"
echo "  Force preprocess: $FORCE_PREPROCESS"
echo ""
echo "╚══════════════════════════════════════╝"
echo ""

# ── 构建 Python 参数 ─────────────────────────────────────────
PYTHON_ARGS=(
    --data-dir "$DATA_DIR"
    --params-file "$PARAMS_FILE"
    --preprocessed-dir "$PREPROCESSED_DIR"
    --seed "$SEED"
    --workers "$WORKERS"
)

if [ "$TARGET_TOKENS" != "0" ]; then
    PYTHON_ARGS+=(--target-tokens "$TARGET_TOKENS")
fi

if [ -n "${OUTPUT:-}" ]; then
    PYTHON_ARGS+=(--output "$OUTPUT")
fi

if [ "$FORCE_PREPROCESS" = "1" ]; then
    PYTHON_ARGS+=(--force)
fi

# ── 执行 ─────────────────────────────────────────────────────
echo "═══════════════════════════════════════════"
echo "  QuaDMix Resample"
echo "═══════════════════════════════════════════"
echo ""

python3 "$QUADMIX_DIR/scripts/resample_with_optimal_params.py" \
    "${PYTHON_ARGS[@]}" || exit $?

echo ""
echo "═══════════════════════════════════════════"
echo "  Resample done!"
echo "═══════════════════════════════════════════"
