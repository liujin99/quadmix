#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Demo: QuaDMix Quick — 快速验证整个流程能跑通
# ──────────────────────────────────────────────────────────────
# 目标：参数最小化，计算/时间开销最少，仅验证端到端流程
# 数据不存在则自动下载 + 预处理
#
# Usage:
#   bash scripts/demo_run_cpu.sh
#
# 缓存控制：
#   CLEAN_PREPROCESSED=1 — 强制清理预处理目录后重新预处理
#   CLEAN_PREPROCESSED=0 — 强制不清理（默认 auto：旧 shard 多于当前时自动清理）
#   CLEAN_TOKEN_CACHE=0  — 不清理 token cache（默认清理）
#
# ──────────────────────────────────────────────────────────────

set -euo pipefail

# ── 使用 conda nano 环境（包含 pyarrow 等依赖）─────────────
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate nano
fi



QUADMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${QUADMIX_DIR}/src:${PYTHONPATH:-}"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=""

# Temp/cache dir: override via QUADMIX_TEMP_DIR env var, defaults to ~/.cache/QuaDMix/temp/
export QUADMIX_TEMP_DIR="${QUADMIX_TEMP_DIR:-$HOME/.cache/QuaDMix/temp}"

PREPROCESSED_DIR="$QUADMIX_TEMP_DIR/preprocessed"
RAW_DATA_DIR="${RAW_DATA_DIR:-$HOME/.cache/QuaDMix/data}"
VAL_FILE="$QUADMIX_DIR/data/core_bmk_10tasks_v3_tokenized.pt"

# ── 驗證集下載（帶版本檢查）──────────────────────────────────
source "$QUADMIX_DIR/scripts/ensure_val_data.sh"
ensure_val_data "liujin99/quadmix-core-bmk-v3" "core_bmk_10tasks_v3_tokenized.pt" "$VAL_FILE"

# ── 数据就绪检查（逐个 shard 检查，补充下载）──────────────────
NUM_SHARDS=2  # quick demo 只需要 2 个 shard
NEED_DOWNLOAD=0
MISSING_SHARDS=()

mkdir -p "$RAW_DATA_DIR"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    IDX=$(printf '%05d' $i)
    SHARD_FILE="$RAW_DATA_DIR/train-${IDX}-of-03291.parquet"
    if [ ! -f "$SHARD_FILE" ]; then
        MISSING_SHARDS+=($i)
        NEED_DOWNLOAD=1
    fi
done

if [ $NEED_DOWNLOAD -eq 1 ]; then
    echo "╔══ 数据就绪: 下载缺失 shard ═══╗"
    echo ""
    EXISTING=$(ls "$RAW_DATA_DIR"/*.parquet 2>/dev/null | wc -l || echo 0)
    echo "  已有 shard: $EXISTING, 需要: $NUM_SHARDS"
    echo "  缺失 ${#MISSING_SHARDS[@]} 个 shard，补充下载..."
    python3 "$QUADMIX_DIR/scripts/preprocess/download_essential_web.py" \
        --num-files "$NUM_SHARDS" --output-dir "$RAW_DATA_DIR" \
        --workers "${DOWNLOAD_WORKERS:-16}"
    echo "  ✓ 下载完成"
    echo ""
    echo "╚══════════════════════════════════════╝"
    echo ""
fi

# ── 缓存清理 + 预处理 ──────────────────────────────────────────────
# CLEAN_PREPROCESSED: 强制清理 preprocessed/ 目录（默认自动检测）
#   - 旧 shard 数量 > 当前 NUM_SHARDS 时自动清理（防止旧文件残留）
#   - 设为 1 强制清理，设为 0 强制不清
# CLEAN_TOKEN_CACHE:  清理 token_cache/ 目录（默认清理）
CLEAN_PREPROCESSED="${CLEAN_PREPROCESSED:-auto}"
CLEAN_TOKEN_CACHE="${CLEAN_TOKEN_CACHE:-1}"
RUN_PREPROCESS=0

TOKEN_CACHE_DIR="$QUADMIX_TEMP_DIR/token_cache"

# 自动检测 shard 数量是否变化（减少时需要清理旧文件）
NEED_CLEAN_PREPROCESSED=0
if [ "$CLEAN_PREPROCESSED" = "1" ]; then
    NEED_CLEAN_PREPROCESSED=1
elif [ "$CLEAN_PREPROCESSED" = "0" ]; then
    NEED_CLEAN_PREPROCESSED=0
    elif [ -f "$PREPROCESSED_DIR/shard_index.json" ]; then
    OLD_SHARDS=$(python3 -c "import json; print(json.load(open('$PREPROCESSED_DIR/shard_index.json'))['num_shards'])" 2>/dev/null || echo 0)
    if [ "$OLD_SHARDS" -gt "$NUM_SHARDS" ]; then
        NEED_CLEAN_PREPROCESSED=1
        echo ""
        echo "  [警告] 旧预处理有 $OLD_SHARDS 个 shard，当前只需 $NUM_SHARDS 个"
        echo "         将自动清理旧预处理目录防止残留文件干扰"
    elif [ "$OLD_SHARDS" -eq 0 ] && [ -d "$PREPROCESSED_DIR" ]; then
        RESIDUAL=$(ls "$PREPROCESSED_DIR"/preprocessed_*.parquet 2>/dev/null | wc -l || echo 0)
        if [ "$RESIDUAL" -gt 0 ]; then
            NEED_CLEAN_PREPROCESSED=1
            echo ""
            echo "  [警告] shard_index.json 损坏（shards=0），但有 $RESIDUAL 个残留文件"
            echo "         将自动清理旧预处理目录"
        fi
    fi
fi

# 是否需要预处理：清理预处理目录 或 索引不存在 或 shard 数量不一致
if [ $NEED_CLEAN_PREPROCESSED -eq 1 ]; then
    RUN_PREPROCESS=1
elif [ ! -f "$PREPROCESSED_DIR/shard_index.json" ]; then
    RUN_PREPROCESS=1
elif [ -f "$PREPROCESSED_DIR/shard_index.json" ]; then
    OLD_SHARDS=$(python3 -c "import json; print(json.load(open('$PREPROCESSED_DIR/shard_index.json'))['num_shards'])" 2>/dev/null || echo 0)
    RAW_SHARDS=$(ls "$RAW_DATA_DIR"/*.parquet 2>/dev/null | wc -l || echo 0)
    if [ "$OLD_SHARDS" -ne "$RAW_SHARDS" ]; then
        RUN_PREPROCESS=1
        echo ""
        echo "  [信息] 预处理有 $OLD_SHARDS 个 shard，raw data 有 $RAW_SHARDS 个"
        echo "         数量不一致，将重新预处理"
    fi
fi

if [ $RUN_PREPROCESS -eq 1 ]; then
    echo ""
    echo "╔══ 预处理数据 ═══╗"
    echo ""
    if [ $NEED_CLEAN_PREPROCESSED -eq 1 ] && [ -d "$PREPROCESSED_DIR" ]; then
        echo "  [清理] 预处理目录: $PREPROCESSED_DIR"
        rm -rf "$PREPROCESSED_DIR"
    fi
    if [ "$CLEAN_TOKEN_CACHE" = "1" ] && [ -d "$TOKEN_CACHE_DIR" ]; then
        echo "  [清理] token cache 目录: $TOKEN_CACHE_DIR"
        rm -f "$TOKEN_CACHE_DIR"/*.pt 2>/dev/null || true
    rm -rf "$TOKEN_CACHE_DIR" 2>/dev/null || true
    fi
    mkdir -p "$PREPROCESSED_DIR"
    echo "  运行多 shard 预处理脚本..."
    python3 "$QUADMIX_DIR/scripts/preprocess/preprocess_essential_web_v1_sharded.py" \
        --input-dir "$RAW_DATA_DIR" \
        --output-dir "$PREPROCESSED_DIR"
    echo "  ✓ 预处理完成 (multi-shard)"
    echo ""
    echo "╚══════════════════════════════════════╝"
    echo ""
elif [ "$CLEAN_TOKEN_CACHE" = "1" ] && [ -d "$TOKEN_CACHE_DIR" ]; then
    echo "  [清理] token cache 目录: $TOKEN_CACHE_DIR"
    rm -f "$TOKEN_CACHE_DIR"/*.pt 2>/dev/null || true
    rm -rf "$TOKEN_CACHE_DIR" 2>/dev/null || true
fi

OUTPUT_DIR="${OUTPUT_DIR:-$QUADMIX_DIR/result/demo_quick_$(date +%Y%m%d_%H%M%S)}"

echo "═══════════════════════════════════════════"
echo "  QuaDMix Demo — Quick"
echo "  验证流程跑通（最小计算/时间开销）"
echo "═══════════════════════════════════════════"
cat << PARAMS

  Output:  $OUTPUT_DIR

  ┌────────────────────────────┬──────────┐
  │ 参数                       │  值      │
  ├────────────────────────────┼──────────┤
  │ 实验数                     │      20  │
  │ 搜索点                     │     200  │
  │ Top-K 平均                 │       2  │
  │ seq_len (block_size)       │      64  │
  │ 训练步数 (tiny_steps)      │       3  │
  │ 全局 batch size            │       8  │
  │ 微批大小                   │       2  │ (ga=4)
  │ 验证集                     │ CORE BMK v3│
  │ 排名参考集大小             │     200  │
  └────────────────────────────┴──────────┘

  ⚡ 预计耗时: ~1-2分钟
PARAMS

python3 "$QUADMIX_DIR/scripts/runners/run_essential_web_v1.py" \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --num-experiments 20 \
    --num-search 200 \
    --top-k 2 \
    --block-size 64 \
    --tiny-steps 3 \
    --micro-batch-size 2 \
    --global-batch-size 8 \
    --rank-ref-size 200 \
    --checkpoint-interval 0 \
    --val-set core_bmk_v3 \
    --output "$OUTPUT_DIR" \
    "$@" || exit $?

echo ""
echo "═══════════════════════════════════════════"
echo "  Demo Quick complete!"
echo "  Output: $OUTPUT_DIR/"
echo "    ├── quadmix_report.md"
echo "    ├── optimal_parameters.json"
echo "    ├── pipeline_summary.json"
echo "    ├── sampled_dataset.parquet"
echo "    └── proxy_experiments/"
echo "═══════════════════════════════════════════"