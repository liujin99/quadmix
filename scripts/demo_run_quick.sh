#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Demo: QuaDMix Quick — 快速验证整个流程能跑通
# ──────────────────────────────────────────────────────────────
# 目标：参数最小化，计算/时间开销最少，仅验证端到端流程
# 数据不存在则自动下载 + 预处理
#
# Usage:
#   bash scripts/demo_run_quick.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

QUADMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=""

PREPROCESSED_DIR="$QUADMIX_DIR/temp/preprocessed"
RAW_DATA_DIR="/home/liujin99/data/essential-web-v1"

# ── 数据就绪检查 ──────────────────────────────────
if [ ! -f "$PREPROCESSED_DIR/shard_index.json" ]; then
    echo "╔══ 数据就绪: 下载 + 预处理 ═══╗"
    echo ""
    echo "  预处理数据不存在: $PREPROCESSED_DIR/"
    RAW_FILES=$(ls "$RAW_DATA_DIR"/*.parquet 2>/dev/null || true)
    if [ -z "$RAW_FILES" ]; then
        echo "  原始数据不存在，下载 2 个 shard..."
        mkdir -p "$RAW_DATA_DIR"
        python3 "$QUADMIX_DIR/scripts/download_essential_web.py" \
            --num-files 2 --output-dir "$RAW_DATA_DIR"
        echo "  ✓ 下载完成"
    else
        echo "  原始数据已存在，跳过下载"
    fi
    echo "  运行多 shard 预处理脚本..."
    python3 "$QUADMIX_DIR/scripts/preprocess_essential_web_v1_sharded.py" \
        --input-dir "$RAW_DATA_DIR" \
        --output-dir "$PREPROCESSED_DIR"
    echo "  ✓ 预处理完成 (multi-shard)"
    echo ""
    echo "╚══════════════════════════════════════╝"
    echo ""
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
  │ 实验数                     │       2  │
  │ 搜索点                     │     200  │
  │ Top-K 平均                 │       2  │
  │ seq_len (block_size)       │      64  │
  │ 文档数 (doc_limit)         │     500  │
  │ 训练步数 (tiny_steps)      │       3  │
  │ 全局 batch size            │       8  │
  │ 微批大小                   │       2  │
  │ 验证集文档数               │      50  │
  │ 排名参考集大小             │     200  │
  └────────────────────────────┴──────────┘

  ⚡ 预计耗时: ~15秒
PARAMS

python3 "$QUADMIX_DIR/scripts/run_essential_web_v1.py" \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --num-experiments 2 \
    --num-search 200 \
    --top-k 2 \
    --doc-limit 500 \
    --block-size 64 \
    --tiny-steps 3 \
    --micro-batch-size 2 \
    --global-batch-size 8 \
    --val-limit 50 \
    --rank-ref-size 200 \
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
