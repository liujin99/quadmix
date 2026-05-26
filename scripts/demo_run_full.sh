#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Demo: QuaDMix Full — 完整论文配置，验证算法正确性 & 有效性
# ──────────────────────────────────────────────────────────────
# 目标：完全按照 arXiv:2504.16511 的配置运行
# 数据不存在则自动下载 + 预处理
#
# 需要 GPU（CPU 不可行，每实验 ~5s → >4 小时）
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/demo_run_full.sh
#
# 8 张 NPU 并行（需要 NPU 环境 + --npu-devices 参数）：
#   bash scripts/demo_run_full.sh --device-type npu --npu-devices 8
#
# 如需限制文档数以便在 CPU 上试跑：
#   bash scripts/demo_run_full.sh --doc-limit 5000
#
# 如需控制下载数据量（默认为 2000 shards，~158B tokens）：
#   NUM_SHARDS=100 bash scripts/demo_run_full.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

QUADMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

PREPROCESSED_DIR="$QUADMIX_DIR/temp/preprocessed"
RAW_DATA_DIR="$QUADMIX_DIR/data/essential-web-v1"
VAL_FILE="$QUADMIX_DIR/data/openhermes_10k_assistant_tokenized.pt"

# ── 驗證集下載 ──────────────────────────────────
if [ ! -f "$VAL_FILE" ]; then
    echo "╔══ 驗證集就緒: 從 HuggingFace 下載 ═══╗"
    echo ""
    echo "  驗證集不存在: $VAL_FILE"
    echo "  從 liujin99/quadmix-openhermes-10k 下載..."
    mkdir -p "$(dirname "$VAL_FILE")"
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download liujin99/quadmix-openhermes-10k \
            openhermes_10k_assistant_tokenized.pt --local-dir "$(dirname "$VAL_FILE")" 2>/dev/null
    elif command -v wget &>/dev/null; then
        wget -q --show-progress \
            "https://huggingface.co/datasets/liujin99/quadmix-openhermes-10k/resolve/main/openhermes_10k_assistant_tokenized.pt?download=true" \
            -O "$VAL_FILE"
    else
        curl -L -o "$VAL_FILE" \
            "https://huggingface.co/datasets/liujin99/quadmix-openhermes-10k/resolve/main/openhermes_10k_assistant_tokenized.pt?download=true"
    fi
    echo "  ✓ 驗證集下載完成: $(du -h "$VAL_FILE" | cut -f1)"
    echo ""
    echo "╚══════════════════════════════════════╝"
    echo ""
fi

# ── 下载规模控制 ──────────────────────────────────
# 每 shard ≈ 79M tokens (char//4) / 246 MB 原始 parquet
# 完整 3291 shards ≈ 260B tokens / 791 GB
# 如果通过环境变量 $NUM_SHARDS 控制下载量：
NUM_SHARDS="${NUM_SHARDS:-2000}"
TOKEN_ESTIMATE=$(( NUM_SHARDS * 79000000 ))
TOKEN_ESTIMATE_B=$(echo "scale=1; $TOKEN_ESTIMATE / 1000000000" | bc 2>/dev/null || echo "~$(( TOKEN_ESTIMATE / 1000000000 ))")
echo "  [配置] 将使用 ~$NUM_SHARDS shards（~$TOKEN_ESTIMATE_B B tokens）"
echo "  [提示] 设 NUM_SHARDS=3291 用满全部数据，NUM_SHARDS=2 快速测试"

# ── 数据就绪检查 ──────────────────────────────────
if [ ! -f "$PREPROCESSED_DIR/shard_index.json" ]; then
    echo "╔══ 数据就绪: 下载 + 预处理 ═══╗"
    echo ""
    echo "  预处理数据不存在: $PREPROCESSED_DIR/"
    RAW_FILES=$(ls "$RAW_DATA_DIR"/*.parquet 2>/dev/null || true)
    if [ -z "$RAW_FILES" ]; then
        echo "  原始数据不存在，下载 $NUM_SHARDS 个 shard... ($((NUM_SHARDS * 246 / 1024)) GB 预计)"
        mkdir -p "$RAW_DATA_DIR"
        python3 "$QUADMIX_DIR/scripts/download_essential_web.py" \
            --num-files "$NUM_SHARDS" --output-dir "$RAW_DATA_DIR" \
            --workers "${DOWNLOAD_WORKERS:-4}"
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

OUTPUT_DIR="${OUTPUT_DIR:-$QUADMIX_DIR/result/demo_full_$(date +%Y%m%d_%H%M%S)}"

echo "═══════════════════════════════════════════"
echo "  QuaDMix Demo — Full"
echo "  完整论文配置 (arXiv:2504.16511)"
echo "═══════════════════════════════════════════"
cat << PARAMS

  Output:  $OUTPUT_DIR

  ┌────────────────────────────┬──────────────┐
  │ 参数                       │  论文值      │
  ├────────────────────────────┼──────────────┤
  │ 实验数                     │       3,000  │
  │ 搜索点                     │     100,000  │
  │ Top-K 平均                 │          10  │
  │ seq_len (block_size)       │       2,048  │
  │ 文档数                     │   全量 167K  │
  │ 训练步数                   │      25,000  │
  │ 全局 batch size            │         512  │
  │ 微批大小                   │           4  │
  │ 验证集文档数               │       1,000  │
  │ 排名参考集大小             │      10,000  │
  │ 代理模型                   │  tinyllama_1M│
  │                            │ (1.3M non-emb│
  │                            │  RMSNorm+    │
  │                            │  SwiGLU)    │
  └────────────────────────────┴──────────────┘

PARAMS

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Check for NPU (Ascend)
    if command -v npu-smi &> /dev/null; then
        DEVICE_ARG="--device-type npu"
        echo "  ✓ NPU 已启用 (Ascend)"
    else
        DEVICE_ARG=""
        echo "  ⚠ 当前为 CPU 模式，预计耗时 >4 小时！"
        echo "     建议: CUDA_VISIBLE_DEVICES=0 bash scripts/demo_run_full.sh"
        echo ""
        echo "     CPU 试跑: bash scripts/demo_run_full.sh --doc-limit 5000 \\"
        echo "                                    --block-size 256 \\"
        echo "                                    --tiny-steps 10"
        echo ""
    fi
else
    DEVICE_ARG="--device-type cuda"
    echo "  ✓ GPU 已启用 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
fi

python3 "$QUADMIX_DIR/scripts/run_essential_web_v1.py" \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --full \
    --block-size 2048 \
    --tiny-steps 0 \
    --micro-batch-size 4 \
    --global-batch-size 512 \
    --val-limit 1000 \
    --rank-ref-size 10000 \
    --top-k 10 \
    --output "$OUTPUT_DIR" \
    $DEVICE_ARG \
    "$@" || exit $?

echo ""
echo "═══════════════════════════════════════════"
echo "  Demo Full complete!"
echo "  Output: $OUTPUT_DIR/"
echo "    ├── quadmix_report.md"
echo "    ├── optimal_parameters.json"
echo "    ├── pipeline_summary.json"
echo "    ├── sampled_dataset.parquet"
echo "    └── proxy_experiments/"
echo "═══════════════════════════════════════════"
