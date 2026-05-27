#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Demo: QuaDMix 8x NPU — 中等规模测试，适合 NPU 集群验证
# ──────────────────────────────────────────────────────────────
# 目标：100 shards (~7.9B tokens)，8 NPU 并行，验证多卡调度
# 数据不存在则自动下载 + 预处理
#
# Usage:
#   bash scripts/demo_run.sh
#
# 自定义 shard 数量：
#   NUM_SHARDS=50 bash scripts/demo_run.sh
#
# 自定义 NPU 设备数：
#   bash scripts/demo_run.sh --npu-devices 4
#
# HF 镜像加速（中国用户）：
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/demo_run.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

QUADMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"

PREPROCESSED_DIR="$QUADMIX_DIR/temp/preprocessed"
RAW_DATA_DIR="$QUADMIX_DIR/data/essential-web-v1"
VAL_FILE="$QUADMIX_DIR/data/openhermes_10k_assistant_tokenized.pt"

# ── 驗證集下載 ──────────────────────────────────
if [ ! -f "$VAL_FILE" ]; then
    echo "╔══ 驗證集就緒: 從 HuggingFace 下載 ═══╗"
    echo ""
    echo "  驗證集不存在: $VAL_FILE"
    echo "  從 liujin99/quadmix-openhermes-10k 下載..."
    
    HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
    if [ "$HF_ENDPOINT" != "https://huggingface.co" ]; then
        echo "  使用 HF 镜像: $HF_ENDPOINT"
    fi
    
    VAL_URL="$HF_ENDPOINT/datasets/liujin99/quadmix-openhermes-10k/resolve/main/openhermes_10k_assistant_tokenized.pt?download=true"
    
    mkdir -p "$(dirname "$VAL_FILE")"
    if command -v wget &>/dev/null; then
        wget -q --show-progress "$VAL_URL" -O "$VAL_FILE"
    else
        curl -L -o "$VAL_FILE" "$VAL_URL"
    fi
    echo "  ✓ 驗證集下載完成: $(du -h "$VAL_FILE" | cut -f1)"
    echo ""
    echo "╚══════════════════════════════════════╝"
    echo ""
fi

# ── 下载规模控制 ──────────────────────────────────
# 每 shard ≈ 79M tokens (char//4) / 246 MB 原始 parquet
# 100 shards ≈ 7.9B tokens / 24.6 GB
NUM_SHARDS="${NUM_SHARDS:-100}"
TOKEN_ESTIMATE=$(( NUM_SHARDS * 79000000 ))
TOKEN_ESTIMATE_B=$(echo "scale=1; $TOKEN_ESTIMATE / 1000000000" | bc 2>/dev/null || echo "~$(( TOKEN_ESTIMATE / 1000000000 ))")

# ── NPU 设备数 ──────────────────────────────────
NPU_DEVICES="${NPU_DEVICES:-8}"

echo "╔══ 配置摘要 ═══╗"
echo ""
echo "  Shards:       $NUM_SHARDS (~$TOKEN_ESTIMATE_B B tokens)"
echo "  NPU 设备:     $NPU_DEVICES 张卡并行"
echo "  预计下载:     ~$(( NUM_SHARDS * 246 / 1024 )) GB (如需)"
echo ""
echo "╚══════════════════════════════════════╝"
echo ""

# ── 数据就绪检查（逐个 shard 检查，补充下载）──────────────────
NEED_DOWNLOAD=0
MISSING_SHARDS=()

mkdir -p "$RAW_DATA_DIR"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    SHARD_FILE="$RAW_DATA_DIR/shard_$(printf '%05d' $i).parquet"
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
    python3 "$QUADMIX_DIR/scripts/download_essential_web.py" \
        --num-files "$NUM_SHARDS" --output-dir "$RAW_DATA_DIR" \
        --workers "${DOWNLOAD_WORKERS:-16}"
    echo "  ✓ 下载完成"
    echo ""
    echo "╚══════════════════════════════════════╝"
    echo ""
fi

# 检查是否需要预处理（shard 数量变化或首次运行）
if [ $NEED_DOWNLOAD -eq 1 ] || [ ! -f "$PREPROCESSED_DIR/shard_index.json" ]; then
    echo "╔══ 预处理数据 ═══╗"
    echo ""
    echo "  运行多 shard 预处理脚本..."
    python3 "$QUADMIX_DIR/scripts/preprocess_essential_web_v1_sharded.py" \
        --input-dir "$RAW_DATA_DIR" \
        --output-dir "$PREPROCESSED_DIR"
    echo "  ✓ 预处理完成 (multi-shard)"
    echo ""
    echo "╚══════════════════════════════════════╝"
    echo ""
fi

OUTPUT_DIR="${OUTPUT_DIR:-$QUADMIX_DIR/result/demo_8xnpu_$(date +%Y%m%d_%H%M%S)}"

echo "═══════════════════════════════════════════"
echo "  QuaDMix Demo — 8x NPU"
echo "  正式训练配置验证 (100 shards, 8 NPU)"
echo "═══════════════════════════════════════════"
cat << PARAMS

  Output:  $OUTPUT_DIR

  ┌────────────────────────────┬──────────────┐
  │ 参数                       │  值          │
  ├────────────────────────────┼──────────────┤
  │ Shards                     │        $NUM_SHARDS  │
  │ NPU 设备                   │         $NPU_DEVICES  │
  │ 实验数                     │          20  │
  │ 搜索点                     │       1,000  │
  │ Top-K 平均                 │           3  │
  │ seq_len (block_size)       │       2,048  │ ← 论文值
  │ 训练步数                   │      25,000  │ ← 论文值 (完整训练)
  │ 全局 batch size            │         512  │ ← 论文值
  │ 微批大小                   │           4  │ ← 论文值
  │ 验证集文档数               │       1,000  │ ← 论文值
  │ 排名参考集大小             │      10,000  │ ← 论文值
  │ 代理模型                   │  tinyllama_1M│
  └────────────────────────────┴──────────────┘

  ⏱ 预计耗时: ~10-20 分钟 (20 exp × ~3-5min/exp on NPU)
  ⚠ 注意: 使用完整训练配置 (25000 steps)，非 tiny 模式
PARAMS

# 检查 NPU 环境
if ! command -v npu-smi &> /dev/null; then
    echo ""
    echo "  ⚠ 未检测到 NPU 环境 (npu-smi 不存在)"
    echo "     将尝试使用 CUDA 或降级 CPU..."
    echo "     ⚠ CPU 模式下完整训练会极慢！建议使用 GPU/NPU"
    echo ""
    DEVICE_ARG=""
else
    echo ""
    echo "  ✓ NPU 已检测 (npu-smi 可用)"
    DEVICE_ARG="--device-type npu --npu-devices $NPU_DEVICES"
fi

python3 "$QUADMIX_DIR/scripts/run_essential_web_v1.py" \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --num-experiments 20 \
    --num-search 1000 \
    --top-k 3 \
    --block-size 2048 \
    --tiny-steps 0 \
    --micro-batch-size 4 \
    --global-batch-size 512 \
    --val-limit 1000 \
    --rank-ref-size 10000 \
    --output "$OUTPUT_DIR" \
    $DEVICE_ARG \
    "$@" || exit $?

echo ""
echo "═══════════════════════════════════════════"
echo "  Demo 8x NPU complete!"
echo "  Output: $OUTPUT_DIR/"
echo "    ├── quadmix_report.md"
echo "    ├── optimal_parameters.json"
echo "    ├── pipeline_summary.json"
echo "    ├── sampled_dataset.parquet"
echo "    └── proxy_experiments/"
echo "═══════════════════════════════════════════"