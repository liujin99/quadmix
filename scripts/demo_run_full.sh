#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Demo: QuaDMix Full — 中等规模验证，适合 NPU 集群
# ──────────────────────────────────────────────────────────────
# 目标：20 shards (~1.6B tokens)，50 实验，5000 步
# 数据不存在则自动下载 + 预处理
#
# 需要 GPU/NPU（CPU 不可行）
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/demo_run_full.sh
#
# 8 张 NPU 并行：
#   bash scripts/demo_run_full.sh --device-type npu --npu-devices 8
#
# 自定义规模：
#   NUM_SHARDS=200 bash scripts/demo_run_full.sh
#   NUM_EXPERIMENTS=100 bash scripts/demo_run_full.sh
#
# 最终输出数据集大小（target_tokens，单位 B）：
#   bash scripts/demo_run_full.sh --target-tokens 10  # 输出 ~10B tokens
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

# ── 锁定 Python 3.11（pyarrow 兼容 Ascend NPU）─────────────
if [ -x "/usr/local/python3.11.13/bin/python3" ]; then
    export PATH="/usr/local/python3.11.13/bin:$PATH"
fi

QUADMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

# NPU allocator: prevent memory fragmentation by blocking splits of large blocks
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:128}"

# Temp/cache dir: override via QUADMIX_TEMP_DIR env var, defaults to ~/.cache/QuaDMix/temp/
export QUADMIX_TEMP_DIR="${QUADMIX_TEMP_DIR:-$HOME/.cache/QuaDMix/temp}"

PREPROCESSED_DIR="$QUADMIX_TEMP_DIR/preprocessed"
RAW_DATA_DIR="$QUADMIX_DIR/data/essential-web-v1"
VAL_FILE="$QUADMIX_DIR/data/openhermes_10k_assistant_tokenized.pt"

# ── 驗證集下載 ──────────────────────────────────
if [ ! -f "$VAL_FILE" ]; then
    echo "╔══ 驗證集就緒: 從 HuggingFace 下載 ═══╗"
    echo ""
    echo "  驗證集不存在: $VAL_FILE"
    echo "  從 liujin99/quadmix-openhermes-10k 下載..."

    # Support HF mirror via environment variable
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
# 完整 3291 shards ≈ 260B tokens / 791 GB
# 如果通过环境变量 $NUM_SHARDS 控制下载量：
NUM_SHARDS="${NUM_SHARDS:-100}"
NUM_EXPERIMENTS="${NUM_EXPERIMENTS:-96}"
TOKEN_ESTIMATE=$(( NUM_SHARDS * 79000000 ))
TOKEN_ESTIMATE_B=$(echo "scale=1; $TOKEN_ESTIMATE / 1000000000" | bc 2>/dev/null || echo "~$(( TOKEN_ESTIMATE / 1000000000 ))")
echo "  [配置] 将使用 ~$NUM_SHARDS shards（~$TOKEN_ESTIMATE_B B tokens）"
echo "  [提示] 设 NUM_SHARDS=3291 用满全部数据，NUM_SHARDS=2 快速测试"

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
        # shard_index.json 存在但解析失败或为 0，目录有残留文件
        RESIDUAL=$(ls "$PREPROCESSED_DIR"/preprocessed_*.parquet 2>/dev/null | wc -l || echo 0)
        if [ "$RESIDUAL" -gt 0 ]; then
            NEED_CLEAN_PREPROCESSED=1
            echo ""
            echo "  [警告] shard_index.json 损坏（shards=0），但有 $RESIDUAL 个残留文件"
            echo "         将自动清理旧预处理目录"
        fi
    fi
fi

# 是否需要预处理：清理预处理目录 或 索引不存在
if [ $NEED_CLEAN_PREPROCESSED -eq 1 ]; then
    RUN_PREPROCESS=1
elif [ ! -f "$PREPROCESSED_DIR/shard_index.json" ]; then
    RUN_PREPROCESS=1
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
    python3 "$QUADMIX_DIR/scripts/preprocess_essential_web_v1_sharded.py" \
        --input-dir "$RAW_DATA_DIR" \
        --output-dir "$PREPROCESSED_DIR"
    echo "  ✓ 预处理完成 (multi-shard)"
    echo ""
    echo "╚══════════════════════════════════════╝"
    echo ""
elif [ "$CLEAN_TOKEN_CACHE" = "1" ] && [ -d "$TOKEN_CACHE_DIR" ]; then
    echo "  [清理] token cache 目录: $TOKEN_CACHE_DIR"
    rm -rf "$TOKEN_CACHE_DIR"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$QUADMIX_DIR/result/demo_full_$(date +%Y%m%d_%H%M%S)}"

echo "═══════════════════════════════════════════"
echo "  QuaDMix Demo — Full (中等规模)"
echo "  $NUM_SHARDS shards, $NUM_EXPERIMENTS 实验, 5000 步"
echo "═══════════════════════════════════════════"
cat << PARAMS

  Output:  $OUTPUT_DIR

  ┌────────────────────────────┬──────────────┐
  │ 参数                       │  值          │
  ├────────────────────────────┼──────────────┤
  │ Shards                     │        $NUM_SHARDS  │
  │ 实验数                     │ $NUM_EXPERIMENTS  │
  │ 搜索点                     │       5,000  │
  │ Top-K 平均                 │           5  │
  │ seq_len (block_size)       │       2,048  │
  │ 训练步数                   │       5,000  │
  │ 全局 batch size            │          64  │
  │ 微批大小                   │          64  │ (ga=1)
  │ warmup                     │         4%   │
  │ 验证集                     │   全量 10k   │
  │ 排名参考集大小             │      10,000  │
  │ 代理模型                   │  tinyllama_1M│
  └────────────────────────────┴──────────────┘

  ⏱ 预计耗时: ~$(( NUM_EXPERIMENTS / 8 + 1 ))h ($NUM_EXPERIMENTS exp, 8 NPU 并行)
PARAMS

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Check for NPU (Ascend)
    if command -v npu-smi &> /dev/null; then
        NPU_DEVICES="${NPU_DEVICES:-8}"
        DEVICE_ARG="--device-type npu --npu-devices $NPU_DEVICES"
        echo "  ✓ NPU 已启用 (Ascend, $NPU_DEVICES 卡)"
    else
        DEVICE_ARG=""
        echo "  ⚠ 当前为 CPU 模式，预计耗时 >4 小时！"
        echo "     建议: CUDA_VISIBLE_DEVICES=0 bash scripts/demo_run_full.sh"
        echo ""
        echo "     CPU 试跑: bash scripts/demo_run_cpu.sh  # 快速验证流程"
        echo ""
    fi
else
    DEVICE_ARG="--device-type cuda"
    echo "  ✓ GPU 已启用 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
fi

# Tokenize parallelism: workers × threads_per_worker ≈ num_cpus
export TOKENIZE_WORKERS="${TOKENIZE_WORKERS:-48}"
export TOKENIZE_THREADS_PER_WORKER="${TOKENIZE_THREADS_PER_WORKER:-4}"

# Performance timer: set to 1 to enable detailed timing report
export QUADMIX_PERF_TIMER="${QUADMIX_PERF_TIMER:-1}"

python3 "$QUADMIX_DIR/scripts/run_essential_web_v1.py" \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --num-experiments "$NUM_EXPERIMENTS" \
    --num-search 5000 \
    --top-k 5 \
    --block-size 2048 \
    --tiny-steps 5000 \
    --micro-batch-size 32 \
    --global-batch-size 64 \
    --rank-ref-size 10000 \
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