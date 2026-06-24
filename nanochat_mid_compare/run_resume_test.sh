#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Resume Test: Isolate whether crash is pure-data or model-state+data
#
# Phase 1: Run normal mid_train.py from step 0 → RESUME_STEP (save checkpoint)
# Phase 2: Resume from checkpoint, replay captured batches through crash step
#
# If Phase 2 crashes → crash is reproducible with correct model state
# If Phase 2 doesn't crash → something about the full 0→330 trajectory differs
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NANOCHAT_REPO="${NANOCHAT_REPO:-/home/ma-user/work/nanochat_midtrain_326}"
NANOCHAT_MODEL_DIR="${NANOCHAT_MODEL_DIR:-/home/ma-user/work/nanochat_model_dir}"
NUM_NPU=8

# ══════ CONFIGURATION ══════
DATA_DIR="${DATA_DIR:-$NANOCHAT_MODEL_DIR/mid_train_data}"
BATCH_DIR="${BATCH_DIR:-$NANOCHAT_MODEL_DIR/mid_checkpoints/replay_batches}"
RESUME_STEP="${RESUME_STEP:-320}"
END_STEP="${END_STEP:-330}"
MODEL_TAG="${MODEL_TAG:-fineweb_edu_resume}"
BASE_MODEL_TAG="${BASE_MODEL_TAG:-d24_0320}"
BASE_MODEL_STEP="${BASE_MODEL_STEP:-6612}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
NUM_ITERATIONS="${NUM_ITERATIONS:-627}"

# ══════ NPU ENVIRONMENT ══════
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export NANOCHAT_BASE_DIR="$NANOCHAT_MODEL_DIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_HCCL_PATH=/usr/local/Ascend/ascend-toolkit/latest/hccl
export LD_LIBRARY_PATH=${ASCEND_HCCL_PATH}/lib64:${LD_LIBRARY_PATH:-}
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_WHITELIST_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

export PYTORCH_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_LOG_LEVEL=3

ASCEND_DEVICE_LIST=$(seq -s, 0 $((NUM_NPU - 1)))
export ASCEND_VISIBLE_DEVICES="$ASCEND_DEVICE_LIST"
export RANK_SIZE=$NUM_NPU
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export HCCL_EXEC_TIMEOUT=1200
export ASCEND_DISABLE_MEM_SWAP=1
export ASCEND_LAUNCH_BLOCKING=0
export NPU_DISABLE_RECORD=1
export PYTHONUNBUFFERED=1
export ASCEND_COMPILE_OPT_LEVEL=O3
export TORCH_NPU_LAZY_COMPILE=1
export PYTHONPRELOAD=torch_npu
export TORCH_NPU_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,memory_pool:True"
export PYTORCH_NPU_ALLOC_MAX_SIZE=60G
export ASCEND_ENABLE_CACHE=1
export ASCEND_CACHE_POLICY=2
export ASCEND_FUSION_ENABLE=1
export ASCEND_GEMM_DTiling=1
export TORCH_NPU_ENABLE_NUMA=1
export ASCEND_MEMORY_COPY_MODE=1
export ASCEND_HBM_ALLOC_TYPE=1
export ASCEND_OPP_LEVEL=O3
export ASCEND_FUSION_PASS_ENABLE=1
export ASCEND_GEMM_BTiling=1
export ASCEND_GEMM_ATiling=1
export ASCEND_CONV_ALGO_SELECTION=1
export ASCEND_ENABLE_TRANSFORMER_FUSION=1
export ASCEND_MEMORY_REUSE_MODE=2
export ASCEND_ENABLE_PREFETCH=1
export ASCEND_NPU_ENABLE_UNIFIED_MEMORY=1
export ASCEND_OPTIMIZER_AGGRESSIVE_MODE=1
export ASCEND_SYNCHRONIZATION_MODE=0
export PYTORCH_NPU_ENABLE_LARGE_CONCAT=1
export PYTORCH_NPU_ENABLE_TORCHscript=1
export NPU_PERF_MODE=high_performance
export NANOCHAT_DTYPE=bfloat16

export PYTHONPATH="$SCRIPT_DIR:$NANOCHAT_REPO:${PYTHONPATH:-}"

# Create symlink for base model if needed
LINK_DIR="$NANOCHAT_MODEL_DIR/base_checkpoints/$MODEL_TAG"
BASE_CKPT_DIR="$NANOCHAT_MODEL_DIR/base_checkpoints/$BASE_MODEL_TAG"
if [ ! -e "$LINK_DIR" ]; then
    echo "Creating symlink: $LINK_DIR -> $BASE_CKPT_DIR"
    ln -s "$BASE_CKPT_DIR" "$LINK_DIR"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Resume Test: step 0 → $RESUME_STEP → $END_STEP"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Data:           $DATA_DIR"
echo "  Replay batches: $BATCH_DIR"
echo "  Model tag:      $MODEL_TAG"
echo "  Base model:     $BASE_MODEL_TAG (step $BASE_MODEL_STEP)"
echo "  Total batch:    $TOTAL_BATCH_SIZE"
echo "  Num iterations: $NUM_ITERATIONS"
echo ""

# ══════════════════════════════════════════════════════════════
#  PHASE 1: Normal training from step 0 → RESUME_STEP
# ══════════════════════════════════════════════════════════════

echo "╔══ Phase 1: Normal training step 0 → $RESUME_STEP ══╗"
echo ""

pushd "$NANOCHAT_REPO" > /dev/null
python3 -m torch.distributed.run --standalone --nproc_per_node="$NUM_NPU" -m scripts.mid_train -- \
    --run="resume_phase1" \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --num-iterations="$RESUME_STEP" \
    --save-every="$RESUME_STEP" \
    --core-metric-every=-1 \
    --eval-every=-1 \
    --sample-every=-1 \
    --data-dir="$DATA_DIR"
popd > /dev/null

echo ""
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════
#  VERIFY CHECKPOINT
# ══════════════════════════════════════════════════════════════

CKPT_DIR="$NANOCHAT_MODEL_DIR/mid_checkpoints/$MODEL_TAG"
if [ ! -f "$CKPT_DIR/model_$(printf '%06d' $RESUME_STEP).pt" ]; then
    echo "ERROR: Phase 1 checkpoint not found at $CKPT_DIR/model_$(printf '%06d' $RESUME_STEP).pt"
    exit 1
fi
echo "Phase 1 checkpoint verified: step $RESUME_STEP"
echo ""

# ══════════════════════════════════════════════════════════════
#  PHASE 2: Resume from checkpoint, replay batches
# ══════════════════════════════════════════════════════════════

echo "╔══ Phase 2: Resume from step $RESUME_STEP, replay through step $END_STEP ══╗"
echo ""

export REPLAY_DIR="$BATCH_DIR"
export REPLAY_START="$RESUME_STEP"
export REPLAY_END="$END_STEP"

echo "Generating replay_mid_train.py (with --resume-step=$RESUME_STEP)..."
python3 "$SCRIPT_DIR/generate_replay_script.py" "$NANOCHAT_REPO" --resume-step="$RESUME_STEP"

pushd "$NANOCHAT_REPO" > /dev/null
python3 -m torch.distributed.run --standalone --nproc_per_node="$NUM_NPU" -m scripts.replay_mid_train -- \
    --run="resume_phase2" \
    --model-tag="$MODEL_TAG" \
    --model-step="$RESUME_STEP" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --num-iterations="$NUM_ITERATIONS" \
    --core-metric-every=-1 \
    --eval-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --data-dir="$BATCH_DIR"
popd > /dev/null

echo ""
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Resume test completed successfully (no crash)!"
