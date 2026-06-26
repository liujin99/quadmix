#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# QuadMix Mid-Training — Eval Only
# ──────────────────────────────────────────────────────────────
#
# Runs CORE evaluation on an already mid-trained checkpoint.
#
# Usage:
#   bash nanochat_mid_compare/run_eval_only.sh \
#     --model-tag d24_0320_quadmix_20260625_193457 \
#     --result-dir nanochat_mid_compare/results/quadmix_only_20260625_193457
#
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ══════════════════════════════════════════════════════════════
#  CLI ARGUMENTS
# ══════════════════════════════════════════════════════════════

MODEL_TAG=""
RESULT_DIR=""
BASE_MODEL_TAG="${BASE_MODEL_TAG:-d24_0320}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-tag) MODEL_TAG="$2"; shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --base-model-tag) BASE_MODEL_TAG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_TAG" ]; then
    echo "ERROR: --model-tag is required."
    echo "  Usage: $0 --model-tag <mid_model_tag> --result-dir <output_dir>"
    exit 1
fi

if [ -z "$RESULT_DIR" ]; then
    RESULT_DIR="$SCRIPT_DIR/results/eval_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S)"
fi

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

NANOCHAT_MODEL_DIR="${NANOCHAT_MODEL_DIR:-/home/ma-user/work/nanochat_model_dir}"
NANOCHAT_REPO="${NANOCHAT_REPO:-/home/ma-user/work/nanochat-npu}"
NUM_NPU="${NUM_NPU:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"

# ══════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════

MID_CKPT_DIR="$NANOCHAT_MODEL_DIR/mid_checkpoints/$MODEL_TAG"
if [ ! -d "$MID_CKPT_DIR" ]; then
    echo "ERROR: Mid checkpoint not found: $MID_CKPT_DIR"
    exit 1
fi

if [ ! -d "$NANOCHAT_REPO" ]; then
    echo "ERROR: Nanochat repo not found: $NANOCHAT_REPO"
    exit 1
fi

BASE_CKPT_DIR="$NANOCHAT_MODEL_DIR/base_checkpoints/$BASE_MODEL_TAG"
if [ ! -d "$BASE_CKPT_DIR" ]; then
    echo "ERROR: Base model checkpoint not found: $BASE_CKPT_DIR"
    exit 1
fi

# ══════════════════════════════════════════════════════════════
#  NPU ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════

export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export NANOCHAT_BASE_DIR="$NANOCHAT_MODEL_DIR"
mkdir -p "$NANOCHAT_MODEL_DIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_HCCL_PATH=/usr/local/Ascend/ascend-toolkit/latest/hccl
export LD_LIBRARY_PATH=${ASCEND_HCCL_PATH}/lib64:${LD_LIBRARY_PATH:-}
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_WHITELIST_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

export PYTORCH_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_LOG_LEVEL=3

if [ -z "${ASCEND_DEVICE_ID:-}" ] && [ -n "${LOCAL_RANK:-}" ]; then
    export ASCEND_DEVICE_ID=$LOCAL_RANK
elif [ -z "${ASCEND_DEVICE_ID:-}" ]; then
    export ASCEND_DEVICE_ID=0
fi

# ══════════════════════════════════════════════════════════════
#  PRINT CONFIG
# ══════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  QuadMix Mid-Training — Eval Only"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Model tag:           $MODEL_TAG"
echo "  Base model tag:      $BASE_MODEL_TAG"
echo "  Mid checkpoint:      $MID_CKPT_DIR"
echo "  Nanochat repo:       $NANOCHAT_REPO"
echo "  Output dir:          $RESULT_DIR"
echo "  NPU cards:           $NUM_NPU"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

mkdir -p "$RESULT_DIR"

# ══════════════════════════════════════════════════════════════
#  SYMLINK: base_checkpoints -> mid_checkpoints
# ══════════════════════════════════════════════════════════════

LINK_DIR="$NANOCHAT_MODEL_DIR/base_checkpoints/$MODEL_TAG"
if [ ! -e "$LINK_DIR" ]; then
    echo "  Creating symlink: $LINK_DIR -> $BASE_CKPT_DIR"
    ln -s "$BASE_CKPT_DIR" "$LINK_DIR"
fi

# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

echo "╔══ Evaluation ══╗"
echo ""
echo "  Evaluating: $MODEL_TAG (mid)"

EVAL_LOG="$RESULT_DIR/eval_quadmix.log"

pushd "$NANOCHAT_REPO" > /dev/null
python3 -m torch.distributed.run --standalone --nproc_per_node="$NUM_NPU" -m scripts.base_eval -- \
    --eval=core \
    --device-batch-size="$EVAL_BATCH_SIZE" \
    --model-tag="$MODEL_TAG" \
    --model-type="mid" \
    2>&1 | tee "$EVAL_LOG"
popd > /dev/null

if [ -L "$LINK_DIR" ]; then
    rm "$LINK_DIR"
fi

echo ""
echo "╚════════════════════════════════════╝"
echo ""

# ══════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════

echo "╔══ Generating Report ══╗"
echo ""

EVAL_LOG="$EVAL_LOG" \
MODEL_TAG="$MODEL_TAG" BASE_MODEL_TAG="$BASE_MODEL_TAG" \
RESULT_DIR="$RESULT_DIR" \
python3 -c "
import os, re
from datetime import datetime

def parse_eval(path):
    info = {'core_metric': None, 'tasks': {}}
    if not path or not os.path.exists(path):
        return info
    task_pat = re.compile(r'Evaluating:\s+(.+?)\s+\(.*?\)\.\.\.\s+accuracy:\s+([\d.]+)\s+\|\s+centered:\s+([\d.-]+)\s+\|\s+time:\s+([\d.]+)s')
    core_pat = re.compile(r'CORE metric:\s+([\d.]+)')
    for line in open(path):
        m = task_pat.search(line)
        if m:
            info['tasks'][m.group(1)] = {'accuracy': float(m.group(2)), 'centered': float(m.group(3)), 'time': float(m.group(4))}
        m = core_pat.search(line)
        if m: info['core_metric'] = float(m.group(1))
    return info

def fmt(v, spec='', suffix=''):
    return f'{v:{spec}}{suffix}' if v is not None else 'N/A'

evl = parse_eval(os.environ['EVAL_LOG'])
result_dir = os.environ['RESULT_DIR']

lines = []
lines.append('# QuadMix Eval-Only Report')
lines.append('')
lines.append(f'**Generated**: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
lines.append(f'**Base Model**: \`{os.environ[\"BASE_MODEL_TAG\"]}\`')
lines.append(f'**Mid Model**: \`{os.environ[\"MODEL_TAG\"]}\`')
lines.append(f'**Result Dir**: \`{result_dir}\`')
lines.append('')

lines.append('## Result')
lines.append('')
lines.append(f'**CORE metric: {fmt(evl[\"core_metric\"], \".4f\")}**')
lines.append('')

if evl['tasks']:
    lines.append('## Per-Task Breakdown')
    lines.append('')
    lines.append('| Task | Accuracy | Centered | Time |')
    lines.append('|---|---|---|---|')
    for task in sorted(evl['tasks']):
        t = evl['tasks'][task]
        lines.append(f'| {task} | {t[\"accuracy\"]:.4f} | {t[\"centered\"]:.4f} | {t[\"time\"]:.1f}s |')
    lines.append('')

report = '\n'.join(lines)
report_path = os.path.join(result_dir, 'eval_report.md')
with open(report_path, 'w') as f:
    f.write(report)
print(f'Report written to: {report_path}')
print()
print(report)
"

echo ""
echo "╚══════════════════════════════════════╝"
echo ""

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Eval Only Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Output:          $RESULT_DIR"
echo "  Report:          $RESULT_DIR/eval_report.md"
echo "  Eval log:        $EVAL_LOG"
