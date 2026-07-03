import os, re, json
from datetime import datetime


def parse_train(path):
    info = {'final_loss': None, 'total_time': None, 'peak_memory': None,
            'final_tok_per_sec': None, 'final_mfu': None, 'num_steps': None,
            'core_metrics': []}
    if not path or not os.path.exists(path):
        return info
    step_pat = re.compile(r'step\s+(\d+)/(\d+)\s+\(.*?\)\s+\|\s+loss:\s+([\d.]+)\s+\|.*?\|\s+tok/sec:\s+([\d,]+)\s+\|\s+bf16_mfu:\s+([\d.]+)')
    core_pat = re.compile(r'Step\s+(\d+)\s+\|\s+CORE metric:\s+([\d.]+)')
    time_pat = re.compile(r'Total training time:\s+([\d.]+)m')
    mem_pat = re.compile(r'Peak memory usage:\s+([\d.]+)MiB')
    for line in open(path):
        m = step_pat.search(line)
        if m:
            info['num_steps'] = int(m.group(2))
            info['final_loss'] = float(m.group(3))
            info['final_tok_per_sec'] = int(m.group(4).replace(',', ''))
            info['final_mfu'] = float(m.group(5))
        m = core_pat.search(line)
        if m:
            info['core_metrics'].append((int(m.group(1)), float(m.group(2))))
        m = time_pat.search(line)
        if m:
            info['total_time'] = float(m.group(1))
        m = mem_pat.search(line)
        if m:
            info['peak_memory'] = float(m.group(1))
    return info


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
        if m:
            info['core_metric'] = float(m.group(1))
    return info


def fmt(v, spec='', suffix=''):
    return f'{v:{spec}}{suffix}' if v is not None else 'N/A'


train = parse_train(os.environ['QUADMIX_LOG'])
evl = parse_eval(os.environ['QUADMIX_EVAL_LOG'])
result_dir = os.environ['RESULT_DIR']
data_dir = os.environ['DATA_DIR']

stats_path = os.path.join(data_dir, 'dataset_stats.json')
stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
q = stats.get('quadmix', {})

lines = []
lines.append('# QuadMix Quick Validation Report')
lines.append('')
lines.append(f'**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
lines.append(f'**Base Model**: `{os.environ["BASE_MODEL_TAG"]}`')
lines.append(f'**Mid Model**: `{os.environ["QUADMIX_MODEL_TAG"]}`')
lines.append(f'**Result Dir**: `{result_dir}`')
lines.append('')

lines.append('## Provenance')
lines.append('')
lines.append('| Key | Value |')
lines.append('|---|---|')
quadmix_source = stats.get('config', {}).get('quadmix_source', 'N/A')
lines.append(f'| Source parquet | `{quadmix_source}` |')
lines.append(f'| Tokenizer | `{os.environ.get("TOKENIZER_PKL", "N/A")}` |')
lines.append(f'| Nanochat repo | `{os.environ.get("NANOCHAT_REPO", "N/A")}` |')
lines.append(f'| Nanochat model dir | `{os.environ.get("NANOCHAT_MODEL_DIR", "N/A")}` |')
lines.append(f'| Mid checkpoint output | `{os.environ.get("MID_CHECKPOINTS_OUTPUT_DIR", "N/A")}` |')
lines.append(f'| QuadMix git | `{os.environ.get("QUADMIX_GIT_HASH", "N/A")}` |')
lines.append(f'| Nanochat git | `{os.environ.get("NANOCHAT_GIT_HASH", "N/A")}` |')
lines.append('')

lines.append('## Result')
lines.append('')
lines.append(f'**CORE metric: {fmt(evl["core_metric"], ".4f")}**')
lines.append('')

lines.append('## Training')
lines.append('')
lines.append('| Metric | Value |')
lines.append('|---|---|')
lines.append(f'| Steps | {fmt(train["num_steps"])} |')
lines.append(f'| Final loss | {fmt(train["final_loss"], ".6f")} |')
lines.append(f'| Total time | {fmt(train["total_time"], ".1f", "m")} |')
lines.append(f'| Throughput | {fmt(train["final_tok_per_sec"], ",", " tok/s")} |')
lines.append(f'| MFU | {fmt(train["final_mfu"], ".2f", "%")} |')
lines.append(f'| Peak memory | {fmt(train["peak_memory"], ".0f", " MiB")} |')
lines.append('')

if train['core_metrics']:
    lines.append('### CORE During Training')
    lines.append('')
    lines.append('| Step | CORE |')
    lines.append('|---|---|')
    for step, val in train['core_metrics']:
        lines.append(f'| {step} | {val:.4f} |')
    lines.append('')

if evl['tasks']:
    lines.append('## Per-Task Breakdown')
    lines.append('')
    lines.append('| Task | Accuracy | Centered | Time |')
    lines.append('|---|---|---|---|')
    for task in sorted(evl['tasks']):
        t = evl['tasks'][task]
        lines.append(f'| {task} | {t["accuracy"]:.4f} | {t["centered"]:.4f} | {t["time"]:.1f}s |')
    lines.append('')

lines.append('## Data')
lines.append('')
lines.append('| Metric | Value |')
lines.append('|---|---|')
lines.append(f'| Train docs | {q.get("train_docs", "N/A"):,} |' if isinstance(q.get('train_docs'), int) else '| Train docs | N/A |')
lines.append(f'| Tokens | {q.get("tokens", "N/A"):,} |' if isinstance(q.get('tokens'), int) else '| Tokens | N/A |')
lines.append(f'| Shards | {q.get("shards", "N/A")} |')
lines.append('')

lines.append('## Training Budget')
lines.append('')
lines.append('| Parameter | Value |')
lines.append('|---|---|')
lines.append(f'| Dataset tokens | {q.get("tokens", "N/A"):,} |' if isinstance(q.get('tokens'), int) else '| Dataset tokens | N/A |')
actual_tokens = os.environ.get('ACTUAL_TOKENS')
actual_ratio = os.environ.get('ACTUAL_RATIO')
num_iterations = os.environ.get('NUM_ITERATIONS')
lines.append(f'| Actual training tokens | {int(actual_tokens):,} |' if actual_tokens else '| Actual training tokens | N/A |')
lines.append(f'| Actual param-data ratio | {actual_ratio} |' if actual_ratio else '| Actual param-data ratio | N/A |')
lines.append(f'| Iterations | {num_iterations} |' if num_iterations else '| Iterations | N/A |')
lines.append('')

lines.append('## Config')
lines.append('')
lines.append('| Parameter | Value |')
lines.append('|---|---|')
lines.append(f'| target-param-data-ratio | {os.environ["TARGET_PARAM_DATA_RATIO"]} |')
lines.append(f'| num-scaling-params | {int(os.environ["NUM_SCALING_PARAMS"]):,} |')
lines.append(f'| device-batch-size | {os.environ["DEVICE_BATCH_SIZE"]} |')
lines.append(f'| total-batch-size | {int(os.environ["TOTAL_BATCH_SIZE"]):,} |')
lines.append(f'| NPU cards | {os.environ["NUM_NPU"]} |')
lines.append('')

report = '\n'.join(lines)
report_path = os.path.join(result_dir, 'midtrain_validation_report.md')
with open(report_path, 'w') as f:
    f.write(report)
print(f'Report written to: {report_path}')
print()
print(report)
