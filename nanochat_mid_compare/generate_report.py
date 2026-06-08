#!/usr/bin/env python3
"""Generate experiment comparison report from QuadMix vs Random mid-training logs."""

import argparse
import json
import os
import re
import sys
from datetime import datetime


def parse_dataset_stats(path):
    with open(path) as f:
        return json.load(f)


def parse_training_log(path):
    info = {
        "final_loss": None,
        "total_time": None,
        "peak_memory": None,
        "final_tok_per_sec": None,
        "final_mfu": None,
        "num_steps": None,
        "core_metrics_during_training": [],
    }

    if not os.path.exists(path):
        return info

    with open(path) as f:
        lines = f.readlines()

    step_pattern = re.compile(
        r"step\s+(\d+)/(\d+)\s+\(.*?\)\s+\|\s+loss:\s+([\d.]+)\s+\|.*?\|\s+tok/sec:\s+([\d,]+)\s+\|\s+bf16_mfu:\s+([\d.]+)"
    )
    core_pattern = re.compile(r"Step\s+(\d+)\s+\|\s+CORE metric:\s+([\d.]+)")
    time_pattern = re.compile(r"Total training time:\s+([\d.]+)m")
    mem_pattern = re.compile(r"Peak memory usage:\s+([\d.]+)MiB")

    for line in lines:
        m = step_pattern.search(line)
        if m:
            step, total, loss, tok_sec, mfu = m.groups()
            info["final_loss"] = float(loss)
            info["num_steps"] = int(total)
            info["final_tok_per_sec"] = int(tok_sec.replace(",", ""))
            info["final_mfu"] = float(mfu)

        m = core_pattern.search(line)
        if m:
            step, metric = m.groups()
            info["core_metrics_during_training"].append((int(step), float(metric)))

        m = time_pattern.search(line)
        if m:
            info["total_time"] = float(m.group(1))

        m = mem_pattern.search(line)
        if m:
            info["peak_memory"] = float(m.group(1))

    return info


def parse_eval_log(path):
    info = {
        "core_metric": None,
        "tasks": {},
    }

    if not os.path.exists(path):
        return info

    task_pattern = re.compile(
        r"Evaluating:\s+(.+?)\s+\(.*?\)\.\.\.\s+accuracy:\s+([\d.]+)\s+\|\s+centered:\s+([\d.-]+)\s+\|\s+time:\s+([\d.]+)s"
    )
    core_pattern = re.compile(r"CORE metric:\s+([\d.]+)")

    with open(path) as f:
        for line in f:
            m = task_pattern.search(line)
            if m:
                label, acc, centered, elapsed = m.groups()
                info["tasks"][label] = {
                    "accuracy": float(acc),
                    "centered": float(centered),
                    "time": float(elapsed),
                }

            m = core_pattern.search(line)
            if m:
                info["core_metric"] = float(m.group(1))

    return info


def generate_report(args):
    stats = parse_dataset_stats(args.dataset_stats)
    quadmix_train = parse_training_log(args.quadmix_train_log)
    random_train = parse_training_log(args.random_train_log)
    quadmix_eval = parse_eval_log(args.quadmix_eval_log)
    random_eval = parse_eval_log(args.random_eval_log)

    q = stats["quadmix"]
    r = stats["random"]
    config = stats["config"]

    q_core = quadmix_eval["core_metric"]
    r_core = random_eval["core_metric"]

    if q_core is not None and r_core is not None:
        diff = q_core - r_core
        winner = "QuadMix" if diff > 0 else "Random" if diff < 0 else "Tie"
        pct = abs(diff) / max(r_core, 1e-9) * 100
    else:
        diff = None
        winner = "N/A"
        pct = 0

    lines = []
    lines.append("# QuadMix vs Random — Mid-Training Experiment Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Base Model**: `{config.get('base_model_tag', 'N/A')}`")
    lines.append(f"**Experiment Dir**: `{args.experiment_dir}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    if diff is not None:
        emoji = "+" if diff > 0 else ""
        lines.append(f"| | QuadMix | Random | Delta |")
        lines.append(f"|---|---|---|---|")
        lines.append(f"| **CORE metric** | **{q_core:.4f}** | **{r_core:.4f}** | **{emoji}{diff:.4f} ({emoji}{pct:.1f}%)** |")
        lines.append("")
        lines.append(f"**Winner: {winner}**")
    else:
        lines.append("CORE metric evaluation results not available.")
    lines.append("")

    lines.append("## Dataset Statistics")
    lines.append("")
    lines.append("| | QuadMix | Random |")
    lines.append("|---|---|---|")
    lines.append(f"| Train docs | {q['train_docs']:,} | {r['train_docs']:,} |")
    lines.append(f"| Tokens | {q['tokens']:,} | {r['tokens']:,} |")
    lines.append(f"| Shards | {q['shards']} | {r['shards']} |")
    lines.append(f"| Val docs | {q['val_docs']:,} | {r['val_docs']:,} |")
    lines.append("")

    lines.append("## Training Statistics")
    lines.append("")
    lines.append("| | QuadMix | Random |")
    lines.append("|---|---|---|")
    def fmt(val, spec="", suffix=""):
        if val is None:
            return "N/A"
        return f"{val:{spec}}{suffix}"

    lines.append(f"| Steps | {fmt(quadmix_train['num_steps'])} | {fmt(random_train['num_steps'])} |")
    lines.append(f"| Final loss | {fmt(quadmix_train['final_loss'], '.6f')} | {fmt(random_train['final_loss'], '.6f')} |")
    lines.append(f"| Training time | {fmt(quadmix_train['total_time'], '.1f', 'm')} | {fmt(random_train['total_time'], '.1f', 'm')} |")
    lines.append(f"| Tokens/sec | {fmt(quadmix_train['final_tok_per_sec'], ',')} | {fmt(random_train['final_tok_per_sec'], ',')} |")
    lines.append(f"| MFU | {fmt(quadmix_train['final_mfu'], '.2f', '%')} | {fmt(random_train['final_mfu'], '.2f', '%')} |")
    lines.append(f"| Peak memory | {fmt(quadmix_train['peak_memory'], '.0f', ' MiB')} | {fmt(random_train['peak_memory'], '.0f', ' MiB')} |")
    lines.append("")

    if quadmix_train["core_metrics_during_training"] or random_train["core_metrics_during_training"]:
        lines.append("## CORE Metric During Training")
        lines.append("")
        lines.append("| Step | QuadMix | Random |")
        lines.append("|---|---|---|")
        q_cm = dict(quadmix_train["core_metrics_during_training"])
        r_cm = dict(random_train["core_metrics_during_training"])
        all_steps = sorted(set(list(q_cm.keys()) + list(r_cm.keys())))
        for s in all_steps:
            qv = f"{q_cm[s]:.4f}" if s in q_cm else "-"
            rv = f"{r_cm[s]:.4f}" if s in r_cm else "-"
            lines.append(f"| {s} | {qv} | {rv} |")
        lines.append("")

    if quadmix_eval["tasks"] or random_eval["tasks"]:
        lines.append("## CORE Metric — Per-Task Breakdown")
        lines.append("")
        lines.append("| Task | QuadMix (centered) | Random (centered) | Delta |")
        lines.append("|---|---|---|---|")
        all_tasks = sorted(set(list(quadmix_eval["tasks"].keys()) + list(random_eval["tasks"].keys())))
        for task in all_tasks:
            qc = quadmix_eval["tasks"].get(task, {}).get("centered")
            rc = random_eval["tasks"].get(task, {}).get("centered")
            qc_str = f"{qc:.4f}" if qc is not None else "-"
            rc_str = f"{rc:.4f}" if rc is not None else "-"
            if qc is not None and rc is not None:
                d = qc - rc
                d_str = f"{d:+.4f}"
            else:
                d_str = "-"
            lines.append(f"| {task} | {qc_str} | {rc_str} | {d_str} |")
        lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Target param-data ratio | {config.get('target_param_data_ratio', 'N/A')} |")
    lines.append(f"| Num scaling params | {config.get('num_scaling_params', 'N/A'):,} |")
    lines.append(f"| Device batch size | {config.get('device_batch_size', 'N/A')} |")
    lines.append(f"| Num NPU | {config.get('num_npu', 'N/A')} |")
    lines.append(f"| Seed | {config.get('seed', 'N/A')} |")
    lines.append(f"| Token method | {config.get('token_method', 'N/A')} |")
    lines.append("")

    report_text = "\n".join(lines)

    report_path = os.path.join(args.experiment_dir, "experiment_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"Report written to: {report_path}")
    print()
    print(report_text)


def main():
    parser = argparse.ArgumentParser(description="Generate QuadMix vs Random experiment report")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--quadmix-train-log", required=True)
    parser.add_argument("--random-train-log", required=True)
    parser.add_argument("--quadmix-eval-log", required=True)
    parser.add_argument("--random-eval-log", required=True)
    args = parser.parse_args()
    generate_report(args)


if __name__ == "__main__":
    main()
