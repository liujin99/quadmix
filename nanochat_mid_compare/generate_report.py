#!/usr/bin/env python3
"""Generate experiment comparison report from mid-training logs.

Supports N baselines: QuadMix, Random, and optionally multiple Quality-Only Top-K methods.
"""

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

    if not path or not os.path.exists(path):
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

    if not path or not os.path.exists(path):
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


def fmt(val, spec="", suffix=""):
    if val is None:
        return "N/A"
    return f"{val:{spec}}{suffix}"


def generate_report(args):
    stats = parse_dataset_stats(args.dataset_stats)
    config = stats["config"]
    baselines = config.get("baselines", ["quadmix", "random"])

    quality_methods = config.get("quality_methods", [])
    if not quality_methods:
        for key in stats:
            if key.startswith("quality_"):
                method = key[len("quality_"):]
                if method not in quality_methods:
                    quality_methods.append(method)

    train_logs = {}
    eval_logs = {}
    log_map = {
        "quadmix": (args.quadmix_train_log, args.quadmix_eval_log),
        "random": (args.random_train_log, args.random_eval_log),
    }
    if args.quality_train_log:
        for i, train_log in enumerate(args.quality_train_log):
            eval_log = args.quality_eval_log[i] if args.quality_eval_log and i < len(args.quality_eval_log) else None
            if i < len(quality_methods):
                log_map[f"quality_{quality_methods[i]}"] = (train_log, eval_log)

    for b in baselines:
        if b in log_map:
            train_logs[b] = parse_training_log(log_map[b][0])
            eval_logs[b] = parse_eval_log(log_map[b][1])

    labels = {"quadmix": "QuadMix", "random": "Random"}
    for m in quality_methods:
        labels[f"quality_{m}"] = f"Quality ({m})"

    lines = []
    title_parts = [labels[b] for b in baselines]
    title = " vs ".join(title_parts)
    lines.append(f"# {title} — Mid-Training Experiment Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Base Model**: `{config.get('base_model_tag', 'N/A')}`")
    lines.append(f"**Experiment Dir**: `{args.experiment_dir}`")
    if quality_methods:
        for m in quality_methods:
            qcol = stats.get(f"quality_{m}", {}).get("quality_column", "N/A")
            lines.append(f"**Quality Method**: `{m}` ({qcol})")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    header = "| | " + " | ".join(labels[b] for b in baselines) + " |"
    sep = "|---|" + "|".join("---" for _ in baselines) + "|"
    lines.append(header)
    lines.append(sep)

    core_vals = {}
    for b in baselines:
        core_vals[b] = eval_logs.get(b, {}).get("core_metric")

    core_row = "| **CORE metric** |"
    for b in baselines:
        v = core_vals[b]
        core_row += f" **{v:.4f}** |" if v is not None else " N/A |"
    lines.append(core_row)
    lines.append("")

    ref_b = baselines[1] if len(baselines) > 1 else baselines[0]
    ref_core = core_vals.get(ref_b)
    if core_vals.get(baselines[0]) is not None and ref_core is not None:
        lines.append("### Pairwise Deltas")
        lines.append("")
        lines.append("| Comparison | Delta | % |")
        lines.append("|---|---|---|")
        for i, b1 in enumerate(baselines):
            for b2 in baselines[i+1:]:
                v1, v2 = core_vals.get(b1), core_vals.get(b2)
                if v1 is not None and v2 is not None:
                    d = v1 - v2
                    pct = abs(d) / max(v2, 1e-9) * 100
                    sign = "+" if d > 0 else ""
                    winner = labels[b1] if d > 0 else labels[b2] if d < 0 else "Tie"
                    lines.append(f"| {labels[b1]} vs {labels[b2]} | {sign}{d:.4f} | {sign}{pct:.1f}% ({winner}) |")
        lines.append("")

    lines.append("## Dataset Statistics")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for field, fmt_spec in [("train_docs", ","), ("tokens", ","), ("shards", ""), ("val_docs", ",")]:
        row = f"| {field.replace('_', ' ').title()} |"
        for b in baselines:
            v = stats.get(b, {}).get(field)
            row += f" {v:{fmt_spec}} |" if v is not None else " N/A |"
        lines.append(row)
    lines.append("")

    lines.append("## Training Statistics")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for field, spec, suffix in [
        ("num_steps", "", ""),
        ("final_loss", ".6f", ""),
        ("total_time", ".1f", "m"),
        ("final_tok_per_sec", ",", ""),
        ("final_mfu", ".2f", "%"),
        ("peak_memory", ".0f", " MiB"),
    ]:
        label = field.replace("_", " ").title()
        row = f"| {label} |"
        for b in baselines:
            v = train_logs.get(b, {}).get(field)
            row += f" {fmt(v, spec, suffix)} |"
        lines.append(row)
    lines.append("")

    any_core_during = any(
        train_logs.get(b, {}).get("core_metrics_during_training") for b in baselines
    )
    if any_core_during:
        lines.append("## CORE Metric During Training")
        lines.append("")
        lines.append(header.replace("| |", "| Step |", 1))
        lines.append(sep)
        all_cm = {b: dict(train_logs.get(b, {}).get("core_metrics_during_training", [])) for b in baselines}
        all_steps = sorted(set(s for cm in all_cm.values() for s in cm.keys()))
        for s in all_steps:
            row = f"| {s} |"
            for b in baselines:
                v = all_cm[b].get(s)
                row += f" {v:.4f} |" if v is not None else " - |"
            lines.append(row)
        lines.append("")

    all_tasks = set()
    for b in baselines:
        all_tasks.update(eval_logs.get(b, {}).get("tasks", {}).keys())

    if all_tasks:
        lines.append("## CORE Metric — Per-Task Breakdown")
        lines.append("")
        task_header = "| Task | " + " (centered) | ".join(labels[b] for b in baselines) + " (centered) |"
        task_sep = "|---|" + "|".join("---" for _ in baselines) + "|"
        lines.append(task_header)
        lines.append(task_sep)
        for task in sorted(all_tasks):
            row = f"| {task} |"
            for b in baselines:
                c = eval_logs.get(b, {}).get("tasks", {}).get(task, {}).get("centered")
                row += f" {c:.4f} |" if c is not None else " - |"
            lines.append(row)
        lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Target param-data ratio | {config.get('target_param_data_ratio', 'N/A')} |")
    lines.append(f"| Num scaling params | {config.get('num_scaling_params', 'N/A'):,} |")
    lines.append(f"| Device batch size | {config.get('device_batch_size', 'N/A')} |")
    lines.append(f"| Num NPU | {config.get('num_npu', 'N/A')} |")
    lines.append(f"| Seed | {config.get('seed', 'N/A')} |")
    lines.append(f"| Token method | {config.get('token_method', 'N/A')} |")
    if quality_methods:
        lines.append(f"| Quality methods | {', '.join(quality_methods)} |")
    lines.append("")

    report_text = "\n".join(lines)

    report_path = os.path.join(args.experiment_dir, "experiment_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"Report written to: {report_path}")
    print()
    print(report_text)


def main():
    parser = argparse.ArgumentParser(description="Generate mid-training experiment comparison report")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--quadmix-train-log", required=True)
    parser.add_argument("--random-train-log", required=True)
    parser.add_argument("--quadmix-eval-log", required=True)
    parser.add_argument("--random-eval-log", required=True)
    parser.add_argument("--quality-train-log", nargs="+", default=None,
                        help="Quality baseline training logs (one per method, in order)")
    parser.add_argument("--quality-eval-log", nargs="+", default=None,
                        help="Quality baseline evaluation logs (one per method, in order)")
    args = parser.parse_args()
    generate_report(args)


if __name__ == "__main__":
    main()
