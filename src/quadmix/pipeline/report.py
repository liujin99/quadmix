"""QuaDMix report generator — produces MD report + companion PNG files.

Figures saved as PNG alongside the report, referenced via standard
markdown ![](filename.png) syntax — supported by all MD viewers.
"""

import json
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quadmix.core.types import ParameterSet

DOMAIN_SHORT = [
    "Industrial", "Social", "Science", "Religion", "Philology",
    "Literature", "History", "General", "Philosophy", "Arts",
]

QUALITY_NAMES = ["DCLM", "FineWeb-Edu", "English", "Math (Gen)", "Math (OpenWeb)"]
QUALITY_SHORT = ["DCLM", "Edu", "Eng", "MathG", "MathO"]

COLOR_ORIG = "#5B9BD5"
COLOR_OPT = "#ED7D31"
QUALITY_COLORS = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5"]


def _setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def _save_fig(fig, output_dir, filename):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  [Figure] Saved: {path}")
    return filename


# ── Figure 1 ──

def _make_fig1(orig_dist, opt_dist, output_dir):
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    m = len(orig_dist)
    x = np.arange(m)
    width = 0.35
    ax.bar(x - width / 2, orig_dist * 100, width,
           label="Original", color=COLOR_ORIG, edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, opt_dist * 100, width,
           label="Optimal", color=COLOR_OPT, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Domain Distribution: Original vs Optimal Sampling")
    ax.set_xticks(x)
    ax.set_xticklabels(DOMAIN_SHORT, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    for bars, color in [(ax.containers[0], COLOR_ORIG), (ax.containers[1], COLOR_OPT)]:
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=6, color=color)
    plt.tight_layout()
    return _save_fig(fig, output_dir, "fig1_domain_distribution.png")


# ── Figure 2 ──

def _make_fig2(domain_weights, num_domains, num_criteria, output_dir):
    _setup_style()
    data = np.zeros((num_domains, num_criteria))
    for m in range(num_domains):
        start = m * num_criteria
        data[m] = domain_weights[start:start + num_criteria]
    labels = DOMAIN_SHORT

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for n in range(num_criteria):
        ax.bar(x, data[:, n], bottom=bottom, width=0.65,
               label=QUALITY_NAMES[n], color=QUALITY_COLORS[n],
               edgecolor="white", linewidth=0.4)
        bottom += data[:, n]
    ax.set_xlabel("Domain")
    ax.set_ylabel("Weight")
    ax.set_title("Quality Signal Weights per Domain (α)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=8, ncol=num_criteria)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    for i in range(len(labels)):
        row = data[i]
        max_idx = np.argmax(row)
        max_val = row[max_idx]
        if max_val > 0.20:
            prefix = sum(row[:max_idx])
            ax.text(i, prefix + max_val / 2, QUALITY_SHORT[max_idx],
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", color="white")
    plt.tight_layout()
    return _save_fig(fig, output_dir, "fig2_quality_weights.png")


# ── Table ──

def _experiment_table(exp_outputs_dir, data_path, num_domains=10, top_k=50,
                       domain_labels_override=None):
    """Generate experiment table. If domain_labels_override is provided, use it
    instead of loading from data_path (supports sharded mode)."""
    if domain_labels_override is not None:
        domain_labels = domain_labels_override
    elif os.path.isdir(data_path):
        return "*(experiment table: sharded mode, see pipeline_summary.json)*"
    else:
        df = pd.read_parquet(data_path)
        domain_labels = df["domain"].to_numpy(dtype=np.int64)
    exp_dirs = sorted([
        d for d in os.listdir(exp_outputs_dir)
        if d.startswith("exp_") and os.path.isdir(os.path.join(exp_outputs_dir, d))
    ])
    if not exp_dirs:
        return "*(no experiment data)*"

    rows = []
    for exp_dir_name in exp_dirs[:top_k]:
        exp_dir = os.path.join(exp_outputs_dir, exp_dir_name)
        meta_path = os.path.join(exp_dir, "meta.json")
        indices_path = os.path.join(exp_dir, "selected_indices.npy")
        if not os.path.exists(indices_path):
            continue
        selected_idx = np.load(indices_path)
        sel_domain = domain_labels[selected_idx]
        dist = np.bincount(sel_domain[sel_domain >= 0],
                           minlength=num_domains).astype(np.float64)
        dist /= max(1, dist.sum())
        top5 = sorted(range(num_domains), key=lambda m: dist[m], reverse=True)[:5]
        top5_cells = " ".join(
            f"**{DOMAIN_SHORT[m]}** {dist[m]*100:.0f}%"
            for m in top5 if dist[m] > 0.01
        )
        val_loss = "?"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            val_loss = f"{meta.get('val_loss', '?'):.4f}"
        rows.append((exp_dir_name.replace("exp_", ""), val_loss, len(selected_idx), top5_cells))

    if not rows:
        return "*(no usable experiment data)*"

    lines = [
        "### Table 1: 各实验的 Top-5 域分布\n",
        "| Exp | Val Loss | Docs | Top-5 Domain Distribution |",
        "|:---:|:--------:|:----:|:--------------------------|",
    ]
    for e, vl, docs, t5 in rows:
        lines.append(f"| {e:>4s} | {vl} | {docs:,} | {t5} |")
    losses = []
    for _, vl, _, _ in rows:
        try:
            losses.append(float(vl))
        except ValueError:
            pass
    if losses:
        lines.append(
            f"| **Avg** | **{np.mean(losses):.4f}** | **{np.mean([row[2] for row in rows]):.0f}** | "
            f"min: **{np.min(losses):.4f}**, max: **{np.max(losses):.4f}** |"
        )
    lines.append("")
    return "\n".join(lines)


# ── Main ──

def generate_report(
    output_dir, data_path, optimal_params, optimal_selected_indices,
    domain_labels, token_counts, num_domains=10, num_criteria=5,
    config=None, metrics=None, elapsed=None,
    use_sharded=False, reliability=None, proxy_loss_stats=None,
    per_task_analysis=None,
):
    """Generate MD report with separate PNG figures."""
    # Compute distributions
    orig_dist = np.bincount(domain_labels[domain_labels >= 0], minlength=num_domains).astype(np.float64)
    orig_dist /= max(1, orig_dist.sum())
    opt_domain = domain_labels[optimal_selected_indices]
    opt_dist = np.bincount(opt_domain[opt_domain >= 0], minlength=num_domains).astype(np.float64)
    opt_dist /= max(1, opt_dist.sum())
    domain_w = optimal_params.merge_config.domain_weights

    # Save figures
    fig1_file = _make_fig1(orig_dist, opt_dist, output_dir)
    fig2_file = _make_fig2(domain_w, num_domains, num_criteria, output_dir)

    sel_tokens = token_counts[optimal_selected_indices].sum()

    # Build markdown
    parts = ["# QuaDMix 最优采样报告\n"]

    if config or metrics:
        parts.append("## 配置与指标\n")
        parts.append("| Parameter | Value |\n|:----------|:------|")
        if config:
            for k, v in config.items():
                parts.append(f"| {k} | {v} |")
        if metrics:
            for k, v in metrics.items():
                if v is None:
                    parts.append(f"| {k} | — |")
                elif isinstance(v, float):
                    parts.append(f"| {k} | {v:.4f} |")
                else:
                    parts.append(f"| {k} | {v} |")
        if elapsed is not None:
            parts.append(f"| Duration | {elapsed:.1f}s |")
        parts.append("")

    if reliability:
        parts.append("## Model Reliability\n")
        parts.append("**Aggregate model** (single LightGBM on total loss, diagnostic only — not used for search):\n")
        bootstrap_mean = reliability.get("val_r2_bootstrap_mean")
        ci_lower = reliability.get("val_r2_ci_lower")
        ci_upper = reliability.get("val_r2_ci_upper")
        ci_width = reliability.get("val_r2_ci_width")
        sample_sufficient = reliability.get("sample_sufficient")
        overfit_gap = reliability.get("overfit_gap")
        n_features = reliability.get("n_features")
        n_train = reliability.get("n_train_samples")

        parts.append("| Metric | Value | Status |")
        parts.append("|:-------|:------|:-------|")

        if bootstrap_mean is not None:
            quality = "✓ Excellent" if bootstrap_mean > 0.8 else ("✓ Good" if bootstrap_mean > 0.6 else "⚠️ Weak signal")
            parts.append(f"| **Val R² (bootstrap mean)** | **{bootstrap_mean:.4f}** | **{quality}** |")

        agg_val_r2 = metrics.get("aggregate_val_r2", metrics.get("val_r2")) if metrics else None
        if agg_val_r2 is not None:
            parts.append(f"| Val R² (single split) | {agg_val_r2:.4f} | — |")

        if ci_lower is not None and ci_upper is not None:
            ci_status = "✓ Stable" if ci_width is not None and ci_width < 0.3 else "⚠️ Wide CI"
            parts.append(f"| 95% CI | [{ci_lower:.3f}, {ci_upper:.3f}] | width={ci_width:.3f} {ci_status} |")

        if overfit_gap is not None:
            gap_status = "✓ OK" if overfit_gap < 0.3 else "⚠️ Overfitting"
            parts.append(f"| Train-Val Gap | {overfit_gap:.3f} | {gap_status} |")

        if n_train is not None and n_features is not None:
            ratio = n_train / n_features if n_features > 0 else 0
            suff_status = "✓ OK" if sample_sufficient else "⚠️ Underdetermined"
            parts.append(f"| Samples/Features | {n_train}/{n_features} ({ratio:.1f}x) | {suff_status} |")

        parts.append("")

        warnings = []
        if ci_width is not None and ci_width > 0.5:
            warnings.append(f"- ⚠️ **CI width = {ci_width:.3f}**: results unreliable, increase experiments to 100+")
        if not sample_sufficient:
            warnings.append(f"- ⚠️ **Samples ({n_train}) < Features ({n_features})**: model is underdetermined, increase experiments to {n_features * 3 if n_features else 200}+")
        if overfit_gap is not None and overfit_gap > 0.3:
            warnings.append(f"- ⚠️ **Train-Val gap = {overfit_gap:.3f}**: possible overfitting, consider more experiments")
        if bootstrap_mean is not None and bootstrap_mean < 0.6:
            warnings.append(f"- ⚠️ **Bootstrap mean R² = {bootstrap_mean:.3f}**: signal too weak, consider larger model or more training steps")

        if warnings:
            parts.append("**Recommendations:**\n")
            parts.extend(warnings)
            parts.append("")

    if metrics:
        ens_r2 = metrics.get("ensemble_val_r2")
        ens_mae = metrics.get("ensemble_val_mae")
        if ens_r2 is not None:
            parts.append("## Ensemble Model (used for search)\n")
            parts.append("Per-task weighted prediction with z-score calibration and R²-adaptive weights.\n")
            parts.append("| Metric | Value |")
            parts.append("|:-------|:------|")
            quality = "✓ Excellent" if ens_r2 > 0.6 else ("✓ Good" if ens_r2 > 0.3 else "⚠️ Weak")
            parts.append(f"| **Val R²** | **{ens_r2:.4f}** ({quality}) |")
            parts.append(f"| Val MAE | {ens_mae:.4f} |")
            parts.append("")

    if proxy_loss_stats:
        parts.append("## Proxy Experiment Loss Stats\n")
        parts.append("| Metric | Mean | Std | Min | Max |")
        parts.append("|:-------|:-----|:----|:----|:----|")
        for name, stats in proxy_loss_stats.items():
            if isinstance(stats, dict):
                parts.append(f"| {name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |")
            elif isinstance(stats, float):
                parts.append(f"| {name} | {stats:.4f} | — | — | — |")
        parts.append("")

    if per_task_analysis:
        parts.append("## Per-Task Analysis (R²-Adaptive Weighting)\n")
        n_active = per_task_analysis.get("n_active", 0)
        n_filtered = per_task_analysis.get("n_filtered", 0)
        parts.append(f"**Active tasks:** {n_active} | **Filtered (R²≤0):** {n_filtered}\n")
        parts.append("| Task | R² | Weight | Std | Status |")
        parts.append("|:-----|---:|-------:|----:|:-------|")
        for task in per_task_analysis.get("tasks", []):
            name = task["name"]
            r2 = task["r2"]
            weight = task["weight"]
            std = task.get("std")
            std_str = f"{std:.4f}" if std is not None else "—"
            if weight == 0:
                status = "⚠️ Filtered"
            elif r2 > 0.6:
                status = "✓ Excellent"
            elif r2 > 0.3:
                status = "✓ Good"
            else:
                status = "⚠️ Weak"
            parts.append(f"| {name} | {r2:.4f} | {weight:.4f} | {std_str} | {status} |")
        parts.append("")

    parts += [
        "## 采样概览\n",
        f"- **原始文档数:** {len(domain_labels):,}",
        f"- **原始总 tokens:** {int(token_counts.sum()):,}",
        f"- **最优采样文档数:** {len(optimal_selected_indices):,}",
        f"- **最优采样总 tokens:** {int(sel_tokens):,}",
        f"- **采样比例 (docs):** {len(optimal_selected_indices) / max(1, len(domain_labels)):.4f}x",
        f"- **采样比例 (tokens):** {int(sel_tokens) / max(1, int(token_counts.sum())):.4f}x",
        "",
        "---\n",
        "## Figure 1: 域分布对比\n",
        "原始数据集 vs 最优参数采样子集的域分布对比。\n",
        f"![]({fig1_file})\n",
        "---\n",
        "## Figure 2: 质量信号权重\n",
        "最优参数下各域的质量信号融合权重 α。\n",
        f"![]({fig2_file})\n",
    ]

    proxy_dir = os.path.join(output_dir, "proxy_experiments")
    if os.path.isdir(proxy_dir):
        parts += ["---\n", _experiment_table(
            proxy_dir, data_path, num_domains,
            domain_labels_override=domain_labels if use_sharded else None,
        )]

    parts += ["---\n", f"*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"]
    return "\n".join(parts)


def save_report(report, output_dir, filename="quadmix_report.md"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        f.write(report)
    print(f"[Report] Saved to: {path}  ({len(report):,} chars)")
