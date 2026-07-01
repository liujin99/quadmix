"""Regenerate Stage 9 report from saved pipeline_summary.json + optimal_parameters.json.

Usage:
    python scripts/regenerate_report.py /path/to/result/demo_full_XXX
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quadmix.pipeline.report import (
    _make_fig1, _make_fig2, _experiment_table, save_report,
)
from quadmix.constants import DOMAIN_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Pipeline output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    params_path = os.path.join(output_dir, "optimal_parameters.json")

    with open(summary_path) as f:
        summary = json.load(f)
    with open(params_path) as f:
        params = json.load(f)

    config = summary["config"]
    metrics = summary["metrics"]
    num_domains = config["num_domains"]
    num_criteria = config["num_quality_criteria"]

    dist_change = summary.get("sampling", {}).get("domain_distribution_change", {})
    domain_names = DOMAIN_NAMES[:num_domains]

    orig_dist = np.zeros(num_domains)
    opt_dist = np.zeros(num_domains)
    for m, name in enumerate(domain_names):
        if name in dist_change:
            orig_dist[m] = dist_change[name]["original"]
            opt_dist[m] = dist_change[name]["selected"]
    orig_total = orig_dist.sum()
    opt_total = opt_dist.sum()
    orig_dist_norm = orig_dist / max(1, orig_total)
    opt_dist_norm = opt_dist / max(1, opt_total)

    dw_flat = np.zeros(num_domains * num_criteria)
    quality_names = ["DCLM", "FineWeb-Edu", "English", "Math (Gen)", "Math (OpenWeb)"]
    for m, name in enumerate(domain_names):
        if name in params["quality_weights"]:
            for n, qn in enumerate(quality_names[:num_criteria]):
                dw_flat[m * num_criteria + n] = params["quality_weights"][name].get(qn, 0)

    print(f"[Regenerate] Generating figures...")
    fig1_file = _make_fig1(orig_dist_norm, opt_dist_norm, output_dir, num_domains)
    fig2_file = _make_fig2(dw_flat, num_domains, num_criteria, output_dir)

    reliability = summary.get("reliability")
    proxy_loss_stats = summary.get("proxy_loss_stats")
    per_task_analysis = summary.get("per_task_analysis")
    elapsed = summary.get("elapsed_seconds")
    sampling = summary.get("sampling", {})

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
        agg_val_r2 = metrics.get("aggregate_val_r2") if metrics else None
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
        parts.append("## Model Evaluation Metrics\n")
        search_mode = config.get("search_weight_mode", "r2_sigma_weighted")
        parts.append(f"**Search mode:** {search_mode}\n")
        parts.append("| Metric | Value | Formula | Purpose |")
        parts.append("|:-------|:------|:--------|:--------|")
        ens_r2 = metrics.get("ensemble_val_r2")
        eq_r2 = metrics.get("equal_weight_r2")
        spearman = metrics.get("spearman_corr")
        top_k_recall = metrics.get("top_k_recall")
        top_k_value = metrics.get("top_k_value", 5)
        search_lift = metrics.get("search_lift")
        if ens_r2 is not None:
            q = "✓ Good" if ens_r2 > 0.3 else "⚠️ Weak"
            parts.append(f"| **Overall Val R²** | **{ens_r2:.4f}** ({q}) | R²(Σ wᵢ·z_predᵢ, Σ wᵢ·z_actualᵢ) | Search objective quality |")
        if eq_r2 is not None:
            q = "✓ Good" if eq_r2 > 0.3 else "⚠️ Weak"
            parts.append(f"| **Equal-Wt Val R²** | **{eq_r2:.4f}** ({q}) | R²((1/K)Σ z_predᵢ, (1/K)Σ z_actualᵢ) | Downstream goal quality |")
        if spearman is not None:
            q = "✓ Good" if spearman > 0.5 else "⚠️ Weak"
            parts.append(f"| **Spearman Rank Corr** | **{spearman:.4f}** ({q}) | corr(rank(pred), rank(actual)) | Ranking ability |")
        if top_k_recall is not None:
            q = "✓ Good" if top_k_recall > 0.3 else "⚠️ Weak"
            parts.append(f"| **Top-{top_k_value} Recall** | **{top_k_recall:.4f}** ({q}) | |pred_top_k ∩ actual_top_k| / k | Search hit rate |")
        if search_lift is not None:
            q = "✓ Good" if search_lift > 0.5 else "⚠️ Weak"
            parts.append(f"| **Search Lift** | **{search_lift:.4f}** σ ({q}) | (μ_random - μ_search_top_k) / σ | Search value vs random |")
        parts.append("")

        status_lines = []
        if spearman is not None and spearman > 0.5:
            status_lines.append("✓ **Spearman > 0.5**: ranking is reliable, search results trustworthy")
        elif spearman is not None and spearman < 0.3:
            status_lines.append("⚠️ **Spearman < 0.3**: Weak ranking ability, search results unreliable")
        if search_lift is not None and search_lift > 0.5:
            status_lines.append("✓ **Search Lift > 0.5σ**: Search moderately outperforms random selection")
        if ens_r2 is not None and eq_r2 is not None:
            if ens_r2 > 0.3 and eq_r2 > 0.3:
                status_lines.append("✓ **Both R² metrics strong**")
            elif ens_r2 > 0.3 and eq_r2 < 0.3:
                status_lines.append("⚠️ **Overall strong but Equal-Wt weak**")
        if status_lines:
            parts.extend(status_lines)
            parts.append("")

    if proxy_loss_stats:
        parts.append("## Proxy Experiment Loss Stats\n")
        parts.append("| Metric | Mean | Std | Min | Max |")
        parts.append("|:-------|:-----|:----|:----|:----|")
        for name, stats in proxy_loss_stats.items():
            if isinstance(stats, dict) and "mean" in stats:
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
        tasks = per_task_analysis.get("tasks", [])
        all_r2s = [t["r2"] for t in tasks]
        if all_r2s:
            mean_r2 = sum(all_r2s) / len(all_r2s)
            n_good = sum(1 for r in all_r2s if r > 0.3)
            parts.append("")
            parts.append(f"**Per-task R²:** mean {mean_r2:.4f} ({len(all_r2s)} tasks, {n_good} with R² > 0.3)")
        parts.append("")

    n_orig = sampling.get("num_original_docs", 0)
    n_sel = sampling.get("num_selected_docs", 0)
    parts += [
        "## 采样概览\n",
        f"- **原始文档数:** {n_orig:,}",
        f"- **最优采样文档数:** {n_sel:,}",
        f"- **采样比例 (docs):** {n_sel / max(1, n_orig):.4f}x",
        "",
        "---\n",
        "## Figure 1: 域分布对比\n",
        f"![]({fig1_file})\n",
        "---\n",
        "## Figure 2: 质量信号权重\n",
        f"![]({fig2_file})\n",
    ]

    proxy_dir = os.path.join(output_dir, "proxy_experiments")
    if os.path.isdir(proxy_dir):
        parts += ["---\n", _experiment_table(proxy_dir, output_dir, num_domains)]

    parts += ["---\n", f"*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"]
    report = "\n".join(parts)
    save_report(report, output_dir)


if __name__ == "__main__":
    main()
