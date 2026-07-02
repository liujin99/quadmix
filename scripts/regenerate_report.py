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
            all_none = all(v is None for v in metrics.values())
            for k, v in metrics.items():
                if v is None:
                    parts.append(f"| {k} | — |")
                elif isinstance(v, float):
                    parts.append(f"| {k} | {v:.4f} |")
                else:
                    parts.append(f"| {k} | {v} |")
            if all_none:
                parts.append("")
                parts.append("> Per-task metrics are N/A: validation set has no task labels (aggregate-only mode).")
        if elapsed is not None:
            parts.append(f"| Duration | {elapsed:.1f}s |")
        parts.append("")

    if reliability:
        parts.append("## Model Reliability\n")
        parts.append("**Aggregate model** (single LightGBM on total loss, diagnostic only — not used for search):\n")
        bootstrap = reliability.get("bootstrap") or {}
        bootstrap_mean = bootstrap.get("mean")
        ci_lower = bootstrap.get("ci_lower")
        ci_upper = bootstrap.get("ci_upper")
        ci_std = bootstrap.get("ci_std")
        n_ensemble = bootstrap.get("n_ensemble_models")
        ci_width = (ci_upper - ci_lower) if (ci_lower is not None and ci_upper is not None) else None
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
        if ci_std is not None:
            parts.append(f"| CI Std | {ci_std:.4f} | — |")
        if n_ensemble is not None:
            parts.append(f"| Ensemble Models | {n_ensemble} | — |")
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
        eq_r2 = metrics.get("equal_weight_r2")
        eq_mae = metrics.get("equal_weight_mae")
        sp = metrics.get("spearman_corr")
        tk = metrics.get("top_k_recall")
        tk_val = metrics.get("top_k_value", 5)
        sl = metrics.get("search_lift")
        if ens_r2 is not None:
            parts.append("## Model Evaluation Metrics\n")
            _swm = config.get("search_weight_mode", "equal_weight")
            if _swm == "equal_weight":
                mode_label = "equal-weight"
            elif _swm == "r2_sigma_weighted":
                mode_label = "R²×σ-weighted"
            else:
                mode_label = "R²-weighted"
            search_desc = "Σ z_predᵢ / K" if mode_label == "equal-weight" else "Σ wᵢ·z_predᵢ"
            parts.append(f"**Search mode:** {mode_label} (optimizes {search_desc}, matches downstream goal)\n")
            parts.append("")
            parts.append("Four complementary metrics assess prediction and search quality:\n")
            parts.append("")
            parts.append("| Metric | Value | Formula | Purpose |")
            parts.append("|:-------|:------|:--------|:--------|")
            quality = "✓ Excellent" if ens_r2 > 0.6 else ("✓ Good" if ens_r2 > 0.3 else "⚠️ Weak")
            parts.append(f"| **Overall Val R²** | **{ens_r2:.4f}** ({quality}) | R²(Σ wᵢ·z_predᵢ, Σ wᵢ·z_actualᵢ) | Search objective quality |")
            parts.append(f"| Overall Val MAE | {ens_mae:.4f} | z-score space | — |")
            if eq_r2 is not None:
                eq_quality = "✓ Excellent" if eq_r2 > 0.6 else ("✓ Good" if eq_r2 > 0.3 else "⚠️ Weak")
                parts.append(f"| **Equal-Wt Val R²** | **{eq_r2:.4f}** ({eq_quality}) | R²((1/K)Σ z_predᵢ, (1/K)Σ z_actualᵢ) | Downstream goal quality |")
                parts.append(f"| Equal-Wt Val MAE | {eq_mae:.4f} | z-score space | — |")
            if sp is not None:
                sp_quality = "✓ Excellent" if sp > 0.7 else ("✓ Good" if sp > 0.5 else ("⚠️ Moderate" if sp > 0.3 else "⚠️ Weak"))
                parts.append(f"| **Spearman Rank Corr** | **{sp:.4f}** ({sp_quality}) | corr(rank(pred), rank(actual)) | Ranking ({mode_label}, auto-adapted) |")
            if tk is not None:
                tk_quality = "✓ Excellent" if tk > 0.7 else ("✓ Good" if tk > 0.5 else ("⚠️ Moderate" if tk > 0.3 else "⚠️ Weak"))
                parts.append(f"| **Top-{tk_val} Recall** | **{tk:.4f}** ({tk_quality}) | |pred_top_k ∩ actual_top_k| / k | Hit rate ({mode_label}, auto-adapted) |")
            if sl is not None:
                sl_quality = "✓ Excellent" if sl > 1.0 else ("✓ Good" if sl > 0.5 else ("⚠️ Moderate" if sl > 0.2 else "⚠️ Weak"))
                parts.append(f"| **Search Lift** | **{sl:.4f}** σ ({sl_quality}) | (μ_random - μ_search_top_k) / σ | vs random ({mode_label}, auto-adapted) |")
            parts.append("")
            parts.append("### Interpretation\n")
            parts.append("")
            parts.append("**Overall Val R²** (search objective):\n")
            parts.append("- Uses R²-weighted z-score combination: high-R² tasks contribute more\n")
            parts.append("- Reduces noise impact from low-signal tasks\n")
            parts.append("- Directly measures how well the search strategy's predictions match reality\n")
            parts.append("- High value → search will find good parameters\n")
            parts.append("")
            parts.append("**Equal-Wt Val R²** (downstream goal):\n")
            parts.append("- Uses equal-weight average in z-score space\n")
            parts.append("- Matches downstream evaluation (21 benchmarks equally weighted, normalized)\n")
            parts.append("- Diagnostic: shows prediction quality when all tasks treated equally\n")
            parts.append("")
            parts.append(f"**Spearman Rank Correlation** (ranking ability, {mode_label}):\n")
            parts.append("- Measures whether the model correctly ranks which parameters are better\n")
            parts.append(f"- Auto-adapts: both pred and actual use {mode_label} aggregation\n")
            parts.append("- Search only needs correct ranking, not accurate absolute values\n")
            parts.append("- R² can be low when loss variance is small, but ranking may still be effective\n")
            parts.append("- High value → model knows which parameters are better\n")
            parts.append("")
            parts.append(f"**Top-K Recall** (search hit rate, {mode_label}):\n")
            parts.append("- Fraction of search's top-K predictions that are actually in the top-K\n")
            parts.append(f"- Auto-adapts: both pred and actual use {mode_label} aggregation\n")
            parts.append("- Directly measures whether the selected parameters are good\n")
            parts.append("- Most practical metric: answers 'are my chosen parameters actually good?'\n")
            parts.append("")
            parts.append(f"**Search Lift** (search value vs random, {mode_label}):\n")
            parts.append("- How many standard deviations better search's top-K is compared to random selection\n")
            parts.append(f"- Auto-adapts: uses {mode_label} actual values for evaluation\n")
            parts.append("- Positive value means search finds better parameters than random\n")
            parts.append("- Most intuitive metric for stakeholders: answers 'how much better is search than random?'\n")
            parts.append("")
            if sp is not None and sp > 0.5:
                parts.append("✓ **Spearman > 0.5**: Ranking is reliable, search results trustworthy\n")
            elif sp is not None and sp > 0.3:
                parts.append("⚠️ **Spearman 0.3-0.5**: Moderate ranking ability, search may find decent parameters\n")
            elif sp is not None:
                parts.append("⚠️ **Spearman < 0.3**: Weak ranking ability, search results unreliable\n")
            if tk is not None and tk > 0.5:
                parts.append(f"✓ **Top-{tk_val} Recall > 0.5**: Search finds good parameters\n")
            elif tk is not None and tk > 0.3:
                parts.append(f"⚠️ **Top-{tk_val} Recall 0.3-0.5**: Search finds some good parameters\n")
            elif tk is not None:
                parts.append(f"⚠️ **Top-{tk_val} Recall < 0.3**: Search struggles to find good parameters\n")
            if sl is not None and sl > 1.0:
                parts.append(f"✓ **Search Lift > 1.0σ**: Search significantly outperforms random selection\n")
            elif sl is not None and sl > 0.5:
                parts.append(f"✓ **Search Lift > 0.5σ**: Search moderately outperforms random selection\n")
            elif sl is not None and sl > 0.2:
                parts.append(f"⚠️ **Search Lift 0.2-0.5σ**: Search slightly outperforms random\n")
            elif sl is not None:
                parts.append(f"⚠️ **Search Lift < 0.2σ**: Search provides minimal advantage over random\n")
            if ens_r2 > 0.3 and eq_r2 is not None and eq_r2 > 0.3:
                parts.append("✓ **Both R² metrics strong**: R²-weighting and equal-weight both effective in z-score space\n")
            elif ens_r2 > 0.3 and eq_r2 is not None and eq_r2 < 0.3:
                parts.append("⚠️ **Overall strong but Equal-Wt weak**: R²-weighting outperforms equal-weight, high-R² tasks carry more signal\n")
            elif ens_r2 < 0.3 and eq_r2 is not None and eq_r2 > 0.3:
                parts.append("⚠️ **Equal-Wt strong but Overall weak**: Equal-weight outperforms R²-weighting, consider using --search-mode equal_weight\n")
            else:
                parts.append("⚠️ **Both R² metrics weak**: But check Spearman/Top-K — ranking may still be effective\n")
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
    else:
        parts.append("## Per-Task Analysis\n")
        parts.append("> Not available: validation set has no task labels. Search uses aggregate model only.\n")
        n_active = per_task_analysis.get("n_active", 0)
        n_filtered = per_task_analysis.get("n_filtered", 0)
        r2_method = per_task_analysis.get("r2_method", "unknown")
        search_mode = per_task_analysis.get("search_weight_mode", config.get("search_weight_mode", ""))
        parts.append(f"**Active tasks:** {n_active} | **Filtered (R²≤0):** {n_filtered} | **R² method:** {r2_method} | **Search mode:** {search_mode}\n")
        tasks = per_task_analysis.get("tasks", [])
        has_train_r2 = any(t.get("train_r2") is not None for t in tasks)
        if has_train_r2:
            parts.append("| Task | Train R² | Val R² | Gap | Weight | Std | Status |")
            parts.append("|:-----|--------:|-------:|----:|-------:|----:|:-------|")
        else:
            parts.append("| Task | R² | Weight | Std | Status |")
            parts.append("|:-----|---:|-------:|----:|:-------|")
        for task in tasks:
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
            if has_train_r2:
                train_r2 = task.get("train_r2")
                gap = task.get("gap")
                train_str = f"{train_r2:.4f}" if train_r2 is not None else "—"
                gap_str = f"{gap:.4f}" if gap is not None else "—"
                parts.append(f"| {name} | {train_str} | {r2:.4f} | {gap_str} | {weight:.4f} | {std_str} | {status} |")
            else:
                parts.append(f"| {name} | {r2:.4f} | {weight:.4f} | {std_str} | {status} |")
        all_r2s = [t["r2"] for t in tasks]
        if all_r2s:
            mean_r2 = sum(all_r2s) / len(all_r2s)
            n_good = sum(1 for r in all_r2s if r > 0.3)
            parts.append("")
            parts.append(f"**Per-task R²:** mean {mean_r2:.4f} ({len(all_r2s)} tasks, {n_good} with R² > 0.3)")
            if has_train_r2:
                all_gaps = [t["gap"] for t in tasks if t.get("gap") is not None]
                if all_gaps:
                    mean_gap = sum(all_gaps) / len(all_gaps)
                    n_overfit = sum(1 for g in all_gaps if g > 0.3)
                    parts.append(f"**Overfit gap:** mean {mean_gap:.4f} ({n_overfit} tasks with gap > 0.3)")
            parts.append("")
            parts.append("Note: Per-task R² measures individual task prediction quality. Aggregate R² is lower due to correlated errors when averaging (mathematical property, not model failure).")
        parts.append("")

    dataset_size_prediction = summary.get("dataset_size_prediction")
    if dataset_size_prediction:
        parts.append("## Dataset Size Prediction\n")
        dsp = dataset_size_prediction
        parts.append("| Parameter | Value |")
        parts.append("|:----------|:------|")
        if dsp.get("total_tokens_est_B") is not None:
            parts.append(f"| Dataset Total Size | {dsp['total_tokens_est_B']:.1f}B tokens |")
        if dsp.get("omega_min") is not None:
            parts.append(f"| ω Range | [{dsp['omega_min']:.6f}, {dsp['omega_max']:.6f}] (avg {dsp['omega_avg']:.6f}) |")
        if dsp.get("estimated_tokens_B") is not None:
            parts.append(f"| Estimated Output | {dsp['estimated_tokens_B']:.2f}B tokens |")
        if dsp.get("target_tokens") is not None:
            parts.append(f"| Target | {dsp['target_tokens']/1e9:.1f}B tokens |")
        if dsp.get("note"):
            parts.append(f"| Note | {dsp['note']} |")
        parts.append("")

    stage_times_data = summary.get("stage_times")
    if stage_times_data:
        parts.append("## Stage Timing\n")
        parts.append("| Stage | Time (s) | Percentage |")
        parts.append("|:------|--------:|----------:|")
        total_time = sum(stage_times_data.values()) or 1
        for name, secs in sorted(stage_times_data.items(), key=lambda x: -x[1]):
            pct = secs / total_time * 100
            parts.append(f"| {name} | {secs:.1f} | {pct:.1f}% |")
        parts.append(f"| **Total** | **{total_time:.1f}** | **100%** |")
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
