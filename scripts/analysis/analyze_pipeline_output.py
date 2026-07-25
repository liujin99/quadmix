#!/usr/bin/env python3
"""Analyze QuaDMix pipeline output: quality score and rank distributions.

Generates three outputs in <exp-dir>/figures/:
  1. fig_quality_score_dist.png — full corpus q̄ distribution by domain (overlaid)
  2. fig_quality_rank_dist.png  — full corpus r̄ (solid) vs selected r̄ (dashed) by domain
  3. analysis_summary.txt       — key diagnostics (tie detection, selection stats, etc.)

The script recomputes merged quality scores and ranks for the FULL corpus using
the optimal parameters from the pipeline output, then compares the full corpus
distribution with the selected documents' distribution.

Usage:
  python scripts/analysis/analyze_pipeline_output.py \
      --exp-dir <pipeline_output> \
      --source-dir <source_data> \
      --schema configs/schema_stem.yaml \
      --seed 42
"""

import argparse
import json
import os
import sys

try:
    import quadmix  # noqa: F401
except ImportError:
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"),
    )

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quadmix.core.types import ParameterSet
from quadmix.core.quality_merger import compute_merged_quality_scores
from quadmix.core.quality_rank import compute_quality_ranks
from quadmix.data.dataset_schema import DatasetSchema
from quadmix.data.metadata_manager import ShardMetadataManager
from quadmix.pipeline.report import (
    _setup_style,
    _save_fig,
    _get_domain_short,
    _str_has_cjk,
    _CJK_FONT_AVAILABLE,
)


# ── CLI ──────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze QuaDMix pipeline output distributions."
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        help="Pipeline output directory (contains optimal_parameters.json, "
        "pipeline_summary.json, sampled_dataset.parquet)",
    )
    parser.add_argument(
        "--source-dir",
        default="/home/ma-user/work/100B_stem_parquet_filtered",
        help="Source data directory with parquet shards (default: stem path)",
    )
    parser.add_argument(
        "--schema",
        default="configs/schema_stem.yaml",
        help="Schema YAML file path (relative to project root or absolute)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for tie-breaking in rank computation (default: 42)",
    )
    return parser.parse_args()


# ── Loaders ──────────────────────────────────────────────────────


def load_optimal_params(params_path):
    with open(params_path) as f:
        data = json.load(f)
    return ParameterSet.from_dict(data["quality_weights"], data["sampling_params"])


def load_pipeline_summary(summary_path):
    with open(summary_path) as f:
        return json.load(f)


def resolve_schema_path(schema_arg):
    if os.path.isabs(schema_arg):
        return schema_arg
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(project_root, schema_arg)


# ── Plot helpers ─────────────────────────────────────────────────


def _get_colors(num_domains):
    cmap = plt.cm.tab10 if num_domains <= 10 else plt.cm.tab20
    return [cmap(i % (10 if num_domains <= 10 else 20)) for i in range(num_domains)]


def _get_top_domains(domain_labels, num_domains, top_n=6):
    counts = np.bincount(
        domain_labels[domain_labels >= 0], minlength=num_domains
    )
    return set(np.argsort(counts)[-top_n:].tolist())


def plot_quality_score_dist(
    merged_scores, domain_labels, domain_names, num_domains, output_dir
):
    """Figure 1: full corpus q̄ distribution by domain (overlaid)."""
    _setup_style()

    colors = _get_colors(num_domains)
    domain_short = _get_domain_short(num_domains, domain_names)
    top_domains = _get_top_domains(domain_labels, num_domains)

    fig, ax = plt.subplots(figsize=(10, 5))

    valid = domain_labels >= 0
    global_min = float(merged_scores[valid].min())
    global_max = float(merged_scores[valid].max())
    if global_max - global_min < 1e-15:
        global_max = global_min + 1.0
    bin_edges = np.linspace(global_min, global_max, 81)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    unique_domains = np.unique(domain_labels)
    unique_domains = unique_domains[unique_domains >= 0]

    for m in unique_domains:
        if m >= num_domains:
            continue
        mask = domain_labels == m
        scores = merged_scores[mask]
        counts, _ = np.histogram(scores, bins=bin_edges, density=True)

        if m in top_domains:
            ax.plot(
                bin_centers, counts, color=colors[m],
                label=domain_short[m], linewidth=1.5, alpha=0.85,
            )
            ax.fill_between(bin_centers, counts, alpha=0.12, color=colors[m])
        else:
            ax.plot(
                bin_centers, counts, color="lightgray",
                linewidth=0.8, alpha=0.5,
            )

    ax.set_xlabel("Merged Quality Score (q̄)")
    ax.set_ylabel("Density")
    ax.set_title("Quality Score Distribution by Domain (Full Corpus)")
    ax.legend(fontsize=8, loc="best", ncol=2 if num_domains > 6 else 1)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    return _save_fig(fig, output_dir, "fig_quality_score_dist.png")


def plot_quality_rank_dist(
    ranks,
    selected_ranks,
    domain_labels,
    selected_domain_labels,
    domain_names,
    num_domains,
    output_dir,
):
    """Figure 2: full corpus r̄ (solid) vs selected r̄ (dashed) by domain."""
    _setup_style()

    colors = _get_colors(num_domains)
    domain_short = _get_domain_short(num_domains, domain_names)
    top_domains = _get_top_domains(domain_labels, num_domains)

    fig, ax = plt.subplots(figsize=(10, 5))

    bin_edges = np.linspace(0, 1, 81)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    unique_domains = np.unique(domain_labels)
    unique_domains = unique_domains[unique_domains >= 0]

    for m in unique_domains:
        if m >= num_domains:
            continue
        mask = domain_labels == m
        full_ranks = ranks[mask]
        counts_full, _ = np.histogram(full_ranks, bins=bin_edges, density=True)

        if m in top_domains:
            color = colors[m]
            label = domain_short[m]
        else:
            color = "lightgray"
            label = None

        ax.plot(
            bin_centers, counts_full, color=color, label=label,
            linewidth=1.5, alpha=0.85,
        )

        sel_mask = selected_domain_labels == m
        sel_ranks = selected_ranks[sel_mask]
        if len(sel_ranks) > 1:
            counts_sel, _ = np.histogram(sel_ranks, bins=bin_edges, density=True)
            ax.plot(
                bin_centers, counts_sel, color=color, linestyle="--",
                linewidth=1.5, alpha=0.85,
            )

    ax.set_xlabel("Quality Rank (r̄)")
    ax.set_ylabel("Density")
    ax.set_title(
        "Quality Rank Distribution: Full Corpus (solid) vs Selected (dashed)"
    )
    ax.legend(fontsize=8, loc="best", ncol=2 if num_domains > 6 else 1)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    return _save_fig(fig, output_dir, "fig_quality_rank_dist.png")


# ── Summary writer ───────────────────────────────────────────────


def write_analysis_summary(
    output_path,
    args,
    summary,
    params,
    mgr,
    merged_scores,
    ranks,
    selected_doc_ids,
    selected_ranks,
    domain_labels,
    selected_domain_labels,
    domain_names,
    quality_names,
    num_domains,
    fig_score,
    fig_rank,
):
    """Write analysis_summary.txt with key diagnostics."""
    lines = []
    lines.append("=" * 70)
    lines.append("QuaDMix Pipeline Output Analysis")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Experiment dir : {args.exp_dir}")
    lines.append(f"Source dir     : {args.source_dir}")
    lines.append(f"Schema         : {args.schema}")
    lines.append(f"Seed (recomp.) : {args.seed}")
    lines.append("")

    # ── Optimal Parameters ──
    lines.append("-" * 70)
    lines.append("Optimal Parameters")
    lines.append("-" * 70)
    lines.append(f"Domains: {num_domains}")
    lines.append(f"Quality criteria: {params.num_criteria} ({quality_names})")
    lines.append("")

    domain_short = _get_domain_short(num_domains, domain_names)

    lines.append("Quality Weights (α):")
    dw = params.merge_config.domain_weights
    N = params.num_criteria
    for m in range(num_domains):
        weights = dw[m * N : (m + 1) * N]
        w_str = ", ".join(
            f"{quality_names[n]}={weights[n]:.4f}" for n in range(N)
        )
        lines.append(f"  {domain_short[m]:>12s}: {w_str}")
    lines.append("")

    lines.append("Sampling Params (λ, ω, η, ε):")
    for m in range(num_domains):
        sc = params.sampling_configs[m]
        lines.append(
            f"  {domain_short[m]:>12s}: "
            f"λ={sc.lambda_:.4f}, ω={sc.omega:.6f}, "
            f"η={sc.eta:.6f}, ε={sc.epsilon:.6f}"
        )
    lines.append("")

    # ── Sampling Statistics ──
    lines.append("-" * 70)
    lines.append("Sampling Statistics (from pipeline_summary.json)")
    lines.append("-" * 70)
    sampling = summary.get("sampling", {})
    n_orig = sampling.get("num_original_docs", mgr.num_docs)
    n_sel = sampling.get("num_selected_docs", len(selected_doc_ids))
    ratio = sampling.get("sampling_ratio", n_sel / max(1, n_orig))
    lines.append(f"Original docs   : {n_orig:,}")
    lines.append(f"Selected docs   : {n_sel:,}")
    lines.append(f"Sampling ratio  : {ratio:.6f}")

    dsp = summary.get("dataset_size_prediction", {})
    if dsp:
        lines.append(
            f"Total tokens est: {dsp.get('total_tokens_est_B', '?')}B"
        )
        lines.append(f"ω range         : [{dsp.get('omega_min', '?')}, {dsp.get('omega_max', '?')}]")
        lines.append(f"ω average       : {dsp.get('omega_avg', '?')}")
        lines.append(f"ε average       : {dsp.get('epsilon_avg', '?')}")
        lines.append(
            f"Estimated output: {dsp.get('estimated_tokens', '?')}"
        )
    lines.append("")

    # ── Quality Score (q̄) Distribution ──
    lines.append("-" * 70)
    lines.append("Quality Score (q̄) Distribution — Full Corpus")
    lines.append("-" * 70)
    header = (
        f"{'Domain':>12s} {'#docs':>12s} {'min':>10s} {'max':>10s} "
        f"{'mean':>10s} {'std':>10s} {'#unique':>10s} {'max_frac':>10s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    tie_warning_domains = []
    for m in range(num_domains):
        mask = domain_labels == m
        scores = merged_scores[mask]
        if len(scores) == 0:
            lines.append(f"  {domain_short[m]:>12s} {'0':>12s} (empty)")
            continue
        unique_scores = np.unique(scores)
        values, counts = np.unique(scores, return_counts=True)
        max_frac = counts.max() / len(scores) if len(scores) > 0 else 0
        lines.append(
            f"  {domain_short[m]:>12s} {len(scores):>12,} "
            f"{scores.min():>10.6f} {scores.max():>10.6f} "
            f"{scores.mean():>10.6f} {scores.std():>10.6f} "
            f"{len(unique_scores):>10,} {max_frac:>10.4%}"
        )
        if max_frac > 0.5:
            tie_warning_domains.append((domain_short[m], max_frac))

    lines.append("")

    # ── Quality Rank (r̄) Distribution ──
    lines.append("-" * 70)
    lines.append("Quality Rank (r̄) Distribution — Full Corpus vs Selected")
    lines.append("-" * 70)
    header = (
        f"{'Domain':>12s} {'#full':>12s} {'#sel':>8s} "
        f"{'full_mean':>10s} {'sel_mean':>10s} "
        f"{'%r>0.99':>8s} {'sel_ratio':>10s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for m in range(num_domains):
        mask = domain_labels == m
        full_r = ranks[mask]
        sel_mask = selected_domain_labels == m
        sel_r = selected_ranks[sel_mask]

        if len(full_r) == 0:
            lines.append(f"  {domain_short[m]:>12s} {'0':>12s} (empty)")
            continue

        pct_high = float((full_r > 0.99).sum() / len(full_r))
        sel_ratio = len(sel_r) / len(full_r) if len(full_r) > 0 else 0
        sel_mean = sel_r.mean() if len(sel_r) > 0 else float("nan")

        lines.append(
            f"  {domain_short[m]:>12s} {len(full_r):>12,} {len(sel_r):>8,} "
            f"{full_r.mean():>10.6f} {sel_mean:>10.6f} "
            f"{pct_high:>7.2%} {sel_ratio:>10.6f}"
        )

    lines.append("")

    # ── Tie-Breaking Diagnosis ──
    lines.append("-" * 70)
    lines.append("Tie-Breaking Diagnosis")
    lines.append("-" * 70)

    if tie_warning_domains:
        lines.append(
            "⚠ TIE PROBLEM DETECTED in the following domains "
            "(>50% docs share the same merged score):"
        )
        for name, frac in tie_warning_domains:
            lines.append(f"    {name}: {frac:.4%} of docs share the same q̄")
        lines.append("")
        lines.append(
            "This means quality scores have little discrimination power."
        )
        lines.append(
            "With the tie-breaking fix (1e-12 noise), tied docs get ranks"
        )
        lines.append(
            "spread across the tie group instead of collapsed to r̄≈1.0."
        )
        lines.append(
            "Check if the fix is applied in quality_rank.py (seed param)."
        )
    else:
        lines.append("✓ No severe tie problem detected (all domains <50% same score).")
    lines.append("")

    # ── Selection Analysis ──
    lines.append("-" * 70)
    lines.append("Selection Analysis (is selection quality-based?)")
    lines.append("-" * 70)

    quality_based_count = 0
    for m in range(num_domains):
        mask = domain_labels == m
        full_r = ranks[mask]
        sel_mask = selected_domain_labels == m
        sel_r = selected_ranks[sel_mask]

        if len(full_r) == 0 or len(sel_r) == 0:
            continue

        full_mean = full_r.mean()
        sel_mean = sel_r.mean()
        is_quality_based = sel_mean < full_mean
        if is_quality_based:
            quality_based_count += 1
            status = "✓ quality-based (sel r̄ < full r̄)"
        else:
            status = "✗ NOT quality-based (sel r̄ ≥ full r̄)"
        lines.append(
            f"  {domain_short[m]:>12s}: full r̄={full_mean:.4f}, "
            f"sel r̄={sel_mean:.4f} → {status}"
        )

    lines.append("")
    lines.append(
        f"Quality-based selection: {quality_based_count}/{num_domains} domains"
    )
    lines.append("")

    # ── CJK Font Note ──
    if not _CJK_FONT_AVAILABLE and domain_names is not None:
        has_cjk = any(_str_has_cjk(n) for n in domain_names[:num_domains])
        if has_cjk:
            lines.append("-" * 70)
            lines.append("Font Note")
            lines.append("-" * 70)
            lines.append(
                "No CJK font found. PNG figures use D{i} labels. Mapping:"
            )
            domain_disp = domain_names[:num_domains]
            for i in range(min(num_domains, len(domain_short), len(domain_disp))):
                if domain_short[i] != domain_disp[i]:
                    lines.append(f"  {domain_short[i]} = {domain_disp[i]}")
            lines.append("")

    # ── Figures ──
    lines.append("-" * 70)
    lines.append("Generated Figures")
    lines.append("-" * 70)
    lines.append(f"  1. {fig_score}")
    lines.append(f"  2. {fig_rank}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("End of Analysis")
    lines.append("=" * 70)

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ─────────────────────────────────────────────────────────


def main():
    args = parse_args()

    params_path = os.path.join(args.exp_dir, "optimal_parameters.json")
    summary_path = os.path.join(args.exp_dir, "pipeline_summary.json")
    sampled_path = os.path.join(args.exp_dir, "sampled_dataset.parquet")

    print("=== QuaDMix Pipeline Output Analysis ===\n")

    # ── Load pipeline outputs ──
    print(f"[1/5] Loading optimal parameters: {params_path}")
    params = load_optimal_params(params_path)

    print(f"[2/5] Loading pipeline summary: {summary_path}")
    summary = load_pipeline_summary(summary_path)
    normalizer = summary.get("config", {}).get("normalizer", "rank")
    print(f"       Normalizer: {normalizer}")

    print(f"[3/5] Loading sampled dataset: {sampled_path}")
    sampled_df = pd.read_parquet(sampled_path)
    selected_doc_ids = sampled_df["doc_id"].to_numpy(dtype=np.int64)
    print(f"       Selected docs: {len(selected_doc_ids):,}")

    # ── Load full corpus metadata ──
    schema_path = resolve_schema_path(args.schema)
    print(f"[4/5] Loading full corpus metadata from: {args.source_dir}")
    print(f"       Schema: {schema_path}")
    schema = DatasetSchema.from_yaml(schema_path)
    mgr = ShardMetadataManager(args.source_dir, schema)

    domain_labels = mgr.domain_labels
    num_domains = mgr.num_domains
    domain_names = mgr.detected_domain_names
    quality_names = mgr.detected_quality_names

    print(f"       Total docs: {mgr.num_docs:,}")
    print(f"       Domains: {num_domains} ({domain_names})")

    # ── Recompute merged scores and ranks ──
    print(f"[5/5] Recomputing merged quality scores (Eq.1)...")
    merged_scores = compute_merged_quality_scores(
        mgr.quality_scores,
        domain_labels,
        params.merge_config,
        normalizer=normalizer,
    )

    print(f"       Computing quality ranks (Eq.2)...")
    token_counts = mgr.estimate_token_counts()
    ranks = compute_quality_ranks(
        merged_scores, domain_labels, token_counts, seed=args.seed
    )

    selected_ranks = ranks[selected_doc_ids]
    selected_domain_labels = domain_labels[selected_doc_ids]

    # ── Generate figures ──
    figures_dir = os.path.join(args.exp_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"\nGenerating figures in: {figures_dir}")
    fig_score = plot_quality_score_dist(
        merged_scores, domain_labels, domain_names, num_domains, figures_dir
    )
    fig_rank = plot_quality_rank_dist(
        ranks,
        selected_ranks,
        domain_labels,
        selected_domain_labels,
        domain_names,
        num_domains,
        figures_dir,
    )

    # ── Generate analysis summary ──
    print("\nGenerating analysis summary...")
    summary_out = os.path.join(figures_dir, "analysis_summary.txt")
    write_analysis_summary(
        summary_out,
        args,
        summary,
        params,
        mgr,
        merged_scores,
        ranks,
        selected_doc_ids,
        selected_ranks,
        domain_labels,
        selected_domain_labels,
        domain_names,
        quality_names,
        num_domains,
        fig_score,
        fig_rank,
    )
    print(f"  Saved: {summary_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
