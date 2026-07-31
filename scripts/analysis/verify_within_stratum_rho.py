#!/usr/bin/env python3
"""Verify within-stratum quality-length ρ to assess stratification effectiveness.

If quality-length correlation is driven by between-stratum differences (long
documents have higher average quality), splitting by length and ranking
within each stratum will dramatically reduce ρ.  If the correlation persists
within each stratum, stratification will not help.

Generates outputs in <exp-dir>:
  - within_stratum_rho.txt        — full comparison table + verdict
  - fig_within_stratum_rho.png    — grouped bar (global vs within) + heatmap

Usage:
  python scripts/analysis/verify_within_stratum_rho.py \
      --exp-dir <pipeline_output> \
      --source-dir <source_data> \
      --schema configs/schema_stem.yaml \
      --n-strata 4
"""

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

try:
    import quadmix  # noqa: F401
except ImportError:
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"),
    )

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quadmix.data.dataset_schema import DatasetSchema
from quadmix.data.metadata_manager import ShardMetadataManager
from quadmix.pipeline.report import _setup_style, _save_fig


# ── CLI ──────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify within-stratum quality-length ρ."
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        help="Experiment output directory (figures/txt saved here).",
    )
    parser.add_argument(
        "--source-dir",
        default="/home/ma-user/work/100B_stem_parquet_filtered",
        help="Source data directory with parquet shards.",
    )
    parser.add_argument(
        "--schema",
        default="configs/schema_stem.yaml",
        help="Schema YAML file path (relative to project root or absolute).",
    )
    parser.add_argument(
        "--n-strata",
        type=int,
        default=4,
        help="Number of length strata (default: 4 = quartiles).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers (default: -1 = all cores).",
    )
    return parser.parse_args()


def resolve_schema_path(schema_arg):
    if os.path.isabs(schema_arg):
        return schema_arg
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(project_root, schema_arg)


# ── Spearman ρ (self-contained copy from analyze_pipeline_output.py) ──


def _spearman_rho_vs_fixed(quality_scores, y, n_criteria, n_jobs):
    """Spearman ρ of each quality criterion vs a FIXED y (e.g. char_counts).

    Optimized: rank of y computed once, each x ranked with single argsort,
    N_criteria columns ranked in parallel via threads.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 2:
        return np.zeros(n_criteria, dtype=np.float64)
    order_y = np.argsort(y, kind="quicksort")
    ry = np.empty(n, dtype=np.float64)
    ry[order_y] = np.arange(n, dtype=np.float64)
    ry_m = ry - ry.mean()
    ry_ss = float(np.sum(ry_m ** 2))

    def _rho_col(k):
        x = np.asarray(quality_scores[:, k], dtype=np.float64)
        if len(x) < 2:
            return 0.0
        order_x = np.argsort(x, kind="quicksort")
        rx = np.empty(n, dtype=np.float64)
        rx[order_x] = np.arange(n, dtype=np.float64)
        rx_m = rx - rx.mean()
        den = (float(np.sum(rx_m ** 2)) * ry_ss) ** 0.5
        return float(np.sum(rx_m * ry_m) / den) if den > 0 else 0.0

    effective = n_jobs if n_jobs != -1 else (os.cpu_count() or 1)
    if effective > 1 and n_criteria > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=min(effective, n_criteria), prefer="threads")(
            delayed(_rho_col)(k) for k in range(n_criteria)
        )
        return np.array(results, dtype=np.float64)
    return np.array([_rho_col(k) for k in range(n_criteria)], dtype=np.float64)


# ── Plot ─────────────────────────────────────────────────────────


def plot_within_stratum_rho(
    global_rhos,
    within_rhos_matrix,
    quality_names,
    n_strata,
    stratum_bounds,
    output_dir,
):
    """Two-panel figure: grouped bar (global vs within) + per-stratum heatmap.

    Parameters
    ----------
    global_rhos : array (n_criteria,)
    within_rhos_matrix : array (n_strata, n_criteria)
    quality_names : list of str
    n_strata : int
    stratum_bounds : list of (lo, hi) tuples
    output_dir : str
    """
    n_criteria = len(quality_names)
    avg_within = within_rhos_matrix.mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    # ── Panel 1: Grouped bar ──
    x = np.arange(n_criteria)
    width = 0.35
    ax1.bar(
        x - width / 2, global_rhos, width, label="Global ρ",
        color="#4472C4", edgecolor="white", linewidth=0.5,
    )
    ax1.bar(
        x + width / 2, avg_within, width, label="Avg within-stratum ρ",
        color="#ED7D31", edgecolor="white", linewidth=0.5,
    )
    ax1.axhline(y=0, color="gray", linewidth=0.8, linestyle="-")
    ax1.set_xticks(x)
    ax1.set_xticklabels(quality_names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Spearman ρ (quality vs length)")
    ax1.set_title("Global vs Within-Stratum ρ")
    ax1.legend(fontsize=9, loc="upper right")

    for i in range(n_criteria):
        g = global_rhos[i]
        w = avg_within[i]
        if abs(g) > 1e-8:
            reduction = (1 - abs(w) / abs(g)) * 100
            txt = f"-{reduction:.0f}%" if reduction > 0 else f"+{abs(reduction):.0f}%"
            color = "green" if reduction > 0 else "red"
            ax1.text(
                i, max(g, w) + 0.02, txt,
                ha="center", fontsize=8, color=color, fontweight="bold",
            )

    # ── Panel 2: Heatmap ──
    vmax = max(abs(global_rhos).max(), abs(within_rhos_matrix).max())
    im = ax2.imshow(
        within_rhos_matrix, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
    )

    stratum_labels = []
    for k in range(n_strata):
        lo, hi = stratum_bounds[k]
        lo_str = f"{int(lo)}" if lo > 0 else "0"
        hi_str = f"{int(hi)}" if hi < 1e9 else "∞"
        stratum_labels.append(f"Q{k + 1}\n[{lo_str}, {hi_str}]")

    ax2.set_xticks(np.arange(n_strata))
    ax2.set_xticklabels(stratum_labels, fontsize=9)
    ax2.set_yticks(np.arange(n_criteria))
    ax2.set_yticklabels(quality_names, fontsize=9)
    ax2.set_title("Per-Stratum ρ (short → long)")

    for k in range(n_strata):
        for j in range(n_criteria):
            val = within_rhos_matrix[k, j]
            color = "white" if abs(val) > 0.3 * vmax else "black"
            ax2.text(
                k, j, f"{val:+.3f}",
                ha="center", va="center", fontsize=8, color=color,
            )

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontsize=9)

    fig.suptitle(
        "Quality-Length ρ: Global vs Within-Stratum",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    return _save_fig(fig, output_dir, "fig_within_stratum_rho.png")


# ── Main ─────────────────────────────────────────────────────────


def main():
    args = parse_args()

    print("=== Within-Stratum Quality-Length ρ Verification ===\n")

    # ── Load metadata ──
    schema_path = resolve_schema_path(args.schema)
    print(f"Loading schema: {schema_path}")
    schema = DatasetSchema.from_yaml(schema_path)

    print(f"Loading metadata from: {args.source_dir}")
    mgr = ShardMetadataManager(args.source_dir, schema)

    char_counts = mgr.doc_char_counts
    quality_scores = mgr.quality_scores
    n_criteria = mgr.num_quality_criteria
    quality_names = mgr.detected_quality_names

    n_docs = mgr.num_docs
    print(f"  Total docs: {n_docs:,}")
    print(f"  Quality criteria: {n_criteria} ({quality_names})")
    print(f"  Strata: {args.n_strata}")

    if char_counts is None or len(char_counts) == 0:
        print("ERROR: doc_char_counts not available.")
        sys.exit(1)
    if quality_scores is None or quality_scores.shape[0] == 0:
        print("ERROR: quality_scores not available.")
        sys.exit(1)

    # ── Global ρ ──
    print(f"\nComputing global ρ...", flush=True)
    global_rhos = _spearman_rho_vs_fixed(
        quality_scores, char_counts, n_criteria, args.n_jobs
    )
    print(f"  Global ρ: {global_rhos}")

    # ── Split into strata by char_count quantiles ──
    K = args.n_strata
    quantiles = np.linspace(0, 1, K + 1)
    boundaries = np.quantile(char_counts.astype(np.float64), quantiles)

    print(f"\nStratum boundaries (char_count):")
    strata_masks = []
    stratum_bounds = []
    for k in range(K):
        lo = boundaries[k]
        hi = boundaries[k + 1]
        if k == 0:
            mask = char_counts <= hi
        elif k == K - 1:
            mask = char_counts > lo
        else:
            mask = (char_counts > lo) & (char_counts <= hi)
        strata_masks.append(mask)
        stratum_bounds.append((lo, hi))
        n = int(mask.sum())
        print(
            f"  Q{k + 1}: [{lo:.0f}, {hi:.0f}]"
            f" — {n:,} docs ({n / n_docs * 100:.1f}%)"
        )

    # ── Within-stratum ρ ──
    print(f"\nComputing within-stratum ρ...", flush=True)
    within_rhos = np.zeros((K, n_criteria), dtype=np.float64)
    for k in range(K):
        mask = strata_masks[k]
        qs_k = quality_scores[mask]
        cc_k = char_counts[mask]
        within_rhos[k] = _spearman_rho_vs_fixed(
            qs_k, cc_k, n_criteria, args.n_jobs
        )
        print(f"  Q{k + 1} ρ: {within_rhos[k]}")

    avg_within = within_rhos.mean(axis=0)

    # ── Summary table ──
    lines = []
    lines.append("=" * 70)
    lines.append("Within-Stratum Quality-Length ρ Verification")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Strata: {K} (by char_count quantiles)")
    lines.append(f"Total docs: {n_docs:,}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("Stratum Boundaries (char_count)")
    lines.append("-" * 70)
    for k in range(K):
        lo, hi = stratum_bounds[k]
        lo_str = f"{int(lo)}" if lo > 0 else "0"
        hi_str = f"{int(hi)}" if hi < 1e9 else "∞"
        n = int(strata_masks[k].sum())
        lines.append(
            f"  Q{k + 1}: [{lo_str:>8s}, {hi_str:>8s}]"
            f" — {n:>12,} docs ({n / n_docs * 100:.1f}%)"
        )
    lines.append("")

    lines.append("-" * 70)
    lines.append("Global vs Within-Stratum ρ")
    lines.append("-" * 70)
    header = (
        f"  {'Dimension':<20s} {'Global':>8s}"
        f" {'Within(avg)':>12s} {'Reduction':>10s} {'Verdict':>12s}"
    )
    lines.append(header)
    lines.append(f"  {'-' * 66}")
    for j, qname in enumerate(quality_names):
        g = global_rhos[j]
        w = avg_within[j]
        if abs(g) > 1e-8:
            reduction = (1 - abs(w) / abs(g)) * 100
        else:
            reduction = 0.0
        if reduction > 60:
            verdict = "effective"
        elif reduction > 30:
            verdict = "partial"
        else:
            verdict = "ineffective"
        lines.append(
            f"  {qname:<20s} {g:>+8.4f} {w:>+12.4f}"
            f" {reduction:>9.1f}% {verdict:>12s}"
        )
    lines.append("")

    lines.append("-" * 70)
    lines.append("Per-Stratum Detail")
    lines.append("-" * 70)
    header = f"  {'Dimension':<20s}" + "".join(
        f"{'Q' + str(k + 1):>10s}" for k in range(K)
    )
    lines.append(header)
    lines.append(f"  {'-' * (20 + 10 * K)}")
    for j, qname in enumerate(quality_names):
        row = f"  {qname:<20s}" + "".join(
            f"{within_rhos[k, j]:>+10.4f}" for k in range(K)
        )
        lines.append(row)
    lines.append("")

    # ── Overall verdict ──
    avg_global = float(np.mean(np.abs(global_rhos)))
    avg_within_abs = float(np.mean(np.abs(avg_within)))
    overall_reduction = (
        (1 - avg_within_abs / avg_global) * 100 if avg_global > 1e-8 else 0.0
    )

    lines.append("-" * 70)
    lines.append("Overall Verdict")
    lines.append("-" * 70)
    lines.append(f"  Avg |global ρ|   = {avg_global:.4f}")
    lines.append(f"  Avg |within ρ|  = {avg_within_abs:.4f}")
    lines.append(f"  Reduction        = {overall_reduction:.1f}%")
    lines.append("")
    if overall_reduction > 60:
        lines.append(
            "  => Stratification EFFECTIVE. Within-stratum ρ << global ρ."
        )
        lines.append(
            "     Length-stratified sampling (A1) is expected to significantly"
        )
        lines.append("     reduce quality-length bias. Proceed with implementation.")
    elif overall_reduction > 30:
        lines.append(
            "  => Stratification PARTIALLY effective. Within-stratum ρ < global ρ"
        )
        lines.append(
            "     but non-negligible. A1 may help but effect could be limited."
        )
    else:
        lines.append(
            "  => Stratification INEFFECTIVE. Within-stratum ρ ≈ global ρ."
        )
        lines.append(
            "     Quality-length correlation is not driven by between-stratum"
        )
        lines.append("     differences. Need alternative approach.")
    lines.append("")
    lines.append("=" * 70)

    text = "\n".join(lines)
    print(text)

    out_path = os.path.join(args.exp_dir, "within_stratum_rho.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved: {out_path}")

    # ── Figure ──
    print("\nGenerating figure...")
    _setup_style()
    fig_name = plot_within_stratum_rho(
        global_rhos, within_rhos, quality_names, K,
        stratum_bounds, args.exp_dir,
    )
    print(f"Done. Figure: {fig_name}")


if __name__ == "__main__":
    main()
