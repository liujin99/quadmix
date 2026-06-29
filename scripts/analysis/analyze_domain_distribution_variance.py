"""
Analyze domain distribution variance across proxy experiments.

For each experiment, maps selected_indices.npy back to domain labels
and computes the domain distribution. Produces visualizations showing
how much (or how little) the domain mix varies across experiments.

Usage:
    python scripts/analysis/analyze_domain_distribution_variance.py \
        --exp-dir result/quadmix_20250620_120000/proxy_experiments \
        --preprocessed-dir /path/to/preprocessed \
        --output-dir result/analysis
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from quadmix.constants import DOMAIN_NAMES

DOMAIN_SHORT = [
    "Industrial", "Social", "Science", "Religion", "Philology",
    "Literature", "History", "General", "Philosophy", "Arts",
]

COLOR_ORIG = "#5B9BD5"
COLOR_OPT = "#ED7D31"
PALETTE = [
    "#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5",
    "#70AD47", "#264478", "#9B59B6", "#E74C3C", "#1ABC9C",
]


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


def load_domain_labels(preprocessed_dir: str) -> np.ndarray:
    print(f"[Load] Reading domain labels from {preprocessed_dir}")
    shard_files = sorted(Path(preprocessed_dir).glob("preprocessed_*.parquet"))
    if not shard_files:
        print(f"ERROR: No preprocessed_*.parquet found in {preprocessed_dir}")
        sys.exit(1)

    domain_chunks = []
    for sf in shard_files:
        df = pd.read_parquet(sf, columns=["domain"])
        domain_chunks.append(df["domain"].to_numpy(dtype=np.int64))

    labels = np.concatenate(domain_chunks)
    print(f"[Load] {len(labels):,} domain labels from {len(shard_files)} shards")
    return labels


def load_experiment_results(exp_dir: str):
    exp_dirs = sorted(Path(exp_dir).glob("exp_*"))
    if not exp_dirs:
        print(f"ERROR: No exp_* directories found in {exp_dir}")
        sys.exit(1)

    results = []
    skipped = 0
    for ed in exp_dirs:
        idx_path = ed / "selected_indices.npy"
        meta_path = ed / "meta.json"
        if not idx_path.exists() or not meta_path.exists():
            skipped += 1
            continue
        indices = np.load(idx_path)
        with open(meta_path) as f:
            meta = json.load(f)
        results.append({
            "experiment_id": meta.get("experiment_id", int(ed.name.split("_")[1])),
            "indices": indices,
            "val_loss": meta.get("val_loss", None),
            "training_docs": meta.get("training_docs", len(indices)),
        })

    print(f"[Load] {len(results)} experiments loaded, {skipped} skipped")
    return results


def compute_domain_distributions(results, domain_labels, num_domains):
    n_exp = len(results)
    dist_matrix = np.zeros((n_exp, num_domains), dtype=np.float64)
    count_matrix = np.zeros((n_exp, num_domains), dtype=np.int64)
    val_losses = np.zeros(n_exp, dtype=np.float64)
    doc_counts = np.zeros(n_exp, dtype=np.int64)

    for i, r in enumerate(results):
        idx = r["indices"]
        idx = idx[idx >= 0]
        idx = idx[idx < len(domain_labels)]
        domains = domain_labels[idx]
        counts = np.bincount(domains, minlength=num_domains)
        count_matrix[i] = counts
        total = counts.sum()
        if total > 0:
            dist_matrix[i] = counts / total
        val_losses[i] = r["val_loss"] if r["val_loss"] is not None else np.nan
        doc_counts[i] = total

    return dist_matrix, count_matrix, val_losses, doc_counts


def plot_boxplot_domain_ratios(dist_matrix, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    data = [dist_matrix[:, d] * 100 for d in range(dist_matrix.shape[1])]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.7)
    ax.set_xticklabels(DOMAIN_SHORT[:dist_matrix.shape[1]], rotation=45, ha="right")
    ax.set_ylabel("Domain Ratio (%)")
    ax.set_title("Domain Distribution Across All Experiments (Boxplot)")
    ax.grid(axis="y", alpha=0.3)

    stats_text = []
    for d in range(dist_matrix.shape[1]):
        col = dist_matrix[:, d] * 100
        stats_text.append(f"{DOMAIN_SHORT[d]}: {col.mean():.1f}% ± {col.std():.2f}%")
    ax.text(1.02, 0.98, "\n".join(stats_text), transform=ax.transAxes,
            fontsize=8, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    _save_fig(fig, output_dir, "fig_domain_boxplot.png")


def plot_heatmap_top_experiments(dist_matrix, val_losses, output_dir):
    sorted_idx = np.argsort(val_losses)
    n_show = min(50, len(sorted_idx))
    top_idx = sorted_idx[:n_show]
    bottom_idx = sorted_idx[-n_show:]

    show_idx = np.concatenate([top_idx, bottom_idx])
    heatmap_data = dist_matrix[show_idx] * 100

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(heatmap_data.shape[1]))
    ax.set_xticklabels(DOMAIN_SHORT[:heatmap_data.shape[1]], rotation=45, ha="right")
    ax.set_xlabel("Domain")
    ax.set_ylabel("Experiment (sorted by val_loss)")
    ax.set_title(f"Domain Distribution Heatmap (Top {n_show} best + Bottom {n_show} worst)")

    ax.axhline(y=n_show - 0.5, color="red", linewidth=2, linestyle="--")
    ax.text(heatmap_data.shape[1] + 0.5, n_show / 2, "Best", fontsize=9,
            verticalalignment="center", color="green")
    ax.text(heatmap_data.shape[1] + 0.5, n_show + n_show / 2, "Worst", fontsize=9,
            verticalalignment="center", color="red")

    plt.colorbar(im, ax=ax, label="Domain Ratio (%)")
    _save_fig(fig, output_dir, "fig_domain_heatmap.png")


def plot_std_across_domains(dist_matrix, output_dir):
    stds = dist_matrix.std(axis=0) * 100
    means = dist_matrix.mean(axis=0) * 100
    cv = stds / np.where(means > 0, means, 1) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.bar(range(len(stds)), stds, color=PALETTE[:len(stds)], alpha=0.8)
    ax1.set_xticks(range(len(stds)))
    ax1.set_xticklabels(DOMAIN_SHORT[:len(stds)], rotation=45, ha="right")
    ax1.set_ylabel("Std Dev (%)")
    ax1.set_title("Domain Ratio Std Dev Across Experiments")
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, stds):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    bars2 = ax2.bar(range(len(cv)), cv, color=PALETTE[:len(cv)], alpha=0.8)
    ax2.set_xticks(range(len(cv)))
    ax2.set_xticklabels(DOMAIN_SHORT[:len(cv)], rotation=45, ha="right")
    ax2.set_ylabel("CV (%)")
    ax2.set_title("Coefficient of Variation (Std/Mean × 100%)")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, cv):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    _save_fig(fig, output_dir, "fig_domain_std_cv.png")


def plot_variance_vs_val_loss(dist_matrix, val_losses, output_dir):
    valid = ~np.isnan(val_losses)
    var_per_exp = dist_matrix[valid].var(axis=1)
    losses = val_losses[valid]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(var_per_exp * 1e4, losses, alpha=0.3, s=10, color=COLOR_ORIG)
    ax.set_xlabel("Domain Distribution Variance (×10⁻⁴)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Domain Distribution Variance vs Validation Loss")
    ax.grid(alpha=0.3)

    if len(losses) > 2:
        corr = np.corrcoef(var_per_exp, losses)[0, 1]
        ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    _save_fig(fig, output_dir, "fig_variance_vs_val_loss.png")


def plot_doc_count_distribution(doc_counts, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(doc_counts, bins=50, color=COLOR_ORIG, alpha=0.7, edgecolor="white")
    ax1.set_xlabel("Number of Selected Documents")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Selected Doc Count per Experiment")
    ax1.axvline(doc_counts.mean(), color="red", linestyle="--", label=f"Mean: {doc_counts.mean():,.0f}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    cv = doc_counts.std() / doc_counts.mean() * 100
    ax2.hist(doc_counts / doc_counts.mean() * 100, bins=50, color=COLOR_ORIG, alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Doc Count / Mean Doc Count (%)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Normalized Doc Count (CV = {cv:.1f}%)")
    ax2.axvline(100, color="red", linestyle="--")
    ax2.grid(alpha=0.3)

    _save_fig(fig, output_dir, "fig_doc_count_distribution.png")


def plot_sampling_rate_per_domain(count_matrix, dist_matrix, output_dir):
    mean_counts = count_matrix.mean(axis=0)
    mean_ratios = dist_matrix.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(mean_ratios))
    width = 0.35
    bars1 = ax.bar(x - width/2, mean_ratios * 100, width, label="Mean Selected Ratio",
                   color=COLOR_ORIG, alpha=0.8)
    orig_ratio = mean_counts / mean_counts.sum()
    bars2 = ax.bar(x + width/2, orig_ratio * 100, width, label="Original Ratio",
                   color=COLOR_ORIG, alpha=0.4, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(DOMAIN_SHORT[:len(x)], rotation=45, ha="right")
    ax.set_ylabel("Ratio (%)")
    ax.set_title("Mean Selected Domain Ratio vs Original Domain Ratio")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    _save_fig(fig, output_dir, "fig_selected_vs_original_ratio.png")


def plot_pairwise_correlation(dist_matrix, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = np.corrcoef(dist_matrix.T)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(corr.shape[1]))
    ax.set_yticks(range(corr.shape[0]))
    ax.set_xticklabels(DOMAIN_SHORT[:corr.shape[1]], rotation=45, ha="right")
    ax.set_yticklabels(DOMAIN_SHORT[:corr.shape[0]])
    ax.set_title("Pairwise Correlation of Domain Ratios Across Experiments")

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr[i, j]) > 0.5 else "black")

    plt.colorbar(im, ax=ax)
    _save_fig(fig, output_dir, "fig_pairwise_correlation.png")


def save_summary_json(dist_matrix, count_matrix, val_losses, doc_counts, output_dir):
    num_domains = dist_matrix.shape[1]
    summary = {
        "num_experiments": len(val_losses),
        "num_domains": num_domains,
        "doc_count": {
            "mean": float(doc_counts.mean()),
            "std": float(doc_counts.std()),
            "min": int(doc_counts.min()),
            "max": int(doc_counts.max()),
            "cv_percent": float(doc_counts.std() / doc_counts.mean() * 100),
        },
        "domain_distribution": {},
        "overall_stats": {
            "mean_std_across_domains_percent": float(dist_matrix.std(axis=0).mean() * 100),
            "max_std_percent": float(dist_matrix.std(axis=0).max() * 100),
            "min_std_percent": float(dist_matrix.std(axis=0).min() * 100),
            "mean_cv_percent": float(
                (dist_matrix.std(axis=0) / np.where(dist_matrix.mean(axis=0) > 0, dist_matrix.mean(axis=0), 1)).mean() * 100
            ),
        },
    }

    for d in range(num_domains):
        col = dist_matrix[:, d] * 100
        name = DOMAIN_SHORT[d] if d < len(DOMAIN_SHORT) else f"D{d}"
        summary["domain_distribution"][name] = {
            "mean_percent": float(col.mean()),
            "std_percent": float(col.std()),
            "min_percent": float(col.min()),
            "max_percent": float(col.max()),
            "cv_percent": float(col.std() / col.mean() * 100) if col.mean() > 0 else 0,
        }

    path = os.path.join(output_dir, "domain_variance_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [Summary] Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze domain distribution variance across proxy experiments")
    parser.add_argument("--exp-dir", required=True, help="Path to proxy_experiments directory")
    parser.add_argument("--preprocessed-dir", required=True, help="Path to preprocessed parquet shards")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <exp-dir>/../analysis)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.exp_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    _setup_style()

    domain_labels = load_domain_labels(args.preprocessed_dir)
    num_domains = int(domain_labels.max()) + 1
    print(f"[Info] {num_domains} domains detected")

    results = load_experiment_results(args.exp_dir)
    dist_matrix, count_matrix, val_losses, doc_counts = compute_domain_distributions(
        results, domain_labels, num_domains
    )

    print("\n=== Domain Distribution Summary ===")
    for d in range(num_domains):
        col = dist_matrix[:, d] * 100
        name = DOMAIN_SHORT[d] if d < len(DOMAIN_SHORT) else f"D{d}"
        print(f"  {name:<15} {col.mean():6.2f}% ± {col.std():.3f}%  "
              f"[{col.min():.2f}%, {col.max():.2f}%]  CV={col.std()/col.mean()*100:.1f}%")

    print(f"\n  Doc count: {doc_counts.mean():,.0f} ± {doc_counts.std():,.0f}  "
          f"[{doc_counts.min():,}, {doc_counts.max():,}]")
    print(f"  Doc count CV: {doc_counts.std()/doc_counts.mean()*100:.1f}%")

    print("\n=== Generating Figures ===")
    plot_boxplot_domain_ratios(dist_matrix, args.output_dir)
    plot_heatmap_top_experiments(dist_matrix, val_losses, args.output_dir)
    plot_std_across_domains(dist_matrix, args.output_dir)
    plot_variance_vs_val_loss(dist_matrix, val_losses, args.output_dir)
    plot_doc_count_distribution(doc_counts, args.output_dir)
    plot_sampling_rate_per_domain(count_matrix, dist_matrix, args.output_dir)
    plot_pairwise_correlation(dist_matrix, args.output_dir)

    save_summary_json(dist_matrix, count_matrix, val_losses, doc_counts, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
