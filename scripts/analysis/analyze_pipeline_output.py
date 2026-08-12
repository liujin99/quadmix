#!/usr/bin/env python3
"""Analyze QuaDMix pipeline output: quality score and rank distributions.

Generates outputs directly in <exp-dir> (the experiment result directory):
  - fig_quality_score_dist.png — full corpus q̄ distribution by domain (overlaid)
  - fig_quality_rank_dist.png  — full corpus r̄ (solid) vs selected r̄ (dashed) by domain
  - fig_duplication_analysis.png — per-domain unique vs duplicate docs + sampling-value buckets
  - fig_quality_length_decomposition.png — weight×ρ(length) stacked bar + sampling ω/S_max
  - fig_token_length_dist.png — token length distribution per domain with T marker (needs --tokenizer)
  - fig_crop_analysis.png      — total vs trained vs cropped tokens + crop rate decomposition
  - fig_packing_boundaries.png — doc boundaries per packed row (BOS-bestfit simulation)
  - analysis_summary.txt       — key diagnostics: tie detection, selection stats,
                                  quality-length ρ decomposition (which quality
                                  dimensions drive length bias via merge weights),
                                  sampling aggressiveness interpretation
                                  (ω/λ/η/ε translated to top%×oversampling),
                                  and token length & crop waste analysis

Optional (when <exp-dir>/proxy_experiments/ exists):
  - proxy_val_loss_analysis.txt — hard-vs-easy val_loss analysis: is reverse-optimization (max val_loss) safe?
  - optimizer_domain_analysis.txt — per-domain proportion vs val_loss + LightGBM feature importance
  - fig_optimizer_domain_vs_loss.png — scatter: domain proportion vs val_loss per domain

The script recomputes merged quality scores and ranks for the FULL corpus using
the optimal parameters from the pipeline output, then compares the full corpus
distribution with the selected documents' distribution.

Token-length & crop analysis requires a nanochat tokenizer (--tokenizer or
$NANOCHAT_MODEL_DIR/tokenizer). When available, the script samples up to
--tokenize-sample docs from the sampled parquet, tokenizes them via
multiprocessing, and simulates BOS-bestfit packing to estimate the crop rate
(tokens discarded by the dataloader's crop-and-discard behavior).

Usage:
  python scripts/analysis/analyze_pipeline_output.py \\
      --exp-dir <pipeline_output> \\
      --source-dir <source_data> \\
      --schema configs/schema_stem.yaml \\
      --seed 42 \\
      --proxy-hard-easy-count 20 \\
      --tokenizer /path/to/tokenizer \\
      --seq-len 2048 \\
      --tokenize-sample 50000
"""

import argparse
import json
import os
import pickle
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
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quadmix.core.types import ParameterSet
from quadmix.core.quality_merger import compute_merged_quality_scores
from quadmix.core.quality_rank import compute_quality_ranks
from quadmix.data.dataset_schema import DatasetSchema
from quadmix.data.metadata_manager import ShardMetadataManager
from quadmix.pipeline import report as _report_mod
from quadmix.pipeline.report import (
    setup_style,
    save_fig,
    get_domain_short,
    str_has_cjk,
)
from quadmix.sampling.batch_sampler import resolve_parquet_source
from analysis._common.tokenize_helpers import init_tok_worker, tokenize_lens, tokenize_batch, resolve_tokenizer_path
from analysis._common.stats_helpers import spearman, spearman_rho_vs_fixed


# ── CLI ──────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze QuaDMix pipeline output distributions."
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        help="Pipeline output directory (contains optimal_parameters.json, "
        "pipeline_summary.json, sampled_dataset/)",
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
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for score/rank computation "
        "(default: -1 = all CPU cores)",
    )
    parser.add_argument(
        "--proxy-hard-easy-count",
        type=int,
        default=20,
        help="Number of experiments at EACH end of the proxy val_loss ranking to "
        "form the 'hard' (highest val_loss) vs 'easy' (lowest val_loss) "
        "comparison groups. Only used when <exp-dir>/proxy_experiments/ exists. "
        "Larger = stabler means but weaker contrast; smaller = sharper contrast "
        "but noisier. Default: 20 (≈6%% of 336 experiments).",
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Baseline pipeline output directory for comparison. "
        "Used for: (1) diversity tradeoff figure (baseline theta* marker), "
        "(2) token length distribution figure (baseline histogram overlay). "
        "Must contain sampled_dataset/ or sampled_dataset.parquet.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="nanochat tokenizer.pkl or its dir (enables token-length & crop "
        "analysis; default: $NANOCHAT_MODEL_DIR/tokenizer)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=2048,
        help="training context length in tokens (default: 2048)",
    )
    parser.add_argument(
        "--pack-buffer", type=int, default=2000,
        help="dataloader packing buffer size (default: 2000, same as mid_train.py)",
    )
    parser.add_argument(
        "--max-pack-docs", type=int, default=20000,
        help="max docs sampled for packing simulation (default: 20000)",
    )
    parser.add_argument(
        "--tokenize-sample", type=int, default=50000,
        help="number of docs to tokenize for length distribution (0 = all, "
        "default: 50000)",
    )
    parser.add_argument(
        "--tokenizer-threads", type=int, default=1,
        help="threads per tokenizing worker (default: 1)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="multiprocessing workers for tokenization (default: min(32, cpu_count))",
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


# ── Packing simulation (BOS-bestfit, mirrors nanochat dataloader) ─


def _packing_crop_analysis(tok_lens, seq_len=2048, buffer_size=2000, max_docs=20000):
    """Simulate BOS-bestfit packing; return crop statistics.

    Mirrors nanochat dataloader's _bos_bestfit algorithm:
    1. Buffer N documents (up to buffer_size)
    2. For each row, greedily pick the LARGEST doc that fits entirely
    3. When nothing fits, crop the SHORTEST doc's tail to fill remaining space

    Returns dict with total_tokens, trained_tokens, cropped_tokens, crop_rate,
    n_rows, boundaries, docs_over_T, tokens_over_T, theoretical_min_waste,
    additional_waste, seq_len, buffer_size, sample_size.
    """
    if tok_lens is None or len(tok_lens) == 0:
        return None
    a = np.asarray(tok_lens)
    if len(a) > max_docs:
        step = max(1, len(a) // max_docs)
        a = a[::step][:max_docs]
    row_capacity = seq_len + 1
    buf = []
    boundaries = []
    total_tokens = int(a.sum())
    trained_tokens = 0
    cropped_tokens = 0
    docs_over_T = int(np.sum(a > row_capacity))
    tokens_over_T = int(np.maximum(a - row_capacity, 0).sum())
    i = 0
    N = len(a)
    while i < N or buf:
        pos = 0
        docs_in_row = 0
        while pos < row_capacity and (i < N or buf):
            while len(buf) < buffer_size and i < N:
                buf.append(int(a[i]))
                i += 1
            if not buf:
                break
            remaining = row_capacity - pos
            best_idx, best_len = -1, 0
            for k in range(len(buf)):
                dl = buf[k]
                if dl <= remaining and dl > best_len:
                    best_idx = k
                    best_len = dl
            if best_idx >= 0:
                buf.pop(best_idx)
                pos += best_len
                trained_tokens += best_len
            else:
                sk = min(range(len(buf)), key=lambda x: buf[x])
                doc_len = buf.pop(sk)
                trained_tokens += remaining
                cropped_tokens += doc_len - remaining
                pos += remaining
            docs_in_row += 1
        boundaries.append(docs_in_row - 1)
    crop_rate = cropped_tokens / max(total_tokens, 1)
    theo_min = tokens_over_T / max(total_tokens, 1)
    return {
        "total_tokens": total_tokens,
        "trained_tokens": trained_tokens,
        "cropped_tokens": cropped_tokens,
        "crop_rate": crop_rate,
        "n_rows": len(boundaries),
        "boundaries": np.array(boundaries, dtype=np.int32),
        "docs_over_T": docs_over_T,
        "tokens_over_T": tokens_over_T,
        "theoretical_min_waste": theo_min,
        "additional_waste": crop_rate - theo_min,
        "seq_len": seq_len,
        "buffer_size": buffer_size,
        "sample_size": len(a),
    }


# ── Tokenization driver ──────────────────────────────────────────


def _tokenize_sampled_docs(sampled_df, tokenizer_path, sample_size,
                           num_workers, tok_threads, domain_col, domain_names):
    """Sample docs from sampled_df, tokenize them, return (tok_lens, domain_labels).

    Samples up to sample_size docs (random, seed=42) from the sampled parquet,
    tokenizes them via multiprocessing, and returns token lengths plus per-doc
    domain labels (integer codes 0..M-1).
    """
    n_total = len(sampled_df)
    if "text" not in sampled_df.columns:
        print("  [warn] sampled_dataset has no 'text' column — skipping token analysis")
        return None, None
    if sample_size > 0 and sample_size < n_total:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_total, size=sample_size, replace=False)
        indices.sort()
    else:
        indices = np.arange(n_total)
    texts_all = sampled_df["text"].to_numpy()[indices]
    domain_labels = None
    if domain_col and domain_col in sampled_df.columns:
        raw = sampled_df[domain_col].to_numpy()[indices]
        if raw.dtype.kind in ("i", "u"):
            domain_labels = raw.astype(np.int64)
        elif raw.dtype.kind in ("U", "S", "O"):
            label_map = {}
            if domain_names:
                for i, name in enumerate(domain_names):
                    label_map[str(name)] = i
                    label_map[i] = i
            domain_labels = np.array(
                [label_map.get(str(d), label_map.get(int(d), -1)
                                if isinstance(d, (int, np.integer)) and not isinstance(d, bool)
                                else -1)
                 for d in raw],
                dtype=np.int64,
            )

    from multiprocessing import Pool
    from tqdm import tqdm

    batch_size = max(256, len(texts_all) // (num_workers * 4))
    tasks = [
        (i, texts_all[i:i + batch_size].tolist())
        for i in range(0, len(texts_all), batch_size)
    ]
    parts = []
    with Pool(num_workers, initializer=init_tok_worker,
              initargs=(tokenizer_path, tok_threads)) as pool:
        for idx, lens in tqdm(
            pool.imap_unordered(tokenize_batch, tasks, chunksize=1),
            total=len(tasks), desc="  tokenize", leave=False,
        ):
            parts.append((idx, lens))
    parts.sort(key=lambda x: x[0])
    tok_lens = np.concatenate([lens for _, lens in parts]) if parts else np.array([], dtype=np.int64)
    return tok_lens, domain_labels


def _tokenize_from_parquet_shards(parquet_path, tokenizer_path, sample_size,
                                   num_workers, tok_threads, domain_col,
                                   domain_names):
    """Tokenize a sample of docs from parquet shards without loading all into memory.

    Reads one shard at a time (only text/domain columns), collects the sampled
    texts, then tokenizes them via multiprocessing.  Produces the same result
    as ``_tokenize_sampled_docs`` but with O(1) memory instead of O(N) —
    peak usage is one shard (~5 GB) + collected sample (~2 GB) rather than
    the full DataFrame (hundreds of GB).
    """
    shards = resolve_parquet_source(parquet_path)

    import pyarrow.parquet as pq
    from tqdm import tqdm as tqdm

    # Get row counts per shard (metadata only, no data loaded)
    shard_counts = [pq.read_metadata(p).num_rows for p in shards]
    total_rows = sum(shard_counts)

    if total_rows == 0:
        return None, None

    # Check schema for text column
    schema_names = set(pq.read_schema(shards[0]).names)
    if "text" not in schema_names:
        print("  [warn] sampled_dataset has no 'text' column — skipping token analysis")
        return None, None

    # Sample indices (same RNG as _tokenize_sampled_docs for reproducibility)
    if sample_size > 0 and sample_size < total_rows:
        rng = np.random.default_rng(42)
        indices = rng.choice(total_rows, size=sample_size, replace=False)
        indices.sort()
    else:
        indices = np.arange(total_rows)

    # Collect texts and domain labels shard by shard
    texts = []
    domain_labels = []
    offset = 0
    cols = ["text"]
    if domain_col and domain_col in schema_names:
        cols.append(domain_col)

    for p, sc in zip(tqdm(shards, desc="  read shards", leave=False),
                     shard_counts):
        mask = (indices >= offset) & (indices < offset + sc)
        if not mask.any():
            offset += sc
            continue
        shard_indices = indices[mask] - offset
        shard_df = pd.read_parquet(p, columns=cols)
        shard_texts = shard_df["text"].to_numpy()[shard_indices]
        texts.extend(shard_texts.tolist())
        if domain_col and domain_col in shard_df.columns:
            raw = shard_df[domain_col].to_numpy()[shard_indices]
            domain_labels.extend(raw.tolist())
        del shard_df
        offset += sc

    texts_arr = np.array(texts, dtype=object)

    # Domain labels
    dom_arr = None
    if domain_labels:
        raw = np.array(domain_labels)
        if raw.dtype.kind in ("i", "u"):
            dom_arr = raw.astype(np.int64)
        elif raw.dtype.kind in ("U", "S", "O"):
            label_map = {}
            if domain_names:
                for i, name in enumerate(domain_names):
                    label_map[str(name)] = i
                    label_map[i] = i
            dom_arr = np.array(
                [label_map.get(str(d), label_map.get(int(d), -1)
                                if isinstance(d, (int, np.integer)) and not isinstance(d, bool)
                                else -1)
                 for d in raw],
                dtype=np.int64,
            )

    # Tokenize collected texts (same multiprocessing logic as _tokenize_sampled_docs)
    from multiprocessing import Pool

    batch_size = max(256, len(texts_arr) // (num_workers * 4))
    tasks = [
        (i, texts_arr[i:i + batch_size].tolist())
        for i in range(0, len(texts_arr), batch_size)
    ]
    parts = []
    with Pool(num_workers, initializer=init_tok_worker,
              initargs=(tokenizer_path, tok_threads)) as pool:
        for idx, lens in tqdm(
            pool.imap_unordered(tokenize_batch, tasks, chunksize=1),
            total=len(tasks), desc="  tokenize", leave=False,
        ):
            parts.append((idx, lens))
    parts.sort(key=lambda x: x[0])
    tok_lens = np.concatenate(
        [lens for _, lens in parts]
    ) if parts else np.array([], dtype=np.int64)
    return tok_lens, dom_arr


# ── Plot helpers ─────────────────────────────────────────────────


def _get_colors(num_domains):
    cmap = plt.cm.tab10 if num_domains <= 10 else plt.cm.tab20
    return [cmap(i % (10 if num_domains <= 10 else 20)) for i in range(num_domains)]


def _get_top_domains(domain_counts, top_n=6):
    return set(np.argsort(domain_counts)[-top_n:].tolist())


def plot_quality_score_dist(
    merged_scores, domain_indices, domain_counts,
    domain_names, num_domains, output_dir,
):
    """Figure 1: full corpus q̄ distribution by domain (overlaid)."""
    colors = _get_colors(num_domains)
    domain_short = get_domain_short(num_domains, domain_names)
    top_domains = _get_top_domains(domain_counts)

    fig, ax = plt.subplots(figsize=(10, 5))

    nonempty = [idx for idx in domain_indices if len(idx) > 0]
    if nonempty:
        global_min = min(float(merged_scores[idx].min()) for idx in nonempty)
        global_max = max(float(merged_scores[idx].max()) for idx in nonempty)
    else:
        global_min, global_max = 0.0, 1.0
    if global_max - global_min < 1e-15:
        global_max = global_min + 1.0
    bin_edges = np.linspace(global_min, global_max, 81)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for m in range(num_domains):
        idx = domain_indices[m]
        if len(idx) == 0:
            continue
        scores = merged_scores[idx]
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
    return save_fig(fig, output_dir, "fig_quality_score_dist.png")


def plot_quality_rank_dist(
    ranks,
    selected_ranks,
    domain_indices,
    selected_domain_labels,
    domain_counts,
    domain_names,
    num_domains,
    output_dir,
):
    """Figure 2: full corpus r̄ (solid) vs selected r̄ (dashed) by domain."""
    colors = _get_colors(num_domains)
    domain_short = get_domain_short(num_domains, domain_names)
    top_domains = _get_top_domains(domain_counts)

    fig, ax = plt.subplots(figsize=(10, 5))

    bin_edges = np.linspace(0, 1, 81)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for m in range(num_domains):
        idx = domain_indices[m]
        if len(idx) == 0:
            continue
        full_ranks = ranks[idx]
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
    return save_fig(fig, output_dir, "fig_quality_rank_dist.png")


def plot_duplication_analysis(
    selected_doc_ids,
    selected_domain_labels,
    sampling_values_col,
    domain_names,
    num_domains,
    output_dir,
):
    """Figure 3: document duplication analysis (2x sampling cap impact).

    Top subplot: per-domain stacked bars (unique vs duplicate docs).
    Bottom subplot: sampling-value distribution in 4 buckets.
    """
    domain_short = get_domain_short(num_domains, domain_names)

    use_horizontal = num_domains > 10
    if use_horizontal:
        height = 12 + num_domains * 0.3
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, height),
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # ── Subplot 1: Per-domain unique vs duplicate stacked bars ──
    unique_domains = np.unique(selected_domain_labels)
    unique_domains = unique_domains[unique_domains >= 0]

    labels = []
    unique_counts = []
    dup_counts = []

    for m in unique_domains:
        if m >= num_domains:
            continue
        mask = selected_domain_labels == m
        ids = selected_doc_ids[mask]
        total = len(ids)
        unique = len(np.unique(ids))
        labels.append(domain_short[m])
        unique_counts.append(unique)
        dup_counts.append(total - unique)

    x = np.arange(len(labels))

    if use_horizontal:
        ax1.barh(x, unique_counts, label="Unique docs", color="steelblue")
        ax1.barh(
            x, dup_counts, left=unique_counts, label="Duplicates", color="coral",
        )
        ax1.set_yticks(x)
        ax1.set_yticklabels(labels)
        ax1.set_xlabel("Document count")
        for i, (u, d) in enumerate(zip(unique_counts, dup_counts)):
            if d > 0:
                rate = d / max(u + d, 1) * 100
                ax1.text(u + d, i, f" {rate:.1f}%", va="center", fontsize=8)
    else:
        ax1.bar(x, unique_counts, label="Unique docs", color="steelblue")
        ax1.bar(
            x, dup_counts, bottom=unique_counts, label="Duplicates", color="coral",
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.set_ylabel("Document count")
        for i, (u, d) in enumerate(zip(unique_counts, dup_counts)):
            if d > 0:
                rate = d / max(u + d, 1) * 100
                ax1.text(i, u + d, f"{rate:.1f}%", ha="center", va="bottom", fontsize=8)

    ax1.set_title("Per-Domain: Unique vs Duplicate Documents")
    ax1.legend(loc="lower right")
    ax1.grid(axis="x" if use_horizontal else "y", alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    # ── Subplot 2: Sampling value distribution (4 buckets) ──
    if sampling_values_col is not None:
        sv = sampling_values_col
        bucket_defs = [
            ("≥ 1.99 (2x cap)", sv >= 1.99, "#d62728"),
            ("1.0 ~ 1.99", (sv >= 1.0) & (sv < 1.99), "#ff7f0e"),
            ("< 1.0 (no repeat)", (sv > 0.01) & (sv < 1.0), "#2ca02c"),
            ("< 0.01 (ε tail)", sv <= 0.01, "#999999"),
        ]
        bucket_labels = [b[0] for b in bucket_defs]
        bucket_counts = [int(b[1].sum()) for b in bucket_defs]
        bucket_colors = [b[2] for b in bucket_defs]

        bar_x = np.arange(len(bucket_labels))
        bars = ax2.bar(bar_x, bucket_counts, color=bucket_colors, edgecolor="white")
        ax2.set_xticks(bar_x)
        ax2.set_xticklabels(bucket_labels)
        ax2.set_ylabel("Number of rows")
        ax2.set_title("Sampling Value Distribution")

        sv_total = sum(bucket_counts)
        for bar, count in zip(bars, bucket_counts):
            pct = count / max(sv_total, 1) * 100
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9,
            )

        ax2.grid(axis="y", alpha=0.3, linestyle="--")
        ax2.set_axisbelow(True)
    else:
        ax2.text(
            0.5, 0.5, "sampling_value column not available",
            ha="center", va="center", transform=ax2.transAxes, fontsize=12,
        )
        ax2.set_title("Sampling Value Distribution (not available)")

    plt.tight_layout()
    return save_fig(fig, output_dir, "fig_duplication_analysis.png")


# ── Quality-length decomposition figure ───────────────────────────


def plot_quality_length_decomposition(
    params,
    quality_length_rhos,
    quality_names,
    domain_names,
    num_domains,
    output_dir,
):
    """Figure: weight×ρ(length) decomposition + sampling aggressiveness.

    Panel 1: stacked bar of w_n × ρ_n per domain, showing which quality
    dimensions drive length bias. Uniform-ρ reference line overlaid.
    Panel 2: bar chart of ω (top%) per domain with S_max annotated.
    """
    N = params.num_criteria
    dw = params.merge_config.domain_weights
    domain_short = get_domain_short(num_domains, domain_names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel 1: weight×ρ stacked bar ──
    x = np.arange(num_domains)
    width = 0.55
    cmap = plt.cm.tab10
    colors = [cmap(i) for i in range(N)]

    contributions = np.zeros((N, num_domains))
    for n in range(N):
        for m in range(num_domains):
            w_n = dw[n + m * N]
            contributions[n, m] = w_n * quality_length_rhos[n]

    pos_contrib = np.where(contributions > 0, contributions, 0)
    neg_contrib = np.where(contributions < 0, contributions, 0)
    pos_bottom = np.zeros(num_domains)
    neg_bottom = np.zeros(num_domains)

    for n in range(N):
        pos_vals = pos_contrib[n]
        neg_vals = neg_contrib[n]
        if pos_vals.any():
            ax1.bar(x, pos_vals, width, bottom=pos_bottom,
                    label=quality_names[n], color=colors[n], alpha=0.85)
            pos_bottom += pos_vals
        if neg_vals.any():
            ax1.bar(x, neg_vals, width, bottom=neg_bottom,
                    label=quality_names[n], color=colors[n], alpha=0.85)
            neg_bottom += neg_vals

    totals = pos_bottom + neg_bottom
    uniform_rho = float(np.mean(quality_length_rhos))
    ax1.axhline(y=uniform_rho, color="red", linestyle="--", linewidth=1.5,
                label=f"Uniform ρ = {uniform_rho:.3f}")

    for i, t in enumerate(totals):
        ax1.text(i, t + 0.005, f"{t:.3f}", ha="center", va="bottom", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(domain_short, rotation=30, ha="right")
    ax1.set_ylabel("Weighted ρ contribution")
    ax1.set_title("Quality-Length Bias Decomposition (w × ρ)")
    ax1.legend(fontsize=8, loc="best", ncol=2)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    # ── Panel 2: ω (top%) + S_max ──
    omega_pct = np.array([params.sampling_configs[m].omega * 100
                          for m in range(num_domains)])
    s_max = np.array([2.0 ** params.sampling_configs[m].eta
                      + params.sampling_configs[m].epsilon
                      for m in range(num_domains)])

    bars = ax2.bar(x, omega_pct, width, color="steelblue", alpha=0.8,
                   edgecolor="white")
    for i, (bar, sv) in enumerate(zip(bars, s_max)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"ω={omega_pct[i]:.1f}%\nS_max={sv:.1f}",
                 ha="center", va="bottom", fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(domain_short, rotation=30, ha="right")
    ax2.set_ylabel("ω (top percentile selected, %)")
    ax2.set_title("Sampling Aggressiveness")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    return save_fig(fig, output_dir, "fig_quality_length_decomposition.png")


# ── Token length & crop analysis figures ──────────────────────────


def plot_token_length_dist(tok_lens, seq_len, domain_labels,
                           domain_names, num_domains, output_dir,
                           baseline_tok_lens=None):
    """Histogram of token lengths with T=seq_len vertical marker.

    If baseline_tok_lens is provided, plots it as a dashed reference line
    for before-vs-after comparison.
    """
    if tok_lens is None or len(tok_lens) == 0:
        print("  [skip] fig_token_length_dist: no token length data")
        return None
    colors = _get_colors(num_domains)
    domain_short = get_domain_short(num_domains, domain_names)

    fig, ax = plt.subplots(figsize=(10, 5))

    lo, hi = np.percentile(tok_lens, [1, 99])
    lo = max(1.0, lo)
    bins = np.logspace(np.log10(lo), np.log10(hi), 50)

    if baseline_tok_lens is not None and len(baseline_tok_lens) > 0:
        ax.hist(baseline_tok_lens, bins=bins, density=True, histtype="step",
                lw=2, ls="--", color="gray", alpha=0.8,
                label=f"Baseline (n={len(baseline_tok_lens):,})")

    ax.hist(tok_lens, bins=bins, density=True, histtype="step", lw=2,
            label=f"This run (n={len(tok_lens):,})", color="black")

    if domain_labels is not None:
        unique_doms = np.unique(domain_labels)
        for d in unique_doms:
            if d < 0 or (num_domains and d >= num_domains):
                continue
            mask = domain_labels == d
            vals = tok_lens[mask]
            if len(vals) < 10:
                continue
            label = domain_short[d] if num_domains and d < len(domain_short) else f"D{d}"
            ax.hist(vals, bins=bins, density=True, histtype="step", lw=1.5,
                    alpha=0.7, label=f"{label} (n={len(vals):,})",
                    color=colors[d] if d < len(colors) else "gray")

    ax.axvline(seq_len, color="red", ls="--", lw=1.5, label=f"T={seq_len}")
    ax.set_xscale("log")
    ax.set_xlabel("Document length (tokens, incl. BOS)")
    ax.set_ylabel("density")
    has_baseline = baseline_tok_lens is not None and len(baseline_tok_lens) > 0
    ax.set_title("Token Length Distribution"
                 + (" (this run vs baseline)" if has_baseline else " (sampled dataset)"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    return save_fig(fig, output_dir, "fig_token_length_dist.png")


def plot_crop_analysis(crop_stats, output_dir):
    """Bar chart: total vs trained vs cropped tokens + crop rate decomposition."""
    if crop_stats is None:
        print("  [skip] fig_crop_analysis: no crop stats")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    categories = ["Total\ntokens", "Trained\ntokens", "Cropped\ntokens"]
    values = [crop_stats["total_tokens"],
              crop_stats["trained_tokens"],
              crop_stats["cropped_tokens"]]
    bar_colors = ["steelblue", "seagreen", "coral"]
    bars = ax1.bar(categories, values, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("Tokens")
    ax1.set_title(
        f"Token Breakdown (T={crop_stats['seq_len']}, "
        f"sample={crop_stats['sample_size']:,})"
    )
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    theo = crop_stats["theoretical_min_waste"] * 100
    addl = crop_stats["additional_waste"] * 100
    ax2.bar(["Theoretical min\n(docs > T+1)", "Additional\n(row-end slack)"],
            [theo, addl], color=["#ff7f0e", "#d62728"], edgecolor="white")
    ax2.set_ylabel("Crop rate (%)")
    ax2.set_title(
        f"Crop Rate Decomposition (total={crop_stats['crop_rate'] * 100:.1f}%)"
    )
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    return save_fig(fig, output_dir, "fig_crop_analysis.png")


def plot_packing_boundaries(boundaries, output_dir, seq_len=2048):
    """Bar histogram: doc boundaries per packed row (docs_in_row - 1)."""
    if boundaries is None or len(boundaries) == 0:
        print("  [skip] fig_packing_boundaries: no boundary data")
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    hi = min(int(boundaries.max()), 8)
    vc = np.bincount(
        np.clip(boundaries.astype(np.int64), 0, hi), minlength=hi + 1
    ).astype(float)
    vc = vc / max(1, vc.sum())

    ax.bar(np.arange(hi + 1), vc, 0.8, color="steelblue")
    ax.set_xticks(range(hi + 1))
    labels = [str(i) if i < hi else f"{hi}+" for i in range(hi + 1)]
    ax.set_xticklabels(labels)
    ax.set_xlabel(f"Doc boundaries per {seq_len}-token row")
    ax.set_ylabel("fraction of rows")
    ax.set_title("Packing Boundary Distribution (BOS-bestfit simulation)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    return save_fig(fig, output_dir, "fig_packing_boundaries.png")


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
    domain_indices,
    domain_counts,
    selected_domain_labels,
    domain_names,
    quality_names,
    num_domains,
    fig_score,
    fig_rank,
    sampling_values_col=None,
    fig_dup=None,
    quality_length_rhos=None,
    fig_decomp=None,
    crop_stats=None,
    tok_lens=None,
    baseline_tok_lens=None,
    fig_token_len=None,
    fig_crop=None,
    fig_boundaries=None,
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

    domain_short = get_domain_short(num_domains, domain_names)

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

    # ── Quality-Length Correlation Decomposition ──
    lines.append("-" * 70)
    lines.append("Quality-Length Correlation Decomposition (Spearman ρ vs char_count)")
    lines.append("-" * 70)
    if quality_length_rhos is not None and len(quality_length_rhos) == N:
        col_w = max(len(qn) for qn in quality_names) if quality_names else 10
        hdr_dim = "Dimension".ljust(col_w)
        hdr = f"  {hdr_dim} {'ρ(len)':>8s}"
        for m in range(num_domains):
            hdr += f"  w({domain_short[m]})"
        hdr += f"  {'w_avg':>8s}  {'w×ρ':>8s}"
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))

        for n in range(N):
            qname = quality_names[n].ljust(col_w) if quality_names else f"q{n}".ljust(col_w)
            rho_n = quality_length_rhos[n]
            w_vals = [dw[n + m * N] for m in range(num_domains)]
            w_avg = float(np.mean(w_vals))
            wr = w_avg * rho_n
            row = f"  {qname} {rho_n:>+8.4f}"
            for m in range(num_domains):
                row += f"  {w_vals[m]:.4f}"
            row += f"  {w_avg:>8.4f}  {wr:>+8.4f}"
            lines.append(row)

        lines.append("  " + "-" * (len(hdr) - 2))
        w_rho_per_domain = []
        for m in range(num_domains):
            weights_m = dw[m * N : (m + 1) * N]
            wr_m = float(np.dot(weights_m, quality_length_rhos))
            w_rho_per_domain.append(wr_m)
        uniform_rho = float(np.mean(quality_length_rhos))

        row_w = "  " + "Weighted ρ:".ljust(col_w + 9)
        for m in range(num_domains):
            row_w += f"  {w_rho_per_domain[m]:.4f}"
        row_w += f"  {float(np.mean(w_rho_per_domain)):>8.4f}"
        lines.append(row_w)

        row_u = "  " + "Uniform ρ (ref):".ljust(col_w + 9)
        for _ in range(num_domains):
            row_u += f"  {uniform_rho:.4f}"
        row_u += f"  {uniform_rho:>8.4f}"
        lines.append(row_u)

        row_a = "  " + "Amplification:".ljust(col_w + 9)
        for m in range(num_domains):
            amp = (w_rho_per_domain[m] / uniform_rho - 1) * 100 if abs(uniform_rho) > 1e-12 else 0.0
            row_a += f"  {amp:>+6.0f}%"
        mean_amp = (float(np.mean(w_rho_per_domain)) / uniform_rho - 1) * 100 if abs(uniform_rho) > 1e-12 else 0.0
        row_a += f"  {mean_amp:>+7.0f}%"
        lines.append(row_a)
        lines.append("")
    else:
        lines.append("  (quality_length_rhos not available or dimension mismatch)")
        lines.append("")

    # ── Sampling Aggressiveness ──
    lines.append("-" * 70)
    lines.append("Sampling Aggressiveness Interpretation")
    lines.append("-" * 70)
    hdr = (
        f"  {'Domain':>12s} {'ω(top%)':>8s} {'λ(mode)':>10s} "
        f"{'η(oversamp)':>12s} {'ε(tail)':>8s} {'S_max':>6s}  {'Effective':>12s}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for m in range(num_domains):
        sc = params.sampling_configs[m]
        omega_pct = sc.omega * 100
        if sc.lambda_ > 400:
            lam_mode = "step"
        elif sc.lambda_ > 100:
            lam_mode = "steep"
        else:
            lam_mode = "smooth"
        s_max = 2.0 ** sc.eta + sc.epsilon
        eff = f"top{omega_pct:.1f}%×{s_max:.1f}"
        lines.append(
            f"  {domain_short[m]:>12s} {omega_pct:>7.1f}% {lam_mode:>10s} "
            f"{sc.eta:>12.4f} {sc.epsilon:>8.6f} {s_max:>6.2f}  {eff:>12s}"
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
        epsilon_avg = dsp.get("epsilon_avg")
        if epsilon_avg is None:
            epsilon_avg = float(np.mean([sc.epsilon for sc in params.sampling_configs]))
        lines.append(f"ε average       : {epsilon_avg}")
        lines.append(
            f"Estimated output: {dsp.get('estimated_tokens', '?')}"
        )
    lines.append("")

    # ── Document Duplication Analysis ──
    lines.append("-" * 70)
    lines.append("Document Duplication Analysis (2x sampling cap impact)")
    lines.append("-" * 70)

    total_rows = len(selected_doc_ids)
    unique_docs = len(np.unique(selected_doc_ids))
    dup_rows = total_rows - unique_docs
    dup_rate = 1 - unique_docs / max(total_rows, 1)

    lines.append(f"Total rows       : {total_rows:,}")
    lines.append(f"Unique docs      : {unique_docs:,}")
    lines.append(f"Duplicate rows   : {dup_rows:,}")
    lines.append(f"Duplication rate : {dup_rate:.1%}")
    lines.append("")

    if sampling_values_col is not None:
        sv = sampling_values_col
        n_2x = int((sv >= 1.99).sum())
        n_partial = int(((sv >= 1.0) & (sv < 1.99)).sum())
        n_no_rep = int(((sv > 0.01) & (sv < 1.0)).sum())
        n_eps = int((sv <= 0.01).sum())
        sv_total = n_2x + n_partial + n_no_rep + n_eps

        lines.append("Sampling value distribution:")
        lines.append(
            f"  ≥ 1.99 (2x cap)  : {n_2x:>10,} rows "
            f"({n_2x / max(sv_total, 1):.1%})"
        )
        lines.append(
            f"  1.0 ~ 1.99       : {n_partial:>10,} rows "
            f"({n_partial / max(sv_total, 1):.1%})"
        )
        lines.append(
            f"  < 1.0 (no repeat): {n_no_rep:>10,} rows "
            f"({n_no_rep / max(sv_total, 1):.1%})"
        )
        lines.append(
            f"  < 0.01 (ε tail)  : {n_eps:>10,} rows "
            f"({n_eps / max(sv_total, 1):.1%})"
        )
        lines.append("")

    lines.append("Per-domain duplication:")
    header = (
        f"  {'Domain':>12s} {'total_rows':>12s} {'unique_docs':>12s} "
        f"{'dup_rate':>10s} {'max_sv(2^η+ε)':>14s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for m in range(num_domains):
        sc = params.sampling_configs[m]
        max_sv = 2.0 ** sc.eta + sc.epsilon

        mask = selected_domain_labels == m
        ids = selected_doc_ids[mask]
        t = len(ids)
        u = len(np.unique(ids))
        dr = 1 - u / max(t, 1) if t > 0 else 0

        lines.append(
            f"  {domain_short[m]:>12s} {t:>12,} {u:>12,} "
            f"{dr:>9.1%} {max_sv:>14.4f}"
        )

    lines.append("")

    if dup_rate > 0.30:
        lines.append(
            f"⚠ HIGH duplication rate ({dup_rate:.1%}) — 2x sigmoid cap causes "
            f"significant"
        )
        lines.append(
            "document repetition. This leads to fewer unique documents and model"
        )
        lines.append(
            "overfitting. Consider: deduplicating before training or lowering"
        )
        lines.append("the sigmoid cap.")
    elif dup_rate > 0.10:
        lines.append(
            f"⚠ MODERATE duplication rate ({dup_rate:.1%}) — some document "
            f"repetition"
        )
        lines.append("from 2x sigmoid cap.")
    else:
        lines.append(
            f"✓ LOW duplication rate ({dup_rate:.1%}) — 2x cap has minimal impact."
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

    def _domain_stats(m):
        idx = domain_indices[m]
        scores = merged_scores[idx]
        if len(scores) == 0:
            return None
        values, counts = np.unique(scores, return_counts=True)
        max_frac = counts.max() / len(scores)
        return (
            len(scores),
            float(scores.min()),
            float(scores.max()),
            float(scores.mean()),
            float(scores.std()),
            int(len(values)),
            float(max_frac),
        )

    n_jobs = getattr(args, "n_jobs", 1)
    effective = n_jobs if n_jobs != -1 else (os.cpu_count() or 1)
    if effective > 1 and num_domains > 1:
        from joblib import Parallel, delayed
        stats = Parallel(n_jobs=min(effective, num_domains), prefer="threads")(
            delayed(_domain_stats)(m) for m in range(num_domains)
        )
    else:
        stats = [_domain_stats(m) for m in range(num_domains)]

    for m, s in enumerate(stats):
        if s is None:
            lines.append(f"  {domain_short[m]:>12s} {'0':>12s} (empty)")
            continue
        n_docs, smin, smax, smean, sstd, n_uniq, max_frac = s
        lines.append(
            f"  {domain_short[m]:>12s} {n_docs:>12,} "
            f"{smin:>10.6f} {smax:>10.6f} "
            f"{smean:>10.6f} {sstd:>10.6f} "
            f"{n_uniq:>10,} {max_frac:>10.4%}"
        )
        if max_frac > 0.05:
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
        idx = domain_indices[m]
        full_r = ranks[idx]
        sel_mask = selected_domain_labels == m
        sel_r = selected_ranks[sel_mask]

        if len(full_r) == 0:
            lines.append(f"  {domain_short[m]:>12s} {'0':>12s} (empty)")
            continue

        pct_high = float((full_r > 0.99).sum() / len(full_r))
        sel_ratio = len(sel_r) / len(full_r) if len(full_r) > 0 else 0
        full_mean = full_r.mean()
        sel_mean = sel_r.mean() if len(sel_r) > 0 else float("nan")

        lines.append(
            f"  {domain_short[m]:>12s} {len(full_r):>12,} {len(sel_r):>8,} "
            f"{full_mean:>10.6f} {sel_mean:>10.6f} "
            f"{pct_high:>7.2%} {sel_ratio:>10.6f}"
        )

    lines.append("")

    # ── Tie-Breaking Diagnosis ──
    lines.append("-" * 70)
    lines.append("Tie-Breaking Diagnosis")
    lines.append("-" * 70)

    if tie_warning_domains:
        max_domain_frac = max(f for _, f in tie_warning_domains)
        if max_domain_frac > 0.5:
            severity = "CRITICAL"
        elif max_domain_frac > 0.10:
            severity = "SEVERE"
        else:
            severity = "MODERATE"
        lines.append(
            f"⚠ {severity} TIE PROBLEM DETECTED in the following domains "
            f"(>5% docs share the same merged score):"
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
        lines.append("✓ No severe tie problem detected (all domains <5% same score).")
    lines.append("")

    # ── Selection Analysis ──
    lines.append("-" * 70)
    lines.append("Selection Analysis (is selection quality-based?)")
    lines.append("-" * 70)

    quality_based_count = 0
    for m in range(num_domains):
        idx = domain_indices[m]
        full_r = ranks[idx]
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

    # ── Token Length & Crop Analysis ──
    if crop_stats is not None:
        lines.append("-" * 70)
        lines.append("Token Length & Crop Analysis (BOS-bestfit packing)")
        lines.append("-" * 70)
        lines.append(f"Sample size      : {crop_stats['sample_size']:,} docs")
        lines.append(f"Seq length (T)   : {crop_stats['seq_len']}")
        lines.append(f"Pack buffer      : {crop_stats['buffer_size']}")
        lines.append("")

        if tok_lens is not None and len(tok_lens) > 0:
            row_cap = crop_stats["seq_len"] + 1
            over_doc_pct = float(np.mean(tok_lens > row_cap) * 100)
            over_tok_pct = crop_stats["theoretical_min_waste"] * 100
            lines.append("Token Length Distribution:")
            lines.append(
                f"  mean={tok_lens.mean():.1f}  median={np.median(tok_lens):.1f}  "
                f"p25={np.percentile(tok_lens, 25):.0f}  "
                f"p75={np.percentile(tok_lens, 75):.0f}  "
                f"p90={np.percentile(tok_lens, 90):.0f}  "
                f"max={tok_lens.max()}"
            )
            lines.append(f"  %docs > T+1={row_cap}: {over_doc_pct:.1f}%")
            lines.append(
                f"  %tokens in docs > T+1: {over_tok_pct:.1f}% "
                f"(theoretical minimum waste)"
            )
            lines.append("")

            # ── Baseline comparison ──
            if baseline_tok_lens is not None and len(baseline_tok_lens) > 0:
                bl_over_doc = float(np.mean(baseline_tok_lens > row_cap) * 100)
                bl_over_tok = float(
                    np.maximum(baseline_tok_lens - row_cap, 0).sum()
                    / baseline_tok_lens.sum() * 100
                )
                delta_mean_pct = (
                    (tok_lens.mean() - baseline_tok_lens.mean())
                    / baseline_tok_lens.mean() * 100
                )
                delta_median_pct = (
                    (np.median(tok_lens) - np.median(baseline_tok_lens))
                    / np.median(baseline_tok_lens) * 100
                )
                lines.append("Baseline Comparison (this run vs --baseline-dir):")
                lines.append(
                    f"  This run  : mean={tok_lens.mean():.1f}  "
                    f"median={np.median(tok_lens):.1f}  "
                    f"%docs>T+1={over_doc_pct:.1f}%  "
                    f"%toks>T+1={over_tok_pct:.1f}%"
                )
                lines.append(
                    f"  Baseline  : mean={baseline_tok_lens.mean():.1f}  "
                    f"median={np.median(baseline_tok_lens):.1f}  "
                    f"%docs>T+1={bl_over_doc:.1f}%  "
                    f"%toks>T+1={bl_over_tok:.1f}%"
                )
                lines.append(
                    f"  Δ mean    : {delta_mean_pct:+.1f}% "
                    f"({tok_lens.mean():.0f} vs {baseline_tok_lens.mean():.0f})"
                )
                lines.append(
                    f"  Δ median  : {delta_median_pct:+.1f}% "
                    f"({np.median(tok_lens):.0f} vs "
                    f"{np.median(baseline_tok_lens):.0f})"
                )
                lines.append(
                    f"  Δ %toks>T+1: {over_tok_pct - bl_over_tok:+.1f}pp "
                    f"({over_tok_pct:.1f}% vs {bl_over_tok:.1f}%)"
                )
                if delta_mean_pct < -5:
                    lines.append(
                        "  → Shorter docs than baseline "
                        "(length-shaping strategy appears effective)"
                    )
                elif delta_mean_pct > 5:
                    lines.append(
                        "  → Longer docs than baseline "
                        "(length-shaping did NOT shorten distribution)"
                    )
                else:
                    lines.append(
                        "  → Similar length to baseline "
                        "(no meaningful length shift)"
                    )
                lines.append("")

        lines.append(
            f"Packing Simulation (BOS-bestfit, buffer={crop_stats['buffer_size']}, "
            f"T={crop_stats['seq_len']}):"
        )
        lines.append(f"  Total tokens (sample)  : {crop_stats['total_tokens']:,}")
        lines.append(f"  Trained tokens          : {crop_stats['trained_tokens']:,}")
        lines.append(f"  Cropped tokens          : {crop_stats['cropped_tokens']:,}")
        lines.append(f"  Crop rate               : {crop_stats['crop_rate'] * 100:.1f}%")
        lines.append(
            f"  Theoretical minimum     : {crop_stats['theoretical_min_waste'] * 100:.1f}% "
            f"(docs > T+1, can never fit)"
        )
        lines.append(
            f"  Additional waste        : {crop_stats['additional_waste'] * 100:.1f}% "
            f"(row-end slack)"
        )
        lines.append(f"  Packed rows             : {crop_stats['n_rows']:,}")

        bd = crop_stats["boundaries"]
        if len(bd) > 0:
            lines.append(
                f"  Docs/row (mean)         : {bd.mean() + 1:.2f}"
            )
            lines.append(
                f"  Rows with 1 doc (long)  : "
                f"{float(np.mean(bd == 0) * 100):.1f}%"
            )
        lines.append("")

        dsp = summary.get("dataset_size_prediction", {})
        est_tokens = dsp.get("estimated_tokens")
        if est_tokens:
            est_int = int(est_tokens)
            cropped_est = int(crop_stats["crop_rate"] * est_int)
            trained_est = est_int - cropped_est
            lines.append(f"Extrapolated to full dataset (~{est_int:,} tokens):")
            lines.append(
                f"  Tokens discarded (est.) : ~{cropped_est:,} "
                f"({crop_stats['crop_rate'] * 100:.1f}% x {est_int:,})"
            )
            lines.append(f"  Tokens trained on (est.): ~{trained_est:,}")
            if trained_est > 0:
                flat_gain = cropped_est / trained_est * 100
                lines.append(
                    f"  Flat loader would save  : ~{cropped_est:,} "
                    f"(+{flat_gain:.1f}% more training tokens)"
                )
            lines.append("")

        if crop_stats["crop_rate"] > 0.20:
            lines.append(
                f"⚠ SIGNIFICANT crop waste ({crop_stats['crop_rate'] * 100:.1f}%) "
                f"— consider --loader flat in mid_train.py"
            )
        elif crop_stats["crop_rate"] > 0.10:
            lines.append(
                f"⚠ MODERATE crop waste ({crop_stats['crop_rate'] * 100:.1f}%) "
                f"— some tokens lost to packing"
            )
        else:
            lines.append(
                f"✓ LOW crop waste ({crop_stats['crop_rate'] * 100:.1f}%) "
                f"— packing is efficient"
            )
        lines.append("")

    # ── CJK Font Note ──
    if not _report_mod._CJK_FONT_AVAILABLE and domain_names is not None:
        has_cjk = any(str_has_cjk(n) for n in domain_names[:num_domains])
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
    if fig_dup:
        lines.append(f"  3. {fig_dup}")
    if fig_decomp:
        lines.append(f"  4. {fig_decomp}")
    if fig_token_len:
        lines.append(f"  5. {fig_token_len}")
    if fig_crop:
        lines.append(f"  6. {fig_crop}")
    if fig_boundaries:
        lines.append(f"  7. {fig_boundaries}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("End of Analysis")
    lines.append("=" * 70)

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Proxy val_loss hard-vs-easy analysis ──────────────────────────


def _domain_block(symbol, name, hard_v, easy_v, mid_v, domain_short, M):
    """Build a per-domain hard/easy/mid comparison table (list of lines)."""
    out = [
        f"  {symbol} {name} per domain:",
        f"    {'Domain':<12s} {'hard':>10s} {'easy':>10s} {'mid':>10s} {'d(h-e)':>10s}",
        f"    {'-' * 58}",
    ]
    for m in range(M):
        out.append(
            f"    {domain_short[m]:<12s} {hard_v[m]:>10.4f} {easy_v[m]:>10.4f} "
            f"{mid_v[m]:>10.4f} {hard_v[m] - easy_v[m]:>+10.4f}"
        )
    out.append(
        f"    {'mean':<12s} {hard_v.mean():>10.4f} {easy_v.mean():>10.4f} "
        f"{mid_v.mean():>10.4f} {hard_v.mean() - easy_v.mean():>+10.4f}"
    )
    out.append("")
    return out


def _analyze_proxy_val_loss(exp_dir, domain_names, quality_names, extreme_count):
    """Hard-vs-easy val_loss analysis: is reverse-optimization (max val_loss) safe?

    Loads <exp-dir>/proxy_experiments/exp_*/meta.json, sorts by val_loss,
    splits into hard (highest) / easy (lowest) / mid, and compares their
    per-domain eta/lambda/omega/epsilon, alpha(noise), lambda-entropy, and
    per-task losses. Outputs proxy_val_loss_analysis.txt.

    Verdict checks whether the highest-loss runs relax quality/noise filtering
    or concentrate on one domain (=> reverse-optimization would pick garbage).
    Only rules out the 'picks garbage' failure mode; does not predict whether
    reversing improves downstream score.
    """
    proxy_dir = os.path.join(exp_dir, "proxy_experiments")
    print(f"\n[Proxy val_loss] Loading experiments from: {proxy_dir}")

    exp_names = sorted(
        d for d in os.listdir(proxy_dir)
        if d.startswith("exp_") and os.path.isdir(os.path.join(proxy_dir, d))
    )
    records = []
    for exp_name in exp_names:
        meta_path = os.path.join(proxy_dir, exp_name, "meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if ("val_loss" not in meta or "per_task_losses" not in meta
                or "sampling_params" not in meta or "quality_weights" not in meta):
            continue
        records.append(meta)

    n = len(records)
    if n < 6:
        print(f"[Proxy val_loss] Only {n} usable experiments (need >=6), skip.")
        return

    k = extreme_count
    if 2 * k >= n:
        k = max(1, n // 3)
        print(
            f"[Proxy val_loss] --proxy-hard-easy-count={extreme_count} too large "
            f"for n={n}, clamped to {k}"
        )

    domain_keys = sorted(records[0]["sampling_params"].keys())
    M = len(domain_keys)
    domain_short = get_domain_short(M, domain_keys)

    noise_criterion = None
    sample_qw = records[0]["quality_weights"][domain_keys[0]]
    for qname in sample_qw:
        if "noise" in str(qname).lower():
            noise_criterion = qname
            break

    tasks = sorted(records[0]["per_task_losses"].keys())
    K = len(tasks)

    val_losses = np.array([r["val_loss"] for r in records], dtype=np.float64)
    eta = np.zeros((n, M))
    lam = np.zeros((n, M))
    omega = np.zeros((n, M))
    eps = np.zeros((n, M))
    alpha_noise = np.zeros((n, M)) if noise_criterion else None
    per_task = np.zeros((n, K))

    for i, r in enumerate(records):
        sp = r["sampling_params"]
        qw = r["quality_weights"]
        for m, dk in enumerate(domain_keys):
            eta[i, m] = float(sp[dk].get("eta", 0.0))
            lam[i, m] = float(sp[dk].get("lambda", 0.0))
            omega[i, m] = float(sp[dk].get("omega", 0.0))
            eps[i, m] = float(sp[dk].get("epsilon", 0.0))
            if noise_criterion:
                alpha_noise[i, m] = float(qw[dk].get(noise_criterion, 0.0))
        for j, t in enumerate(tasks):
            per_task[i, j] = float(r["per_task_losses"].get(t, 0.0))

    lam_pos = np.maximum(lam, 1e-12)
    lam_norm = lam_pos / lam_pos.sum(axis=1, keepdims=True)
    lam_entropy = -(lam_norm * np.log(lam_norm)).sum(axis=1) / np.log(max(M, 2))

    order = np.argsort(val_losses)
    easy_idx = order[:k]
    hard_idx = order[-k:]
    mid_mask = np.ones(n, dtype=bool)
    mid_mask[easy_idx] = False
    mid_mask[hard_idx] = False
    mid_idx = np.where(mid_mask)[0]
    has_mid = len(mid_idx) > 0

    hard_eta = eta[hard_idx].mean(axis=0)
    easy_eta = eta[easy_idx].mean(axis=0)
    mid_eta = eta[mid_idx].mean(axis=0) if has_mid else np.zeros(M)
    hard_lam = lam[hard_idx].mean(axis=0)
    easy_lam = lam[easy_idx].mean(axis=0)
    mid_lam = lam[mid_idx].mean(axis=0) if has_mid else np.zeros(M)
    hard_omega = omega[hard_idx].mean(axis=0)
    easy_omega = omega[easy_idx].mean(axis=0)
    mid_omega = omega[mid_idx].mean(axis=0) if has_mid else np.zeros(M)
    hard_eps = eps[hard_idx].mean(axis=0)
    easy_eps = eps[easy_idx].mean(axis=0)
    mid_eps = eps[mid_idx].mean(axis=0) if has_mid else np.zeros(M)
    if noise_criterion:
        hard_an = alpha_noise[hard_idx].mean(axis=0)
        easy_an = alpha_noise[easy_idx].mean(axis=0)
        mid_an = alpha_noise[mid_idx].mean(axis=0) if has_mid else np.zeros(M)
    hard_task = per_task[hard_idx].mean(axis=0)
    easy_task = per_task[easy_idx].mean(axis=0)
    hard_entropy = float(lam_entropy[hard_idx].mean())
    easy_entropy = float(lam_entropy[easy_idx].mean())
    mid_entropy = float(lam_entropy[mid_idx].mean()) if has_mid else 0.0

    # ── Build text output ──
    lines = []
    lines.append("=" * 70)
    lines.append("Proxy val_loss Hard-vs-Easy Analysis")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Proxy dir       : {proxy_dir}")
    lines.append(f"Experiments     : {n} (with val_loss + per_task_losses)")
    lines.append(f"Hard/Easy count : k={k} (hard=highest {k}, easy=lowest {k}, mid={len(mid_idx)})")
    lines.append(f"Domains         : {M} ({domain_keys})")
    lines.append(f"Tasks           : {K} ({tasks})")
    if noise_criterion:
        lines.append(f"Noise criterion : {noise_criterion}")
    else:
        lines.append("Noise criterion : (none detected, alpha_noise skipped)")
    lines.append("")

    lines.append("-" * 70)
    lines.append("val_loss Distribution")
    lines.append("-" * 70)
    vl_mean = float(val_losses.mean())
    lines.append(
        f"  mean={vl_mean:.4f}, std={float(val_losses.std()):.4f}, "
        f"CV={float(val_losses.std()) / max(vl_mean, 1e-12):.4f}"
    )
    lines.append(
        f"  min={float(val_losses.min()):.4f}, max={float(val_losses.max()):.4f}, "
        f"range={float(val_losses.max() - val_losses.min()):.4f}"
    )
    lines.append(
        f"  easy mean={float(val_losses[easy_idx].mean()):.4f}, "
        f"hard mean={float(val_losses[hard_idx].mean()):.4f}"
    )
    lines.append("")

    lines.append("-" * 70)
    lines.append("Per-Domain Group Means (hard vs easy vs mid)")
    lines.append("-" * 70)
    lines += _domain_block("eta", "quality exponent", hard_eta, easy_eta, mid_eta, domain_short, M)
    lines += _domain_block("lambda", "domain weight", hard_lam, easy_lam, mid_lam, domain_short, M)
    lines += _domain_block("omega", "sampling rate", hard_omega, easy_omega, mid_omega, domain_short, M)
    lines += _domain_block("epsilon", "tail prob", hard_eps, easy_eps, mid_eps, domain_short, M)
    if noise_criterion:
        lines += _domain_block("alpha_noise", "noise weight", hard_an, easy_an, mid_an, domain_short, M)

    lines.append("-" * 70)
    lines.append("lambda Diversity (normalized entropy across domains, 0=niche, 1=balanced)")
    lines.append("-" * 70)
    lines.append(
        f"  hard entropy={hard_entropy:.4f}, easy entropy={easy_entropy:.4f}, "
        f"mid entropy={mid_entropy:.4f}, d(h-e)={hard_entropy - easy_entropy:+.4f}"
    )
    lines.append("")

    lines.append("-" * 70)
    lines.append("Spearman Correlation: val_loss vs each parameter (|rho| sorted desc)")
    lines.append("-" * 70)
    sp_rows = [("lambda-entropy (diversity)", spearman(val_losses, lam_entropy))]
    for m in range(M):
        sp_rows.append((f"eta[{domain_short[m]}]", spearman(val_losses, eta[:, m])))
        sp_rows.append((f"lambda[{domain_short[m]}]", spearman(val_losses, lam[:, m])))
        sp_rows.append((f"omega[{domain_short[m]}]", spearman(val_losses, omega[:, m])))
        sp_rows.append((f"epsilon[{domain_short[m]}]", spearman(val_losses, eps[:, m])))
        if noise_criterion:
            sp_rows.append(
                (f"alpha_noise[{domain_short[m]}]", spearman(val_losses, alpha_noise[:, m]))
            )
    for j, t in enumerate(tasks):
        sp_rows.append((f"task:{t}", spearman(val_losses, per_task[:, j])))
    sp_rows.sort(key=lambda r: -abs(r[1]))
    lines.append(f"  {'Param':<30s} {'rho':>8s}")
    lines.append(f"  {'-' * 40}")
    for label, rho in sp_rows:
        lines.append(f"  {label:<30s} {rho:>+8.4f}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("Per-Task Loss: hard vs easy (which tasks drive high val_loss?)")
    lines.append("-" * 70)
    lines.append(f"  {'Task':<28s} {'hard':>10s} {'easy':>10s} {'d(h-e)':>10s}")
    lines.append(f"  {'-' * 60}")
    task_delta = hard_task - easy_task
    order_t = np.argsort(-task_delta)
    for j in order_t:
        lines.append(
            f"  {tasks[j]:<28s} {hard_task[j]:>10.4f} {easy_task[j]:>10.4f} "
            f"{task_delta[j]:>+10.4f}"
        )
    lines.append("")

    # ── Verdict ──
    hard_eta_mean = float(hard_eta.mean())
    easy_eta_mean = float(easy_eta.mean())
    low_eta = hard_eta_mean < easy_eta_mean
    low_alpha_noise = False
    hard_an_mean = easy_an_mean = 0.0
    if noise_criterion:
        hard_an_mean = float(alpha_noise[hard_idx].mean())
        easy_an_mean = float(alpha_noise[easy_idx].mean())
        low_alpha_noise = hard_an_mean < easy_an_mean
    low_entropy = hard_entropy < easy_entropy

    lines.append("-" * 70)
    lines.append("Verdict: Is reverse-optimization (picking MAX val_loss) safe?")
    lines.append("-" * 70)
    lines.append(
        f"  quality filter (eta)    : hard={hard_eta_mean:.4f}  easy={easy_eta_mean:.4f}  "
        f"-> {'RELAXED (may pick low-quality)' if low_eta else 'maintained (still filters) OK'}"
    )
    if noise_criterion:
        lines.append(
            f"  noise penalty (alpha)   : hard={hard_an_mean:.4f}  easy={easy_an_mean:.4f}  "
            f"-> {'RELAXED (may let noise in)' if low_alpha_noise else 'maintained (still penalizes) OK'}"
        )
    else:
        lines.append("  noise penalty (alpha)   : (no noise criterion detected, skipped)")
    lines.append(
        f"  domain balance (entropy): hard={hard_entropy:.4f}  easy={easy_entropy:.4f}  "
        f"-> {'LOW (may pick one domain)' if low_entropy else 'HIGH (balanced) OK'}"
    )
    lines.append("")

    if low_eta and low_alpha_noise:
        verdict = (
            "[!] NOT SAFE: highest-loss runs relaxed BOTH quality and noise "
            "filtering -> reverse-optimization would pick low-quality/noisy "
            "data. Do NOT reverse without a quality floor."
        )
    elif low_entropy:
        verdict = (
            "[!] RISKY: highest-loss runs concentrate on one domain -> "
            "reverse-optimization would pick niche data; domain imbalance risk."
        )
    else:
        verdict = (
            "[+] SAFE: highest-loss runs keep quality + noise filtering + "
            "domain balance -> reverse-optimization would not pick garbage. "
            "Note: this only rules out the 'picks garbage' failure mode; it "
            "does NOT predict whether reversing improves downstream score."
        )
    lines.append("  " + verdict)
    lines.append("")
    lines.append("=" * 70)
    lines.append("End of Proxy val_loss Analysis")
    lines.append("=" * 70)

    out_path = os.path.join(exp_dir, "proxy_val_loss_analysis.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [Text]  Saved: {out_path}")


# ── Optimizer domain-proportion analysis ──────────────────────────


def _analyze_optimizer_domain_proportions(
    exp_dir,
    domain_labels,
    domain_names,
    quality_names,
    num_domains,
    num_criteria,
):
    """Analyze why the optimizer chose the domain proportions it did.

    For each proxy experiment:
    1. Load selected_indices.npy → compute per-domain proportion
    2. Load meta.json → get val_loss and parameters

    Then:
    3. Correlate domain proportions with val_loss (Spearman)
    4. Retrain LightGBM → feature_importance()
    5. Scatter plot: domain proportion vs val_loss
    6. Output: optimizer_domain_analysis.txt
    """
    proxy_dir = os.path.join(exp_dir, "proxy_experiments")
    print(f"\n[Optimizer] Loading experiments from: {proxy_dir}")

    exp_names = sorted(
        d for d in os.listdir(proxy_dir)
        if d.startswith("exp_") and os.path.isdir(os.path.join(proxy_dir, d))
    )

    records = []
    n_missing_npy = 0
    n_oob = 0
    for exp_name in exp_names:
        meta_path = os.path.join(proxy_dir, exp_name, "meta.json")
        npy_path = os.path.join(proxy_dir, exp_name, "selected_indices.npy")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if ("val_loss" not in meta or "sampling_params" not in meta
                or "quality_weights" not in meta):
            continue
        if not os.path.exists(npy_path):
            n_missing_npy += 1
            continue
        sel_idx = np.load(npy_path)
        if len(sel_idx) == 0:
            continue
        if sel_idx.max() >= len(domain_labels):
            n_oob += 1
            continue

        sel_domains = domain_labels[sel_idx]
        valid = sel_domains >= 0
        counts = np.bincount(
            sel_domains[valid], minlength=num_domains
        ).astype(np.float64)
        total = counts.sum()
        props = counts / total if total > 0 else counts

        records.append({
            "exp_name": exp_name,
            "val_loss": float(meta["val_loss"]),
            "domain_props": props,
            "meta": meta,
        })

    n = len(records)
    if n_missing_npy > 0:
        print(f"[Optimizer] Skipped {n_missing_npy} experiments without selected_indices.npy")
    if n_oob > 0:
        print(f"[Optimizer] Skipped {n_oob} experiments with out-of-bounds indices")
    if n < 6:
        print(f"[Optimizer] Only {n} usable experiments (need >=6), skip.")
        return
    print(f"[Optimizer] Loaded {n} experiments")

    val_losses = np.array([r["val_loss"] for r in records], dtype=np.float64)
    domain_proportions = np.array(
        [r["domain_props"] for r in records]
    )  # (n, M)

    domain_short = get_domain_short(num_domains, list(domain_names))

    # ── Retrain LightGBM ──
    params_list = []
    for r in records:
        pset = ParameterSet.from_dict(
            r["meta"]["quality_weights"], r["meta"]["sampling_params"]
        )
        params_list.append(pset)

    print(
        f"[Optimizer] Retraining LightGBM ({n} samples, "
        f"{(num_criteria + 4) * num_domains} features)..."
    )
    from quadmix.pipeline.regression_model import RegressionModel

    model = RegressionModel(model_type="lightgbm")
    model.fit(
        params_list, val_losses,
        num_domains=num_domains, num_criteria=num_criteria,
        verbose=False,
    )
    importance = model.feature_importance()
    r2_score = model.score(params_list, val_losses)

    # ── Scatter plot: domain proportion vs val_loss ──
    fig, axes = plt.subplots(
        1, num_domains, figsize=(5 * num_domains, 4), squeeze=False
    )
    for m in range(num_domains):
        ax = axes.flat[m]
        ax.scatter(
            domain_proportions[:, m] * 100, val_losses,
            alpha=0.4, s=12, c="#5B9BD5",
        )
        rho_m = spearman(domain_proportions[:, m], val_losses)
        ax.set_xlabel(f"{domain_short[m]} proportion (%)")
        ax.set_ylabel("val_loss")
        ax.set_title(f"{domain_short[m]}: ρ={rho_m:+.4f}")
    fig.suptitle(
        "Domain proportion vs val_loss (proxy experiments)", fontsize=13
    )
    fig.tight_layout()
    save_fig(fig, exp_dir, "fig_optimizer_domain_vs_loss.png")

    # ── Easy vs hard groups ──
    order = np.argsort(val_losses)
    k = min(5, max(1, n // 3))
    easy_idx = order[:k]
    hard_idx = order[-k:]

    # ── Build text output ──
    lines = []
    lines.append("=" * 70)
    lines.append("Optimizer Domain Proportion Analysis")
    lines.append("=" * 70)
    lines.append(f"Proxy dir        : {proxy_dir}")
    lines.append(f"Experiments      : {n}")
    lines.append(f"Domains          : {num_domains} ({list(domain_names)})")
    lines.append(f"Quality criteria : {num_criteria} ({list(quality_names)})")
    lines.append("")

    lines.append("-" * 70)
    lines.append("Domain Proportion Summary (across all experiments)")
    lines.append("-" * 70)
    lines.append(
        f"  {'Domain':<15s} {'mean%':>8s} {'std%':>8s} {'min%':>8s} "
        f"{'max%':>8s} {'Spearman ρ':>12s}"
    )
    lines.append(f"  {'-' * 60}")
    for m in range(num_domains):
        props_m = domain_proportions[:, m] * 100
        rho = spearman(domain_proportions[:, m], val_losses)
        lines.append(
            f"  {domain_short[m]:<15s} {props_m.mean():>8.2f} "
            f"{props_m.std():>8.2f} {props_m.min():>8.2f} "
            f"{props_m.max():>8.2f} {rho:>+12.4f}"
        )
    lines.append("")

    lines.append("-" * 70)
    lines.append(
        f"Easy (lowest val_loss) vs Hard (highest val_loss) — top {k}"
    )
    lines.append("-" * 70)
    lines.append(
        f"  {'Domain':<15s} {'easy%':>8s} {'hard%':>8s} {'d(h-e)':>10s}"
    )
    lines.append(f"  {'-' * 45}")
    for m in range(num_domains):
        easy_mean = domain_proportions[easy_idx, m].mean() * 100
        hard_mean = domain_proportions[hard_idx, m].mean() * 100
        lines.append(
            f"  {domain_short[m]:<15s} {easy_mean:>8.2f} {hard_mean:>8.2f} "
            f"{hard_mean - easy_mean:>+10.2f}"
        )
    lines.append(
        f"  {'val_loss':<15s} {val_losses[easy_idx].mean():>8.4f} "
        f"{val_losses[hard_idx].mean():>8.4f}"
    )
    lines.append("")

    # ── LightGBM feature importance ──
    lines.append("-" * 70)
    lines.append(f"LightGBM Feature Importance (R² = {r2_score:.4f})")
    lines.append("-" * 70)
    domain_imp = {}
    if importance:
        imp_sorted = sorted(importance.items(), key=lambda x: -x[1])
        lines.append(f"  {'Feature':<25s} {'Importance':>12s}")
        lines.append(f"  {'-' * 40}")
        for name, imp in imp_sorted[:20]:
            lines.append(f"  {name:<25s} {imp:>12.1f}")
        lines.append("")

        lines.append("  Grouped by parameter type:")
        groups = {}
        for name, imp in importance.items():
            prefix = name.split("_")[0]
            groups[prefix] = groups.get(prefix, 0.0) + imp
        total_imp = sum(groups.values())
        for prefix in sorted(groups.keys(), key=lambda p: -groups[p]):
            pct = groups[prefix] / total_imp * 100 if total_imp > 0 else 0
            lines.append(
                f"    {prefix:<15s} {groups[prefix]:>10.1f} ({pct:>5.1f}%)"
            )
        lines.append("")

        lines.append("  Grouped by domain (sum of all params):")
        for name, imp in importance.items():
            parts = name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                m_idx = int(parts[1])
                domain_imp[m_idx] = domain_imp.get(m_idx, 0.0) + imp
        dom_total = sum(domain_imp.values())
        for m in sorted(domain_imp.keys()):
            pct = domain_imp[m] / dom_total * 100 if dom_total > 0 else 0
            short = domain_short[m] if m < len(domain_short) else f"D{m}"
            lines.append(
                f"    {short:<15s} {domain_imp[m]:>10.1f} ({pct:>5.1f}%)"
            )
        lines.append("")
    else:
        lines.append("  (feature importance not available)")
        lines.append("")

    # ── Diagnosis ──
    lines.append("-" * 70)
    lines.append(
        "Diagnosis: Why did the optimizer choose high math proportion?"
    )
    lines.append("-" * 70)

    math_idx = None
    for m, name in enumerate(domain_names):
        if "math" in str(name).lower() or "数学" in str(name):
            math_idx = m
            break

    if math_idx is not None:
        rho_math = spearman(domain_proportions[:, math_idx], val_losses)
        easy_math = domain_proportions[easy_idx, math_idx].mean() * 100
        hard_math = domain_proportions[hard_idx, math_idx].mean() * 100
        lines.append(
            f"  Math proportion vs val_loss: Spearman ρ = {rho_math:+.4f}"
        )
        lines.append(
            f"  Easy group math%: {easy_math:.2f}, "
            f"Hard group math%: {hard_math:.2f}"
        )
        if rho_math < -0.1:
            lines.append(
                f"  -> More math → lower val_loss (ρ < -0.1)"
            )
            lines.append(
                f"     The optimizer favors math because math-heavy experiments"
            )
            lines.append(
                f"     have lower val_loss. Possible causes:"
            )
            lines.append(
                f"     (a) math text is more predictable/compressible,"
            )
            lines.append(
                f"     (b) the validation set is math-heavy,"
            )
            lines.append(
                f"     (c) math quality scores correlate with lower loss."
            )
        elif rho_math > 0.1:
            lines.append(
                f"  -> More math → HIGHER val_loss (ρ > 0.1)"
            )
            lines.append(
                f"     The optimizer does NOT favor math via val_loss;"
            )
            lines.append(
                f"     the high math proportion in optimal parameters is"
            )
            lines.append(
                f"     a side effect of other parameter choices."
            )
        else:
            lines.append(
                f"  -> Math proportion has weak correlation with val_loss"
            )
            lines.append(
                f"     (|ρ| ≤ 0.1); the high math proportion is likely"
            )
            lines.append(
                f"     a side effect of other parameters."
            )
        lines.append("")

    if importance and domain_imp:
        dom_total = sum(domain_imp.values())
        for m in sorted(domain_imp.keys(), key=lambda x: -domain_imp[x]):
            pct = domain_imp[m] / dom_total * 100 if dom_total > 0 else 0
            if pct > 40:
                short = domain_short[m] if m < len(domain_short) else f"D{m}"
                lines.append(
                    f"  Domain {short} dominates feature importance "
                    f"({pct:.1f}%)"
                )
                lines.append(
                    f"  -> The optimizer's val_loss prediction is"
                )
                lines.append(
                    f"     primarily driven by {short} parameters."
                )
                break

    if importance:
        top_feature = max(importance.items(), key=lambda x: x[1])
        lines.append(
            f"  Top individual feature: {top_feature[0]} "
            f"(importance={top_feature[1]:.1f})"
        )
    lines.append("")
    lines.append("=" * 70)
    lines.append("End of Optimizer Domain Proportion Analysis")
    lines.append("=" * 70)

    out_path = os.path.join(exp_dir, "optimizer_domain_analysis.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [Text]  Saved: {out_path}")


# ── Diversity analysis (quality-diversity tradeoff) ───────────────


def _analyze_diversity(
    exp_dir,
    mgr,
    domain_names,
    num_domains,
    *,
    summary=None,
    selected_doc_ids=None,
    baseline_dir=None,
):
    """Compute diversity_score for each proxy experiment and analyze
    the quality-diversity tradeoff.

    For each experiment:
      n_unique = len(np.unique(selected_indices))
      unique_tokens = (doc_char_counts[unique_idx] // 4).sum()
      diversity_score = n_unique / max(n_unique)   # normalized [0, 1]

    Outputs:
      diversity_analysis.txt
      fig_diversity_tradeoff.png
      fig_diversity_lambda_sweep.png
    """
    proxy_dir = os.path.join(exp_dir, "proxy_experiments")
    print(f"\n[Diversity] Loading experiments from: {proxy_dir}")

    char_counts = mgr.doc_char_counts
    if char_counts is None:
        print("[Diversity] doc_char_counts not available — skipping")
        return

    exp_names = sorted(
        d for d in os.listdir(proxy_dir)
        if d.startswith("exp_") and os.path.isdir(os.path.join(proxy_dir, d))
    )

    records = []
    n_missing_npy = 0
    n_oob = 0
    for exp_name in exp_names:
        meta_path = os.path.join(proxy_dir, exp_name, "meta.json")
        npy_path = os.path.join(proxy_dir, exp_name, "selected_indices.npy")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if "val_loss" not in meta:
            continue
        if not os.path.exists(npy_path):
            n_missing_npy += 1
            continue
        sel_idx = np.load(npy_path)
        if len(sel_idx) == 0:
            continue
        if sel_idx.max() >= len(char_counts):
            n_oob += 1
            continue

        n_total = len(sel_idx)
        unique_idx = np.unique(sel_idx)
        n_unique = len(unique_idx)
        unique_tokens = int(np.maximum(char_counts[unique_idx] // 4, 1).sum())
        if unique_tokens == 0:
            continue
        dup_rate = 1.0 - n_unique / max(n_total, 1)
        records.append({
            "exp_id": meta.get("experiment_id", exp_name),
            "exp_name": exp_name,
            "val_loss": float(meta["val_loss"]),
            "n_total": n_total,
            "n_unique": n_unique,
            "unique_tokens": unique_tokens,
            "dup_rate": dup_rate,
            "doc_density": n_unique / unique_tokens,
            "sampling_params": meta.get("sampling_params", {}),
        })

    n = len(records)
    if n_missing_npy > 0:
        print(f"[Diversity] Skipped {n_missing_npy} experiments without selected_indices.npy")
    if n_oob > 0:
        print(f"[Diversity] Skipped {n_oob} experiments with out-of-bounds indices")
    if n < 4:
        print(f"[Diversity] Only {n} usable experiments (need >=4), skipping.")
        return
    print(f"[Diversity] Loaded {n} experiments")

    max_n_unique = max(r["n_unique"] for r in records)
    for r in records:
        r["diversity_score"] = r["n_unique"] / max_n_unique

    val_losses = np.array([r["val_loss"] for r in records])
    div_scores = np.array([r["diversity_score"] for r in records])
    n_uniques = np.array([r["n_unique"] for r in records])
    densities = np.array([r["doc_density"] for r in records])

    rho_vd = spearman(val_losses, div_scores)
    rho_vn = spearman(val_losses, n_uniques)
    rho_density = spearman(val_losses, densities)

    order = np.argsort(val_losses)
    pareto_idx = []
    best_div_so_far = -1.0
    for i in order:
        d = div_scores[i]
        if d > best_div_so_far:
            pareto_idx.append(int(i))
            best_div_so_far = d
    pareto_set = set(pareto_idx)

    lambdas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    lambda_results = []
    for lam in lambdas:
        adjusted = val_losses + lam * (1.0 - div_scores)
        best_i = int(np.argmin(adjusted))
        lambda_results.append({
            "lambda": lam,
            "best_idx": best_i,
            "exp_id": records[best_i]["exp_id"],
            "val_loss": records[best_i]["val_loss"],
            "diversity_score": records[best_i]["diversity_score"],
            "n_unique": records[best_i]["n_unique"],
        })

    min_vl = float(val_losses.min())
    candidates = []
    candidate_ids = set()
    for i in pareto_idx:
        r = records[i]
        if r["val_loss"] <= min_vl * 1.05 and r["diversity_score"] > 0.85:
            candidates.append(r)
            candidate_ids.add(r["exp_id"])
    for lr in lambda_results:
        r = records[lr["best_idx"]]
        if r["exp_id"] not in candidate_ids and lr["diversity_score"] > 0.80:
            candidates.append(r)
            candidate_ids.add(r["exp_id"])

    lines = []
    lines.append("=" * 70)
    lines.append("Diversity Analysis (Quality-Diversity Tradeoff)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Experiments       : {n}")
    lines.append(f"max(n_unique)     : {max_n_unique:,} (D_ref)")
    lines.append(f"doc_char_counts   : available ({len(char_counts):,} docs in source)")
    lines.append("")

    lines.append("-" * 70)
    lines.append("Diversity Metrics Summary")
    lines.append("-" * 70)
    for key, label in [
        ("val_loss", "val_loss"),
        ("n_unique", "n_unique"),
        ("unique_tokens", "unique_tokens"),
        ("doc_density", "doc_density"),
        ("diversity_score", "diversity_score"),
        ("dup_rate", "dup_rate"),
    ]:
        vals = np.array([r[key] for r in records])
        lines.append(
            f"  {label:16s} min={vals.min():.4f}  max={vals.max():.4f}  "
            f"mean={vals.mean():.4f}  std={vals.std():.4f}"
        )
    lines.append("")

    lines.append("-" * 70)
    lines.append("Spearman Correlations (val_loss vs diversity metrics)")
    lines.append("-" * 70)
    lines.append(f"  rho(val_loss, diversity_score)  = {rho_vd:+.4f}")
    lines.append(f"  rho(val_loss, n_unique)         = {rho_vn:+.4f}")
    lines.append(f"  rho(val_loss, doc_density)      = {rho_density:+.4f}")
    lines.append("")
    if rho_vd < -0.3:
        lines.append("  -> Tradeoff CONFIRMED: lower val_loss <-> lower diversity (anti-correlated)")
    elif abs(rho_vd) < 0.1:
        lines.append("  -> NO tradeoff: val_loss and diversity uncorrelated; theta* may be an outlier")
    elif rho_vd > 0.3:
        lines.append("  -> UNEXPECTED: val_loss and diversity positively correlated")
    else:
        lines.append(f"  -> Weak tradeoff (|rho|={abs(rho_vd):.4f})")
    lines.append("")

    lines.append("-" * 70)
    lines.append("Per-Experiment Diversity (sorted by val_loss, top 30)")
    lines.append("-" * 70)
    header = (
        f"  {'exp_id':>6s} {'val_loss':>10s} {'n_total':>10s} {'n_unique':>10s} "
        f"{'uniq_tok':>10s} {'density':>10s} {'div_score':>10s} {'dup_rate':>8s} {'pareto':>6s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    sorted_idx = np.argsort(val_losses)
    for i in sorted_idx[:30]:
        r = records[i]
        p_tag = "*" if i in pareto_set else ""
        lines.append(
            f"  {str(r['exp_id']):>6s} {r['val_loss']:>10.4f} {r['n_total']:>10,} {r['n_unique']:>10,} "
            f"{r['unique_tokens']:>10,} {r['doc_density']:>10.6f} {r['diversity_score']:>10.4f} "
            f"{r['dup_rate']:>8.2%} {p_tag:>6s}"
        )
    lines.append("")

    lines.append("-" * 70)
    lines.append(f"Pareto Frontier ({len(pareto_idx)} non-dominated points)")
    lines.append("-" * 70)
    lines.append(f"  {'exp_id':>6s} {'val_loss':>10s} {'div_score':>10s} {'n_unique':>10s}")
    lines.append("  " + "-" * 40)
    for i in pareto_idx:
        r = records[i]
        lines.append(
            f"  {str(r['exp_id']):>6s} {r['val_loss']:>10.4f} {r['diversity_score']:>10.4f} {r['n_unique']:>10,}"
        )
    lines.append("")

    lines.append("-" * 70)
    lines.append("Lambda Sweep: adjusted_loss = val_loss + lambda * (1 - diversity_score)")
    lines.append("-" * 70)
    lines.append(f"  {'lambda':>6s} {'exp_id':>6s} {'val_loss':>10s} {'div_score':>10s} {'n_unique':>10s} {'d_val_loss':>10s}")
    lines.append("  " + "-" * 58)
    for lr in lambda_results:
        dv = lr["val_loss"] - min_vl
        lines.append(
            f"  {lr['lambda']:>6.2f} {str(lr['exp_id']):>6s} {lr['val_loss']:>10.4f} "
            f"{lr['diversity_score']:>10.4f} {lr['n_unique']:>10,} {dv:>+10.4f}"
        )
    lines.append("")

    lines.append("-" * 70)
    lines.append("Phase 2 Candidates (high diversity, minimal val_loss sacrifice)")
    lines.append("-" * 70)
    if not candidates:
        lines.append("  (no candidates found -- see Pareto frontier and lambda sweep)")
    else:
        for r in candidates[:10]:
            lines.append(
                f"  exp_id={r['exp_id']}  val_loss={r['val_loss']:.4f}  "
                f"diversity={r['diversity_score']:.4f}  n_unique={r['n_unique']:,}"
            )
            sp = r.get("sampling_params", {})
            if sp:
                lines.append("    sampling_params:")
                for dk in sorted(sp):
                    p = sp[dk]
                    lines.append(
                        f"      {dk:>12s}: lambda={p.get('lambda','?')}, "
                        f"omega={p.get('omega','?')}, eta={p.get('eta','?')}, "
                        f"epsilon={p.get('epsilon','?')}"
                    )
    lines.append("")

    lines.append("-" * 70)
    lines.append("Interpretation")
    lines.append("-" * 70)
    if rho_vd < -0.3:
        lines.append("  Tradeoff confirmed. The diversity penalty is needed.")
        found = False
        for lr in lambda_results:
            if lr["diversity_score"] > 0.85 and lr["lambda"] > 0:
                dv_pct = (lr["val_loss"] - min_vl) / min_vl * 100
                lines.append(
                    f"  lambda={lr['lambda']:.2f}: diversity={lr['diversity_score']:.4f}, "
                    f"val_loss sacrifice={dv_pct:.1f}%"
                )
                found = True
                break
        if not found:
            lines.append("  WARNING: no lambda achieves diversity > 0.85 -- tradeoff too steep")
    elif abs(rho_vd) < 0.1:
        lines.append("  No tradeoff. theta* is likely an outlier; diversity penalty will easily fix this.")
    elif rho_vd > 0.3:
        lines.append("  Unexpected positive correlation -- re-examine the theory.")
    else:
        lines.append(f"  Weak tradeoff (|rho|={abs(rho_vd):.4f}). Penalty may help but effect unclear.")
    lines.append("")

    # ── Extract actual θ* (search-selected) info ──
    cur_dpw = 0.0
    theta_unique = None
    theta_pred = None
    if summary is not None:
        cur_dpw = summary.get("config", {}).get("diversity_penalty_weight", 0.0)
        theta_unique = summary.get("sampling", {}).get("num_unique_docs")
        theta_pred = summary.get("metrics", {}).get("best_predicted_loss")
    if theta_unique is None and selected_doc_ids is not None:
        theta_unique = int(len(np.unique(selected_doc_ids)))

    # ── Extract baseline θ* info ──
    baseline_info = None
    if baseline_dir:
        bs_path = os.path.join(baseline_dir, "pipeline_summary.json")
        bd_dir = os.path.join(baseline_dir, "sampled_dataset")
        bd_file = os.path.join(baseline_dir, "sampled_dataset.parquet")
        bd_path = bd_dir if os.path.isdir(bd_dir) else bd_file
        if os.path.exists(bs_path) and os.path.exists(bd_path):
            with open(bs_path) as f:
                bs = json.load(f)
            b_dpw = bs.get("config", {}).get("diversity_penalty_weight", 0.0)
            b_unique = bs.get("sampling", {}).get("num_unique_docs")
            if b_unique is None:
                bdf = pd.read_parquet(resolve_parquet_source(bd_path), columns=["doc_id"])
                b_unique = int(len(np.unique(bdf["doc_id"].to_numpy())))
            b_pred = bs.get("metrics", {}).get("best_predicted_loss")
            baseline_info = {"dpw": b_dpw, "n_unique": b_unique, "pred": b_pred}
            print(f"[Diversity] Baseline θ*: λ={b_dpw}, n_unique={b_unique:,}, "
                  f"pred={b_pred}")
        else:
            print(f"[Diversity] Baseline files not found in {baseline_dir}"
                  " — skipping baseline comparison")

    lines.append("-" * 70)
    lines.append("Actual θ* (Search-Selected) vs Baseline")
    lines.append("-" * 70)
    lines.append(f"  Current θ*:")
    lines.append(f"    diversity_penalty_weight = {cur_dpw}")
    if theta_unique:
        lines.append(f"    num_unique_docs          = {theta_unique:,}")
    else:
        lines.append(f"    num_unique_docs          = N/A")
    if theta_pred is not None:
        lines.append(f"    best_predicted_loss      = {theta_pred:.4f}")
    if baseline_info:
        lines.append("")
        lines.append(f"  Baseline θ*:")
        lines.append(f"    diversity_penalty_weight = {baseline_info['dpw']}")
        lines.append(f"    num_unique_docs          = {baseline_info['n_unique']:,}")
        if baseline_info["pred"] is not None:
            lines.append(f"    best_predicted_loss      = {baseline_info['pred']:.4f}")
        if theta_pred is not None and baseline_info["pred"] is not None:
            q_change = ((theta_pred - baseline_info["pred"])
                        / baseline_info["pred"] * 100
                        if baseline_info["pred"] != 0 else 0.0)
            lines.append("")
            lines.append(f"  Improvement:")
            lines.append(f"    quality change = {q_change:+.2f}%  "
                         f"({baseline_info['pred']:.4f} -> {theta_pred:.4f})")
    lines.append("")

    lines.append("=" * 70)
    lines.append("End of Diversity Analysis")
    lines.append("=" * 70)

    out_path = os.path.join(exp_dir, "diversity_analysis.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [Text]  Saved: {out_path}")

    fig, ax = plt.subplots(figsize=(10, 7))
    dominated = [i for i in range(n) if i not in pareto_set]
    if dominated:
        ax.scatter(
            val_losses[dominated], div_scores[dominated],
            c="steelblue", alpha=0.5, s=20, label=f"dominated ({len(dominated)})",
        )
    ax.scatter(
        val_losses[pareto_idx], div_scores[pareto_idx],
        c="red", s=40, zorder=5, label=f"Pareto frontier ({len(pareto_idx)})",
    )
    pareto_vl = val_losses[pareto_idx]
    pareto_ds = div_scores[pareto_idx]
    ax.plot(pareto_vl, pareto_ds, "r--", lw=1.0, alpha=0.6)

    best_vl_idx = int(np.argmin(val_losses))
    ax.annotate(
        f"best proxy exp (exp {records[best_vl_idx]['exp_id']})\nval_loss={val_losses[best_vl_idx]:.4f}\ndiv={div_scores[best_vl_idx]:.4f}",
        xy=(val_losses[best_vl_idx], div_scores[best_vl_idx]),
        xytext=(30, -30), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=8, ha="left",
    )

    for r in candidates[:5]:
        ax.annotate(
            f"exp {r['exp_id']}",
            xy=(r["val_loss"], r["diversity_score"]),
            xytext=(15, 15), textcoords="offset points",
            fontsize=7, ha="left",
            arrowprops=dict(arrowstyle="->", color="green", alpha=0.6),
        )

    ax.set_xlabel("Proxy val_loss (lower = better quality)")
    ax.set_ylabel("Diversity score (higher = more unique docs)")
    ax.set_title(f"Quality-Diversity Tradeoff (Spearman rho={rho_vd:+.4f}, n={n})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_fig(fig, exp_dir, "fig_diversity_tradeoff.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    lam_arr = np.array([lr["lambda"] for lr in lambda_results])
    vl_arr = np.array([lr["val_loss"] for lr in lambda_results])
    ds_arr = np.array([lr["diversity_score"] for lr in lambda_results])

    ax1.plot(lam_arr, vl_arr, "o-", color="steelblue", lw=1.5)
    ax1.axhline(min_vl, color="gray", ls="--", lw=0.8, label=f"min val_loss={min_vl:.4f}")
    ax1.set_ylabel("val_loss of selected exp")
    ax1.set_title("Lambda Sweep: adjusted_loss = val_loss + lambda*(1-diversity)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    ax2.plot(lam_arr, ds_arr, "s-", color="coral", lw=1.5)
    ax2.axhline(0.85, color="green", ls="--", lw=0.8, label="diversity=0.85")
    ax2.set_xlabel("lambda (penalty weight)")
    ax2.set_ylabel("diversity_score of selected exp")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    plt.tight_layout()
    save_fig(fig, exp_dir, "fig_diversity_lambda_sweep.png")

    print(f"  [Diversity] Spearman(val_loss, diversity) = {rho_vd:+.4f}")
    print(f"  [Diversity] Pareto frontier: {len(pareto_idx)} points")
    print(f"  [Diversity] Candidates: {len(candidates)}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    args = parse_args()

    params_path = os.path.join(args.exp_dir, "optimal_parameters.json")
    summary_path = os.path.join(args.exp_dir, "pipeline_summary.json")
    sampled_dir = os.path.join(args.exp_dir, "sampled_dataset")
    sampled_file = os.path.join(args.exp_dir, "sampled_dataset.parquet")
    if os.path.isdir(sampled_dir):
        sampled_path = sampled_dir
    elif os.path.isfile(sampled_file):
        sampled_path = sampled_file
        print(f"  [info] Using legacy single-file format: {sampled_file}")
    else:
        sampled_path = sampled_dir

    print("=== QuaDMix Pipeline Output Analysis ===\n")

    # ── Load pipeline outputs ──
    print(f"[1/5] Loading optimal parameters: {params_path}")
    params = load_optimal_params(params_path)

    print(f"[2/5] Loading pipeline summary: {summary_path}")
    summary = load_pipeline_summary(summary_path)
    normalizer = summary.get("config", {}).get("normalizer", "rank")
    print(f"       Normalizer: {normalizer}")

    print(f"[3/5] Loading sampled dataset: {sampled_path}")
    sampled_df = pd.read_parquet(resolve_parquet_source(sampled_path))
    selected_doc_ids = sampled_df["doc_id"].to_numpy(dtype=np.int64)
    print(f"       Selected docs: {len(selected_doc_ids):,}")
    sampling_values_col = None
    if "sampling_value" in sampled_df.columns:
        sampling_values_col = sampled_df["sampling_value"].to_numpy(dtype=np.float64)

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

    # ── Pre-compute domain indices (avoids 28+ redundant domain_labels==m scans) ──
    print(f"\nPre-computing domain indices...")
    domain_indices = [
        np.where(domain_labels == m)[0] for m in range(num_domains)
    ]
    domain_counts = np.bincount(
        domain_labels[domain_labels >= 0], minlength=num_domains
    )

    # ── Recompute merged scores and ranks ──
    print(f"       Indexing unique domains...", flush=True)
    unique_domains = np.unique(domain_labels)
    print(f"[5/5] Recomputing merged quality scores (Eq.1)...", flush=True)
    merged_scores = compute_merged_quality_scores(
        mgr.quality_scores,
        domain_labels,
        params.merge_config,
        normalizer=normalizer,
        n_jobs=args.n_jobs,
        unique_domains=unique_domains,
    )

    print(f"       Computing quality ranks (Eq.2)...", flush=True)
    token_counts = mgr.estimate_token_counts()
    ranks = compute_quality_ranks(
        merged_scores, domain_labels, token_counts,
        seed=args.seed, n_jobs=args.n_jobs,
        unique_domains=unique_domains,
    )

    selected_ranks = ranks[selected_doc_ids]
    selected_domain_labels = domain_labels[selected_doc_ids]

    # ── Quality-length Spearman ρ per dimension (full corpus) ──
    char_counts = mgr.doc_char_counts
    quality_scores = mgr.quality_scores
    N_criteria = params.num_criteria
    print(f"       Computing quality-length ρ ({N_criteria} criteria)...", flush=True)
    quality_length_rhos = spearman_rho_vs_fixed(
        quality_scores, char_counts, N_criteria, args.n_jobs
    )
    print(f"       Quality-length ρ: {quality_length_rhos}")

    # ── Token length & crop analysis (optional, needs tokenizer) ──
    crop_stats = None
    tok_lens = None
    baseline_tok_lens = None
    tokenizer_path = resolve_tokenizer_path(args)
    if tokenizer_path:
        num_workers = args.num_workers or min(32, os.cpu_count() or 1)
        domain_col = None
        for c in ("category_name", "domain"):
            if c in sampled_df.columns:
                domain_col = c
                break
        print(f"\n  Tokenizing sampled docs (sample={args.tokenize_sample:,}, "
              f"workers={num_workers}, domain_col={domain_col})...")
        tok_lens, tok_dom_labels = _tokenize_sampled_docs(
            sampled_df, tokenizer_path, args.tokenize_sample,
            num_workers, args.tokenizer_threads, domain_col, domain_names,
        )
        if tok_lens is not None and len(tok_lens) > 0:
            print(f"  Tokenized {len(tok_lens):,} docs, "
                  f"mean={tok_lens.mean():.1f} tokens/doc")
            crop_stats = _packing_crop_analysis(
                tok_lens, args.seq_len, args.pack_buffer, args.max_pack_docs,
            )
            if crop_stats:
                print(f"  Crop rate: {crop_stats['crop_rate'] * 100:.1f}% "
                      f"(theoretical min: {crop_stats['theoretical_min_waste'] * 100:.1f}%, "
                      f"additional: {crop_stats['additional_waste'] * 100:.1f}%)")

        # Free text column — no longer needed (tok_lens already extracted).
        # This releases ~hundreds of GB before loading baseline.
        if "text" in sampled_df.columns:
            del sampled_df["text"]

        # ── Tokenize baseline for comparison (if --baseline-dir has sampled_dataset) ──
        baseline_tok_lens = None
        if args.baseline_dir:
            bl_dir = os.path.join(args.baseline_dir, "sampled_dataset")
            bl_file = os.path.join(args.baseline_dir, "sampled_dataset.parquet")
            if os.path.isdir(bl_dir):
                bl_path = bl_dir
            elif os.path.isfile(bl_file):
                bl_path = bl_file
            else:
                bl_path = None
            if bl_path:
                print(f"\n  Tokenizing baseline docs (streaming) from: {bl_path}")
                baseline_tok_lens, _ = _tokenize_from_parquet_shards(
                    bl_path, tokenizer_path, args.tokenize_sample,
                    num_workers, args.tokenizer_threads, domain_col, domain_names,
                )
                if baseline_tok_lens is not None and len(baseline_tok_lens) > 0:
                    print(f"  Baseline: {len(baseline_tok_lens):,} docs, "
                          f"mean={baseline_tok_lens.mean():.1f} tokens/doc")
                    if tok_lens is not None and len(tok_lens) > 0:
                        delta = (tok_lens.mean() - baseline_tok_lens.mean()) \
                                / baseline_tok_lens.mean() * 100
                        print(f"  Mean length delta: {delta:+.1f}% "
                              f"(this run {tok_lens.mean():.0f} vs "
                              f"baseline {baseline_tok_lens.mean():.0f})")
            else:
                print(f"\n  [info] --baseline-dir has no sampled_dataset/, "
                      f"skipping baseline tokenization")
    else:
        print("\n  [skip] No tokenizer found — token-length & crop analysis skipped "
              "(pass --tokenizer or set $NANOCHAT_MODEL_DIR)")

    # ── Generate figures ──
    print(f"\nGenerating outputs in: {args.exp_dir}")
    setup_style()
    fig_score = plot_quality_score_dist(
        merged_scores, domain_indices, domain_counts,
        domain_names, num_domains, args.exp_dir,
    )
    fig_rank = plot_quality_rank_dist(
        ranks,
        selected_ranks,
        domain_indices,
        selected_domain_labels,
        domain_counts,
        domain_names,
        num_domains,
        args.exp_dir,
    )

    print("  Generating duplication analysis figure...")
    fig_dup = plot_duplication_analysis(
        selected_doc_ids,
        selected_domain_labels,
        sampling_values_col,
        domain_names,
        num_domains,
        args.exp_dir,
    )

    print("  Generating quality-length decomposition figure...")
    fig_decomp = plot_quality_length_decomposition(
        params,
        quality_length_rhos,
        quality_names,
        domain_names,
        num_domains,
        args.exp_dir,
    )

    if tok_lens is not None and len(tok_lens) > 0:
        print("  Generating token length distribution figure...")
        fig_token_len = plot_token_length_dist(
            tok_lens, args.seq_len, tok_dom_labels,
            domain_names, num_domains, args.exp_dir,
            baseline_tok_lens=baseline_tok_lens,
        )
        if crop_stats:
            print("  Generating crop analysis figure...")
            fig_crop = plot_crop_analysis(crop_stats, args.exp_dir)
            print("  Generating packing boundary figure...")
            fig_boundaries = plot_packing_boundaries(
                crop_stats["boundaries"], args.exp_dir, args.seq_len,
            )
    else:
        fig_token_len = None
        fig_crop = None
        fig_boundaries = None

    # ── Generate analysis summary ──
    print("\nGenerating analysis summary...")
    summary_out = os.path.join(args.exp_dir, "analysis_summary.txt")
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
        domain_indices,
        domain_counts,
        selected_domain_labels,
        domain_names,
        quality_names,
        num_domains,
        fig_score,
        fig_rank,
        sampling_values_col=sampling_values_col,
        fig_dup=fig_dup,
        quality_length_rhos=quality_length_rhos,
        fig_decomp=fig_decomp,
        crop_stats=crop_stats,
        tok_lens=tok_lens,
        baseline_tok_lens=baseline_tok_lens,
        fig_token_len=fig_token_len,
        fig_crop=fig_crop,
        fig_boundaries=fig_boundaries,
    )
    print(f"  Saved: {summary_out}")

    # ── Proxy val_loss hard-vs-easy analysis (optional) ──
    proxy_dir = os.path.join(args.exp_dir, "proxy_experiments")
    if os.path.isdir(proxy_dir):
        _analyze_proxy_val_loss(
            args.exp_dir, domain_names, quality_names, args.proxy_hard_easy_count
        )
    else:
        print("\n[skip] proxy_experiments/ not found — val_loss hard-vs-easy analysis skipped")

    # ── Optimizer domain-proportion analysis (optional) ──
    if os.path.isdir(proxy_dir):
        _analyze_optimizer_domain_proportions(
            args.exp_dir, domain_labels, domain_names, quality_names,
            num_domains, N_criteria,
        )
    else:
        print("[skip] proxy_experiments/ not found — optimizer domain analysis skipped")

    # ── Diversity analysis (quality-diversity tradeoff) ──
    if os.path.isdir(proxy_dir):
        _analyze_diversity(
            args.exp_dir, mgr, domain_names, num_domains,
            summary=summary,
            selected_doc_ids=selected_doc_ids,
            baseline_dir=args.baseline_dir,
        )
    else:
        print("[skip] proxy_experiments/ not found — diversity analysis skipped")

    print("\nDone.")


if __name__ == "__main__":
    main()
