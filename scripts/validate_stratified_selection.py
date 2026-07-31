#!/usr/bin/env python3
"""Length-stratified selection — post-hoc validation script.

Uses existing optimal_parameters.json + the full data pool to produce a
sampled_dataset.parquet where quality ranking (Eq.2) is done within
(domain × length-stratum) cells instead of within the full domain.

This is a standalone script: no pipeline code is modified.  It reuses
the same Eq.1 (merge), Eq.3 (sigmoid sampling), and save logic as the
production pipeline, only swapping Eq.2 (compute_quality_ranks →
compute_stratified_quality_ranks).

Usage:
  python scripts/validate_stratified_selection.py \
      --preprocessed-dir /path/to/stem/parquets \
      --params-file result/quadmix_20260728_201517/optimal_parameters.json \
      --schema configs/schema_stem.yaml \
      --output result/stratified_validation
"""

import argparse
import json
import os
import shutil
import sys
import time

os.environ.setdefault('MALLOC_ARENA_MAX', '4')

try:
    import quadmix  # noqa: F401
except ImportError:
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'),
    )

import numpy as np

from quadmix.core.types import ParameterSet
from quadmix.data.metadata_manager import ShardMetadataManager
from quadmix.data.dataset_schema import DatasetSchema
from quadmix.core.quality_merger import compute_merged_quality_scores
from quadmix.core.quality_rank import compute_stratified_quality_ranks
from quadmix.sampling.batch_sampler import (
    sample_with_optimal_params,
    save_sampled_dataset,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)


def reconstruct_params_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return ParameterSet.from_dict(data["quality_weights"], data["sampling_params"])


def resolve_schema_path(schema_arg):
    if os.path.isabs(schema_arg):
        return schema_arg
    return os.path.join(_PROJECT_ROOT, schema_arg)


def parse_args():
    p = argparse.ArgumentParser(
        description="Length-stratified selection validation",
    )
    p.add_argument("--preprocessed-dir", required=True,
                   help="Directory with parquet shards (same as pipeline --preprocessed-dir)")
    p.add_argument("--params-file", required=True,
                   help="Path to optimal_parameters.json from a prior QuaDMix run")
    p.add_argument("--schema", required=True,
                   help="Schema YAML path (relative to project root or absolute)")
    p.add_argument("--output", "-o", default=None,
                   help="Output directory (default: result/stratified_<timestamp>)")
    p.add_argument("--n-strata", type=int, default=4,
                   help="Number of length strata per domain (default: 4)")
    p.add_argument("--target-tokens", type=float, default=0.0,
                   help="Target tokens in billions (0 = no limit)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = args.output or os.path.join(
        _PROJECT_ROOT, f"result/stratified_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  Length-Stratified Selection Validation")
    print(f"  Data:    {args.preprocessed_dir}")
    print(f"  Params:  {args.params_file}")
    print(f"  Schema:  {args.schema}")
    print(f"  Output:  {output_dir}")
    print(f"  Strata:  {args.n_strata}")
    print(f"  Seed:    {args.seed}")
    print("=" * 70)

    t_start = time.time()

    # ── Stage 1: Load metadata ──────────────────────────────
    _t = time.time()
    print(f"\n[Stage 1] Loading metadata...")
    schema = DatasetSchema.from_yaml(resolve_schema_path(args.schema))
    mm = ShardMetadataManager(args.preprocessed_dir, schema=schema)
    domain_names = mm.detected_domain_names
    quality_names = mm.detected_quality_names
    domain_labels = mm.domain_labels
    quality_scores = mm.quality_scores
    token_counts = mm.estimate_token_counts()
    char_counts = mm.doc_char_counts
    n_docs = mm.num_docs

    if char_counts is None or len(char_counts) == 0:
        print("ERROR: doc_char_counts not available in metadata.")
        return 1

    print(f"  {n_docs:,} docs across {mm.num_shards} shards")
    print(f"  {mm.num_domains} domains, {mm.num_quality_criteria} quality criteria")
    print(f"  Stage 1: {time.time()-_t:.1f}s")

    # ── Stage 2: Load optimal parameters ────────────────────
    _t = time.time()
    print(f"\n[Stage 2] Loading optimal parameters...")
    optimal_params = reconstruct_params_from_json(args.params_file)
    for m, sc in enumerate(optimal_params.sampling_configs):
        name = domain_names[m] if m < len(domain_names) else f"D{m}"
        print(f"  [{m}] {name}: λ={sc.lambda_:.2f}, ω={sc.omega:.6f}, "
              f"η={sc.eta:.4f}, ε={sc.epsilon:.6f}")
    print(f"  Stage 2: {time.time()-_t:.1f}s")

    # ── Stage 3: Eq.1 — Merge quality scores ────────────────
    _t = time.time()
    print(f"\n[Stage 3] Merging quality scores (Eq.1)...")
    merged = compute_merged_quality_scores(
        quality_scores, domain_labels, optimal_params.merge_config, n_jobs=-1,
    )
    print(f"  Merged: [{merged.min():.4f}, {merged.max():.4f}]")
    print(f"  Stage 3: {time.time()-_t:.1f}s")

    # ── Stage 4: Eq.2 (stratified) — Stratified quality ranks
    _t = time.time()
    print(f"\n[Stage 4] Computing stratified quality ranks (Eq.2-stratified)...")
    print(f"  {args.n_strata} strata per domain, per-domain char_count boundaries")
    final_ranks = compute_stratified_quality_ranks(
        merged, domain_labels, token_counts, char_counts,
        num_strata=args.n_strata, seed=args.seed, n_jobs=-1,
    )
    print(f"  Ranks: [{final_ranks.min():.4f}, {final_ranks.max():.4f}]")
    print(f"  Stage 4: {time.time()-_t:.1f}s")

    # ── Stage 5: Eq.3 — Sigmoid sampling ────────────────────
    _t = time.time()
    print(f"\n[Stage 5] Applying sigmoid sampling (Eq.3)...")
    rng = np.random.default_rng(args.seed)
    selected_indices, sampling_values, _ = sample_with_optimal_params(
        final_ranks, domain_labels, optimal_params, rng=rng,
    )
    print(f"  Original docs:  {n_docs:,}")
    print(f"  Selected docs:  {len(selected_indices):,}")
    print(f"  Sampling ratio: {len(selected_indices)/n_docs:.4f}x")
    print(f"  Stage 5: {time.time()-_t:.1f}s")

    # ── Stage 6: Target token adjustment ────────────────────
    target_tokens = int(args.target_tokens * 1e9) if args.target_tokens > 0 else 0
    if target_tokens > 0:
        actual_tokens = float(np.sum(token_counts[selected_indices]))
        print(f"\n[Stage 6] Target token adjustment:")
        print(f"  θ* produces: {actual_tokens/1e9:.2f}B tokens")
        print(f"  Target:      {target_tokens/1e9:.1f}B tokens")
        if actual_tokens > target_tokens:
            keep_prob = target_tokens / actual_tokens
            rng_discard = np.random.default_rng(args.seed + 1)
            keep_mask = rng_discard.random(len(selected_indices)) < keep_prob
            selected_indices = selected_indices[keep_mask]
            final_tokens = float(np.sum(token_counts[selected_indices]))
            print(f"  Uniform discard (keep_prob={keep_prob:.4f}) → "
                  f"{final_tokens/1e9:.2f}B tokens")
        elif actual_tokens < target_tokens * 0.95:
            print(f"  [WARN] θ* produces less than target")
        else:
            print(f"  Accept θ* result (within tolerance)")

    # ── Distribution analysis ───────────────────────────────
    print(f"\n{'─' * 70}")
    print("Distribution Analysis")
    print(f"{'─' * 70}")

    num_domains = optimal_params.num_domains
    orig_dist = np.bincount(domain_labels[domain_labels >= 0],
                            minlength=num_domains)
    sel_dist = np.bincount(
        domain_labels[selected_indices][domain_labels[selected_indices] >= 0],
        minlength=num_domains,
    )

    print(f"\n  Domain distribution:")
    for m in range(num_domains):
        if orig_dist[m] > 0:
            name = domain_names[m] if m < len(domain_names) else f"D{m}"
            ratio = sel_dist[m] / orig_dist[m]
            print(f"    [{m}] {name:>10s}: {orig_dist[m]:>7,} → "
                  f"{sel_dist[m]:>7,}  ({ratio:.2f}x)")

    sel_chars = char_counts[selected_indices]
    total_tokens_est = float(np.sum(token_counts[selected_indices]))
    unique_indices = np.unique(selected_indices)

    print(f"\n  Length distribution (selected docs):")
    print(f"    char_count  mean={sel_chars.mean():.0f}  "
          f"median={np.median(sel_chars):.0f}")
    print(f"                p25={np.percentile(sel_chars, 25):.0f}  "
          f"p75={np.percentile(sel_chars, 75):.0f}  "
          f"p90={np.percentile(sel_chars, 90):.0f}")
    print(f"                min={sel_chars.min()}  max={sel_chars.max()}")
    print(f"    Total tokens:  {total_tokens_est/1e9:.2f}B")
    print(f"    Total docs:    {len(selected_indices):,}")
    print(f"    Unique docs:   {len(unique_indices):,}")
    print(f"    Repeat rate:   "
          f"{(1 - len(unique_indices)/len(selected_indices))*100:.1f}%")

    # ── Stage 7: Save outputs ────────────────────────────────
    _t = time.time()
    print(f"\n[Stage 7] Saving outputs...")

    sampled_path = os.path.join(output_dir, "sampled_dataset.parquet")

    quality_dict = None
    if quality_scores is not None and quality_names:
        quality_dict = {
            name: quality_scores[:, i]
            for i, name in enumerate(quality_names)
        }

    save_sampled_dataset(
        get_text_fn=mm.read_texts,
        num_total_docs=n_docs,
        selected_indices=selected_indices,
        output_path=sampled_path,
        domain_labels=domain_labels,
        quality_ranks=final_ranks,
        sampling_values=sampling_values,
        format="parquet",
        text_col=schema.text_col,
        domain_col=schema.domain_col,
        quality_scores=quality_dict,
    )

    shutil.copy2(args.params_file,
                 os.path.join(output_dir, "optimal_parameters.json"))

    summary = {
        "method": "length_stratified",
        "params_file": args.params_file,
        "preprocessed_dir": args.preprocessed_dir,
        "n_strata": args.n_strata,
        "seed": args.seed,
        "num_original_docs": n_docs,
        "num_selected_docs": len(selected_indices),
        "num_unique_docs": len(unique_indices),
        "sampling_ratio": len(selected_indices) / n_docs,
        "estimated_tokens": total_tokens_est,
        "estimated_tokens_billions": round(total_tokens_est / 1e9, 3),
        "char_count_stats": {
            "mean": float(sel_chars.mean()),
            "median": float(np.median(sel_chars)),
            "p25": float(np.percentile(sel_chars, 25)),
            "p75": float(np.percentile(sel_chars, 75)),
            "p90": float(np.percentile(sel_chars, 90)),
        },
        "domain_distribution": {
            domain_names[m] if m < len(domain_names) else f"D{m}": {
                "original": int(orig_dist[m]),
                "selected": int(sel_dist[m]),
                "ratio": round(sel_dist[m] / orig_dist[m], 4)
                         if orig_dist[m] > 0 else 0,
            }
            for m in range(num_domains) if orig_dist[m] > 0
        },
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    summary_path = os.path.join(output_dir, "stratified_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating,))
                  else int(x) if isinstance(x, (np.integer,)) else x)

    print(f"\n{'=' * 70}")
    print(f"  Stratified Selection Complete! ({time.time()-t_start:.1f}s)")
    print(f"  Selected: {len(selected_indices):,} docs "
          f"({total_tokens_est/1e9:.2f}B tokens)")
    print(f"  Output: {output_dir}/")
    print(f"    ├── sampled_dataset.parquet")
    print(f"    ├── optimal_parameters.json")
    print(f"    └── stratified_summary.json")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
