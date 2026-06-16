#!/usr/bin/env python3
"""
Resume QuaDMix pipeline from Stage 5 (LightGBM regression).

Loads proxy experiment results from disk (meta.json) and runs:
  Stage 5: LightGBM regression
  Stage 6: Optimal parameter search
  Stage 7: Final sampling
  Stage 8: Save outputs + report

Usage:
  python scripts/runners/resume_from_stage5.py \
      --proxy-dir /path/to/output/proxy_experiments \
      --preprocessed-dir /path/to/preprocessed \
      --output /path/to/output
"""

import argparse
import json
import os
import sys
import time
try:
    import quadmix
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import numpy as np

from quadmix import QuaDMixConfig
from quadmix.core.types import ParameterSet, ProxyResult
from quadmix.data.metadata_manager import ShardMetadataManager
from quadmix.pipeline.real_pipeline import QuaDMixPipeline
from quadmix.constants import DOMAIN_NAMES, QUALITY_NAMES


def load_proxy_results(proxy_dir: str):
    results = []
    exp_dirs = sorted(
        d for d in os.listdir(proxy_dir)
        if d.startswith("exp_") and os.path.isdir(os.path.join(proxy_dir, d))
    )
    for exp_name in exp_dirs:
        meta_path = os.path.join(proxy_dir, exp_name, "meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        params = ParameterSet.from_dict(meta["quality_weights"], meta["sampling_params"])
        per_task_losses = meta.get("per_task_losses")
        results.append(ProxyResult(
            parameters=params,
            validation_loss=meta["val_loss"],
            per_task_losses=per_task_losses,
            metadata=meta,
        ))

    return results


def build_parser():
    p = argparse.ArgumentParser(description="Resume QuaDMix from Stage 5")
    p.add_argument("--proxy-dir", required=True,
                   help="Path to proxy_experiments/ directory")
    p.add_argument("--preprocessed-dir", required=True,
                   help="Path to preprocessed shards directory")
    p.add_argument("--output", "-o", default=None,
                   help="Output directory (default: result/reopt_<timestamp>)")
    p.add_argument("--num-search", type=int, default=5000)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--target-tokens", type=float, default=0.0,
                   help="Target tokens in billions (0 = no target)")
    return p


def main():
    args = build_parser().parse_args()

    if args.output is None:
        args.output = os.path.join(
            "result",
            f"reopt_{time.strftime('%Y%m%d_%H%M%S')}",
        )

    print(f"[Resume] Loading proxy results from: {args.proxy_dir}")
    results = load_proxy_results(args.proxy_dir)
    print(f"[Resume] Loaded {len(results)} experiment results")

    losses = np.array([r.validation_loss for r in results])
    print(f"  Loss stats: mean={losses.mean():.4f}, std={losses.std():.4f}, "
          f"min={losses.min():.4f}, max={losses.max():.4f}")

    has_per_task = all(r.per_task_losses is not None for r in results)
    if has_per_task:
        tasks = sorted(results[0].per_task_losses.keys())
        per_task_means = {}
        per_task_stds = {}
        for task in tasks:
            task_losses = np.array([r.per_task_losses[task] for r in results])
            per_task_means[task] = float(np.mean(task_losses))
            per_task_stds[task] = float(np.std(task_losses))
        print(f"\n  Per-task loss stats ({len(tasks)} tasks):")
        print(f"    {'Task':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"    {'-'*70}")
        for task in sorted(tasks, key=lambda t: -per_task_means[t]):
            task_losses = np.array([r.per_task_losses[task] for r in results])
            print(f"    {task:<30} {per_task_means[task]:>8.4f} {per_task_stds[task]:>8.4f} "
                  f"{np.min(task_losses):>8.4f} {np.max(task_losses):>8.4f}")

    n_exp = len(results)
    n_search = args.num_search
    top_k = args.top_k

    config = QuaDMixConfig(
        num_domains=10, num_quality_criteria=5,
        num_proxy_experiments=n_exp, num_search_points=n_search,
        top_k_average=top_k,
        target_tokens=int(args.target_tokens * 1e9) if args.target_tokens > 0 else 0,
    )

    pipeline = QuaDMixPipeline(config)

    print(f"\n[Resume] Loading metadata from: {args.preprocessed_dir}")
    mm = ShardMetadataManager(args.preprocessed_dir)
    print(f"[Resume] {mm.num_docs:,} docs across {mm.num_shards} shards")

    domain_labels = mm.domain_labels
    quality_scores = mm.quality_scores
    token_counts = mm.estimate_token_counts()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()
    stage_times = {}

    # ── Stage 5: LightGBM Regression ────────────────────
    _t = time.time()
    print(f"\n[Stage 5] Training LightGBM regressor...")
    from quadmix.pipeline.optimizer import QuaDMixOptimizer
    pipeline._optimizer = QuaDMixOptimizer(config)
    pipeline._optimizer.add_proxy_results(results)
    pipeline._optimizer.train_regressor()
    stage_times["stage5_lightgbm"] = time.time() - _t
    print(f"[Stage 5] LightGBM: {stage_times['stage5_lightgbm']:.1f}s")

    # ── Stage 6: Optimal Parameter Search ───────────────
    _t = time.time()
    print(f"\n[Stage 6] Searching optimal parameters ({n_search} points)...")
    optimal_params, candidates, predicted_losses = pipeline._optimizer.search_optimal(
        n_search_points=n_search, top_k=top_k,
    )
    stage_times["stage6_search"] = time.time() - _t
    print(f"[Stage 6] Search: {stage_times['stage6_search']:.1f}s")
    print(f"  Best predicted loss: {predicted_losses.min():.4f}")
    k = config.top_k_average
    top_indices = np.argsort(predicted_losses)[:k]
    top_k_avg_loss = float(predicted_losses[top_indices].mean())

    # ── Stage 7: Final Sampling ─────────────────────────
    _t = time.time()
    print(f"\n[Stage 7] Applying optimal parameters for final sampling...")
    from quadmix.core.quality_merger import compute_merged_quality_scores
    from quadmix.core.quality_rank import compute_quality_ranks
    from quadmix.sampling.batch_sampler import sample_with_optimal_params

    print(f"[Stage 7] Merging quality scores (Eq.1)...")
    merged = compute_merged_quality_scores(
        quality_scores, domain_labels, optimal_params.merge_config,
    )
    print(f"  Merged scores: [{merged.min():.4f}, {merged.max():.4f}]")

    print(f"[Stage 7] Computing quality ranks (Eq.2)...")
    final_ranks = compute_quality_ranks(merged, domain_labels, token_counts)
    print(f"  Quality ranks: [{final_ranks.min():.4f}, {final_ranks.max():.4f}]")
    selected_indices, sampling_values, _ = sample_with_optimal_params(
        final_ranks, domain_labels, optimal_params,
    )

    n_docs = len(domain_labels)
    print(f"  Original documents: {n_docs:,}")
    print(f"  Selected samples:   {len(selected_indices):,}")
    print(f"  Sampling ratio:     {len(selected_indices)/n_docs:.4f}x")
    stage_times["stage7_final_sampling"] = time.time() - _t
    print(f"[Stage 7] Final sampling: {stage_times['stage7_final_sampling']:.1f}s")

    # ── Stage 8: Save Outputs ───────────────────────────
    _t = time.time()
    params_path = os.path.join(output_dir, "optimal_parameters.json")
    serialized = pipeline._serialize_params(optimal_params, DOMAIN_NAMES, QUALITY_NAMES)
    with open(params_path, "w") as f:
        json.dump(serialized, f, indent=2)
    print(f"\n[Stage 8] Optimal parameters saved to: {params_path}")

    elapsed = time.time() - t_start
    summary = {
        "config": {
            "num_domains": config.num_domains,
            "num_quality_criteria": config.num_quality_criteria,
            "num_proxy_experiments": n_exp,
            "num_search_points": n_search,
        },
        "metrics": {
            "aggregate_train_r2": pipeline._optimizer.train_r2,
            "aggregate_val_r2": pipeline._optimizer.val_r2,
            "aggregate_val_mae": pipeline._optimizer.val_mae,
            "ensemble_val_r2": pipeline._optimizer.ensemble_val_r2,
            "ensemble_val_mae": pipeline._optimizer.ensemble_val_mae,
            "best_predicted_loss": float(predicted_losses.min()),
            "top_k_avg_loss": top_k_avg_loss,
        },
        "sampling": {
            "num_original_docs": n_docs,
            "num_selected_docs": len(selected_indices),
            "sampling_ratio": len(selected_indices) / n_docs,
        },
        "per_task_analysis": pipeline._optimizer.per_task_analysis,
        "elapsed_seconds": elapsed,
        "stage_times": {k: round(v, 1) for k, v in stage_times.items()},
        "resumed_from": "stage5",
    }
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)

    sampled_texts = mm.read_texts(selected_indices)
    import pandas as pd
    sel_domain = domain_labels[selected_indices]
    sel_rank = final_ranks[selected_indices]
    sel_sv = sampling_values[selected_indices]
    sel_weights = 1.0 / np.maximum(sel_sv, 1e-10)

    sampled_path = os.path.join(output_dir, "sampled_dataset.parquet")
    pd.DataFrame({
        "text": sampled_texts,
        "doc_id": selected_indices,
        "domain": sel_domain,
        "quality_rank": sel_rank,
        "sampling_weight": sel_weights,
        "sampling_value": sel_sv,
    }).to_parquet(sampled_path, index=False)
    print(f"[Stage 8] Sampled dataset saved: {sampled_path}")
    stage_times["stage8_save"] = time.time() - _t
    print(f"[Stage 8] Save outputs: {stage_times['stage8_save']:.1f}s")

    # ── Stage 9: Report ──
    _t = time.time()
    print(f"\n[Stage 9] Generating comparison report...")
    from quadmix.pipeline.report import generate_report, save_report
    report = generate_report(
        output_dir=output_dir,
        data_path=args.preprocessed_dir,
        optimal_params=optimal_params,
        optimal_selected_indices=selected_indices,
        domain_labels=domain_labels,
        token_counts=token_counts,
        num_domains=config.num_domains,
        num_criteria=config.num_quality_criteria,
        config=summary["config"],
        metrics=summary["metrics"],
        elapsed=elapsed,
        use_sharded=True,
        reliability=summary.get("reliability"),
        per_task_analysis=summary.get("per_task_analysis"),
    )
    save_report(report, output_dir)
    stage_times["stage9_report"] = time.time() - _t
    print(f"[Stage 9] Report: {stage_times['stage9_report']:.1f}s")

    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Resume Complete! ({total_elapsed:.1f}s)")
    print(f"  Aggregate Val R² = {pipeline._optimizer.val_r2:.4f} (diagnostic)")
    ens_r2 = pipeline._optimizer.ensemble_val_r2
    ens_mae = pipeline._optimizer.ensemble_val_mae
    if ens_r2 is not None:
        print(f"  Overall   Val R² = {ens_r2:.4f}, MAE = {ens_mae:.4f} (used for search)")
    print(f"  Output: {output_dir}/")
    print(f"    ├── optimal_parameters.json")
    print(f"    ├── pipeline_summary.json")
    print(f"    ├── sampled_dataset.parquet")
    print(f"    └── quadmix_report.md")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
