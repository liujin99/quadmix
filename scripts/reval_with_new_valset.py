#!/usr/bin/env python3
"""
Re-evaluate saved proxy experiments on a new validation set.

Loads trained 1M proxy model weights from a previous pipeline run,
switches to a different validation set, re-evaluates all experiments,
then re-runs LightGBM fitting, optimal search, final sampling, and
report generation.

Usage:
  python scripts/reval_with_new_valset.py \
      --result-dir result/quadmix_20260609_120000 \
      --val-set core \
      --preprocessed-dir /path/to/preprocessed \
      --output result/reval_core_20260610_150000

  python scripts/reval_with_new_valset.py \
      --result-dir result/quadmix_20260609_120000 \
      --val-path /path/to/custom_val.pt \
      --preprocessed-dir /path/to/preprocessed
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

from quadmix import QuaDMixConfig
from quadmix.core.types import ParameterSet, ProxyResult
from quadmix.data.metadata_manager import ShardMetadataManager
from quadmix.pipeline.real_pipeline import QuaDMixPipeline
from quadmix.constants import (
    DOMAIN_NAMES, QUALITY_NAMES, PROJECT_DIR, DEFAULT_TEMP_DIR,
    DEFAULT_VAL_DIR, HF_ENDPOINT, HF_RESOLVE,
    HF_OPENHERMES_DATASET, HF_OPENHERMES_FILENAME,
    HF_CORE_DATASET, HF_CORE_FILENAME, DEFAULT_EVAL_BUNDLE,
)

QUADMIX_DIR = PROJECT_DIR
QUADMIX_TEMP_DIR = DEFAULT_TEMP_DIR


def _hf_remote_size(repo_id: str, filename: str) -> int:
    url = f"{HF_ENDPOINT}/datasets/{repo_id}/resolve/main/{filename}"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return int(resp.headers.get("Content-Length", 0))
    except Exception:
        pass
    return 0


def _download_hf_file(repo_id: str, filename: str, local_path: str) -> bool:
    url = HF_RESOLVE.format(repo=repo_id, file=filename)
    print(f"[Setup] Downloading from:\n  {url}")
    try:
        urllib.request.urlretrieve(url, local_path)
        size_mb = os.path.getsize(local_path) / 1024**2
        print(f"[Setup] Downloaded: {local_path} ({size_mb:.0f} MB)")
        return True
    except Exception as e:
        print(f"[Setup] Download failed: {e}")
        return False


def _check_and_download(local_path: str, repo_id: str, filename: str) -> str:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    remote_size = _hf_remote_size(repo_id, filename)
    if os.path.exists(local_path):
        if remote_size > 0 and os.path.getsize(local_path) == remote_size:
            print(f"[Setup] Validation set OK: {local_path}")
            return local_path
        print(f"[Setup] Local file size mismatch, re-downloading...")
    if _download_hf_file(repo_id, filename, local_path):
        return local_path
    if os.path.exists(local_path):
        print(f"[Setup] Download failed, using existing local file")
        return local_path
    raise FileNotFoundError(f"Cannot obtain validation file: {local_path}")


def resolve_val_path(val_set: str, val_path: str) -> str:
    if val_path:
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation file not found: {val_path}")
        return val_path
    if val_set == "core":
        local = os.path.join(DEFAULT_VAL_DIR, HF_CORE_FILENAME)
        return _check_and_download(local, HF_CORE_DATASET, HF_CORE_FILENAME)
    local = os.path.join(DEFAULT_VAL_DIR, HF_OPENHERMES_FILENAME)
    return _check_and_download(local, HF_OPENHERMES_DATASET, HF_OPENHERMES_FILENAME)


def reconstruct_params_from_meta(meta: dict) -> ParameterSet:
    return ParameterSet.from_dict(meta["quality_weights"], meta["sampling_params"])


def build_parser():
    p = argparse.ArgumentParser(
        description="Re-evaluate saved proxy experiments on a new validation set",
    )
    p.add_argument("--result-dir", required=True,
                   help="Path to original pipeline result directory "
                        "(e.g. result/quadmix_20260609_120000)")
    p.add_argument("--preprocessed-dir", required=True,
                   help="Path to preprocessed shards directory")
    p.add_argument("--val-set", type=str, default="core",
                   choices=["openhermes", "core"],
                   help="New validation set to evaluate on (default: core)")
    p.add_argument("--val-path", type=str, default=None,
                   help="Path to custom validation .pt file (overrides --val-set)")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Output directory (default: result/reval_<valset>_<timestamp>)")
    p.add_argument("--device-type", type=str, default="cpu",
                   choices=["cpu", "cuda", "npu"],
                   help="Device for re-evaluation (default: cpu)")
    p.add_argument("--num-search", type=int, default=100000)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--target-tokens", type=float, default=0.0,
                   help="Target tokens in billions (0 = no target)")
    p.add_argument("--block-size", type=int, default=2048,
                   help="Block size for validation (must match training)")
    p.add_argument("--model-variant", type=str, default="tinyllama_1M",
                   help="Proxy model variant (must match original training)")
    return p


def main():
    args = build_parser().parse_args()

    result_dir = args.result_dir
    proxy_dir = os.path.join(result_dir, "proxy_experiments")
    if not os.path.isdir(proxy_dir):
        print(f"[Error] proxy_experiments not found: {proxy_dir}")
        return 1

    val_path = resolve_val_path(args.val_set, args.val_path)
    val_set_name = args.val_set if not args.val_path else os.path.basename(args.val_path)

    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(
            QUADMIX_DIR,
            f"result/reval_{val_set_name}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  QuaDMix Re-evaluation Pipeline")
    print(f"  Source:  {result_dir}")
    print(f"  Val set: {val_set_name} ({val_path})")
    print(f"  Output:  {output_dir}")
    print("=" * 70)

    t_start = time.time()
    stage_times = {}

    # ── Stage 0: Load metadata ─────────────────────────────
    _t = time.time()
    print(f"\n[Stage 0] Loading metadata from: {args.preprocessed_dir}")
    mm = ShardMetadataManager(args.preprocessed_dir)
    print(f"[Stage 0] {mm.num_docs:,} docs across {mm.num_shards} shards")
    domain_labels = mm.domain_labels
    quality_scores = mm.quality_scores
    token_counts = mm.estimate_token_counts()
    stage_times["stage0_load"] = time.time() - _t

    # ── Stage 1: Scan experiments ──────────────────────────
    _t = time.time()
    exp_dirs = sorted(
        d for d in os.listdir(proxy_dir)
        if d.startswith("exp_") and os.path.isdir(os.path.join(proxy_dir, d))
    )
    experiments = []
    skipped = 0
    for exp_name in exp_dirs:
        exp_path = os.path.join(proxy_dir, exp_name)
        meta_path = os.path.join(exp_path, "meta.json")
        model_path = os.path.join(exp_path, "model.pt")
        if not os.path.exists(meta_path):
            skipped += 1
            continue
        if not os.path.exists(model_path):
            skipped += 1
            continue
        experiments.append((exp_name, exp_path, meta_path, model_path))
    stage_times["stage1_scan"] = time.time() - _t
    print(f"\n[Stage 1] Found {len(experiments)} experiments with model.pt "
          f"({skipped} skipped)")

    if not experiments:
        print("[Error] No experiments with model.pt found.")
        print("  Ensure the original run saved model weights (model.pt in each exp dir).")
        return 1

    # ── Stage 2: Re-evaluate on new validation set ─────────
    _t = time.time()
    print(f"\n[Stage 2] Re-evaluating {len(experiments)} models on {val_set_name}...")

    from essential_proxy_runner import EssentialWebProxyRunner

    config = QuaDMixConfig(
        num_domains=10, num_quality_criteria=5,
        num_proxy_experiments=len(experiments),
        num_search_points=args.num_search,
        top_k_average=args.top_k,
        target_tokens=int(args.target_tokens * 1e9) if args.target_tokens > 0 else 0,
    )

    runner = EssentialWebProxyRunner(
        config=config,
        metadata_manager=mm,
        val_data_path=val_path,
        output_dir=os.path.join(output_dir, "proxy_experiments"),
        device_type=args.device_type,
        model_variant=args.model_variant,
        test_block_size=args.block_size,
        token_cache_dir=os.path.join(QUADMIX_TEMP_DIR, "token_cache"),
        checkpoint_interval=0,
    )

    results = []
    reval_meta = []
    for i, (exp_name, exp_path, meta_path, model_path) in enumerate(experiments):
        with open(meta_path) as f:
            meta = json.load(f)

        new_val_loss = runner.revalidate_from_saved(
            model_path, device_type=args.device_type,
        )

        old_val_loss = meta["val_loss"]
        params = reconstruct_params_from_meta(meta)

        new_meta = dict(meta)
        new_meta["val_loss"] = new_val_loss
        new_meta["val_ppl"] = float(np.exp(new_val_loss))
        new_meta["original_val_loss"] = old_val_loss
        new_meta["reval_source"] = result_dir
        new_meta["reval_val_set"] = val_set_name

        results.append(ProxyResult(
            parameters=params,
            validation_loss=new_val_loss,
            metadata=new_meta,
        ))
        reval_meta.append({
            "exp_name": exp_name,
            "old_val_loss": old_val_loss,
            "new_val_loss": new_val_loss,
            "delta": new_val_loss - old_val_loss,
        })

        if (i + 1) % 50 == 0 or i == len(experiments) - 1:
            losses_so_far = [r.validation_loss for r in results]
            print(f"  [{i+1}/{len(experiments)}] "
                  f"mean={np.mean(losses_so_far):.4f}, "
                  f"std={np.std(losses_so_far):.4f}")

    stage_times["stage2_reval"] = time.time() - _t
    print(f"[Stage 2] Re-evaluation: {stage_times['stage2_reval']:.1f}s")

    losses = np.array([r.validation_loss for r in results])
    print(f"  New loss stats: mean={losses.mean():.4f}, std={losses.std():.4f}, "
          f"min={losses.min():.4f}, max={losses.max():.4f}")

    old_losses = np.array([m["old_val_loss"] for m in reval_meta])
    corr = np.corrcoef(old_losses, losses)[0, 1] if len(losses) > 1 else 0
    print(f"  Correlation (old vs new): {corr:.4f}")

    # Save re-evaluation results
    os.makedirs(os.path.join(output_dir, "proxy_experiments"), exist_ok=True)
    for r, rm in zip(results, reval_meta):
        exp_out_dir = os.path.join(output_dir, "proxy_experiments", rm["exp_name"])
        os.makedirs(exp_out_dir, exist_ok=True)
        with open(os.path.join(exp_out_dir, "meta.json"), "w") as f:
            json.dump(r.metadata, f, indent=2)
        idx_path = os.path.join(
            proxy_dir, rm["exp_name"], "selected_indices.npy",
        )
        if os.path.exists(idx_path):
            import shutil
            shutil.copy2(idx_path, os.path.join(exp_out_dir, "selected_indices.npy"))

    # ── Stage 5: LightGBM Regression ────────────────────
    _t = time.time()
    print(f"\n[Stage 5] Training LightGBM regressor...")
    from quadmix.pipeline.optimizer import QuaDMixOptimizer
    pipeline = QuaDMixPipeline(config)
    pipeline._optimizer = QuaDMixOptimizer(config)
    pipeline._optimizer.add_proxy_results(results)
    pipeline._optimizer.train_regressor()
    stage_times["stage5_lightgbm"] = time.time() - _t
    print(f"[Stage 5] LightGBM: {stage_times['stage5_lightgbm']:.1f}s")

    # ── Stage 6: Optimal Parameter Search ───────────────
    _t = time.time()
    n_search = args.num_search
    top_k = args.top_k
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

    merged = compute_merged_quality_scores(
        quality_scores, domain_labels, optimal_params.merge_config,
    )
    final_ranks = compute_quality_ranks(merged, domain_labels, token_counts)
    selected_indices, sampling_values, _ = sample_with_optimal_params(
        final_ranks, domain_labels, optimal_params,
    )

    n_docs = len(domain_labels)
    print(f"  Original documents: {n_docs:,}")
    print(f"  Selected samples:   {len(selected_indices):,}")
    print(f"  Sampling ratio:     {len(selected_indices)/n_docs:.4f}x")

    orig_dist = np.bincount(domain_labels[domain_labels >= 0],
                             minlength=config.num_domains)
    sel_dist = np.bincount(
        domain_labels[selected_indices][domain_labels[selected_indices] >= 0],
        minlength=config.num_domains)
    print("\n  Domain distribution change:")
    for m in range(config.num_domains):
        if orig_dist[m] > 0:
            ratio = sel_dist[m] / orig_dist[m]
            name = DOMAIN_NAMES[m] if m < len(DOMAIN_NAMES) else f"D{m}"
            print(f"    [{m}] {name:>10s}: {orig_dist[m]:>7,} -> {sel_dist[m]:>7,}  ({ratio:.2f}x)")
    stage_times["stage7_final_sampling"] = time.time() - _t
    print(f"[Stage 7] Final sampling: {stage_times['stage7_final_sampling']:.1f}s")

    # ── Stage 8: Save Outputs ───────────────────────────
    _t = time.time()
    params_path = os.path.join(output_dir, "optimal_parameters.json")
    serialized = pipeline._serialize_params(optimal_params, DOMAIN_NAMES, QUALITY_NAMES)
    with open(params_path, "w") as f:
        json.dump(serialized, f, indent=2)

    elapsed = time.time() - t_start
    summary = {
        "config": {
            "num_domains": config.num_domains,
            "num_quality_criteria": config.num_quality_criteria,
            "num_proxy_experiments": len(experiments),
            "num_search_points": n_search,
            "val_set": val_set_name,
        },
        "metrics": {
            "train_r2": pipeline._optimizer.train_r2,
            "val_r2": pipeline._optimizer.val_r2,
            "val_mae": pipeline._optimizer.val_mae,
            "best_predicted_loss": float(predicted_losses.min()),
            "top_k_avg_loss": top_k_avg_loss,
        },
        "proxy_loss_stats": {
            "new_val_loss": {
                "mean": float(losses.mean()), "std": float(losses.std()),
                "min": float(losses.min()), "max": float(losses.max()),
            },
            "old_val_loss": {
                "mean": float(old_losses.mean()), "std": float(old_losses.std()),
                "min": float(old_losses.min()), "max": float(old_losses.max()),
            },
            "old_new_correlation": float(corr),
        },
        "reliability": {
            "val_r2_bootstrap_mean": pipeline._optimizer.val_r2_bootstrap_mean,
            "val_r2_ci_lower": pipeline._optimizer.val_r2_ci_lower,
            "val_r2_ci_upper": pipeline._optimizer.val_r2_ci_upper,
            "val_r2_ci_width": (
                pipeline._optimizer.val_r2_ci_upper - pipeline._optimizer.val_r2_ci_lower
                if pipeline._optimizer.val_r2_ci_lower is not None
                and pipeline._optimizer.val_r2_ci_upper is not None
                else None
            ),
            "sample_sufficient": pipeline._optimizer.sample_sufficient,
            "overfit_gap": pipeline._optimizer.overfit_gap,
            "n_features": pipeline._optimizer.n_features,
            "n_train_samples": getattr(pipeline._optimizer, "_n_train", None),
        },
        "sampling": {
            "num_original_docs": n_docs,
            "num_selected_docs": len(selected_indices),
            "sampling_ratio": len(selected_indices) / n_docs,
        },
        "reval": {
            "source_result_dir": result_dir,
            "original_val_set": _detect_original_val_set(result_dir),
            "new_val_set": val_set_name,
            "new_val_path": val_path,
            "num_experiments_reevaluated": len(experiments),
            "num_experiments_skipped": skipped,
            "old_loss_mean": float(old_losses.mean()),
            "old_loss_std": float(old_losses.std()),
            "new_loss_mean": float(losses.mean()),
            "new_loss_std": float(losses.std()),
            "loss_correlation": float(corr),
        },
        "elapsed_seconds": elapsed,
        "stage_times": {k: round(v, 1) for k, v in stage_times.items()},
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
    )

    reval_header = (
        f"\n## Re-evaluation Info\n\n"
        f"This report is based on re-evaluating saved proxy model weights "
        f"from a previous pipeline run on a **new validation set**.\n\n"
        f"| Field | Value |\n"
        f"|:------|:------|\n"
        f"| Source result | `{result_dir}` |\n"
        f"| Original val set | {_detect_original_val_set(result_dir)} |\n"
        f"| New val set | **{val_set_name}** |\n"
        f"| Experiments re-evaluated | {len(experiments)} |\n"
        f"| Loss correlation (old vs new) | {corr:.4f} |\n"
        f"| Old loss (mean +/- std) | {old_losses.mean():.4f} +/- {old_losses.std():.4f} |\n"
        f"| New loss (mean +/- std) | {losses.mean():.4f} +/- {losses.std():.4f} |\n\n"
    )
    report = report.replace("# QuaDMix", "# QuaDMix (Re-evaluated)\n" + reval_header + "\n# QuaDMix", 1)

    save_report(report, output_dir)
    stage_times["stage9_report"] = time.time() - _t
    print(f"[Stage 9] Report: {stage_times['stage9_report']:.1f}s")

    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Re-evaluation Complete! ({total_elapsed:.1f}s)")
    print(f"  Train R² = {pipeline._optimizer.train_r2:.4f}")
    print(f"  Val   R² = {pipeline._optimizer.val_r2:.4f}")
    print(f"  Val  MAE = {pipeline._optimizer.val_mae:.4f}")
    print(f"  Loss correlation (old vs new): {corr:.4f}")
    print(f"  Output: {output_dir}/")
    print(f"    ├── optimal_parameters.json")
    print(f"    ├── pipeline_summary.json")
    print(f"    ├── sampled_dataset.parquet")
    print(f"    ├── quadmix_report.md")
    print(f"    └── proxy_experiments/")
    print("=" * 70)
    return 0


def _detect_original_val_set(result_dir: str) -> str:
    summary_path = os.path.join(result_dir, "pipeline_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        return s.get("reval", {}).get("new_val_set",
               s.get("config", {}).get("val_set", "unknown"))
    exp_dirs = sorted(
        d for d in os.listdir(os.path.join(result_dir, "proxy_experiments"))
        if d.startswith("exp_")
    ) if os.path.isdir(os.path.join(result_dir, "proxy_experiments")) else []
    if exp_dirs:
        meta_path = os.path.join(result_dir, "proxy_experiments", exp_dirs[0], "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("assistant_loss"):
                return "openhermes (assistant-only loss)"
    return "unknown"


if __name__ == "__main__":
    sys.exit(main())
