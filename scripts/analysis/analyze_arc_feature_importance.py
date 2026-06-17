"""
Analyze feature importance for arc_easy and arc_challenge per-task LightGBM models.

Loads experiment meta.json files from proxy_validation directory, reconstructs
the 90-dim feature vector, trains LightGBM, and prints feature importance
grouped by domain.

Usage:
    python scripts/analysis/analyze_arc_feature_importance.py \
        --exp_dir /path/to/proxy_validation
"""

import argparse
import json
import os
import sys

import numpy as np
from lightgbm import LGBMRegressor

DOMAIN_NAMES = [
    "Industrial arts, Technology, and Engineering",
    "Social sciences",
    "Science and Natural history",
    "Religion",
    "Philology; or, Language and languages",
    "Literature",
    "History and Geography",
    "General works, books and libraries, information sciences",
    "Philosophy and psychology",
    "Arts",
]

DOMAIN_SHORT = [
    "Industrial", "Social", "Science", "Religion", "Philology",
    "Literature", "History", "General", "Philosophy", "Arts",
]

QUALITY_KEYS = ["dclm", "fineweb_edu_approx", "english", "eai_general_math", "eai_open_web_math"]
SAMPLING_KEYS = ["lambda", "omega", "eta", "epsilon"]

TARGET_TASKS = ["arc_easy", "arc_challenge"]


def find_exp_root(exp_dir: str) -> str:
    exp_dirs = [d for d in os.listdir(exp_dir)
                if d.startswith("exp_") and os.path.isdir(os.path.join(exp_dir, d))]
    if exp_dirs:
        return exp_dir
    for sub in os.listdir(exp_dir):
        sub_path = os.path.join(exp_dir, sub)
        if os.path.isdir(sub_path):
            sub_exps = [d for d in os.listdir(sub_path)
                        if d.startswith("exp_") and os.path.isdir(os.path.join(sub_path, d))]
            if sub_exps:
                return sub_path
    return exp_dir


def load_experiments(exp_dir: str):
    root = find_exp_root(exp_dir)
    if root != exp_dir:
        print(f"  Auto-detected experiment root: {root}")
    exp_dirs = sorted([
        d for d in os.listdir(root)
        if d.startswith("exp_") and os.path.isdir(os.path.join(root, d))
    ])
    print(f"  Found {len(exp_dirs)} experiment directories")

    features = []
    all_losses = {}
    skipped_no_meta = 0
    skipped_no_tasks = 0

    for exp_name in exp_dirs:
        meta_path = os.path.join(root, exp_name, "meta.json")
        if not os.path.exists(meta_path):
            skipped_no_meta += 1
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        per_task = meta.get("per_task_losses")
        if not per_task:
            skipped_no_tasks += 1
            continue

        qw = meta.get("quality_weights", {})
        sp = meta.get("sampling_params", {})

        if not qw and not sp:
            if len(features) == 0:
                print(f"  DEBUG: First meta.json keys: {list(meta.keys())}")
                print(f"  DEBUG: quality_weights keys: {list(qw.keys()) if qw else 'MISSING'}")
                print(f"  DEBUG: sampling_params keys: {list(sp.keys()) if sp else 'MISSING'}")

        feat = []
        for domain_name in DOMAIN_NAMES:
            dw = qw.get(domain_name, {})
            for k in QUALITY_KEYS:
                feat.append(dw.get(k, 0.0))

        for domain_name in DOMAIN_NAMES:
            s = sp.get(domain_name, {})
            for k in SAMPLING_KEYS:
                feat.append(s.get(k, 0.0))

        features.append(feat)

        for task, loss in per_task.items():
            if task not in all_losses:
                all_losses[task] = []
            all_losses[task].append(loss)

    if skipped_no_meta:
        print(f"  Skipped {skipped_no_meta} experiments (no meta.json)")
    if skipped_no_tasks:
        print(f"  Skipped {skipped_no_tasks} experiments (no per_task_losses)")

    if not features:
        print("  ERROR: No experiments loaded!")
        if exp_dirs:
            sample_meta_path = os.path.join(root, exp_dirs[0], "meta.json")
            if os.path.exists(sample_meta_path):
                with open(sample_meta_path) as f:
                    sample = json.load(f)
                print(f"  DEBUG: Sample meta.json keys: {list(sample.keys())}")
                print(f"  DEBUG: Has per_task_losses: {'per_task_losses' in sample}")
                print(f"  DEBUG: Has quality_weights: {'quality_weights' in sample}")
                if "quality_weights" in sample:
                    print(f"  DEBUG: quality_weights domain keys: {list(sample['quality_weights'].keys())[:3]}...")
        return np.array([]).reshape(0, 90), {}

    return np.array(features), all_losses


def build_feature_names():
    names = []
    for m, domain in enumerate(DOMAIN_SHORT):
        for n, qk in enumerate(QUALITY_KEYS):
            names.append(f"alpha_{m}_{n} ({domain}/{qk})")
    for m, domain in enumerate(DOMAIN_SHORT):
        for sk in SAMPLING_KEYS:
            names.append(f"{sk}_{m} ({domain})")
    return names


def train_and_analyze(X, losses, task_name, feature_names):
    y = np.array(losses)
    n = len(y)

    print(f"\n{'='*80}")
    print(f"Task: {task_name}")
    print(f"{'='*80}")
    print(f"Samples: {n}")
    print(f"Loss stats: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")

    rng = np.random.RandomState(42)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    model = LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X[train_idx], y[train_idx])

    train_r2 = model.score(X[train_idx], y[train_idx])
    val_r2 = model.score(X[val_idx], y[val_idx])
    print(f"Train R²: {train_r2:.4f}")
    print(f"Val   R²: {val_r2:.4f}")

    importances = model.feature_importances_
    total = importances.sum()

    print(f"\n--- Top 20 Features ---")
    print(f"{'Rank':<5} {'Feature':<55} {'Importance':>10} {'%':>7}")
    print("-" * 80)
    sorted_idx = np.argsort(importances)[::-1]
    for rank, i in enumerate(sorted_idx[:20]):
        pct = importances[i] / total * 100
        print(f"{rank+1:<5} {feature_names[i]:<55} {importances[i]:>10} {pct:>6.2f}%")

    print(f"\n--- Domain-level Aggregation ---")
    print(f"{'Domain':<15} {'Alpha %':>10} {'Sampling %':>10} {'Total %':>10}")
    print("-" * 50)
    domain_totals = []
    for m in range(10):
        alpha_start = m * 5
        alpha_end = alpha_start + 5
        alpha_imp = importances[alpha_start:alpha_end].sum()

        samp_start = 50 + m * 4
        samp_end = samp_start + 4
        samp_imp = importances[samp_start:samp_end].sum()

        total_imp = alpha_imp + samp_imp
        domain_totals.append((m, alpha_imp, samp_imp, total_imp))

    domain_totals.sort(key=lambda x: -x[3])
    for m, alpha_imp, samp_imp, total_imp in domain_totals:
        alpha_pct = alpha_imp / total * 100
        samp_pct = samp_imp / total * 100
        total_pct = total_imp / total * 100
        print(f"{DOMAIN_SHORT[m]:<15} {alpha_pct:>9.2f}% {samp_pct:>9.2f}% {total_pct:>9.2f}%")

    return model, importances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True, help="Path to proxy_validation directory")
    parser.add_argument("--tasks", nargs="+", default=TARGET_TASKS, help="Tasks to analyze")
    args = parser.parse_args()

    print(f"Loading experiments from: {args.exp_dir}")
    X, all_losses = load_experiments(args.exp_dir)
    if len(X) == 0:
        print("ERROR: No experiments loaded. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(X)} experiments with {X.shape[1]} features")

    feature_names = build_feature_names()

    for task in args.tasks:
        if task not in all_losses:
            print(f"\nWARNING: Task '{task}' not found in experiment data")
            print(f"Available tasks: {sorted(all_losses.keys())}")
            continue
        train_and_analyze(X, all_losses[task], task, feature_names)


if __name__ == "__main__":
    main()
