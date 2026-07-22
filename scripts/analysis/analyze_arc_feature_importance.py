"""
Analyze feature importance for arc_easy and arc_challenge per-task LightGBM models.

Loads experiment meta.json files from proxy_validation directory, reconstructs
the feature vector based on dataset schema, trains LightGBM, and prints
feature importance grouped by domain.

Usage:
    python scripts/analysis/analyze_arc_feature_importance.py \
        --schema configs/schema_essential_web.yaml \
        --exp_dir /path/to/proxy_validation
"""

import argparse
import json
import os
import sys

import numpy as np
from lightgbm import LGBMRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from quadmix.data.dataset_schema import DatasetSchema

SAMPLING_KEYS = ["lambda", "omega", "eta", "epsilon"]

TARGET_TASKS = ["arc_easy", "arc_challenge"]


def find_exp_root(exp_dir: str) -> str:
    exp_dirs = [d for d in os.listdir(exp_dir)
                if d.startswith("exp_") and os.path.isdir(os.path.join(exp_dir, d))]
    if exp_dirs:
        return os.path.join(exp_dir, exp_dirs[0])
    return exp_dir


def load_experiments(exp_dir: str, domain_names, quality_keys, num_domains, num_criteria):
    X = []
    all_losses = {}

    for exp_name in sorted(os.listdir(exp_dir)):
        if not exp_name.startswith("exp_"):
            continue
        exp_path = os.path.join(exp_dir, exp_name)
        meta_path = os.path.join(exp_path, "meta.json")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        qw = meta.get("quality_weights", {})
        sp = meta.get("sampling_params", {})
        losses = meta.get("task_losses", {})

        features = []

        for domain_name in domain_names:
            dw = qw.get(domain_name, {})
            for k in quality_keys:
                features.append(dw.get(k, 0.0))

        for domain_name in domain_names:
            s = sp.get(domain_name, {})
            for k in SAMPLING_KEYS:
                features.append(s.get(k, 0.0))

        expected_dim = num_domains * num_criteria + num_domains * len(SAMPLING_KEYS)
        if len(features) != expected_dim:
            print(f"  SKIP {exp_name}: feature dim {len(features)} != expected {expected_dim}")
            if len(features) == 0:
                print(f"  DEBUG: First meta.json keys: {list(meta.keys())}")
                print(f"  DEBUG: quality_weights keys: {list(qw.keys()) if qw else 'MISSING'}")
                print(f"  DEBUG: sampling_params keys: {list(sp.keys()) if sp else 'MISSING'}")
            continue

        for task, loss in losses.items():
            if task not in all_losses:
                all_losses[task] = []
            all_losses[task].append(loss)

        X.append(features)

    if not X:
        return np.array([]).reshape(0, 0), {}

    return np.array(X), all_losses


def build_feature_names(domain_short, quality_keys, num_domains, num_criteria):
    names = []
    for m in range(num_domains):
        ds = domain_short[m] if m < len(domain_short) else f"D{m}"
        for n, qk in enumerate(quality_keys):
            names.append(f"alpha_{m}_{n} ({ds}/{qk})")
    for m in range(num_domains):
        ds = domain_short[m] if m < len(domain_short) else f"D{m}"
        for sk in SAMPLING_KEYS:
            names.append(f"{sk}_{m} ({ds})")
    return names


def train_and_analyze(X, losses, task_name, feature_names, domain_short, num_domains, num_criteria):
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
    for m in range(num_domains):
        alpha_start = m * num_criteria
        alpha_end = alpha_start + num_criteria
        alpha_imp = importances[alpha_start:alpha_end].sum()

        samp_start = num_domains * num_criteria + m * len(SAMPLING_KEYS)
        samp_end = samp_start + len(SAMPLING_KEYS)
        samp_imp = importances[samp_start:samp_end].sum()

        total_imp = alpha_imp + samp_imp
        domain_totals.append((m, alpha_imp, samp_imp, total_imp))

    domain_totals.sort(key=lambda x: -x[3])
    for m, alpha_imp, samp_imp, total_imp in domain_totals:
        alpha_pct = alpha_imp / total * 100
        samp_pct = samp_imp / total * 100
        total_pct = total_imp / total * 100
        ds = domain_short[m] if m < len(domain_short) else f"D{m}"
        print(f"{ds:<15} {alpha_pct:>9.2f}% {samp_pct:>9.2f}% {total_pct:>9.2f}%")

    return model, importances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=str, required=True,
                       help="YAML dataset schema config file (必填)")
    parser.add_argument("--exp_dir", required=True, help="Path to proxy_validation directory")
    parser.add_argument("--tasks", nargs="+", default=TARGET_TASKS, help="Tasks to analyze")
    args = parser.parse_args()

    schema = DatasetSchema.from_yaml(args.schema)
    domain_names = schema.domain_names or [f"D{i}" for i in range(100)]
    domain_short = [n.replace("_and_", " & ")[:12] if "_" in n else n[:12] for n in domain_names]
    quality_keys = schema.quality_names or schema.quality_cols
    num_domains = len(domain_names)
    num_criteria = len(quality_keys)

    print(f"Loading experiments from: {args.exp_dir}")
    print(f"  Schema: M={num_domains} domains, N={num_criteria} quality criteria")
    X, all_losses = load_experiments(args.exp_dir, domain_names, quality_keys, num_domains, num_criteria)
    if len(X) == 0:
        print("ERROR: No experiments loaded. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(X)} experiments with {X.shape[1]} features")

    feature_names = build_feature_names(domain_short, quality_keys, num_domains, num_criteria)

    for task in args.tasks:
        if task not in all_losses:
            print(f"\nWARNING: Task '{task}' not found in experiment data")
            print(f"Available tasks: {sorted(all_losses.keys())}")
            continue
        train_and_analyze(X, all_losses[task], task, feature_names, domain_short, num_domains, num_criteria)


if __name__ == "__main__":
    main()
