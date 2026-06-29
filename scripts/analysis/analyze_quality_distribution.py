#!/usr/bin/env python3
"""Analyze distribution of 5 quality signals in essential-web-v1."""

import sys
import os
try:
    import quadmix
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import numpy as np
import pandas as pd
from scripts.preprocess.preprocess_essential_web_v1_sharded import extract_quality_signals
from quadmix.constants import FASTTEXT_FIELDS

DATA_DIR = "/home/ma-user/work/QuaDMix/data/essential-web"
NUM_SHARDS = 10

print(f"Loading {NUM_SHARDS} shards from {DATA_DIR}...")

all_quality = []
total_docs = 0

for i in range(NUM_SHARDS):
    path = os.path.join(DATA_DIR, f"train-{i:05d}-of-03291.parquet")
    if not os.path.exists(path):
        print(f"  Shard {i} not found, stopping")
        break
    
    df = pd.read_parquet(path, columns=["quality_signals"])
    quality = df["quality_signals"].apply(extract_quality_signals)
    quality_matrix = np.stack(quality.to_numpy())
    all_quality.append(quality_matrix)
    total_docs += len(df)
    print(f"  Shard {i}: {len(df):,} docs")

quality_all = np.concatenate(all_quality, axis=0)
print(f"\nTotal: {total_docs:,} docs, {quality_all.shape[1]} quality criteria\n")

print("=" * 80)
print(f"{'Criterion':<25} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} {'median':>10} {'IQR':>10}")
print("=" * 80)

for i, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, i]
    q25, q75 = np.percentile(col, [25, 75])
    print(f"{name:<25} {col.min():10.4f} {col.max():10.4f} {col.mean():10.4f} "
          f"{col.std():10.4f} {np.median(col):10.4f} {q75-q25:10.4f}")

print("\n" + "=" * 80)
print("Percentiles:")
print("=" * 80)
print(f"{'Criterion':<25} {'5%':>10} {'25%':>10} {'50%':>10} {'75%':>10} {'95%':>10}")
print("-" * 80)

for i, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, i]
    p5, p25, p50, p75, p95 = np.percentile(col, [5, 25, 50, 75, 95])
    print(f"{name:<25} {p5:10.4f} {p25:10.4f} {p50:10.4f} {p75:10.4f} {p95:10.4f}")

print("\n" + "=" * 80)
print("Outlier analysis (values beyond 3σ):")
print("=" * 80)

for i, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, i]
    mean, std = col.mean(), col.std()
    outliers = np.abs(col - mean) > 3 * std
    n_outliers = outliers.sum()
    pct = n_outliers / len(col) * 100
    print(f"{name:<25} {n_outliers:>8,} ({pct:5.2f}%)  "
          f"range: [{col[outliers].min():.4f}, {col[outliers].max():.4f}]")

print("\n" + "=" * 80)
print("Zero/near-zero values:")
print("=" * 80)

for i, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, i]
    n_zero = (col == 0).sum()
    n_near_zero = (np.abs(col) < 1e-6).sum()
    print(f"{name:<25} exact zeros: {n_zero:>8,} ({n_zero/len(col)*100:5.2f}%)  "
          f"near-zero: {n_near_zero:>8,} ({n_near_zero/len(col)*100:5.2f}%)")

print("\n" + "=" * 80)
print("Correlation matrix:")
print("=" * 80)

corr = np.corrcoef(quality_all.T)
print(f"{'':>25}", end="")
for name in FASTTEXT_FIELDS:
    print(f" {name[:8]:>8}", end="")
print()

for i, name in enumerate(FASTTEXT_FIELDS):
    print(f"{name:<25}", end="")
    for j in range(len(FASTTEXT_FIELDS)):
        print(f" {corr[i,j]:8.3f}", end="")
    print()

print("\nDone.")
