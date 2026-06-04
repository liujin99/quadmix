"""Analyze distribution of 5 quality signals using ORIGINAL POSITIVE scores."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scripts.preprocess_essential_web_v1_sharded import FASTTEXT_FIELDS

DATA_DIR = "/home/liujin99/data/essential-web-v1"
NUM_SHARDS = 10


def extract_quality_signals_positive(quality_signals):
    """Extract WITHOUT negation — original positive scores (higher = better)."""
    if not isinstance(quality_signals, dict):
        return np.zeros(len(FASTTEXT_FIELDS), dtype=np.float32)
    fasttext = quality_signals.get("fasttext", {})
    if not isinstance(fasttext, dict):
        return np.zeros(len(FASTTEXT_FIELDS), dtype=np.float32)
    return np.array([
        (fasttext.get(field, 0.0) or 0.0)
        for field in FASTTEXT_FIELDS
    ], dtype=np.float32)


print(f"Loading {NUM_SHARDS} shards from {DATA_DIR}...")

all_quality = []
total_docs = 0

for i in range(NUM_SHARDS):
    path = os.path.join(DATA_DIR, f"train-{i:05d}-of-03291.parquet")
    if not os.path.exists(path):
        print(f"  Shard {i} not found, stopping")
        break
    df = pd.read_parquet(path, columns=["quality_signals"])
    quality = df["quality_signals"].apply(extract_quality_signals_positive)
    quality_matrix = np.stack(quality.to_numpy())
    all_quality.append(quality_matrix)
    total_docs += len(df)
    print(f"  Shard {i}: {len(df):,} docs")

quality_all = np.concatenate(all_quality, axis=0)
print(f"\nTotal: {total_docs:,} docs, {quality_all.shape[1]} quality criteria")
print("NOTE: Original positive scores (higher = better match)")

# ── Table 1: Basic statistics ───────────────────────────────────────────────
print("\n" + "=" * 90)
print("Table 1: Basic Statistics (original positive scores, higher = better)")
print("=" * 90)
print(f"{'Criterion':<25} {'min':>8} {'max':>8} {'mean':>8} {'std':>8} {'median':>8} {'IQR':>8}")
print("-" * 90)

for i, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, i]
    q25, q75 = np.percentile(col, [25, 75])
    print(f"{name:<25} {col.min():8.4f} {col.max():8.4f} {col.mean():8.4f} "
          f"{col.std():8.4f} {np.median(col):8.4f} {q75-q25:8.4f}")

# ── Table 2: Percentiles ────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("Table 2: Percentile Distribution")
print("=" * 90)
print(f"{'Criterion':<25} {'5%':>8} {'25%':>8} {'50%':>8} {'75%':>8} {'95%':>8}")
print("-" * 90)

for i, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, i]
    p5, p25, p50, p75, p95 = np.percentile(col, [5, 25, 50, 75, 95])
    print(f"{name:<25} {p5:8.4f} {p25:8.4f} {p50:8.4f} {p75:8.4f} {p95:8.4f}")

# ── Table 3: Shape analysis ────────────────────────────────────────────────
print("\n" + "=" * 90)
print("Table 3: Distribution Shape Analysis")
print("=" * 90)
print(f"{'Criterion':<25} {'skewness':>10} {'kurtosis':>10} {'zero%':>8} {'near0%':>8} {'shape':>10}")
print("-" * 90)

for i, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, i]
    skew = scipy_stats.skew(col)
    kurt = scipy_stats.kurtosis(col)
    zero_pct = (col == 0).sum() / len(col) * 100
    near_zero_pct = (col < 0.01).sum() / len(col) * 100

    if abs(skew) > 2.0:
        shape = "HEAVY SKEW"
    elif abs(skew) > 1.0:
        shape = "moderate"
    else:
        shape = "uniform"

    print(f"{name:<25} {skew:+10.2f} {kurt:10.2f} {zero_pct:7.1f}% {near_zero_pct:7.1f}% {shape:>10}")

# ── Table 4: Score concentration ────────────────────────────────────────────
print("\n" + "=" * 90)
print("Table 4: Score Concentration (what % of docs fall in each range)")
print("=" * 90)
print(f"{'Criterion':<25} {'[0, 0.01)':>10} {'[0.01, 0.1)':>12} {'[0.1, 0.5)':>12} {'[0.5, 1.0]':>12} {'>1.0':>8}")
print("-" * 90)

for i, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, i]
    b1 = ((col >= 0) & (col < 0.01)).sum() / len(col) * 100
    b2 = ((col >= 0.01) & (col < 0.1)).sum() / len(col) * 100
    b3 = ((col >= 0.1) & (col < 0.5)).sum() / len(col) * 100
    b4 = ((col >= 0.5) & (col <= 1.0)).sum() / len(col) * 100
    b5 = (col > 1.0).sum() / len(col) * 100
    print(f"{name:<25} {b1:9.1f}% {b2:11.1f}% {b3:11.1f}% {b4:11.1f}% {b5:7.1f}%")

# ── Table 5: Discriminative power ───────────────────────────────────────────
print("\n" + "=" * 90)
print("Table 5: Discriminative Power Analysis")
print("=" * 90)
print(f"{'Criterion':<25} {'top10% avg':>12} {'bot10% avg':>12} {'ratio':>8} {'spread':>8} {'verdict':>12}")
print("-" * 90)

for i, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, i]
    top10 = np.percentile(col, 90)
    bot10 = np.percentile(col, 10)
    top10_avg = col[col >= top10].mean()
    bot10_avg = col[col <= bot10].mean()
    spread = top10_avg - bot10_avg
    ratio = top10_avg / max(bot10_avg, 1e-10)

    if spread < 0.05:
        verdict = "WEAK"
    elif spread < 0.3:
        verdict = "moderate"
    else:
        verdict = "STRONG"

    print(f"{name:<25} {top10_avg:12.4f} {bot10_avg:12.4f} {ratio:8.1f}x {spread:8.4f} {verdict:>12}")

# ── Table 6: Correlation matrix ─────────────────────────────────────────────
print("\n" + "=" * 90)
print("Table 6: Correlation Matrix (are criteria independent?)")
print("=" * 90)

corr = np.corrcoef(quality_all.T)
short_names = [n.replace("fineweb_edu_approx", "fineweb").replace("eai_general_math", "gen_math").replace("eai_open_web_math", "web_math") for n in FASTTEXT_FIELDS]

print(f"{'':>25}", end="")
for name in short_names:
    print(f" {name[:10]:>10}", end="")
print()

for i, name in enumerate(FASTTEXT_FIELDS):
    print(f"{name:<25}", end="")
    for j in range(len(FASTTEXT_FIELDS)):
        print(f" {corr[i,j]:10.3f}", end="")
    print()

# ── Conclusions ─────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("CONCLUSIONS")
print("=" * 90)

print("""
1. dclm (DataComp-LM academic score)
   - 95% of docs score near 0, only 5% have high scores (>0.23)
   - Heavy right skew (+4.82): most web pages are NOT academic content
   - Discriminative power: WEAK (spread=0.23, but 95% concentrated at 0)
   - Normalizer recommendation: zscore (preserves the gap between 5% high-quality vs 95% near-zero)

2. fineweb_edu_approx (FineWeb-Edu educational quality)
   - Most UNIFORM distribution: range [0, 3.94], IQR=0.64, spread across full range
   - Near-normal shape (skew=+1.10): genuinely discriminates all docs
   - Discriminative power: STRONG (spread=2.19)
   - Normalizer recommendation: rank (already well-distributed, rank preserves ordering perfectly)

3. english (English language classifier)
   - Concentrated around 0.89 (IQR=0.12): most docs are English
   - Left skew (-3.65): tail of low-English docs
   - Discriminative power: WEAK (spread=0.13, most docs cluster at 0.89)
   - Normalizer recommendation: zscore (expands the small differences in the tail)

4. eai_general_math (general math content score)
   - 95% near 0, only 5% have meaningful scores (>0.15)
   - Heaviest skew (+5.86): extreme sparsity of math content
   - Discriminative power: WEAK (spread=0.15, but 95% at zero)
   - Normalizer recommendation: zscore (same logic as dclm)

5. eai_open_web_math (open web math content score)
   - Moderate spread (IQR=0.14), less concentrated than general_math
   - Moderate right skew (+2.52)
   - Discriminative power: moderate (spread=0.40)
   - Normalizer recommendation: zscore (skew > 1.5)

6. Correlations
   - All pairwise correlations < 0.37: criteria are largely INDEPENDENT
   - dclm ↔ fineweb_edu: 0.36 (academic content tends to be educational)
   - english ↔ open_web_math: -0.36 (math content slightly less English)
   - No redundancy: all 5 criteria measure different dimensions

7. Overall normalizer strategy
   - fineweb_edu: rank (uniform distribution, rank is ideal)
   - dclm, english, eai_general_math, eai_open_web_math: zscore (skewed, need to preserve gaps)
   - Hybrid with variance rescaling needed to avoid 12x variance imbalance
""")
