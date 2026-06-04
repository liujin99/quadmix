"""
Per-criterion normalizer analysis: what σ() is best for each quality criterion?

Core goal of σ in Eq.1: ¯q = Σ σ(q_n) · α_{n,m}
  - α should control RELATIVE IMPORTANCE, not fight scale differences
  - All σ(q_n) must have comparable variance → equal α = equal importance
  - σ should preserve meaningful differences within each criterion
  - σ should be robust to outliers

We test 7 normalization strategies:
  1. rank         — rank percentile [0,1]
  2. zscore       — (x - mean) / std
  3. robust_z     — (x - median) / MAD (robust to outliers)
  4. log1p_z      — log(1+x) then zscore (compresses right tail)
  5. quantile     — empirical CDF → uniform [0,1] (like rank but interpolated)
  6. clip_zscore  — clip to [-3, 3] after zscore (caps outlier influence)
  7. sigmoid_iqr  — 1/(1+exp(-(x-median)/IQR)) (soft rank, preserves density)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scripts.preprocess_essential_web_v1_sharded import (
    extract_domain_level_1, FASTTEXT_FIELDS,
)

DATA_DIR = "/home/liujin99/data/essential-web-v1"
NUM_SHARDS = 2
SEED = 42


def extract_quality_positive(quality_signals):
    if not isinstance(quality_signals, dict):
        return np.zeros(len(FASTTEXT_FIELDS), dtype=np.float32)
    fasttext = quality_signals.get("fasttext", {})
    if not isinstance(fasttext, dict):
        return np.zeros(len(FASTTEXT_FIELDS), dtype=np.float32)
    return np.array([
        (fasttext.get(field, 0.0) or 0.0)
        for field in FASTTEXT_FIELDS
    ], dtype=np.float32)


# ── Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
all_quality = []
for i in range(NUM_SHARDS):
    path = os.path.join(DATA_DIR, f"train-{i:05d}-of-03291.parquet")
    if not os.path.exists(path):
        break
    df = pd.read_parquet(path, columns=["quality_signals"])
    quality = df["quality_signals"].apply(extract_quality_positive)
    all_quality.append(np.stack(quality.to_numpy()))

quality_all = np.concatenate(all_quality, axis=0).astype(np.float64)
N = len(quality_all)
print(f"Loaded {N:,} docs\n")


# ── Define normalizers ──────────────────────────────────────────────────────
def rank_normalize(x):
    n = len(x)
    ranks = np.argsort(np.argsort(x))
    return ranks.astype(np.float64) / n


def zscore_normalize(x):
    std = np.std(x)
    if std < 1e-10:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def robust_z_normalize(x):
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    if mad < 1e-10:
        return np.zeros_like(x)
    return (x - median) / (mad * 1.4826)


def log1p_z_normalize(x):
    shifted = x - x.min()
    logged = np.log1p(shifted)
    std = np.std(logged)
    if std < 1e-10:
        return np.zeros_like(logged)
    return (logged - np.mean(logged)) / std


def quantile_normalize(x):
    n = len(x)
    sorted_idx = np.argsort(x)
    result = np.zeros(n, dtype=np.float64)
    result[sorted_idx] = np.linspace(0, 1, n)
    return result


def clip_zscore_normalize(x):
    std = np.std(x)
    if std < 1e-10:
        return np.zeros_like(x)
    z = (x - np.mean(x)) / std
    return np.clip(z, -3, 3)


def sigmoid_iqr_normalize(x):
    median = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    if iqr < 1e-10:
        return np.full_like(x, 0.5)
    t = (x - median) / iqr
    t = np.clip(t, -10, 10)
    return 1.0 / (1.0 + np.exp(-t))


NORMALIZERS = {
    "rank": rank_normalize,
    "zscore": zscore_normalize,
    "robust_z": robust_z_normalize,
    "log1p_z": log1p_z_normalize,
    "quantile": quantile_normalize,
    "clip_zscore": clip_zscore_normalize,
    "sigmoid_iqr": sigmoid_iqr_normalize,
}


# ── Analysis ────────────────────────────────────────────────────────────────
print("=" * 100)
print("ANALYSIS: What σ() is best for each quality criterion?")
print("=" * 100)
print("\nCore goal: α_{n,m} should represent RELATIVE IMPORTANCE of criterion n in domain m.")
print("Requirements for fair α learning:")
print("  R1. Equal variance: all σ(q_n) should have similar variance → equal α = equal importance")
print("  R2. Preserve signal: meaningful differences within criterion should be preserved")
print("  R3. Outlier robust: extreme values should not dominate the scale")
print("  R4. Cross-criterion comparability: σ(q_n) values should be on similar scale")

# ── Table 1: Distribution diagnosis ─────────────────────────────────────────
print("\n" + "=" * 100)
print("Table 1: Distribution Diagnosis (original positive scores)")
print("=" * 100)
print(f"{'Criterion':<22} {'range':>14} {'median':>8} {'IQR':>8} {'skew':>8} {'kurt':>8} {'near0%':>8} {'shape':>12}")
print("-" * 100)

shapes = {}
for j, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, j]
    skew = scipy_stats.skew(col)
    kurt = scipy_stats.kurtosis(col)
    near0 = (col < 0.01).sum() / N * 100
    q25, q75 = np.percentile(col, [25, 75])

    if abs(skew) > 3.0:
        shape = "extreme skew"
    elif abs(skew) > 1.5:
        shape = "heavy skew"
    elif abs(skew) > 0.5:
        shape = "moderate"
    else:
        shape = "symmetric"

    if near0 > 50:
        shape += " + spike@0"
    elif (col > np.percentile(col, 75) - 0.1).sum() / N > 0.8:
        shape += " + ceiling"

    shapes[name] = shape
    print(f"{name:<22} [{col.min():.3f},{col.max():.2f}] {np.median(col):8.4f} {q75-q25:8.4f} {skew:+8.2f} {kurt:8.1f} {near0:7.1f}% {shape:>12}")

# ── Table 2: Variance after each normalizer ─────────────────────────────────
print("\n" + "=" * 100)
print("Table 2: Variance After Normalization (R1: equal variance → fair α)")
print("=" * 100)

header = f"{'Criterion':<22}"
for norm_name in NORMALIZERS:
    header += f" {norm_name:>12}"
print(header)
print("-" * 100)

norm_data = {norm_name: np.zeros_like(quality_all) for norm_name in NORMALIZERS}

for norm_name, norm_fn in NORMALIZERS.items():
    for j in range(len(FASTTEXT_FIELDS)):
        norm_data[norm_name][:, j] = norm_fn(quality_all[:, j])

for j, name in enumerate(FASTTEXT_FIELDS):
    row = f"{name:<22}"
    for norm_name in NORMALIZERS:
        v = norm_data[norm_name][:, j].var()
        row += f" {v:12.4f}"
    print(row)

print("-" * 100)
for norm_name in NORMALIZERS:
    vars_ = np.array([norm_data[norm_name][:, j].var() for j in range(len(FASTTEXT_FIELDS))])
    ratio = vars_.max() / max(vars_.min(), 1e-10)
    print(f"{'var ratio (max/min)':<22}", end="")
    for nn in NORMALIZERS:
        if nn == norm_name:
            print(f" {ratio:11.1f}x", end="")
        else:
            print(f" {'':>12}", end="")
    print()

# ── Table 3: Outlier sensitivity ────────────────────────────────────────────
print("\n" + "=" * 100)
print("Table 3: Outlier Sensitivity (R3: robust to extreme values)")
print("=" * 100)
print("Inject 1 extreme outlier at 100x max, measure mean abs change in all other normalized values\n")

header = f"{'Criterion':<22}"
for norm_name in NORMALIZERS:
    header += f" {norm_name:>12}"
print(header)
print("-" * 100)

for j, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, j].copy()
    col_outlier = col.copy()
    col_outlier[0] = col.max() * 100

    row = f"{name:<22}"
    for norm_name, norm_fn in NORMALIZERS.items():
        norm_orig = norm_fn(col)
        norm_out = norm_fn(col_outlier)
        change = np.abs(norm_out[1:] - norm_orig[1:]).mean()
        row += f" {change:12.6f}"
    print(row)

# ── Table 4: Signal preservation ────────────────────────────────────────────
print("\n" + "=" * 100)
print("Table 4: Signal Preservation (R2: meaningful differences preserved)")
print("=" * 100)
print("For each criterion: correlation between raw rank-order and normalized rank-order\n")
print("High corr = preserves relative ordering. Low corr = distorts ranking.\n")

header = f"{'Criterion':<22}"
for norm_name in NORMALIZERS:
    header += f" {norm_name:>12}"
print(header)
print("-" * 100)

for j, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, j]
    raw_ranks = np.argsort(np.argsort(col))

    row = f"{name:<22}"
    for norm_name in NORMALIZERS:
        norm_col = norm_data[norm_name][:, j]
        norm_ranks = np.argsort(np.argsort(norm_col))
        corr = np.corrcoef(raw_ranks, norm_ranks)[0, 1]
        row += f" {corr:12.6f}"
    print(row)

# ── Table 5: Normalized value range ─────────────────────────────────────────
print("\n" + "=" * 100)
print("Table 5: Normalized Value Range (R4: cross-criterion comparability)")
print("=" * 100)
print("If ranges differ wildly, same α value means different effective weight\n")

header = f"{'Criterion':<22}"
for norm_name in NORMALIZERS:
    header += f" {norm_name:>12}"
print(header)
print("-" * 100)

for j, name in enumerate(FASTTEXT_FIELDS):
    row = f"{name:<22}"
    for norm_name in NORMALIZERS:
        col = norm_data[norm_name][:, j]
        rng = f"[{col.min():.2f},{col.max():.2f}]"
        row += f" {rng:>12}"
    print(row)

# ── Table 6: Per-criterion deep dive ────────────────────────────────────────
print("\n" + "=" * 100)
print("Table 6: Per-Criterion Normalized Distribution Detail")
print("=" * 100)

for j, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, j]
    print(f"\n  {name} (raw: median={np.median(col):.4f}, IQR={np.percentile(col,75)-np.percentile(col,25):.4f}, "
          f"skew={scipy_stats.skew(col):+.2f})")
    print(f"  {'Normalizer':<14} {'mean':>8} {'std':>8} {'min':>8} {'5%':>8} {'50%':>8} {'95%':>8} {'max':>8} {'IQR':>8}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for norm_name in NORMALIZERS:
        nc = norm_data[norm_name][:, j]
        q5, q50, q95 = np.percentile(nc, [5, 50, 95])
        q25, q75 = np.percentile(nc, [25, 75])
        print(f"  {norm_name:<14} {nc.mean():8.4f} {nc.std():8.4f} {nc.min():8.4f} {q5:8.4f} {q50:8.4f} {q95:8.4f} {nc.max():8.4f} {q75-q25:8.4f}")

# ── Table 7: Top-5% separation ──────────────────────────────────────────────
print("\n" + "=" * 100)
print("Table 7: Top-5% Separation (can normalizer distinguish top-quality docs?)")
print("=" * 100)
print("Gap = normalized(top 5%) - normalized(bottom 50%). Larger = better separation.\n")

header = f"{'Criterion':<22}"
for norm_name in NORMALIZERS:
    header += f" {norm_name:>12}"
print(header)
print("-" * 100)

for j, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, j]
    p95 = np.percentile(col, 95)
    p50 = np.percentile(col, 50)
    top_mask = col >= p95
    bot_mask = col <= p50

    row = f"{name:<22}"
    for norm_name in NORMALIZERS:
        nc = norm_data[norm_name][:, j]
        top_mean = nc[top_mask].mean()
        bot_mean = nc[bot_mask].mean()
        gap = top_mean - bot_mean
        row += f" {gap:12.4f}"
    print(row)

# ── Table 8: Scoring ────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("Table 8: Overall Scoring (weighted evaluation per criterion)")
print("=" * 100)

criteria_scores = {name: {} for name in FASTTEXT_FIELDS}

for j, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, j]

    # R1: Variance balance (relative to other criteria under same normalizer)
    # Computed globally below

    # R2: Rank preservation
    raw_ranks = np.argsort(np.argsort(col))
    for norm_name in NORMALIZERS:
        nc = norm_data[norm_name][:, j]
        norm_ranks = np.argsort(np.argsort(nc))
        rank_corr = np.corrcoef(raw_ranks, norm_ranks)[0, 1]
        criteria_scores[name][f"{norm_name}_rank_corr"] = rank_corr

    # R3: Outlier robustness
    col_outlier = col.copy()
    col_outlier[0] = col.max() * 100
    for norm_name, norm_fn in NORMALIZERS.items():
        norm_orig = norm_fn(col)
        norm_out = norm_fn(col_outlier)
        change = np.abs(norm_out[1:] - norm_orig[1:]).mean()
        criteria_scores[name][f"{norm_name}_outlier"] = change

    # Top-5% separation
    p95 = np.percentile(col, 95)
    p50 = np.percentile(col, 50)
    top_mask = col >= p95
    bot_mask = col <= p50
    for norm_name in NORMALIZERS:
        nc = norm_data[norm_name][:, j]
        gap = nc[top_mask].mean() - nc[bot_mask].mean()
        criteria_scores[name][f"{norm_name}_top5_gap"] = gap

# Variance balance score (per normalizer, across all criteria)
var_balance = {}
for norm_name in NORMALIZERS:
    vars_ = np.array([norm_data[norm_name][:, j].var() for j in range(len(FASTTEXT_FIELDS))])
    var_balance[norm_name] = vars_.max() / max(vars_.min(), 1e-10)

print(f"\n  Variance balance (max/min ratio, lower = better):")
for norm_name in NORMALIZERS:
    print(f"    {norm_name:<14}: {var_balance[norm_name]:.1f}x")

# Per-criterion recommendation
print(f"\n  Per-criterion analysis:")
print(f"  {'Criterion':<22} {'best norm':>14} {'2nd best':>14} {'reason':>30}")
print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*30}")

for j, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, j]
    skew = abs(scipy_stats.skew(col))
    near0 = (col < 0.01).sum() / N * 100

    scores = {}
    for norm_name in NORMALIZERS:
        s = 0
        # Rank preservation (higher = better)
        s += criteria_scores[name][f"{norm_name}_rank_corr"] * 30

        # Outlier robustness (lower change = better, invert)
        outlier_change = criteria_scores[name][f"{norm_name}_outlier"]
        max_outlier = max(criteria_scores[name][f"{nn}_outlier"] for nn in NORMALIZERS)
        if max_outlier > 0:
            s += (1 - outlier_change / max_outlier) * 25

        # Top-5% separation (higher = better)
        gap = abs(criteria_scores[name][f"{norm_name}_top5_gap"])
        max_gap = max(abs(criteria_scores[name][f"{nn}_top5_gap"]) for nn in NORMALIZERS)
        if max_gap > 0:
            s += (gap / max_gap) * 25

        # Variance balance (global, lower ratio = better)
        s += (1.0 / var_balance[norm_name]) * 20

        scores[norm_name] = s

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best = ranked[0]
    second = ranked[1]

    if skew > 3 and near0 > 50:
        reason = f"extreme skew + {near0:.0f}% near 0"
    elif skew > 1.5:
        reason = f"heavy skew ({skew:.1f})"
    elif near0 > 50:
        reason = f"{near0:.0f}% near 0 (spike)"
    else:
        reason = f"moderate skew ({skew:.1f})"

    print(f"  {name:<22} {best[0]:>14} {second[0]:>14} {reason:>30}")

    print(f"    Scores:", end="")
    for nn, sc in ranked:
        print(f" {nn}={sc:.1f}", end="")
    print()

# ── Final recommendation ────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("FINAL RECOMMENDATION")
print("=" * 100)

print("""
Analysis framework:
  R1. Variance balance (weight 20%): all σ(q_n) should have similar variance
  R2. Rank preservation (weight 30%): σ should not distort relative ordering
  R3. Outlier robustness (weight 25%): extreme values should not dominate
  R4. Top-5% separation (weight 25%): σ should distinguish high-quality docs
""")

for j, name in enumerate(FASTTEXT_FIELDS):
    col = quality_all[:, j]
    skew = scipy_stats.skew(col)
    near0 = (col < 0.01).sum() / N * 100
    q25, q75 = np.percentile(col, [25, 75])

    print(f"\n  {name}:")
    print(f"    Distribution: skew={skew:+.2f}, near0={near0:.1f}%, IQR=[{q25:.4f}, {q75:.4f}]")

    scores = {}
    for norm_name in NORMALIZERS:
        s = 0
        s += criteria_scores[name][f"{norm_name}_rank_corr"] * 30
        outlier_change = criteria_scores[name][f"{norm_name}_outlier"]
        max_outlier = max(criteria_scores[name][f"{nn}_outlier"] for nn in NORMALIZERS)
        if max_outlier > 0:
            s += (1 - outlier_change / max_outlier) * 25
        gap = abs(criteria_scores[name][f"{norm_name}_top5_gap"])
        max_gap = max(abs(criteria_scores[name][f"{nn}_top5_gap"]) for nn in NORMALIZERS)
        if max_gap > 0:
            s += (gap / max_gap) * 25
        s += (1.0 / var_balance[norm_name]) * 20
        scores[norm_name] = s

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    print(f"    Ranking: {' > '.join(f'{nn}({sc:.0f})' for nn, sc in ranked)}")
    print(f"    → Recommended: {ranked[0][0]}")
