"""
Analyze hybrid normalizer: per-criterion zscore vs rank based on distribution shape.

Compares:
  1. All rank (current)
  2. All zscore
  3. Hybrid: zscore for skewed, rank for uniform

Metrics:
  - Distribution classification (skewness, kurtosis, zero-fraction)
  - Variance balance across criteria
  - α-parameter sensitivity per criterion
  - Merged score distribution (std, IQR)
  - LightGBM learnability (R²)
  - Outlier robustness
"""

import sys
import os
try:
    import quadmix
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scripts.preprocess_essential_web_v1_sharded import (
    extract_domain_level_1, extract_quality_signals,
)
from quadmix.constants import FASTTEXT_FIELDS, DOMAIN_MAP
from quadmix.utils.normalization import zscore_normalize, rank_normalize

DATA_PATH = "/home/liujin99/data/essential-web-v1/train-00000-of-03291.parquet"
N_DOCS = 20000
N_EXPERIMENTS = 100
SAMPLE_SIZE = 5000
N_CRITERIA = 5
N_DOMAINS = 10
SEED = 42


def hybrid_normalize(quality_matrix, skew_threshold=1.5):
    """Per-criterion: zscore if |skewness| > threshold, else rank."""
    n_docs, n_criteria = quality_matrix.shape
    normalized = np.zeros_like(quality_matrix)
    choices = []

    for j in range(n_criteria):
        col = quality_matrix[:, j]
        skew = scipy_stats.skew(col)
        kurt = scipy_stats.kurtosis(col)
        zero_frac = (np.abs(col) < 1e-6).sum() / len(col)

        if abs(skew) > skew_threshold:
            normalized[:, j] = zscore_normalize(col)
            choices.append(("zscore", skew, kurt, zero_frac))
        else:
            normalized[:, j] = rank_normalize(col)
            choices.append(("rank", skew, kurt, zero_frac))

    return normalized, choices


def load_data():
    print(f"Loading {N_DOCS} docs from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH, columns=["eai_taxonomy", "quality_signals"])
    df = df.head(N_DOCS)

    domain_labels = df["eai_taxonomy"].apply(extract_domain_level_1).to_numpy(dtype=np.int64)
    quality_list = df["quality_signals"].apply(extract_quality_signals)
    quality_matrix = np.stack(quality_list.to_numpy()).astype(np.float64)

    valid = domain_labels >= 0
    print(f"  {valid.sum()} with valid domain ({valid.sum()*100/N_DOCS:.1f}%)")
    return quality_matrix, domain_labels


def simulate_experiment(rng):
    alpha_matrix = np.zeros((N_DOMAINS, N_CRITERIA), dtype=np.float64)
    for m in range(N_DOMAINS):
        alpha_matrix[m] = rng.dirichlet(np.ones(N_CRITERIA))
    lambda_ = rng.uniform(0.1, 5.0)
    omega = rng.uniform(0.0, 1.0)
    eta = rng.uniform(0.1, 2.0)
    epsilon = rng.uniform(0.0, 0.5)
    return alpha_matrix, lambda_, omega, eta, epsilon


def compute_merged(normalized, domain_labels, alpha_matrix):
    n_docs = normalized.shape[0]
    merged = np.zeros(n_docs, dtype=np.float64)
    for m in range(N_DOMAINS):
        mask = domain_labels == m
        if mask.sum() == 0:
            continue
        merged[mask] = normalized[mask] @ alpha_matrix[m]
    return merged


def sample_documents(merged_scores, domain_labels, lambda_, omega, eta, epsilon, sample_size, rng):
    n_docs = len(merged_scores)
    ranks = np.zeros(n_docs, dtype=np.float64)
    for m in range(N_DOMAINS):
        mask = domain_labels == m
        if mask.sum() == 0:
            continue
        domain_scores = merged_scores[mask]
        order = np.argsort(domain_scores)
        domain_ranks = np.zeros(mask.sum(), dtype=np.float64)
        domain_ranks[order] = np.arange(mask.sum()) / mask.sum()
        ranks[mask] = domain_ranks

    probs = np.zeros(n_docs, dtype=np.float64)
    for m in range(N_DOMAINS):
        mask = domain_labels == m
        if mask.sum() == 0:
            continue
        r = ranks[mask]
        sigmoid_input = lambda_ * (r - omega)
        sigmoid_input = np.clip(sigmoid_input, -50, 50)
        s = 1.0 / (1.0 + np.exp(-sigmoid_input))
        s = s * (1.0 - epsilon) + epsilon * eta
        probs[mask] = s

    total = probs.sum()
    if total < 1e-10:
        probs = np.ones(n_docs) / n_docs
    else:
        probs /= total

    actual_size = min(sample_size, n_docs)
    selected = rng.choice(n_docs, size=actual_size, replace=False, p=probs)
    return np.sort(selected), probs


def normalize_all(quality_matrix, strategy, skew_threshold=1.5):
    n_docs, n_criteria = quality_matrix.shape
    normalized = np.zeros_like(quality_matrix)

    if strategy == "rank":
        for j in range(n_criteria):
            normalized[:, j] = rank_normalize(quality_matrix[:, j])
    elif strategy == "zscore":
        for j in range(n_criteria):
            normalized[:, j] = zscore_normalize(quality_matrix[:, j])
    elif strategy == "hybrid":
        normalized, _ = hybrid_normalize(quality_matrix, skew_threshold)

    return normalized


def main():
    quality_matrix, domain_labels = load_data()

    # ── Section 1: Distribution classification ──────────────────────────────
    print("\n" + "=" * 80)
    print("1. Distribution Shape Classification")
    print("=" * 80)
    print(f"  {'Criterion':25s} | {'skewness':>10s} | {'kurtosis':>10s} | {'zero%':>8s} | {'IQR':>8s} | {'shape':>8s}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    criterion_shapes = []
    for j, field in enumerate(FASTTEXT_FIELDS):
        col = quality_matrix[:, j]
        skew = scipy_stats.skew(col)
        kurt = scipy_stats.kurtosis(col)
        zero_frac = (np.abs(col) < 1e-6).sum() / len(col) * 100
        q25, q75 = np.percentile(col, [25, 75])
        iqr = q75 - q25

        shape = "SKEWED" if abs(skew) > 1.5 else "uniform"
        criterion_shapes.append(shape)
        print(f"  {field:25s} | {skew:10.2f} | {kurt:10.2f} | {zero_frac:7.1f}% | {iqr:8.4f} | {shape:>8s}")

    print(f"\n  Skew threshold: |skewness| > 1.5 → zscore, else → rank")
    print(f"  Hybrid choices: {[f'{FASTTEXT_FIELDS[j]}={criterion_shapes[j]}' for j in range(N_CRITERIA)]}")

    # ── Section 2: Normalized variance balance ──────────────────────────────
    print("\n" + "=" * 80)
    print("2. Variance Balance Across Criteria")
    print("=" * 80)

    strategies = ["rank", "zscore", "hybrid"]
    normalized_data = {}
    for strat in strategies:
        normalized_data[strat] = normalize_all(quality_matrix, strat)

    print(f"  {'Criterion':25s} | {'rank var':>10s} | {'zscore var':>12s} | {'hybrid var':>12s}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
    for j, field in enumerate(FASTTEXT_FIELDS):
        rv = normalized_data["rank"][:, j].var()
        zv = normalized_data["zscore"][:, j].var()
        hv = normalized_data["hybrid"][:, j].var()
        print(f"  {field:25s} | {rv:10.6f} | {zv:12.6f} | {hv:12.6f}")

    for strat in strategies:
        varr = np.array([normalized_data[strat][:, j].var() for j in range(N_CRITERIA)])
        ratio = varr.max() / max(varr.min(), 1e-10)
        print(f"  {strat:25s} var ratio (max/min): {ratio:.1f}x")

    # ── Section 3: α-parameter sensitivity per criterion ────────────────────
    print("\n" + "=" * 80)
    print("3. α-Parameter Sensitivity Per Criterion")
    print("=" * 80)
    print("  Perturbing each α criterion by ±0.3, measuring doc selection change:\n")

    n_perturb = 200
    base_rng = np.random.RandomState(SEED)
    base_alpha, base_lambda, base_omega, base_eta, base_epsilon = simulate_experiment(base_rng)

    print(f"  {'Criterion':25s} | {'rank':>8s} | {'zscore':>8s} | {'hybrid':>8s} | {'best':>8s}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for criterion_j in range(N_CRITERIA):
        sensitivities = {"rank": [], "zscore": [], "hybrid": []}

        for strat in strategies:
            norm_base = normalized_data[strat]
            merged_base = compute_merged(norm_base, domain_labels, base_alpha)
            sel_base, _ = sample_documents(
                merged_base, domain_labels, base_lambda, base_omega, base_eta, base_epsilon,
                SAMPLE_SIZE, np.random.RandomState(SEED))

            for _ in range(n_perturb):
                perturbed_alpha = base_alpha.copy()
                delta = np.random.uniform(-0.3, 0.3)
                perturbed_alpha[:, criterion_j] = np.clip(
                    perturbed_alpha[:, criterion_j] + delta, 0.01, 1.0)
                perturbed_alpha /= perturbed_alpha.sum(axis=1, keepdims=True)

                merged_p = compute_merged(norm_base, domain_labels, perturbed_alpha)
                sel_p, _ = sample_documents(
                    merged_p, domain_labels, base_lambda, base_omega, base_eta, base_epsilon,
                    SAMPLE_SIZE, np.random.RandomState(SEED))

                overlap = len(set(sel_base) & set(sel_p)) / SAMPLE_SIZE
                sensitivities[strat].append(1.0 - overlap)

        r_sens = np.mean(sensitivities["rank"]) * 100
        z_sens = np.mean(sensitivities["zscore"]) * 100
        h_sens = np.mean(sensitivities["hybrid"]) * 100
        best = max(r_sens, z_sens, h_sens)
        best_name = ["rank", "zscore", "hybrid"][np.argmax([r_sens, z_sens, h_sens])]

        print(f"  {FASTTEXT_FIELDS[criterion_j]:25s} | {r_sens:7.2f}% | {z_sens:7.2f}% | {h_sens:7.2f}% | {best_name:>8s}")

    # ── Section 4: Merged score distribution ────────────────────────────────
    print("\n" + "=" * 80)
    print("4. Merged Score Distribution (50 random experiments)")
    print("=" * 80)

    for strat in strategies:
        merged_stds = []
        merged_iqrs = []
        for exp_i in range(50):
            exp_rng = np.random.RandomState(SEED + exp_i)
            alpha_matrix, _, _, _, _ = simulate_experiment(exp_rng)
            merged = compute_merged(normalized_data[strat], domain_labels, alpha_matrix)
            merged_stds.append(merged.std())
            merged_iqrs.append(np.percentile(merged, 75) - np.percentile(merged, 25))

        print(f"  {strat:25s}: std={np.mean(merged_stds):.4f}±{np.std(merged_stds):.4f}, "
              f"IQR={np.mean(merged_iqrs):.4f}±{np.std(merged_iqrs):.4f}")

    # ── Section 5: LightGBM learnability ────────────────────────────────────
    print("\n" + "=" * 80)
    print("5. LightGBM Learnability (parameter → proxy loss mapping)")
    print("=" * 80)

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    n_grid = N_EXPERIMENTS
    param_vectors = []
    proxy_losses = {strat: [] for strat in strategies}

    for i in range(n_grid):
        grid_rng = np.random.RandomState(SEED + i * 7)
        alpha_matrix, lambda_, omega, eta, epsilon = simulate_experiment(grid_rng)
        flat = np.concatenate([alpha_matrix.flatten(), [lambda_, omega, eta, epsilon]])
        param_vectors.append(flat)

        for strat in strategies:
            merged = compute_merged(normalized_data[strat], domain_labels, alpha_matrix)
            grid_rng2 = np.random.RandomState(SEED + i * 7)
            _ = simulate_experiment(grid_rng2)
            sel, _ = sample_documents(
                merged, domain_labels, lambda_, omega, eta, epsilon,
                SAMPLE_SIZE, grid_rng2)
            proxy_losses[strat].append(merged[sel].mean())

    param_vectors = np.array(param_vectors)

    print(f"\n  {'Strategy':25s} | {'Train R²':>10s} | {'CV R²':>10s} | {'Test R²':>10s}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    n_train = min(80, n_grid)
    X_train = param_vectors[:n_train]
    X_test = param_vectors[n_train:]

    for strat in strategies:
        y_all = np.array(proxy_losses[strat])
        y_train = y_all[:n_train]
        y_test = y_all[n_train:]

        gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=SEED)
        gbr.fit(X_train, y_train)
        train_r2 = gbr.score(X_train, y_train)
        test_r2 = gbr.score(X_test, y_test)
        cv_scores = cross_val_score(
            GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=SEED),
            X_train, y_train, cv=5)

        print(f"  {strat:25s} | {train_r2:10.4f} | {cv_scores.mean():10.4f}±{cv_scores.std():.3f} | {test_r2:10.4f}")

    # ── Section 6: Feature importance comparison ────────────────────────────
    print("\n" + "=" * 80)
    print("6. Feature Importance Per Strategy")
    print("=" * 80)

    for strat in strategies:
        y_train = np.array(proxy_losses[strat])[:n_train]
        gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=SEED)
        gbr.fit(X_train, y_train)
        imp = gbr.feature_importances_

        alpha_imp = imp[:N_DOMAINS * N_CRITERIA].reshape(N_DOMAINS, N_CRITERIA).mean(axis=0)
        sampling_imp = imp[N_DOMAINS * N_CRITERIA:]

        print(f"\n  {strat}:")
        for j, field in enumerate(FASTTEXT_FIELDS):
            print(f"    {field:25s}: α_imp={alpha_imp[j]:.4f}")
        for k, pname in enumerate(["lambda", "omega", "eta", "epsilon"]):
            print(f"    {pname:25s}: imp={sampling_imp[k]:.4f}")

    # ── Section 7: Outlier robustness ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("7. Outlier Robustness")
    print("=" * 80)

    exp_rng = np.random.RandomState(SEED)
    alpha_matrix, _, _, _, _ = simulate_experiment(exp_rng)

    quality_outlier = quality_matrix.copy()
    for j in range(N_CRITERIA):
        quality_outlier[0, j] = quality_matrix[:, j].mean() + 10 * quality_matrix[:, j].std()

    for strat in strategies:
        norm_orig = normalize_all(quality_matrix, strat)
        norm_outlier = normalize_all(quality_outlier, strat)

        merged_orig = compute_merged(norm_orig, domain_labels, alpha_matrix)
        merged_outlier = compute_merged(norm_outlier, domain_labels, alpha_matrix)

        change = np.abs(merged_outlier - merged_orig).mean()
        print(f"  {strat:25s}: mean abs change = {change:.6f}")

    # ── Section 8: Cross-experiment diversity ───────────────────────────────
    print("\n" + "=" * 80)
    print("8. Cross-Experiment Diversity (pairwise doc selection overlap)")
    print("=" * 80)

    for strat in strategies:
        all_selected = []
        for exp_i in range(50):
            exp_rng = np.random.RandomState(SEED + exp_i * 1000)
            alpha_matrix, lambda_, omega, eta, epsilon = simulate_experiment(exp_rng)
            merged = compute_merged(normalized_data[strat], domain_labels, alpha_matrix)
            exp_rng2 = np.random.RandomState(SEED + exp_i * 1000)
            _ = simulate_experiment(exp_rng2)
            sel, _ = sample_documents(
                merged, domain_labels, lambda_, omega, eta, epsilon,
                SAMPLE_SIZE, exp_rng2)
            all_selected.append(set(sel))

        overlaps = []
        for i in range(50):
            for j in range(i + 1, 50):
                overlaps.append(len(all_selected[i] & all_selected[j]) / SAMPLE_SIZE)

        print(f"  {strat:25s}: pairwise overlap = {np.mean(overlaps)*100:.1f}% ± {np.std(overlaps)*100:.1f}%")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Hybrid choices (|skew| > 1.5 → zscore):")
    for j, field in enumerate(FASTTEXT_FIELDS):
        col = quality_matrix[:, j]
        skew = scipy_stats.skew(col)
        choice = "zscore" if abs(skew) > 1.5 else "rank"
        print(f"    {field:25s}: skew={skew:+.2f} → {choice}")

    print(f"\n  Recommendation: see Section 5 (CV R²) and Section 3 (α sensitivity)")
    print(f"  Higher CV R² = better learnability")
    print(f"  Higher α sensitivity = more meaningful parameter space")


if __name__ == "__main__":
    main()
