"""
Analyze zscore vs rank normalizer impact on document selection and LightGBM signal space.

Uses real quality scores from essential-web-v1 raw parquet.
Reuses extract functions from preprocess_essential_web_v1_sharded.py.
"""

import sys, os
try:
    import quadmix
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import numpy as np
import pandas as pd

from quadmix.utils.normalization import zscore_normalize, rank_normalize
from quadmix.constants import FASTTEXT_FIELDS, DOMAIN_MAP, QUALITY_COLUMNS
from scripts.preprocess.preprocess_essential_web_v1_sharded import (
    extract_domain_level_1, extract_quality_signals,
)

N_DOCS = 20000
N_EXPERIMENTS = 50
SAMPLE_SIZE = 5000
N_CRITERIA = 5
N_DOMAINS = 10
SEED = 42


def load_data(path, n_docs):
    df = pd.read_parquet(path, columns=["eai_taxonomy", "quality_signals"])
    n = min(n_docs, len(df))
    df = df.head(n)

    domain_labels = df["eai_taxonomy"].apply(extract_domain_level_1).to_numpy(dtype=np.int64)
    quality_list = df["quality_signals"].apply(extract_quality_signals)
    quality_matrix = np.stack(quality_list.to_numpy()).astype(np.float64)

    valid = domain_labels >= 0
    print(f"Loaded {n} docs, {valid.sum()} with valid domain ({valid.sum()*100/n:.1f}%)")
    print(f"Quality matrix: {quality_matrix.shape}, Domain labels: {domain_labels.shape}")

    domain_counts = {}
    for d in range(N_DOMAINS):
        c = (domain_labels == d).sum()
        if c > 0:
            domain_counts[d] = c
    print(f"Domain distribution: {domain_counts}")

    return quality_matrix, domain_labels


def compute_merged_scores(quality_matrix, domain_labels, alpha_matrix, normalizer_fn, name):
    n_docs, n_criteria = quality_matrix.shape
    normalized = np.zeros_like(quality_matrix)
    for j in range(n_criteria):
        normalized[:, j] = normalizer_fn(quality_matrix[:, j])

    merged = np.zeros(n_docs, dtype=np.float64)
    for m in range(N_DOMAINS):
        mask = domain_labels == m
        if mask.sum() == 0:
            continue
        merged[mask] = normalized[mask] @ alpha_matrix[m]

    return merged, normalized


def simulate_experiment(quality_matrix, domain_labels, rng):
    alpha_matrix = np.zeros((N_DOMAINS, N_CRITERIA), dtype=np.float64)
    for m in range(N_DOMAINS):
        a = rng.dirichlet(np.ones(N_CRITERIA))
        alpha_matrix[m] = a

    lambda_ = rng.uniform(0.1, 5.0)
    omega = rng.uniform(0.0, 1.0)
    eta = rng.uniform(0.1, 2.0)
    epsilon = rng.uniform(0.0, 0.5)

    return alpha_matrix, lambda_, omega, eta, epsilon


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


def main():
    rng = np.random.RandomState(SEED)

    print("=" * 70)
    print("Normalizer Impact Analysis: zscore vs rank")
    print("=" * 70)

    quality_matrix, domain_labels = load_data(
        "/home/liujin99/data/essential-web-v1/train-00000-of-03291.parquet",
        N_DOCS
    )

    print()
    print("=" * 70)
    print("1. Raw quality score statistics")
    print("=" * 70)
    for j, field in enumerate(FASTTEXT_FIELDS):
        col = quality_matrix[:, j]
        print(f"  {field:25s}: mean={col.mean():.4f}, std={col.std():.4f}, "
              f"min={col.min():.4f}, max={col.max():.4f}, "
              f"skew={((col - col.mean())**3).mean() / col.std()**3:.2f}")

    print()
    print("=" * 70)
    print("2. Normalized score distributions")
    print("=" * 70)
    zscore_norm = np.zeros_like(quality_matrix)
    rank_norm = np.zeros_like(quality_matrix)
    for j in range(N_CRITERIA):
        zscore_norm[:, j] = zscore_normalize(quality_matrix[:, j])
        rank_norm[:, j] = rank_normalize(quality_matrix[:, j])

    print(f"  {'Criterion':25s} | {'zscore range':>15s} | {'zscore IQR':>12s} | {'rank range':>12s} | {'rank IQR':>10s}")
    print(f"  {'-'*25}-+-{'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
    for j, field in enumerate(FASTTEXT_FIELDS):
        z = zscore_norm[:, j]
        r = rank_norm[:, j]
        z_iqr = np.percentile(z, 75) - np.percentile(z, 25)
        r_iqr = np.percentile(r, 75) - np.percentile(r, 25)
        print(f"  {field:25s} | [{z.min():6.2f}, {z.max():5.2f}] | {z_iqr:10.4f} | "
              f"[{r.min():5.2f}, {r.max():4.2f}] | {r_iqr:8.4f}")

    print()
    print("=" * 70)
    print("3. Experiment simulation: same α, different normalizer")
    print("=" * 70)

    overlap_same_alpha = []
    merged_std_zscore = []
    merged_std_rank = []
    merged_iqr_zscore = []
    merged_iqr_rank = []

    for exp_i in range(N_EXPERIMENTS):
        exp_rng = np.random.RandomState(SEED + exp_i)
        alpha_matrix, lambda_, omega, eta, epsilon = simulate_experiment(quality_matrix, domain_labels, exp_rng)

        merged_z, _ = compute_merged_scores(quality_matrix, domain_labels, alpha_matrix, zscore_normalize, "zscore")
        merged_r, _ = compute_merged_scores(quality_matrix, domain_labels, alpha_matrix, rank_normalize, "rank")

        merged_std_zscore.append(merged_z.std())
        merged_std_rank.append(merged_r.std())
        merged_iqr_zscore.append(np.percentile(merged_z, 75) - np.percentile(merged_z, 25))
        merged_iqr_rank.append(np.percentile(merged_r, 75) - np.percentile(merged_r, 25))

        sel_z, _ = sample_documents(merged_z, domain_labels, lambda_, omega, eta, epsilon, SAMPLE_SIZE, exp_rng)
        exp_rng2 = np.random.RandomState(SEED + exp_i)
        _ = simulate_experiment(quality_matrix, domain_labels, exp_rng2)
        sel_r, _ = sample_documents(merged_r, domain_labels, lambda_, omega, eta, epsilon, SAMPLE_SIZE, exp_rng2)

        overlap = len(set(sel_z) & set(sel_r)) / SAMPLE_SIZE
        overlap_same_alpha.append(overlap)

    print(f"  Same α parameters, different normalizer:")
    print(f"  Document selection overlap: {np.mean(overlap_same_alpha)*100:.1f}% ± {np.std(overlap_same_alpha)*100:.1f}%")
    print(f"  Merged score std (zscore): {np.mean(merged_std_zscore):.4f} ± {np.std(merged_std_zscore):.4f}")
    print(f"  Merged score std (rank):   {np.mean(merged_std_rank):.4f} ± {np.std(merged_std_rank):.4f}")
    print(f"  Merged score IQR (zscore): {np.mean(merged_iqr_zscore):.4f} ± {np.std(merged_iqr_zscore):.4f}")
    print(f"  Merged score IQR (rank):   {np.mean(merged_iqr_rank):.4f} ± {np.std(merged_iqr_rank):.4f}")

    print()
    print("=" * 70)
    print("4. Cross-experiment diversity: different α, same normalizer")
    print("=" * 70)

    all_selected_zscore = []
    all_selected_rank = []
    all_merged_means_zscore = []
    all_merged_means_rank = []

    for exp_i in range(N_EXPERIMENTS):
        exp_rng = np.random.RandomState(SEED + exp_i * 1000)
        alpha_matrix, lambda_, omega, eta, epsilon = simulate_experiment(quality_matrix, domain_labels, exp_rng)

        merged_z, _ = compute_merged_scores(quality_matrix, domain_labels, alpha_matrix, zscore_normalize, "zscore")
        merged_r, _ = compute_merged_scores(quality_matrix, domain_labels, alpha_matrix, rank_normalize, "rank")

        sel_z, probs_z = sample_documents(merged_z, domain_labels, lambda_, omega, eta, epsilon, SAMPLE_SIZE, exp_rng)
        exp_rng2 = np.random.RandomState(SEED + exp_i * 1000)
        _ = simulate_experiment(quality_matrix, domain_labels, exp_rng2)
        sel_r, probs_r = sample_documents(merged_r, domain_labels, lambda_, omega, eta, epsilon, SAMPLE_SIZE, exp_rng2)

        all_selected_zscore.append(set(sel_z))
        all_selected_rank.append(set(sel_r))

        all_merged_means_zscore.append(merged_z[sel_z].mean())
        all_merged_means_rank.append(merged_r[sel_r].mean())

    zscore_means = np.array(all_merged_means_zscore)
    rank_means = np.array(all_merged_means_rank)

    print(f"  Merged score mean across experiments (zscore): {zscore_means.mean():.4f} ± {zscore_means.std():.4f}")
    print(f"  Merged score mean across experiments (rank):   {rank_means.mean():.4f} ± {rank_means.std():.4f}")
    print(f"  Ratio (rank/zscore): {rank_means.std() / zscore_means.std():.2f}x")

    pairwise_overlap_z = []
    pairwise_overlap_r = []
    for i in range(N_EXPERIMENTS):
        for j in range(i + 1, N_EXPERIMENTS):
            pairwise_overlap_z.append(len(all_selected_zscore[i] & all_selected_zscore[j]) / SAMPLE_SIZE)
            pairwise_overlap_r.append(len(all_selected_rank[i] & all_selected_rank[j]) / SAMPLE_SIZE)

    print(f"  Pairwise doc selection overlap (zscore): {np.mean(pairwise_overlap_z)*100:.1f}% ± {np.std(pairwise_overlap_z)*100:.1f}%")
    print(f"  Pairwise doc selection overlap (rank):   {np.mean(pairwise_overlap_r)*100:.1f}% ± {np.std(pairwise_overlap_r)*100:.1f}%")

    print()
    print("=" * 70)
    print("5. LightGBM signal space analysis")
    print("=" * 70)

    proxy_losses_z = []
    proxy_losses_r = []
    param_vectors = []

    for exp_i in range(N_EXPERIMENTS):
        exp_rng = np.random.RandomState(SEED + exp_i * 1000)
        alpha_matrix, lambda_, omega, eta, epsilon = simulate_experiment(quality_matrix, domain_labels, exp_rng)

        merged_z, _ = compute_merged_scores(quality_matrix, domain_labels, alpha_matrix, zscore_normalize, "zscore")
        merged_r, _ = compute_merged_scores(quality_matrix, domain_labels, alpha_matrix, rank_normalize, "rank")

        sel_z, _ = sample_documents(merged_z, domain_labels, lambda_, omega, eta, epsilon, SAMPLE_SIZE, exp_rng)
        exp_rng2 = np.random.RandomState(SEED + exp_i * 1000)
        _ = simulate_experiment(quality_matrix, domain_labels, exp_rng2)
        sel_r, _ = sample_documents(merged_r, domain_labels, lambda_, omega, eta, epsilon, SAMPLE_SIZE, exp_rng2)

        proxy_z = merged_z[sel_z].mean() + 0.01 * rng.randn()
        proxy_r = merged_r[sel_r].mean() + 0.01 * rng.randn()
        proxy_losses_z.append(proxy_z)
        proxy_losses_r.append(proxy_r)

        flat = np.concatenate([alpha_matrix.flatten(), [lambda_, omega, eta, epsilon]])
        param_vectors.append(flat)

    proxy_losses_z = np.array(proxy_losses_z)
    proxy_losses_r = np.array(proxy_losses_r)

    print(f"  Proxy loss (zscore): range=[{proxy_losses_z.min():.4f}, {proxy_losses_z.max():.4f}], "
          f"std={proxy_losses_z.std():.4f}, IQR={np.percentile(proxy_losses_z, 75) - np.percentile(proxy_losses_z, 25):.4f}")
    print(f"  Proxy loss (rank):   range=[{proxy_losses_r.min():.4f}, {proxy_losses_r.max():.4f}], "
          f"std={proxy_losses_r.std():.4f}, IQR={np.percentile(proxy_losses_r, 75) - np.percentile(proxy_losses_r, 25):.4f}")
    print(f"  Signal ratio (rank/zscore): {proxy_losses_r.std() / proxy_losses_z.std():.2f}x")

    print()
    print("=" * 70)
    print("6. Outlier sensitivity analysis")
    print("=" * 70)

    exp_rng = np.random.RandomState(SEED)
    alpha_matrix, lambda_, omega, eta, epsilon = simulate_experiment(quality_matrix, domain_labels, exp_rng)

    merged_z_orig, norm_z_orig = compute_merged_scores(quality_matrix, domain_labels, alpha_matrix, zscore_normalize, "zscore")
    merged_r_orig, norm_r_orig = compute_merged_scores(quality_matrix, domain_labels, alpha_matrix, rank_normalize, "rank")

    quality_outlier = quality_matrix.copy()
    outlier_idx = 0
    for j in range(N_CRITERIA):
        quality_outlier[outlier_idx, j] = quality_matrix[:, j].mean() + 10 * quality_matrix[:, j].std()

    merged_z_outlier, _ = compute_merged_scores(quality_outlier, domain_labels, alpha_matrix, zscore_normalize, "zscore")
    merged_r_outlier, _ = compute_merged_scores(quality_outlier, domain_labels, alpha_matrix, rank_normalize, "rank")

    z_change = np.abs(merged_z_outlier - merged_z_orig).mean()
    r_change = np.abs(merged_r_outlier - merged_r_orig).mean()
    print(f"  After injecting 1 extreme outlier (10σ):")
    print(f"  Mean abs change in merged scores (zscore): {z_change:.6f}")
    print(f"  Mean abs change in merged scores (rank):   {r_change:.6f}")
    print(f"  Ratio (zscore/rank): {z_change/r_change:.1f}x")

    print()
    print("=" * 70)
    print("7. Per-criterion normalized variance (WHY zscore kills signal)")
    print("=" * 70)

    print(f"\n  {'Criterion':25s} | {'zscore var':>12s} | {'rank var':>10s} | {'ratio':>8s}")
    print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")
    for j, field in enumerate(FASTTEXT_FIELDS):
        zv = zscore_norm[:, j].var()
        rv = rank_norm[:, j].var()
        print(f"  {field:25s} | {zv:12.6f} | {rv:10.6f} | {zv/rv:8.2f}x")

    print(f"\n  zscore total var: {zscore_norm.var():.6f}")
    print(f"  rank total var:   {rank_norm.var():.6f}")
    print(f"  zscore var ratio (max/min): {zscore_norm.var(axis=0).max()/zscore_norm.var(axis=0).min():.1f}x")
    print(f"  rank var ratio (max/min):   {rank_norm.var(axis=0).max()/rank_norm.var(axis=0).min():.1f}x")

    print()
    print("=" * 70)
    print("8. α-parameter sensitivity: which α dimensions actually matter?")
    print("=" * 70)

    n_perturb = 200
    base_rng = np.random.RandomState(SEED)
    base_alpha, base_lambda, base_omega, base_eta, base_epsilon = simulate_experiment(quality_matrix, domain_labels, base_rng)

    merged_z_base, _ = compute_merged_scores(quality_matrix, domain_labels, base_alpha, zscore_normalize, "zscore")
    merged_r_base, _ = compute_merged_scores(quality_matrix, domain_labels, base_alpha, rank_normalize, "rank")

    sel_z_base, _ = sample_documents(merged_z_base, domain_labels, base_lambda, base_omega, base_eta, base_epsilon, SAMPLE_SIZE, np.random.RandomState(SEED))
    sel_r_base, _ = sample_documents(merged_r_base, domain_labels, base_lambda, base_omega, base_eta, base_epsilon, SAMPLE_SIZE, np.random.RandomState(SEED))

    zscore_sensitivity = []
    rank_sensitivity = []

    for criterion_j in range(N_CRITERIA):
        z_overlaps = []
        r_overlaps = []
        for _ in range(n_perturb):
            perturbed_alpha = base_alpha.copy()
            delta = np.random.uniform(-0.3, 0.3)
            perturbed_alpha[:, criterion_j] = np.clip(perturbed_alpha[:, criterion_j] + delta, 0.01, 1.0)
            row_sums = perturbed_alpha.sum(axis=1, keepdims=True)
            perturbed_alpha /= row_sums

            merged_z_p, _ = compute_merged_scores(quality_matrix, domain_labels, perturbed_alpha, zscore_normalize, "zscore")
            merged_r_p, _ = compute_merged_scores(quality_matrix, domain_labels, perturbed_alpha, rank_normalize, "rank")

            sel_z_p, _ = sample_documents(merged_z_p, domain_labels, base_lambda, base_omega, base_eta, base_epsilon, SAMPLE_SIZE, np.random.RandomState(SEED))
            sel_r_p, _ = sample_documents(merged_r_p, domain_labels, base_lambda, base_omega, base_eta, base_epsilon, SAMPLE_SIZE, np.random.RandomState(SEED))

            z_overlaps.append(len(set(sel_z_base) & set(sel_z_p)) / SAMPLE_SIZE)
            r_overlaps.append(len(set(sel_r_base) & set(sel_r_p)) / SAMPLE_SIZE)

        zscore_sensitivity.append(1.0 - np.mean(z_overlaps))
        rank_sensitivity.append(1.0 - np.mean(r_overlaps))

    print(f"  Perturbing each α criterion by ±0.3, measuring doc selection change:")
    print(f"  {'Criterion':25s} | {'zscore change':>15s} | {'rank change':>13s} | {'ratio':>8s}")
    print(f"  {'-'*25}-+-{'-'*15}-+-{'-'*13}-+-{'-'*8}")
    for j, field in enumerate(FASTTEXT_FIELDS):
        print(f"  {field:25s} | {zscore_sensitivity[j]*100:14.2f}% | {rank_sensitivity[j]*100:12.2f}% | {rank_sensitivity[j]/max(zscore_sensitivity[j], 1e-10):8.2f}x")

    print(f"\n  Total zscore sensitivity: {np.mean(zscore_sensitivity)*100:.2f}%")
    print(f"  Total rank sensitivity:   {np.mean(rank_sensitivity)*100:.2f}%")
    print(f"  Ratio (rank/zscore): {np.mean(rank_sensitivity)/max(np.mean(zscore_sensitivity), 1e-10):.2f}x")

    print()
    print("=" * 70)
    print("9. LightGBM learnability: parameter → loss mapping complexity")
    print("=" * 70)

    n_grid = 100
    param_to_loss_z = []
    param_to_loss_r = []
    param_vectors_grid = []

    for i in range(n_grid):
        grid_rng = np.random.RandomState(SEED + i * 7)
        alpha_matrix, lambda_, omega, eta, epsilon = simulate_experiment(quality_matrix, domain_labels, grid_rng)

        merged_z, _ = compute_merged_scores(quality_matrix, domain_labels, alpha_matrix, zscore_normalize, "zscore")
        merged_r, _ = compute_merged_scores(quality_matrix, domain_labels, alpha_matrix, rank_normalize, "rank")

        sel_z, _ = sample_documents(merged_z, domain_labels, lambda_, omega, eta, epsilon, SAMPLE_SIZE, grid_rng)
        grid_rng2 = np.random.RandomState(SEED + i * 7)
        _ = simulate_experiment(quality_matrix, domain_labels, grid_rng2)
        sel_r, _ = sample_documents(merged_r, domain_labels, lambda_, omega, eta, epsilon, SAMPLE_SIZE, grid_rng2)

        proxy_z = merged_z[sel_z].mean()
        proxy_r = merged_r[sel_r].mean()
        param_to_loss_z.append(proxy_z)
        param_to_loss_r.append(proxy_r)

        flat = np.concatenate([alpha_matrix.flatten(), [lambda_, omega, eta, epsilon]])
        param_vectors_grid.append(flat)

    param_to_loss_z = np.array(param_to_loss_z)
    param_to_loss_r = np.array(param_to_loss_r)
    param_vectors_grid = np.array(param_vectors_grid)

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    n_train = min(80, n_grid)
    X_train = param_vectors_grid[:n_train]
    X_test = param_vectors_grid[n_train:]

    for name, y_arr in [("zscore", param_to_loss_z), ("rank", param_to_loss_r)]:
        y_train = y_arr[:n_train]
        y_test = y_arr[n_train:]

        gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=SEED)
        gbr.fit(X_train, y_train)
        train_r2 = gbr.score(X_train, y_train)
        test_r2 = gbr.score(X_test, y_test)

        cv_scores = cross_val_score(GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=SEED),
                                     X_train, y_train, cv=5)

        importances = gbr.feature_importances_
        alpha_imp = importances[:N_DOMAINS * N_CRITERIA].reshape(N_DOMAINS, N_CRITERIA).mean(axis=0)
        sampling_imp = importances[N_DOMAINS * N_CRITERIA:]

        print(f"\n  {name}:")
        print(f"    Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        print(f"    CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"    Feature importance (α per criterion, averaged across domains):")
        for j, field in enumerate(FASTTEXT_FIELDS):
            print(f"      {field:25s}: {alpha_imp[j]:.4f}")
        print(f"    Feature importance (sampling params):")
        for k, pname in enumerate(["lambda", "omega", "eta", "epsilon"]):
            print(f"      {pname:25s}: {sampling_imp[k]:.4f}")

    print()
    print("=" * 70)
    print("10. Effective dimensionality: how many α dimensions LightGBM can use")
    print("=" * 70)

    n_dim_test = 200
    dim_params = []
    dim_losses_z = []
    dim_losses_r = []

    for i in range(n_dim_test):
        dim_rng = np.random.RandomState(SEED + i * 13)
        alpha_matrix, lambda_, omega, eta, epsilon = simulate_experiment(quality_matrix, domain_labels, dim_rng)

        merged_z, _ = compute_merged_scores(quality_matrix, domain_labels, alpha_matrix, zscore_normalize, "zscore")
        merged_r, _ = compute_merged_scores(quality_matrix, domain_labels, alpha_matrix, rank_normalize, "rank")

        sel_z, _ = sample_documents(merged_z, domain_labels, lambda_, omega, eta, epsilon, SAMPLE_SIZE, dim_rng)
        dim_rng2 = np.random.RandomState(SEED + i * 13)
        _ = simulate_experiment(quality_matrix, domain_labels, dim_rng2)
        sel_r, _ = sample_documents(merged_r, domain_labels, lambda_, omega, eta, epsilon, SAMPLE_SIZE, dim_rng2)

        flat = np.concatenate([alpha_matrix.flatten(), [lambda_, omega, eta, epsilon]])
        dim_params.append(flat)
        dim_losses_z.append(merged_z[sel_z].mean())
        dim_losses_r.append(merged_r[sel_r].mean())

    dim_params = np.array(dim_params)
    dim_losses_z = np.array(dim_losses_z)
    dim_losses_r = np.array(dim_losses_r)

    for name, y_arr in [("zscore", dim_losses_z), ("rank", dim_losses_r)]:
        from sklearn.decomposition import PCA
        corr_matrix = np.corrcoef(dim_params.T)
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        effective_dim = np.sum(eigenvalues > 0.01 * eigenvalues.max())

        param_corr = np.array([np.corrcoef(dim_params[:, j], y_arr)[0, 1] for j in range(dim_params.shape[1])])
        significant_dims = np.sum(np.abs(param_corr) > 0.1)

        print(f"\n  {name}:")
        print(f"    Effective param dimensions (eigenvalue > 1% max): {effective_dim}/{dim_params.shape[1]}")
        print(f"    Dims with |corr| > 0.1 to loss: {significant_dims}/{dim_params.shape[1]}")
        print(f"    Top 10 correlated dims:")
        top_idx = np.argsort(-np.abs(param_corr))[:10]
        for idx in top_idx:
            if idx < N_DOMAINS * N_CRITERIA:
                d = idx // N_CRITERIA
                c = idx % N_CRITERIA
                label = f"α[domain={d}, {FASTTEXT_FIELDS[c]}]"
            else:
                label = ["lambda", "omega", "eta", "epsilon"][idx - N_DOMAINS * N_CRITERIA]
            print(f"      {label:45s}: corr={param_corr[idx]:+.4f}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  1. Same α → different normalizer → {np.mean(overlap_same_alpha)*100:.0f}% doc overlap")
    print(f"  2. Different α → zscore pairwise overlap: {np.mean(pairwise_overlap_z)*100:.1f}%")
    print(f"     Different α → rank pairwise overlap:   {np.mean(pairwise_overlap_r)*100:.1f}%")
    print(f"  3. Proxy loss std ratio (rank/zscore): {proxy_losses_r.std() / proxy_losses_z.std():.2f}x")
    print(f"  4. Outlier sensitivity ratio (zscore/rank): {z_change/r_change:.1f}x")
    print(f"  5. zscore var imbalance (max/min criterion): {zscore_norm.var(axis=0).max()/zscore_norm.var(axis=0).min():.1f}x")
    print(f"     rank var imbalance: {rank_norm.var(axis=0).max()/rank_norm.var(axis=0).min():.1f}x")
    print(f"  6. α sensitivity (zscore): {np.mean(zscore_sensitivity)*100:.2f}%")
    print(f"     α sensitivity (rank):   {np.mean(rank_sensitivity)*100:.2f}%")


if __name__ == "__main__":
    main()
