#!/usr/bin/env python3
"""
诊断 aggregate loss 方差塌缩根因。

分析内容：
  1. Aggregate loss vs per-task loss 分布对比
  2. 参数空间覆盖度（参数变化分析）
  3. 等权平均 vs 其他聚合方式的方差对比
  4. 参数-损失相关性（哪些参数真正影响 loss）
  5. 信号强度：Train R² / Adjusted R² / Cross-validated R²

用法：
  python diagnose_variance.py /path/to/proxy_experiments/ [--k 5]
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict
from numpy.linalg import lstsq


def load_experiments(proxy_dir):
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
        results.append(meta)
    return results


def flatten_params(meta):
    qw = meta["quality_weights"]
    sp = meta["sampling_params"]
    flat = []
    for domain in sorted(qw.keys()):
        for criterion in sorted(qw[domain].keys()):
            flat.append(qw[domain][criterion])
    for domain in sorted(sp.keys()):
        flat.append(sp[domain]["lambda"])
        flat.append(sp[domain]["omega"])
        flat.append(sp[domain]["eta"])
        flat.append(sp[domain]["epsilon"])
    return flat


def compute_cv_r2(X, y, k=5, seed=42):
    """Compute k-fold cross-validated R²."""
    n = len(y)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // k
    r2_scores = []

    for fold in range(k):
        test_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[: fold * fold_size], indices[(fold + 1) * fold_size :]])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        y_train_c = y_train - y_train.mean()
        coef, _, _, _ = lstsq(X_train_b, y_train_c, rcond=None)

        X_test_b = np.column_stack([X_test, np.ones(len(X_test))])
        y_pred = X_test_b @ coef + y_train.mean()

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2_scores.append(r2)

    return np.mean(r2_scores), np.std(r2_scores)


def main():
    parser = argparse.ArgumentParser(description="诊断 aggregate loss 方差塌缩根因")
    parser.add_argument("proxy_dir", help="Path to proxy_experiments directory")
    parser.add_argument("--k", type=int, default=5, help="Number of CV folds (default: 5)")
    args = parser.parse_args()

    results = load_experiments(args.proxy_dir)
    n = len(results)
    print(f"Loaded {n} experiments\n")

    # ── 1. Aggregate vs Per-task loss 分布 ──────────────────
    agg_losses = np.array([r["val_loss"] for r in results])
    tasks = sorted(results[0]["per_task_losses"].keys())
    K = len(tasks)
    per_task = np.array([[r["per_task_losses"][t] for t in tasks] for r in results])

    print("=" * 80)
    print("1. AGGREGATE vs PER-TASK LOSS 分布")
    print("=" * 80)

    print(f"\n  Aggregate loss (val_loss from meta.json):")
    print(f"    mean={agg_losses.mean():.4f}, std={agg_losses.std():.4f}, "
          f"CV={agg_losses.std()/agg_losses.mean():.4f}")
    print(f"    min={agg_losses.min():.4f}, max={agg_losses.max():.4f}, "
          f"range={agg_losses.max()-agg_losses.min():.4f}")
    print(f"    p25={np.percentile(agg_losses, 25):.4f}, "
          f"p50={np.percentile(agg_losses, 50):.4f}, "
          f"p75={np.percentile(agg_losses, 75):.4f}")

    # 检查 aggregate loss 是否就是 per-task 的等权平均
    per_task_mean = per_task.mean(axis=1)
    corr_with_mean = np.corrcoef(agg_losses, per_task_mean)[0, 1]
    diff = agg_losses - per_task_mean
    print(f"\n  Per-task equal-weight mean:")
    print(f"    mean={per_task_mean.mean():.4f}, std={per_task_mean.std():.4f}")
    print(f"    corr(agg_loss, per_task_mean) = {corr_with_mean:.6f}")
    print(f"    diff(agg - mean): max_abs={np.abs(diff).max():.6f}, "
          f"mean_abs={np.abs(diff).mean():.6f}")

    print(f"\n  Per-task loss stats (sorted by std):")
    print(f"    {'Task':<30} {'Mean':>8} {'Std':>8} {'CV':>8} {'Min':>8} {'Max':>8} {'Range':>8}")
    print(f"    {'-'*80}")
    task_stds = []
    for i, task in enumerate(tasks):
        tl = per_task[:, i]
        std = tl.std()
        task_stds.append((task, tl.mean(), std, std/tl.mean(), tl.min(), tl.max(), tl.max()-tl.min()))
    task_stds.sort(key=lambda x: -x[2])
    for task, mean, std, cv, mn, mx, rng in task_stds:
        print(f"    {task:<30} {mean:>8.4f} {std:>8.4f} {cv:>8.4f} {mn:>8.4f} {mx:>8.4f} {rng:>8.4f}")

    # ── 2. 等权平均如何压缩方差 ──────────────────────────────
    print(f"\n{'='*80}")
    print("2. 等权平均如何压缩方差（方差分解）")
    print("=" * 80)

    task_vars = np.var(per_task, axis=0)
    task_cov_matrix = np.cov(per_task.T)
    mean_var = np.mean(task_vars)
    mean_cov = (np.sum(task_cov_matrix) - np.trace(task_cov_matrix)) / (K * (K - 1))
    avg_var_of_mean = np.var(per_task_mean)

    print(f"\n  单个 task 平均方差:  {mean_var:.6f}")
    print(f"  task 间平均协方差:  {mean_cov:.6f}")
    print(f"  等权平均的方差:     {avg_var_of_mean:.6f}")
    print(f"  方差压缩比:         {mean_var / avg_var_of_mean:.1f}x")
    print(f"  理论公式: Var(mean) = (1/K)*mean_var + (K-1)/K*mean_cov")
    theoretical = (1/K)*mean_var + (K-1)/K*mean_cov
    print(f"  理论值: {theoretical:.6f}, 实际值: {avg_var_of_mean:.6f}")

    # task 间相关性
    task_corr_matrix = np.corrcoef(per_task.T)
    off_diag = []
    for i in range(K):
        for j in range(i+1, K):
            off_diag.append(task_corr_matrix[i, j])
    off_diag = np.array(off_diag)
    print(f"\n  Task 间相关系数:")
    print(f"    mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")
    print(f"    min={off_diag.min():.4f}, max={off_diag.max():.4f}")
    print(f"    负相关对数: {(off_diag < 0).sum()}/{len(off_diag)}")

    # z-score 标准化后的分析
    print(f"\n  Z-score 标准化后:")
    z_per_task = (per_task - per_task.mean(axis=0)) / per_task.std(axis=0)
    z_mean = z_per_task.mean(axis=1)
    z_task_vars = np.var(z_per_task, axis=0)
    z_cov_matrix = np.cov(z_per_task.T)
    z_mean_var = np.mean(z_task_vars)
    z_mean_cov = (np.sum(z_cov_matrix) - np.trace(z_cov_matrix)) / (K * (K - 1))
    z_avg_var_of_mean = np.var(z_mean)
    print(f"    单个 task 平均方差(z):  {z_mean_var:.6f}")
    print(f"    task 间平均协方差(z):   {z_mean_cov:.6f}")
    print(f"    等权平均的方差(z):      {z_avg_var_of_mean:.6f}")
    print(f"    方差压缩比(z):          {z_mean_var / z_avg_var_of_mean:.1f}x")

    z_corr = np.corrcoef(z_per_task.T)
    z_off_diag = []
    for i in range(K):
        for j in range(i+1, K):
            z_off_diag.append(z_corr[i, j])
    z_off_diag = np.array(z_off_diag)
    print(f"    Z-score task 间相关: mean={z_off_diag.mean():.4f}, "
          f"负相关对数: {(z_off_diag < 0).sum()}/{len(z_off_diag)}")

    # ── 3. 参数空间覆盖度 ────────────────────────────────────
    print(f"\n{'='*80}")
    print("  3. 参数空间覆盖度（参数变化分析）")
    print("=" * 80)

    param_matrix = np.array([flatten_params(r) for r in results])
    n_params = param_matrix.shape[1]
    print(f"\n  总参数维度: {n_params}")
    print(f"  实验数量: {n}")

    param_stds = param_matrix.std(axis=0)
    param_ranges = param_matrix.max(axis=0) - param_matrix.min(axis=0)
    param_means = param_matrix.mean(axis=0)
    param_cvs = param_stds / np.maximum(np.abs(param_means), 1e-10)

    n_zero_std = (param_stds < 1e-6).sum()
    n_low_cv = (param_cvs < 0.01).sum()
    n_effective = (param_cvs >= 0.01).sum()

    print(f"\n  零方差参数（std < 1e-6）: {n_zero_std}/{n_params}")
    print(f"  低变异参数（CV < 0.01）:  {n_low_cv}/{n_params}")
    print(f"  有效变化参数（CV >= 0.01）: {n_effective}/{n_params}")

    # 按参数类型分组（动态检测域数和质量标准数）
    M = len(results[0]["quality_weights"])
    N = len(next(iter(results[0]["quality_weights"].values())))
    alpha_stds = []
    for m in range(M):
        for nn in range(N):
            idx = m * N + nn
            alpha_stds.append(param_stds[idx])

    lambda_stds = [param_stds[M*N + m*4] for m in range(M)]
    omega_stds = [param_stds[M*N + m*4 + 1] for m in range(M)]
    eta_stds = [param_stds[M*N + m*4 + 2] for m in range(M)]
    epsilon_stds = [param_stds[M*N + m*4 + 3] for m in range(M)]

    print(f"\n  参数类型统计:")
    print(f"    {'Type':<15} {'Mean Std':>10} {'Min Std':>10} {'Max Std':>10} {'Zero':>6}")
    print(f"    {'-'*55}")
    for name, stds in [(f"alpha({M*N})", alpha_stds), (f"lambda({M})", lambda_stds),
                        (f"omega({M})", omega_stds), (f"eta({M})", eta_stds),
                        (f"epsilon({M})", epsilon_stds)]:
        stds = np.array(stds)
        zero = (stds < 1e-6).sum()
        print(f"    {name:<15} {stds.mean():>10.4f} {stds.min():>10.4f} "
              f"{stds.max():>10.4f} {zero:>6}")

    # 参数与 aggregate loss 的相关性
    print(f"\n  参数与 aggregate loss 的相关性 (top 15):")
    param_agg_corr = np.array([np.corrcoef(param_matrix[:, i], agg_losses)[0, 1]
                                for i in range(n_params)])
    abs_corr = np.abs(param_agg_corr)
    top_idx = np.argsort(-abs_corr)[:15]
    print(f"    {'Param Idx':>10} {'Corr':>8} {'AbsCorr':>8} {'Std':>8}")
    for idx in top_idx:
        print(f"    {idx:>10} {param_agg_corr[idx]:>8.4f} {abs_corr[idx]:>8.4f} "
              f"{param_stds[idx]:>8.4f}")

    n_sig_corr = (abs_corr > 0.1).sum()
    print(f"\n  |corr| > 0.1 的参数数: {n_sig_corr}/{n_params}")
    print(f"  |corr| > 0.2 的参数数: {(abs_corr > 0.2).sum()}/{n_params}")
    print(f"  |corr| > 0.3 的参数数: {(abs_corr > 0.3).sum()}/{n_params}")

    # ── 4. 不同聚合方式的方差对比 ─────────────────────────────
    print(f"\n{'='*80}")
    print("4. 不同聚合方式的方差对比")
    print("=" * 80)

    # 4a. 等权平均
    eq_mean = per_task.mean(axis=1)
    print(f"\n  {'聚合方式':<35} {'Std':>8} {'CV':>8} {'Range':>8}")
    print(f"  {'-'*65}")
    print(f"  {'等权平均 (1/K Σ task_i)':<35} {eq_mean.std():>8.4f} "
          f"{eq_mean.std()/eq_mean.mean():>8.4f} {eq_mean.max()-eq_mean.min():>8.4f}")

    # 4b. 按 task 方差加权（高方差 task 权重大）
    weights_var = task_stds_arr = np.std(per_task, axis=0)
    weights_var = weights_var / weights_var.sum()
    var_weighted = per_task @ weights_var
    print(f"  {'方差加权 (w_i ∝ σ_i)':<35} {var_weighted.std():>8.4f} "
          f"{var_weighted.std()/var_weighted.mean():>8.4f} "
          f"{var_weighted.max()-var_weighted.min():>8.4f}")

    # 4c. 仅高方差 task (top 5)
    top5_idx = np.argsort(-np.std(per_task, axis=0))[:5]
    top5_mean = per_task[:, top5_idx].mean(axis=1)
    top5_names = [tasks[i] for i in top5_idx]
    print(f"  {'Top-5 高方差 task 平均':<35} {top5_mean.std():>8.4f} "
          f"{top5_mean.std()/top5_mean.mean():>8.4f} {top5_mean.max()-top5_mean.min():>8.4f}")
    print(f"    → tasks: {top5_names}")

    # 4d. 仅低方差 task (bottom 5)
    bot5_idx = np.argsort(np.std(per_task, axis=0))[:5]
    bot5_mean = per_task[:, bot5_idx].mean(axis=1)
    bot5_names = [tasks[i] for i in bot5_idx]
    print(f"  {'Bot-5 低方差 task 平均':<35} {bot5_mean.std():>8.4f} "
          f"{bot5_mean.std()/bot5_mean.mean():>8.4f} {bot5_mean.max()-bot5_mean.min():>8.4f}")
    print(f"    → tasks: {bot5_names}")

    # 4e. PCA 第一主成分
    centered = per_task - per_task.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = U[:, 0] * S[0]
    print(f"  {'PCA PC1':<35} {pc1.std():>8.4f} "
          f"{'N/A':>8} {pc1.max()-pc1.min():>8.4f}")
    explained_var_ratio = (S**2) / (S**2).sum()
    print(f"    → PC1 解释方差比: {explained_var_ratio[0]:.4f}")
    print(f"    → PC1-5 累积解释: {explained_var_ratio[:5].sum():.4f}")

    # 4f. 仅用 aggregate loss (原始 val_loss)
    print(f"  {'原始 aggregate val_loss':<35} {agg_losses.std():>8.4f} "
          f"{agg_losses.std()/agg_losses.mean():>8.4f} {agg_losses.max()-agg_losses.min():>8.4f}")

    # ── 5. 信号强度分析 ──────────────────────────────────────
    print(f"\n{'='*80}")
    print("5. 信号强度：参数变化能解释多少 loss 变化？")
    print("=" * 80)

    X = param_matrix - param_matrix.mean(axis=0)
    X_b = np.column_stack([X, np.ones(n)])
    p = X.shape[1]

    print(f"\n  Config: n={n}, p={p}, n/p ratio={n/p:.2f}, k={args.k}")
    print(f"\n  {'Task':<30} {'Train R²':>10} {'Adj R²':>10} {'CV R²':>10} {'CV std':>10}")
    print(f"  {'-'*74}")

    cv_results = {}
    for i, task in enumerate(tasks):
        y = per_task[:, i]
        y_c = y - y.mean()
        coef, _, _, _ = lstsq(X_b, y_c, rcond=None)
        y_pred = X_b @ coef
        ss_res = np.sum((y_c - y_pred) ** 2)
        ss_tot = np.sum(y_c ** 2)
        train_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        adj_r2 = 1 - (1 - train_r2) * (n - 1) / (n - p - 1) if n > p + 1 else float("nan")
        cv_r2, cv_std = compute_cv_r2(X, y, k=args.k)
        cv_results[task] = {"train_r2": train_r2, "adj_r2": adj_r2, "cv_r2": cv_r2, "cv_std": cv_std}
        print(f"  {task:<30} {train_r2:>10.4f} {adj_r2:>10.4f} {cv_r2:>10.4f} {cv_std:>10.4f}")

    # aggregate
    y_agg = agg_losses - agg_losses.mean()
    coef_agg, _, _, _ = lstsq(X_b, y_agg, rcond=None)
    y_pred_agg = X_b @ coef_agg
    ss_res_agg = np.sum((y_agg - y_pred_agg) ** 2)
    ss_tot_agg = np.sum(y_agg ** 2)
    r2_agg = 1 - ss_res_agg / ss_tot_agg if ss_tot_agg > 0 else 0
    adj_r2_agg = 1 - (1 - r2_agg) * (n - 1) / (n - p - 1) if n > p + 1 else float("nan")
    cv_r2_agg, cv_std_agg = compute_cv_r2(X, agg_losses, k=args.k)
    cv_results["__aggregate__"] = {"train_r2": r2_agg, "adj_r2": adj_r2_agg, "cv_r2": cv_r2_agg, "cv_std": cv_std_agg}

    print(f"  {'-'*74}")
    print(f"  {'Aggregate (val_loss)':<30} {r2_agg:>10.4f} {adj_r2_agg:>10.4f} {cv_r2_agg:>10.4f} {cv_std_agg:>10.4f}")

    # equal-weight mean
    y_eq = eq_mean - eq_mean.mean()
    coef_eq, _, _, _ = lstsq(X_b, y_eq, rcond=None)
    y_pred_eq = X_b @ coef_eq
    ss_res_eq = np.sum((y_eq - y_pred_eq) ** 2)
    ss_tot_eq = np.sum(y_eq ** 2)
    r2_eq = 1 - ss_res_eq / ss_tot_eq if ss_tot_eq > 0 else 0
    adj_r2_eq = 1 - (1 - r2_eq) * (n - 1) / (n - p - 1) if n > p + 1 else float("nan")
    cv_r2_eq, cv_std_eq = compute_cv_r2(X, eq_mean, k=args.k)
    cv_results["__equal_weight_mean__"] = {"train_r2": r2_eq, "adj_r2": adj_r2_eq, "cv_r2": cv_r2_eq, "cv_std": cv_std_eq}

    print(f"  {'Equal-weight mean':<30} {r2_eq:>10.4f} {adj_r2_eq:>10.4f} {cv_r2_eq:>10.4f} {cv_std_eq:>10.4f}")

    # Summary
    train_r2s = [cv_results[t]["train_r2"] for t in tasks]
    adj_r2s = [cv_results[t]["adj_r2"] for t in tasks]
    cv_r2s = [cv_results[t]["cv_r2"] for t in tasks]

    print(f"\n  {'Summary':<30} {'Train R²':>10} {'Adj R²':>10} {'CV R²':>10}")
    print(f"    {'Per-task mean':<30} {np.mean(train_r2s):>10.4f} {np.mean(adj_r2s):>10.4f} {np.mean(cv_r2s):>10.4f}")
    print(f"    {'Per-task median':<30} {np.median(train_r2s):>10.4f} {np.median(adj_r2s):>10.4f} {np.median(cv_r2s):>10.4f}")
    print(f"    {'Aggregate':<30} {r2_agg:>10.4f} {adj_r2_agg:>10.4f} {cv_r2_agg:>10.4f}")

    print(f"\n  Overfitting gap:")
    print(f"    Train R² - Adj R² (aggregate): {r2_agg - adj_r2_agg:.4f}")
    print(f"    Train R² - CV R² (aggregate):  {r2_agg - cv_r2_agg:.4f}")
    print(f"    Train R² - CV R² (per-task mean): {np.mean(train_r2s) - np.mean(cv_r2s):.4f}")

    # ── 6. 关键结论 ──────────────────────────────────────────
    print(f"\n{'='*80}")
    print("6. 关键结论")
    print("=" * 80)

    print(f"""
  A. Aggregate loss std = {agg_losses.std():.4f} (CV = {agg_losses.std()/agg_losses.mean():.4f})
     Per-task 平均 std = {np.mean([s[2] for s in task_stds]):.4f}
     → 等权平均将方差压缩了 {mean_var/avg_var_of_mean:.1f}x

  B. Task 间平均相关系数 = {off_diag.mean():.4f}
     → {'正相关为主，等权平均不会完全抵消' if off_diag.mean() > 0.3 else '相关性较低，等权平均大幅抵消信号'}

  C. 有效变化参数: {n_effective}/{n_params} (CV >= 0.01)
     |corr(参数, agg_loss)| > 0.1: {n_sig_corr}/{n_params}
     → {'参数空间覆盖充分' if n_effective > n_params * 0.8 else '部分参数未有效变化'}

   D. 信号强度 (n={n}, p={p}):
     Aggregate: Train R²={r2_agg:.4f}, Adj R²={adj_r2_agg:.4f}, CV R²={cv_r2_agg:.4f}
     Per-task mean: Train R²={np.mean(train_r2s):.4f}, Adj R²={np.mean(adj_r2s):.4f}, CV R²={np.mean(cv_r2s):.4f}
     → {'CV R² > 0.3: 参数变化能有效预测 loss' if cv_r2_agg > 0.3 else 'CV R² < 0.3: 信号弱，参数变化难以预测 loss'}
     → {'过拟合风险: Train-CV gap > 0.2' if r2_agg - cv_r2_agg > 0.2 else '过拟合可控'}
""")


if __name__ == "__main__":
    main()
