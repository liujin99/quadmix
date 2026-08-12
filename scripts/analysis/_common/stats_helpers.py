"""Shared statistics helpers for analysis scripts.

Extracted from analyze_pipeline_output.py and analyze_arm_data.py to
eliminate duplication of _spearman, _spearman_rho_vs_fixed.
"""

import os
import numpy as np


def spearman(x, y):
    """Spearman rank correlation between two arrays."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx_m = rx - rx.mean()
    ry_m = ry - ry.mean()
    den = np.sqrt(np.sum(rx_m ** 2) * np.sum(ry_m ** 2))
    return float(np.sum(rx_m * ry_m) / den) if den > 0 else 0.0


def spearman_rho_vs_fixed(quality_scores, y, n_criteria, n_jobs):
    """Spearman ρ of each quality criterion vs a FIXED y (e.g. char_counts).

    Optimized over a per-criterion spearman loop:
      - the rank of y is computed ONCE (not N_criteria times),
      - each x is ranked with a single argsort + scatter (not double argsort),
      - the N_criteria columns are ranked in parallel via threads.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 2:
        return np.zeros(n_criteria, dtype=np.float64)
    order_y = np.argsort(y, kind="quicksort")
    ry = np.empty(n, dtype=np.float64)
    ry[order_y] = np.arange(n, dtype=np.float64)
    ry_m = ry - ry.mean()
    ry_ss = float(np.sum(ry_m ** 2))

    def _rho_col(k):
        x = np.asarray(quality_scores[:, k], dtype=np.float64)
        if len(x) < 2:
            return 0.0
        order_x = np.argsort(x, kind="quicksort")
        rx = np.empty(n, dtype=np.float64)
        rx[order_x] = np.arange(n, dtype=np.float64)
        rx_m = rx - rx.mean()
        rx_ss = float(np.sum(rx_m ** 2))
        return float(np.sum(rx_m * ry_m) / np.sqrt(rx_ss * ry_ss)) if rx_ss > 0 and ry_ss > 0 else 0.0

    from joblib import Parallel, delayed
    effective = n_jobs if n_jobs != -1 else (os.cpu_count() or 1)
    if effective > 1 and n_criteria > 1:
        results = Parallel(n_jobs=min(effective, n_criteria), prefer="threads")(
            delayed(_rho_col)(k) for k in range(n_criteria)
        )
        return np.array(results, dtype=np.float64)
    return np.array([_rho_col(k) for k in range(n_criteria)], dtype=np.float64)
