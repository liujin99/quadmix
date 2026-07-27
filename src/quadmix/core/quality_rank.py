"""
Quality rank computation — Equation (2) from the paper.

    ¯r = |{x | d_x = m, ¯q_x >= ¯q}| / |{x | d_x = m}|

The merged quality rank ¯r is the percentile of a document within its domain,
where TOKEN counts are used for the denominator (not document counts).

Lower ¯r means higher quality (0 = best in domain, 1 = worst).

QuaDMix Input Contract:
    merged_scores: ndarray of shape (num_docs,)
    Convention: higher = better (e.g., higher probability, higher confidence).
    Users pass raw scores directly — no negation needed.
"""

import os
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple


def _rank_one_domain(
    merged_scores: npt.NDArray[np.float64],
    domain_labels: npt.NDArray[np.int64],
    token_counts: npt.NDArray[np.int64],
    m: int,
    seed: Optional[int],
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Compute quality ranks for a single domain.

    Returns (indices, rank_values) where indices are the original
    document indices belonging to domain m, and rank_values are
    their quality ranks.

    Uses a per-domain RNG (seed + m + 1) for thread-safe tie-breaking.
    """
    mask = domain_labels == m
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return indices, np.array([], dtype=np.float64)

    rng = np.random.default_rng(None if seed is None else seed + m + 1)
    domain_scores = merged_scores[indices]
    domain_tokens = token_counts[indices].astype(np.float64)
    total_tokens = domain_tokens.sum()

    if total_tokens < 1e-10:
        return indices, np.full(len(indices), 0.5, dtype=np.float64)

    jittered_scores = domain_scores + rng.uniform(0, 1e-12, len(domain_scores))
    sort_order = np.argsort(-jittered_scores, kind='mergesort')
    sorted_scores = jittered_scores[sort_order]
    sorted_tokens = domain_tokens[sort_order]
    cumulative = np.cumsum(sorted_tokens)

    diff = np.diff(sorted_scores)
    if (diff == 0).any():
        tie_start = np.concatenate([[True], diff != 0])
        group_starts = np.where(tie_start)[0]
        group_ends = np.append(group_starts[1:], len(cumulative)) - 1
        group_end_cumsum = cumulative[group_ends]
        groups = np.cumsum(tie_start) - 1
        tied_ranks = group_end_cumsum[groups] / total_tokens
    else:
        tied_ranks = cumulative / total_tokens

    inv_sort = np.argsort(sort_order)

    return indices, tied_ranks[inv_sort]


def compute_quality_ranks(
    merged_scores: npt.NDArray[np.float64],
    domain_labels: npt.NDArray[np.int64],
    token_counts: Optional[npt.NDArray[np.int64]] = None,
    seed: Optional[int] = None,
    n_jobs: int = 1,
) -> npt.NDArray[np.float64]:
    """
    Compute quality percentile ranks ¯r within each domain (Equation 2).

    Args:
        merged_scores: Array of merged quality scores ¯q.
                       Shape: (num_docs,) — higher = better.
        domain_labels: Array of domain labels.
                       Shape: (num_docs,).
        token_counts: Optional per-document token counts.
                      Shape: (num_docs,).
                      If None, each document gets equal weight.
                      If provided, token-count-weighted percentiles are
                      used, as in the paper: "calculate the size of the set
                      by adding up the number of tokens for all samples within the set."
        seed: Optional RNG seed for deterministic tie-breaking.
              If None, uses OS entropy (non-deterministic).
        n_jobs: Number of parallel workers.
                Default 1 (sequential). Use -1 for all CPU cores.
                The per-domain rank computation is parallelized
                across domains using threads. When n_jobs > 1,
                each domain uses its own RNG (seed + m + 1) for
                thread safety.

    Returns:
        Quality rank ¯r for each document.
        Shape: (num_docs,) — 0 = best, 1 = worst in domain.
    """
    num_docs = len(merged_scores)
    ranks = np.zeros(num_docs, dtype=np.float64)

    if token_counts is None:
        token_counts = np.ones(num_docs, dtype=np.int64)

    unique_domains = np.unique(domain_labels)

    effective_jobs = n_jobs if n_jobs != -1 else (os.cpu_count() or 1)
    if effective_jobs > 1 and len(unique_domains) > 1:
        from joblib import Parallel, delayed
        results = Parallel(
            n_jobs=min(effective_jobs, len(unique_domains)), prefer="threads"
        )(
            delayed(_rank_one_domain)(
                merged_scores, domain_labels, token_counts, int(m), seed
            )
            for m in unique_domains
        )
        for indices, domain_ranks in results:
            if len(indices) > 0:
                ranks[indices] = domain_ranks
    else:
        rng = np.random.default_rng(seed)
        for m in unique_domains:
            mask = domain_labels == m
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            domain_scores = merged_scores[indices]
            domain_tokens = token_counts[indices].astype(np.float64)
            total_tokens = domain_tokens.sum()

            if total_tokens < 1e-10:
                ranks[indices] = 0.5
                continue

            # Break exact ties with negligible noise (~8600x smaller than
            # the minimum rank-normalized score gap of 1/num_docs) so that
            # max-rank tie handling does not collapse tied docs to rank ≈ 1.0.
            jittered_scores = domain_scores + rng.uniform(0, 1e-12, len(domain_scores))
            sort_order = np.argsort(-jittered_scores, kind='mergesort')
            sorted_scores = jittered_scores[sort_order]
            sorted_tokens = domain_tokens[sort_order]
            cumulative = np.cumsum(sorted_tokens)

            diff = np.diff(sorted_scores)
            if (diff == 0).any():
                tie_start = np.concatenate([[True], diff != 0])
                group_starts = np.where(tie_start)[0]
                group_ends = np.append(group_starts[1:], len(cumulative)) - 1
                group_end_cumsum = cumulative[group_ends]
                groups = np.cumsum(tie_start) - 1
                tied_ranks = group_end_cumsum[groups] / total_tokens
            else:
                tied_ranks = cumulative / total_tokens

            inv_sort = np.argsort(sort_order)
            ranks[indices] = tied_ranks[inv_sort]

    return ranks
