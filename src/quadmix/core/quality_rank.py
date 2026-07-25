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

import numpy as np
import numpy.typing as npt
from typing import Optional


def compute_quality_ranks(
    merged_scores: npt.NDArray[np.float64],
    domain_labels: npt.NDArray[np.int64],
    token_counts: Optional[npt.NDArray[np.int64]] = None,
    seed: Optional[int] = None,
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

    Returns:
        Quality rank ¯r for each document.
        Shape: (num_docs,) — 0 = best, 1 = worst in domain.
    """
    num_docs = len(merged_scores)
    ranks = np.zeros(num_docs, dtype=np.float64)
    rng = np.random.default_rng(seed)

    if token_counts is None:
        token_counts = np.ones(num_docs, dtype=np.int64)

    unique_domains = np.unique(domain_labels)

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
        tie_start = np.concatenate([[True], diff != 0])
        groups = np.cumsum(tie_start) - 1

        group_end_cumsum = np.zeros(np.max(groups) + 1, dtype=np.float64)
        np.maximum.at(group_end_cumsum, groups, cumulative)

        tied_ranks = group_end_cumsum[groups] / total_tokens

        inv_sort = np.argsort(sort_order)
        ranks[indices] = tied_ranks[inv_sort]

    return ranks
