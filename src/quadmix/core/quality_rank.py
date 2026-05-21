"""
Quality rank computation — Equation (2) from the paper.

    ¯r = |{x | d_x = m, ¯q_x <= ¯q}| / |{x | d_x = m}|

The merged quality rank ¯r is the percentile of a document within its domain,
where TOKEN counts are used for the denominator (not document counts).

Lower ¯r means higher quality (0 = best in domain, 1 = worst).
"""

import numpy as np
import numpy.typing as npt
from typing import Optional

from quadmix.core.types import QualityScore


def compute_quality_ranks(
    merged_scores: npt.NDArray[np.float64],
    domain_labels: npt.NDArray[np.int64],
    token_counts: Optional[npt.NDArray[np.int64]] = None,
) -> npt.NDArray[np.float64]:
    """
    Compute quality percentile ranks ¯r within each domain (Equation 2).

    Args:
        merged_scores: Array of merged quality scores ¯q.
                       Shape: (num_docs,) — smaller = better.
        domain_labels: Array of domain labels.
                       Shape: (num_docs,).
        token_counts: Optional per-document token counts.
                      Shape: (num_docs,).
                      If None, each document gets equal weight.
                      If provided, token-count-weighted percentiles are
                      used, as in the paper: "calculate the size of the set
                      by adding up the number of tokens for all samples within the set."

    Returns:
        Quality rank ¯r for each document.
        Shape: (num_docs,) — 0 = best, 1 = worst in domain.
    """
    num_docs = len(merged_scores)
    ranks = np.zeros(num_docs, dtype=np.float64)

    # Default: equal weight per document
    if token_counts is None:
        token_counts = np.ones(num_docs, dtype=np.int64)

    unique_domains = np.unique(domain_labels)

    for m in unique_domains:
        mask = domain_labels == m
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        # Get scores and token counts for this domain
        domain_scores = merged_scores[indices]
        domain_tokens = token_counts[indices].astype(np.float64)
        total_tokens = domain_tokens.sum()

        if total_tokens < 1e-10:
            ranks[indices] = 0.5  # Edge case: all tokens 0
            continue

        # Sort by merged quality (ascending: best quality first)
        sort_order = np.argsort(domain_scores)

        # Compute cumulative token percentage (token-weighted percentile)
        # For each document, ¯r = (cumulative tokens for same or worse quality) / total tokens
        # Per paper: |{x | d_x = m, ¯q_x <= ¯q}|
        sorted_tokens = domain_tokens[sort_order]
        cumulative = np.cumsum(sorted_tokens)

        # Map back to original order
        # The rank for document i at sorted position pos is cumulative[pos] / total
        # Since cumulative[pos] includes document itself (¯q_x <= ¯q)
        sorted_ranks = cumulative / total_tokens

        # Remove the "self" token to get strictly less-than fraction
        # This gives 0 for the best document instead of token_fraction_of_best
        # Actually, per paper: ¯r = |{x | d_x = m, ¯q_x <= ¯q}| / |{x | d_x = m}|
        # The set includes the document itself, so cumulative[t] / total is correct.
        # Best document has ¯r ≈ tokens_of_best / total (very small but > 0)
        # We'll keep it as-is which matches the paper definition.

        # Map sorted positions back to original indices
        inv_sort = np.argsort(sort_order)
        ranks[indices] = sorted_ranks[inv_sort]

    return ranks
