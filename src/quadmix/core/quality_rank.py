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
    sort_order = np.argsort(-jittered_scores, kind='quicksort')
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

    inv_sort = np.empty_like(sort_order)
    inv_sort[sort_order] = np.arange(len(sort_order))

    return indices, tied_ranks[inv_sort]


def compute_quality_ranks(
    merged_scores: npt.NDArray[np.float64],
    domain_labels: npt.NDArray[np.int64],
    token_counts: Optional[npt.NDArray[np.int64]] = None,
    seed: Optional[int] = None,
    n_jobs: int = 1,
    unique_domains: Optional[npt.NDArray[np.int64]] = None,
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
        unique_domains: Optional precomputed sorted unique domain labels.
                       If None (default), computed via np.unique(domain_labels).
                       Pass to avoid a redundant full-corpus sort.

    Returns:
        Quality rank ¯r for each document.
        Shape: (num_docs,) — 0 = best, 1 = worst in domain.
    """
    num_docs = len(merged_scores)
    ranks = np.zeros(num_docs, dtype=np.float64)

    if token_counts is None:
        token_counts = np.ones(num_docs, dtype=np.int64)

    if unique_domains is None:
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
            sort_order = np.argsort(-jittered_scores, kind='quicksort')
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

            inv_sort = np.empty_like(sort_order)
            inv_sort[sort_order] = np.arange(len(sort_order))
            ranks[indices] = tied_ranks[inv_sort]

    return ranks


# ── Length-stratified variant ──────────────────────────────


def _rank_within_subset(
    scores: npt.NDArray[np.float64],
    tokens: npt.NDArray[np.int64],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Token-weighted cumulative percentile rank for a pre-filtered subset.

    This is the core ranking logic extracted from ``_rank_one_domain``:
    sort by score descending, then cumulative token fraction = rank.
    Lower rank = higher quality (0 = best in subset, 1 = worst).

    Parameters
    ----------
    scores : (n,)  Merged quality scores (higher = better).
    tokens : (n,)  Per-doc token counts.
    rng    :        Generator for tie-breaking jitter.

    Returns
    -------
    ranks : (n,)  Percentile ranks in [0, 1], 0 = best.
    """
    n = len(scores)
    if n == 0:
        return np.array([], dtype=np.float64)

    tokens_f = tokens.astype(np.float64)
    total_tokens = tokens_f.sum()
    if total_tokens < 1e-10:
        return np.full(n, 0.5, dtype=np.float64)

    jittered = scores + rng.uniform(0, 1e-12, n)
    sort_order = np.argsort(-jittered, kind='quicksort')
    sorted_tokens = tokens_f[sort_order]
    cumulative = np.cumsum(sorted_tokens)

    diff = np.diff(jittered[sort_order])
    if (diff == 0).any():
        tie_start = np.concatenate([[True], diff != 0])
        group_starts = np.where(tie_start)[0]
        group_ends = np.append(group_starts[1:], len(cumulative)) - 1
        group_end_cumsum = cumulative[group_ends]
        groups = np.cumsum(tie_start) - 1
        tied_ranks = group_end_cumsum[groups] / total_tokens
    else:
        tied_ranks = cumulative / total_tokens

    inv_sort = np.empty_like(sort_order)
    inv_sort[sort_order] = np.arange(n)
    return tied_ranks[inv_sort]


def _rank_one_domain_stratified(
    merged_scores: npt.NDArray[np.float64],
    domain_labels: npt.NDArray[np.int64],
    token_counts: npt.NDArray[np.int64],
    char_counts: npt.NDArray[np.int64],
    m: int,
    num_strata: int,
    seed: Optional[int],
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Compute length-stratified quality ranks for a single domain.

    Splits the domain's documents into *num_strata* groups by char_count
    quantiles, then ranks within each group using token-weighted percentiles.
    """
    mask = domain_labels == m
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return indices, np.array([], dtype=np.float64)

    domain_scores = merged_scores[indices]
    domain_tokens = token_counts[indices]
    domain_chars = char_counts[indices].astype(np.float64)

    K = num_strata
    quantiles = np.linspace(0, 1, K + 1)
    boundaries = np.quantile(domain_chars, quantiles)

    stratum_ids = np.zeros(len(indices), dtype=np.int64)
    for k in range(1, K):
        stratum_ids[domain_chars > boundaries[k]] = k

    rng = np.random.default_rng(None if seed is None else seed + m + 1)

    ranks_local = np.zeros(len(indices), dtype=np.float64)
    for k in range(K):
        s_mask = stratum_ids == k
        if not s_mask.any():
            continue
        ranks_local[s_mask] = _rank_within_subset(
            domain_scores[s_mask], domain_tokens[s_mask], rng,
        )

    return indices, ranks_local


def compute_stratified_quality_ranks(
    merged_scores: npt.NDArray[np.float64],
    domain_labels: npt.NDArray[np.int64],
    token_counts: npt.NDArray[np.int64],
    char_counts: npt.NDArray[np.int64],
    num_strata: int = 4,
    seed: Optional[int] = None,
    n_jobs: int = 1,
    unique_domains: Optional[npt.NDArray[np.int64]] = None,
) -> npt.NDArray[np.float64]:
    """Compute length-stratified quality percentile ranks (stratified variant of Eq.2).

    Purpose
    -------
    Standard ``compute_quality_ranks`` ranks each document within its full
    domain.  When quality scores correlate with document length (longer docs
    get higher quality scores), high-rank (selected) docs skew long, reducing
    diversity and packing efficiency in downstream training.

    This function splits each domain into *K* length strata (by char_count
    quantiles) and ranks documents **within each (domain × stratum) cell**.
    The top-ω fraction of every stratum is then equally likely to be
    selected, producing a length-balanced output without changing the
    sampling formula S(r̄) or the per-domain parameters (ω, λ, η, ε).

    Relationship to ``compute_quality_ranks``
    -----------------------------------------
    * **Same input contract**: ``merged_scores`` (higher = better),
      ``domain_labels``, ``token_counts`` — all identical.
    * **Same output format**: shape ``(num_docs,)``, values in [0, 1],
      0 = best.  Drop-in replacement for the ranks argument passed to
      ``compute_sampling_values`` / ``sample_with_optimal_params``.
    * **Same seed scheme**: per-domain RNG (``seed + m + 1``) for
      tie-breaking, matching ``_rank_one_domain``.
    * **Additional parameter**: ``char_counts`` (required) and
      ``num_strata`` (default 4).

    Key design decisions
    --------------------
    * **Per-domain stratum boundaries**: each domain uses its own
      char_count quantiles, so strata are balanced within domain.
      Global boundaries would make strata extremely unbalanced because
      domains like chemistry have median char_count ≈ 171 while math ≈ 2611.

    * **Sampler is unchanged**: the downstream sampler
      (``compute_sampling_values``) uses the **original** ``domain_labels``
      to look up per-domain (ω, λ, η, ε) and applies S(r̄) to the stratified
      rank.  No sampler code changes are needed — simply pass the stratified
      ranks in place of the standard ranks.

    * **S(r̄) formula is unchanged**: the sigmoid threshold ω now selects
      the top-ω% of *each stratum* rather than the top-ω% of the full
      domain, which is the desired length-balancing behaviour.

    Parameters
    ----------
    merged_scores : (num_docs,)
        Merged quality scores ¯q (higher = better, from Eq.1).
    domain_labels : (num_docs,)
        Domain label for each doc.
    token_counts : (num_docs,)
        Per-doc token counts (char_count // 4).
    char_counts : (num_docs,)
        Per-doc character counts — used for length stratification.
        Must be the same length as ``merged_scores``.
    num_strata : int, default 4
        Number of length strata per domain.  Each domain is split into
        this many equal-population groups by char_count quantiles.
        K=4 (quartiles) is the default, matching the value validated
        in the within-stratum ρ analysis (63% ρ reduction).
    seed : int | None
        RNG seed for deterministic tie-breaking.  If None, uses OS entropy.
    n_jobs : int
        Parallel workers (default 1).  Use -1 for all CPU cores.
        Parallelised across domains, matching ``compute_quality_ranks``.
    unique_domains : (M,) | None
        Precomputed sorted unique domain labels.  If None, computed
        via ``np.unique(domain_labels)``.

    Returns
    -------
    ranks : (num_docs,)
        Stratified quality rank per doc, [0, 1], 0 = best.
        Within each (domain × stratum) cell, ranks follow a
        token-weighted cumulative percentile distribution.
    """
    num_docs = len(merged_scores)
    ranks = np.zeros(num_docs, dtype=np.float64)

    if token_counts is None:
        token_counts = np.ones(num_docs, dtype=np.int64)

    if unique_domains is None:
        unique_domains = np.unique(domain_labels)

    effective_jobs = n_jobs if n_jobs != -1 else (os.cpu_count() or 1)
    if effective_jobs > 1 and len(unique_domains) > 1:
        from joblib import Parallel, delayed
        results = Parallel(
            n_jobs=min(effective_jobs, len(unique_domains)), prefer="threads"
        )(
            delayed(_rank_one_domain_stratified)(
                merged_scores, domain_labels, token_counts, char_counts,
                int(m), num_strata, seed,
            )
            for m in unique_domains
        )
        for indices, domain_ranks in results:
            if len(indices) > 0:
                ranks[indices] = domain_ranks
    else:
        for m in unique_domains:
            indices, domain_ranks = _rank_one_domain_stratified(
                merged_scores, domain_labels, token_counts, char_counts,
                int(m), num_strata, seed,
            )
            if len(indices) > 0:
                ranks[indices] = domain_ranks

    return ranks
