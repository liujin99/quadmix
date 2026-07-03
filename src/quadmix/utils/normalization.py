"""
Normalization function σ for quality scores.

The paper uses σ to align the scales of different quality criteria
before merging (Equation 1).

QuaDMix Input Contract:
    quality_matrix: ndarray of shape (num_docs, num_criteria)
    Convention: higher = better (e.g., higher probability, higher confidence).
    Users pass raw scores directly — no negation needed.
    Most real-world quality scorers output "higher = better" naturally.

    Normalization functions here are mathematical transforms. They do
    not enforce or validate direction. rank_normalize assigns rank 0
    to the smallest value and rank ~1 to the largest, which is correct
    under the "higher = better" convention (best doc gets highest rank).
    threshold_rank detects signal layer as the top tail (largest values).

rank is the default normalizer because it is robust across all distribution
shapes and produces the best predictive performance (R²=0.808).

threshold_rank is available as an alternative for right-skewed distributions:
- For skewed distributions (skew > 4): uses percentile to separate
  noise from signal. Noise layer → 0, signal layer → rank [0, 1].
- For moderate/uniform distributions: falls back to pure rank.
- Caution: changes α weight semantics (only signal layer docs affected).
"""

import numpy as np
import numpy.typing as npt
from scipy import stats as sp_stats
from typing import Callable


def zscore_normalize(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Z-score normalization: (x - μ) / σ.
    Preserves relative ordering. Higher = better (direction-preserving).
    """
    mean = np.mean(scores)
    std = np.std(scores)
    if std < 1e-10:
        return np.zeros_like(scores)
    return (scores - mean) / std


def minmax_normalize(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Min-max normalization to [0, 1].
    Higher = better (largest value maps to 1.0, smallest to 0.0).
    """
    s_min, s_max = np.min(scores), np.max(scores)
    if s_max - s_min < 1e-10:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def rank_normalize(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Rank-based normalization to [0, 1].
    Higher = better (largest value gets rank ~1, smallest gets rank ~0).
    Tied scores receive the average rank.
    """
    n = len(scores)
    if n == 0:
        return scores
    ranks = sp_stats.rankdata(scores, method='average')
    return (ranks - 1).astype(np.float64) / n


def log1p_z_normalize(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Log1p + Z-score normalization: zscore(log(1 + x - min(x))).
    Shifts to non-negative, applies log to compress heavy tails, then standardizes.
    Best for skewed distributions where rank would distort signal.
    """
    if len(scores) == 0:
        return scores
    shifted = scores - scores.min()
    logged = np.log1p(shifted)
    mean = np.mean(logged)
    std = np.std(logged)
    if std < 1e-10:
        return np.zeros_like(scores)
    return (logged - mean) / std


def _detect_threshold(
    scores: npt.NDArray[np.float64],
    skew: float,
) -> tuple[float, int]:
    """Detect noise/signal boundary via percentile for skewed distributions.

    In "higher = better" convention:
    - Right-skewed (skew > 4): most docs have low scores (noise), few have high scores (signal)
      → threshold = p75, signal = scores > p75
    - Left-skewed (skew < -4): most docs have high scores (saturated, no clear noise layer)
      → fall back to pure rank (return direction = 0)

    Returns:
        (threshold_value, direction): direction=1 for right-skew, 0 for fallback.
    """
    if skew <= 4:
        return 0.0, 0

    return float(np.percentile(scores, 75)), 1


def threshold_rank_normalize(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Threshold + rank: noise layer → 0, signal layer → rank [0, 1].

    For right-skewed distributions (skew > 4): uses percentile to separate
    noise from signal. Noise layer (low scores) gets 0, signal layer
    (high scores) gets rank within [0, 1]. Falls back to pure rank otherwise.
    """
    n = len(scores)
    if n == 0:
        return scores

    skew = float(sp_stats.skew(scores))
    threshold, direction = _detect_threshold(scores, skew)

    if direction == 0:
        return rank_normalize(scores)

    result = np.zeros(n, dtype=np.float64)

    signal_mask = scores > threshold

    signal_count = signal_mask.sum()
    if signal_count < max(10, n // 10):
        return rank_normalize(scores)

    signal_scores = scores[signal_mask]
    signal_ranks = sp_stats.rankdata(signal_scores, method='average')

    result[signal_mask] = (signal_ranks - 1).astype(np.float64) / signal_count

    return result


# Registry of available normalization functions
NORMALIZATION_REGISTRY: dict[str, Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = {
    "zscore": zscore_normalize,
    "minmax": minmax_normalize,
    "rank": rank_normalize,
    "log1p_z": log1p_z_normalize,
    "threshold_rank": threshold_rank_normalize,
}


def get_normalizer(name: str = "rank") -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Get a normalization function by name."""
    if name not in NORMALIZATION_REGISTRY:
        raise ValueError(
            f"Unknown normalizer '{name}'. Available: {list(NORMALIZATION_REGISTRY.keys())}"
        )
    return NORMALIZATION_REGISTRY[name]
