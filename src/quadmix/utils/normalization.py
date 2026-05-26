"""
Normalization function σ for quality scores.

The paper uses σ to align the scales of different quality criteria
before merging (Equation 1). Smaller values indicate better quality.
"""

import numpy as np
import numpy.typing as npt
from typing import Callable


def zscore_normalize(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Z-score normalization: (x - μ) / σ.
    Preserves relative ordering. Smaller = better (when mean-subtracted).
    """
    mean = np.mean(scores)
    std = np.std(scores)
    if std < 1e-10:
        return np.zeros_like(scores)
    return (scores - mean) / std


def minmax_normalize(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Min-max normalization to [0, 1].
    Smaller = better (inverted: 1 - minmax).
    """
    s_min, s_max = np.min(scores), np.max(scores)
    if s_max - s_min < 1e-10:
        return np.zeros_like(scores)
    return 1.0 - (scores - s_min) / (s_max - s_min)


def rank_normalize(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Rank-based normalization to [0, 1].
    Smaller = better (rank percentile where 0 = best).
    This is most faithful to the paper's approach since the quality
    values get re-ranked later anyway.
    """
    n = len(scores)
    if n == 0:
        return scores
    ranks = np.argsort(np.argsort(scores))  # 0 = smallest (best)
    return ranks.astype(np.float64) / n


# Registry of available normalization functions
NORMALIZATION_REGISTRY: dict[str, Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = {
    "zscore": zscore_normalize,
    "minmax": minmax_normalize,
    "rank": rank_normalize,
}


def get_normalizer(name: str = "rank") -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Get a normalization function by name."""
    if name not in NORMALIZATION_REGISTRY:
        raise ValueError(
            f"Unknown normalizer '{name}'. Available: {list(NORMALIZATION_REGISTRY.keys())}"
        )
    return NORMALIZATION_REGISTRY[name]
