"""
Normalization function σ for quality scores.

The paper uses σ to align the scales of different quality criteria
before merging (Equation 1). Smaller values indicate better quality.

σ must preserve numerical relationships so that α weights in Eq.1
can meaningfully control the relative importance of each criterion.
Log1p-zscore is chosen as default because:
- Variance balance: all criteria get var=1.0 → equal α = equal importance
- Preserves signal in skewed distributions (dclm/math: 95% near-zero docs
  stay clustered, 5% high-quality docs retain their gap)
- log(1+x) compresses heavy tails before zscore standardization
- Outlier more robust than pure zscore (log dampens extreme values)
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


# Registry of available normalization functions
NORMALIZATION_REGISTRY: dict[str, Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = {
    "zscore": zscore_normalize,
    "minmax": minmax_normalize,
    "rank": rank_normalize,
    "log1p_z": log1p_z_normalize,
}


def get_normalizer(name: str = "log1p_z") -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Get a normalization function by name."""
    if name not in NORMALIZATION_REGISTRY:
        raise ValueError(
            f"Unknown normalizer '{name}'. Available: {list(NORMALIZATION_REGISTRY.keys())}"
        )
    return NORMALIZATION_REGISTRY[name]
