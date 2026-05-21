"""Sampling helpers for large-scale datasets.

Provides:
  - sample_with_optimal_params: apply optimal QuaDMix params to select documents
  - save_sampled_dataset: save selected documents to parquet/jsonl
"""

from typing import List, Optional, Tuple
import os

import numpy as np
import numpy.typing as npt
import pandas as pd

from quadmix.core.types import ParameterSet
from quadmix.core.sampler import compute_sampling_values


def _select_documents_vectorized(
    sampling_values: npt.NDArray[np.float64],
    rng: np.random.Generator = np.random.default_rng(),
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Select documents based on sampling values (fully vectorized).

    Args:
        sampling_values: Fractional sampling expectations per document.
        rng: Random number generator.

    Returns:
        Tuple of (selected_indices, selection_weights).
        Each index may appear multiple times (if sampling_value > 1).
    """
    # Integer part: deterministic repeats
    int_part = np.floor(sampling_values).astype(np.int64)
    # Fractional part: stochastic Bernoulli
    frac_part = sampling_values - int_part
    random_mask = rng.uniform(size=len(sampling_values)) < frac_part

    # Build selected indices using repeat + concatenation (fully vectorized)
    repeats = int_part + random_mask.astype(np.int64)
    doc_indices = np.arange(len(sampling_values), dtype=np.int64)
    selected = np.repeat(doc_indices, repeats)

    # Compute selection weights (1 / original sampling_value for importance sampling)
    weights = 1.0 / np.maximum(sampling_values[selected], 1e-10)

    return selected, weights


def sample_with_optimal_params(
    quality_ranks: npt.NDArray[np.float64],
    domain_labels: npt.NDArray[np.int64],
    params: ParameterSet,
    rng: np.random.Generator = np.random.default_rng(),
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Apply optimal QuaDMix parameters to produce a sampled dataset.

    Args:
        quality_ranks: Per-document quality ranks [0, 1], 0 = best.
        domain_labels: Per-document domain labels.
        params: Optimal QuaDMix parameter set.
        rng: Random number generator.

    Returns:
        Tuple of (selected_indices, sampling_values, selection_weights).
    """
    # Compute sampling values (Eq.3) for all documents
    sampling_values = compute_sampling_values(quality_ranks, domain_labels, params)
    # Select documents based on sampling values
    selected_indices, selection_weights = _select_documents_vectorized(sampling_values, rng)
    return selected_indices, sampling_values, selection_weights


def save_sampled_dataset(
    original_texts: List[str],
    selected_indices: npt.NDArray[np.int64],
    output_path: str,
    domain_labels: Optional[npt.NDArray[np.int64]] = None,
    quality_ranks: Optional[npt.NDArray[np.float64]] = None,
    sampling_values: Optional[npt.NDArray[np.float64]] = None,
    doc_ids: Optional[List[str]] = None,
    format: str = "parquet",
):
    """Save the sampled dataset with metadata.

    Args:
        original_texts: Full list of original documents.
        selected_indices: Indices of selected documents (may repeat).
        output_path: Where to save the sampled dataset.
        domain_labels: Original domain labels (for joining).
        quality_ranks: Original quality ranks (for metadata).
        sampling_values: Sampling values at selection time.
        doc_ids: Original document IDs.
        format: Output format ("parquet" or "jsonl").
    """
    selected_texts = [original_texts[i] for i in selected_indices]

    records = {"text": selected_texts}

    if doc_ids is not None:
        records["doc_id"] = [doc_ids[i] for i in selected_indices]

    if domain_labels is not None:
        records["domain"] = domain_labels[selected_indices].tolist()

    if quality_ranks is not None:
        records["quality_rank"] = quality_ranks[selected_indices].tolist()

    if sampling_values is not None:
        records["sampling_weight"] = 1.0 / np.maximum(sampling_values[selected_indices], 1e-10)
        records["sampling_value"] = sampling_values[selected_indices].tolist()

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if format == "parquet":
        df.to_parquet(output_path, index=False)
    elif format == "jsonl":
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"[Save] Sampled dataset saved to: {output_path}")
    print(f"[Save]   Original docs: {len(original_texts)}")
    print(f"[Save]   Selected docs: {len(selected_indices)}")
    print(f"[Save]   Sampling ratio: {len(selected_indices) / max(1, len(original_texts)):.4f}x")
