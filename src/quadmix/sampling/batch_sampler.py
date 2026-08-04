"""Sampling helpers for large-scale datasets.

Provides:
  - sample_with_optimal_params: apply optimal QuaDMix params to select documents
  - save_sampled_dataset: save selected documents to parquet/jsonl
"""

from typing import Callable, Dict, List, Optional, Tuple
import glob
import json
import os
import time

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quadmix.core.types import ParameterSet
from quadmix.core.sampler import compute_sampling_values


def resolve_parquet_source(path: str) -> List[str]:
    """Resolve a parquet source path to a list of file paths.

    Handles both sharded (directory) and legacy (single file) outputs.

    Args:
        path: Path to a directory containing ``shard_*.parquet`` files,
            or a single ``.parquet`` file.

    Returns:
        Sorted list of parquet file paths.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If *path* is a directory but contains no ``.parquet``
            files.
    """
    if os.path.isdir(path):
        shards = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not shards:
            raise ValueError(
                f"Directory '{path}' contains no .parquet files"
            )
        return shards
    elif os.path.isfile(path):
        return [path]
    else:
        raise FileNotFoundError(
            f"Parquet source not found: {path} (not a file or directory)"
        )


def _select_documents_vectorized(
    sampling_values: npt.NDArray[np.float64],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Select documents based on sampling values (fully vectorized).

    Args:
        sampling_values: Fractional sampling expectations per document.
        rng: Random number generator.

    Returns:
        Tuple of (selected_indices, selection_weights).
        Each index may appear multiple times (if sampling_value > 1).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    int_part = np.floor(sampling_values).astype(np.int64)
    frac_part = sampling_values - int_part
    random_mask = rng.uniform(size=len(sampling_values)) < frac_part

    repeats = int_part + random_mask.astype(np.int64)
    doc_indices = np.arange(len(sampling_values), dtype=np.int64)
    selected = np.repeat(doc_indices, repeats)

    weights = 1.0 / np.maximum(sampling_values[selected], 1e-10)

    return selected, weights


def sample_with_optimal_params(
    quality_ranks: npt.NDArray[np.float64],
    domain_labels: npt.NDArray[np.int64],
    params: ParameterSet,
    rng: Optional[np.random.Generator] = None,
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
    sampling_values = compute_sampling_values(quality_ranks, domain_labels, params)
    selected_indices, selection_weights = _select_documents_vectorized(sampling_values, rng)
    return selected_indices, sampling_values, selection_weights


def save_sampled_dataset(
        get_text_fn: Callable[[npt.NDArray[np.int64]], List[str]],
        num_total_docs: int,
    selected_indices: npt.NDArray[np.int64],
    output_path: str,
    domain_labels: Optional[npt.NDArray[np.int64]] = None,
    quality_ranks: Optional[npt.NDArray[np.float64]] = None,
    sampling_values: Optional[npt.NDArray[np.float64]] = None,
    doc_id_fn: Optional[Callable[[int], str]] = None,
    format: str = "parquet",
    text_col: str = "text",
    domain_col: str = "domain",
    batch_size: int = 100000,
    quality_scores: Optional[Dict[str, npt.NDArray[np.float64]]] = None,
):
    """Save the sampled dataset with metadata.

    Reads each unique document's text exactly once via get_text_fn, then
    expands back to the selected output order (which may contain repeats
    when sampling_value > 1). This avoids re-opening shard files once per
    batch and keeps the read phase to a single callback invocation.

    Args:
        get_text_fn: Callable accepting a numpy array of indices and returning
            a list of text strings. For sharded datasets, use
            metadata_manager.read_texts directly. For in-memory datasets,
            wrap with lambda: lambda idx: [texts[i] for i in idx].
            May optionally accept a `verbose` keyword (silenced here to keep
            the save log concise).
        num_total_docs: Total number of documents in the original corpus.
        selected_indices: Indices of selected documents (may repeat).
        output_path: Where to save the sampled dataset.
            For parquet format, this is a directory path (sharded output
            with ``shard_NNNNN.parquet`` files and ``manifest.json``).
            For jsonl format, this is a file path.
        domain_labels: Original domain labels (for joining).
        quality_ranks: Original quality ranks (for metadata).
        sampling_values: Sampling values at selection time.
        doc_id_fn: Callable returning doc_id for a given index. If None,
            uses the index itself as doc_id.
        format: Output format ("parquet" or "jsonl").
        text_col: Column name for text.
        domain_col: Column name for domain.
        batch_size: Rows per parquet shard when writing sharded parquet
            output. Ignored for jsonl format.
    """

    n_selected = len(selected_indices)

    # Dedup indices so each unique document is read exactly once, then map
    # back to the (possibly repeated) output order via the inverse permutation.
    unique_indices, inverse = np.unique(selected_indices, return_inverse=True)
    n_unique = len(unique_indices)

    print(f"[Save] Reading {n_unique:,} unique texts "
          f"({n_selected:,} selected rows w/ repeats)...")
    t0 = time.time()

    try:
        unique_texts = get_text_fn(unique_indices, verbose=False)
    except TypeError:
        # Callback does not accept the verbose kwarg (e.g. in-memory lambda).
        unique_texts = get_text_fn(unique_indices)

    texts_arr = np.asarray(unique_texts, dtype=object)
    # Reconstruct the selected order; this is a pointer copy (no string dup).
    records = {text_col: texts_arr[inverse]}

    unique_char_counts = np.array([len(t) for t in unique_texts], dtype=np.int64)
    records["char_count"] = unique_char_counts[inverse]

    if doc_id_fn is not None:
        records["doc_id"] = [doc_id_fn(i) for i in selected_indices]
    else:
        records["doc_id"] = selected_indices

    if domain_labels is not None:
        records[domain_col] = domain_labels[selected_indices]

    if quality_ranks is not None:
        records["quality_rank"] = quality_ranks[selected_indices]

    if sampling_values is not None:
        records["sampling_weight"] = 1.0 / np.maximum(sampling_values[selected_indices], 1e-10)
        records["sampling_value"] = sampling_values[selected_indices]

    if quality_scores is not None:
        for name, arr in quality_scores.items():
            records[name] = arr[selected_indices]

    t_write = time.time()
    if format == "parquet":
        # Build the arrow Table directly, bypassing pandas. The text column is
        # constructed from the 4.7M unique strings + pa.compute.take(inverse)
        # so arrow copies only the unique string contents into its data buffer
        # once; take() then reuses that buffer via a new offsets array (no
        # per-row string re-copy). Numeric columns are plain numpy → arrow
        # memcpy. zstd + row_group_size lets pyarrow compress in chunks.
        arrays = {}
        for name, val in records.items():
            if name == text_col:
                unique_arrow = pa.array(texts_arr, type=pa.large_string())
                arrays[name] = pa.compute.take(unique_arrow, inverse)
            else:
                arrays[name] = pa.array(val)
        table = pa.table(arrays)

        n_rows = table.num_rows
        shard_size = max(1, batch_size)
        n_shards = (n_rows + shard_size - 1) // shard_size
        os.makedirs(output_path, exist_ok=True)
        for i in range(n_shards):
            start = i * shard_size
            end = min(start + shard_size, n_rows)
            shard = table.slice(start, end - start)
            shard_path = os.path.join(output_path, f"shard_{i:05d}.parquet")
            pq.write_table(
                shard,
                shard_path,
                row_group_size=500_000,
                compression="zstd",
                compression_level=3,
                use_dictionary=True,
            )
        manifest = {
            "n_shards": n_shards,
            "total_rows": n_rows,
            "n_unique": n_unique,
            "shard_size": shard_size,
            "columns": list(records.keys()),
        }
        with open(os.path.join(output_path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        del table
    elif format == "jsonl":
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pd.DataFrame(records).to_json(
            output_path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    write_elapsed = time.time() - t_write
    total_elapsed = time.time() - t0
    if format == "parquet":
        print(f"[Save] Sampled dataset saved to: {output_path}/ "
              f"({n_shards} shard{'s' if n_shards != 1 else ''})")
    else:
        print(f"[Save] Sampled dataset saved to: {output_path}")
    print(f"[Save]   Original docs: {num_total_docs}")
    print(f"[Save]   Selected docs: {n_selected} (unique: {n_unique})")
    print(f"[Save]   Sampling ratio: {n_selected / max(1, num_total_docs):.4f}x")
    print(f"[Save]   Text read: {total_elapsed - write_elapsed:.1f}s | "
          f"Write: {write_elapsed:.1f}s | Total: {total_elapsed:.1f}s")
