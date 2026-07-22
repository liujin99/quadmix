"""
ShardMetadataManager — loads metadata (domain + quality signals) from
preprocessed multi-shard parquet files into memory, supporting
on-demand text loading for selected documents.

Usage:
    mgr = ShardMetadataManager(preprocessed_dir)
    dom = mgr.domain_labels       # [N] int64
    qs  = mgr.quality_scores      # [N, 5] float64
    tx  = mgr.read_texts(indices)  # List[str] for selected docs
"""

import json, os, glob, time, re
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from quadmix.constants import QUALITY_COLUMNS

CHAR_COUNT_COL = "doc_char_count"
STEM_DOMAIN_TO_ID = {"数学": 0, "化学": 1, "生物学": 2, "物理": 3}


def _parse_shard_idx(path: str) -> int:
    groups = re.findall(r"\d+", os.path.splitext(os.path.basename(path))[0])
    if not groups:
        raise ValueError(f"Cannot parse shard index from filename: {path}")
    return int(groups[-1])


def read_parquet_text_rows(
    shard_path: str,
    rows: npt.NDArray[np.int64],
) -> Tuple[npt.NDArray[np.int64], List[str]]:
    """Read selected text rows from either QuaDMix or raw STEM parquet.

    Preprocessed shards are filtered through their physical ``row_in_shard``
    column. Raw STEM shards have no such column, so only the overlapping row
    groups are read and their physical row offsets are selected with Arrow.
    Returned rows are unique and sorted.
    """
    requested = np.unique(np.asarray(rows, dtype=np.int64))
    if len(requested) == 0:
        return requested, []
    if requested[0] < 0:
        raise IndexError(f"Negative row requested from {shard_path}")

    parquet_file = pq.ParquetFile(shard_path)
    if "row_in_shard" in parquet_file.schema_arrow.names:
        frame = pd.read_parquet(
            shard_path,
            columns=["row_in_shard", "text"],
            filters=[("row_in_shard", "in", requested.tolist())],
        ).sort_values("row_in_shard")
        parsed = frame["row_in_shard"].to_numpy(dtype=np.int64)
        texts = frame["text"].tolist()
    else:
        total_rows = parquet_file.metadata.num_rows
        if requested[-1] >= total_rows:
            raise IndexError(
                f"Row {requested[-1]} out of range for {shard_path} ({total_rows} rows)"
            )

        parsed_parts: List[np.ndarray] = []
        text_parts: List[str] = []
        row_group_start = 0
        for row_group_idx in range(parquet_file.num_row_groups):
            row_group_rows = parquet_file.metadata.row_group(row_group_idx).num_rows
            row_group_end = row_group_start + row_group_rows
            mask = (requested >= row_group_start) & (requested < row_group_end)
            if np.any(mask):
                group_rows = requested[mask]
                local_rows = group_rows - row_group_start
                text_column = parquet_file.read_row_group(
                    row_group_idx, columns=["text"]
                ).column("text").combine_chunks()
                selected = pc.take(text_column, pa.array(local_rows, type=pa.int64()))
                parsed_parts.append(group_rows)
                text_parts.extend(selected.to_pylist())
            row_group_start = row_group_end

        parsed = np.concatenate(parsed_parts) if parsed_parts else np.empty(0, dtype=np.int64)
        texts = text_parts

    if not np.array_equal(parsed, requested):
        missing = np.setdiff1d(requested, parsed)
        raise RuntimeError(
            f"Failed to read {len(missing)} requested rows from {shard_path}; "
            f"examples: {missing[:10].tolist()}"
        )
    if any(not isinstance(text, str) for text in texts):
        raise ValueError(f"Null or non-string text found in selected rows of {shard_path}")
    return parsed, texts

_METADATA_COLUMNS = ["domain", *QUALITY_COLUMNS, CHAR_COUNT_COL]


def _read_shard_metadata_pyarrow(shard_path: str) -> dict:
    import pyarrow.parquet as pq
    basename = os.path.basename(shard_path)
    parsed_idx = _parse_shard_idx(shard_path)
    pf = pq.ParquetFile(shard_path)
    table = pf.read(columns=_METADATA_COLUMNS, use_threads=False)
    n = len(table)
    domain = table.column("domain").to_numpy(zero_copy_only=False).astype(np.int64)
    quality = np.column_stack([
        table.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
        for c in QUALITY_COLUMNS
    ])
    char_count = table.column(CHAR_COUNT_COL).to_numpy(zero_copy_only=False).astype(np.int64)
    return {
        "shard_idx": parsed_idx,
        "path": shard_path,
        "num_docs": n,
        "domain": domain,
        "quality": quality,
        "char_count": char_count,
    }


_CACHE_FILENAME = "metadata_cache.npz"


class ShardMetadataManager:
    """
    Manages metadata from a directory of preprocessed parquet shards.

    Lazy loading:
      - __init__ reads **only** metadata columns (domain, qs_*) from all shards
      - text is NEVER loaded upfront; call read_texts() to get text for specific docs

    Shard layout:
      Each parquet has: text, domain, shard_idx, row_in_shard, qs_*
    """

    def __init__(
        self,
        preprocessed_dir: str,
        index_file: Optional[str] = None,
        input_format: str = "preprocessed",
        batch_size: int = 65536,
        shard_limit: Optional[int] = None,
        max_workers: Optional[int] = None,
    ):
        self._dir = preprocessed_dir
        self.input_format = input_format

        if input_format not in {"preprocessed", "stem_raw"}:
            raise ValueError(
                f"Unsupported input_format={input_format!r}; "
                "expected 'preprocessed' or 'stem_raw'"
            )
        if input_format == "stem_raw":
            self._init_stem_raw(batch_size=batch_size, shard_limit=shard_limit)
            return

        self._shard_files: List[str] = sorted(
            glob.glob(os.path.join(preprocessed_dir, "*.parquet"))
        )
        if not self._shard_files:
            raise FileNotFoundError(
                f"No .parquet files found in {preprocessed_dir}"
            )

        self._shard_index: Optional[dict] = None
        if index_file is None:
            index_candidate = os.path.join(preprocessed_dir, "shard_index.json")
            if os.path.exists(index_candidate):
                index_file = index_candidate
        if index_file:
            with open(index_file, encoding="utf-8") as f:
                self._shard_index = json.load(f)

        if self._shard_index:
            expected_shards = self._shard_index.get("num_shards", 0)
            actual_shards = len(self._shard_files)
            if expected_shards != actual_shards:
                print(f"[ShardMetadataManager] WARNING: shard_index.json says {expected_shards} shards, "
                      f"but found {actual_shards} files. "
                      f"May need to re-run preprocessing.")

        total_shards = len(self._shard_files)
        load_t0 = time.time()

        cache_path = os.path.join(preprocessed_dir, _CACHE_FILENAME)
        shard_info_path = os.path.join(preprocessed_dir, "metadata_shard_info.json")

        current_shard_stats = {
            os.path.basename(f): {"size": os.path.getsize(f), "mtime": os.path.getmtime(f)}
            for f in self._shard_files
        }
        current_basenames = sorted(current_shard_stats.keys())

        cache_valid = False
        if os.path.exists(cache_path) and os.path.exists(shard_info_path):
            try:
                with open(shard_info_path) as f:
                    cached_info = json.load(f)
                cached_basenames = cached_info.get("shard_basenames", [])
                cached_stats = cached_info.get("shard_stats", {})
                if cached_basenames == current_basenames:
                    mismatches = []
                    for bn in current_basenames:
                        cs = cached_stats.get(bn, {})
                        cr = current_shard_stats[bn]
                        if cs.get("size") != cr["size"] or cs.get("mtime") != cr["mtime"]:
                            mismatches.append(bn)
                    if not mismatches:
                        cache_valid = True
                    else:
                        print(f"[ShardMetadataManager] Cache invalid: {len(mismatches)} shard(s) changed "
                              f"(e.g. {mismatches[:3]})")
                else:
                    added = [b for b in current_basenames if b not in cached_basenames]
                    removed = [b for b in cached_basenames if b not in current_basenames]
                    print(f"[ShardMetadataManager] Cache invalid: shard list changed "
                          f"(+{len(added)} new, -{len(removed)} removed)")
            except Exception as e:
                print(f"[ShardMetadataManager] Cache read error: {e}")

        if cache_valid:
            print(f"[ShardMetadataManager] Cache valid, loading from: {cache_path}")
            cached = np.load(cache_path, allow_pickle=False)
            self._domain_labels = cached["domain_labels"]
            self._quality_scores = cached["quality_scores"]
            self._doc_char_counts = cached["doc_char_counts"]
            self._num_docs = len(self._domain_labels)
            self._num_shards = total_shards

            self._per_shard_info = cached_info["per_shard_info"]
            self._shard_starts = np.array(
                [s["start_idx"] for s in self._per_shard_info], dtype=np.int64
            )

            total_time = time.time() - load_t0
            print(f"[ShardMetadataManager] Loaded {self._num_docs:,} docs "
                  f"({self._num_shards} shards) from cache in {total_time:.1f}s")
            print(f"[ShardMetadataManager] Quality scores: {self._quality_scores.shape}")
            return

        n_workers = max_workers if max_workers is not None else min(32, total_shards)
        print(f"[ShardMetadataManager] Discovered {total_shards} shards, "
              f"loading metadata with {n_workers} ProcessPoolExecutor workers")

        self._per_shard_info: List[Optional[dict]] = [None] * total_shards
        shard_data: List[Optional[dict]] = [None] * total_shards

        done = 0
        log_interval = max(1, total_shards // 20)

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            future_to_idx = {
                pool.submit(_read_shard_metadata_pyarrow, sf): i
                for i, sf in enumerate(self._shard_files)
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                result = future.result()
                shard_data[i] = result
                done += 1
                if done % log_interval == 0 or done == total_shards:
                    elapsed = time.time() - load_t0
                    pct = done / total_shards * 100
                    docs_so_far = sum(r["num_docs"] for r in shard_data if r is not None)
                    eta = elapsed / done * (total_shards - done)
                    print(f"[ShardMetadataManager] {done}/{total_shards} "
                          f"({pct:.0f}%) — {docs_so_far:,} docs, "
                          f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

        global_start = 0
        domain_list: List[np.ndarray] = []
        quality_list: List[np.ndarray] = []
        char_count_list: List[np.ndarray] = []

        for i, data in enumerate(shard_data):
            domain_list.append(data["domain"])
            quality_list.append(data["quality"])
            char_count_list.append(data["char_count"])
            self._per_shard_info[i] = {
                "shard_idx": data["shard_idx"],
                "path": data["path"],
                "num_docs": data["num_docs"],
                "start_idx": global_start,
                "end_idx": global_start + data["num_docs"],
            }
            global_start += data["num_docs"]

        self._domain_labels = np.concatenate(domain_list)
        self._quality_scores = np.concatenate(quality_list)
        self._doc_char_counts = np.concatenate(char_count_list)
        self._num_docs = global_start
        self._num_shards = total_shards

        self._shard_starts = np.array(
            [s["start_idx"] for s in self._per_shard_info], dtype=np.int64
        )

        total_time = time.time() - load_t0
        print(f"[ShardMetadataManager] Loaded {self._num_docs:,} docs "
              f"({self._num_shards} shards) in {total_time:.1f}s")
        print(f"[ShardMetadataManager] Quality scores: {self._quality_scores.shape}")

        try:
            np.savez(cache_path,
                     domain_labels=self._domain_labels,
                     quality_scores=self._quality_scores,
                     doc_char_counts=self._doc_char_counts)
            cache_meta = {
                "shard_basenames": current_basenames,
                "shard_stats": current_shard_stats,
                "per_shard_info": self._per_shard_info,
            }
            with open(shard_info_path, "w") as f:
                json.dump(cache_meta, f)
            cache_size = os.path.getsize(cache_path) / (1024 ** 3)
            print(f"[ShardMetadataManager] Saved metadata cache: {cache_path} "
                  f"({cache_size:.2f} GB)")
        except Exception as e:
            print(f"[ShardMetadataManager] Failed to save cache: {e}")

    def _init_stem_raw(self, batch_size: int, shard_limit: Optional[int]) -> None:
        """Load metadata directly from filtered STEM parquet shards.

        This is intentionally strict: rows are not filtered or renumbered at
        runtime because local row IDs must remain identical to physical source
        row offsets for on-demand text loading and token caches.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        candidates = [
            path for path in glob.glob(os.path.join(self._dir, "*.parquet"))
            if not os.path.basename(path).startswith("_")
        ]
        indexed = sorted(
            ((_parse_shard_idx(path), path) for path in candidates),
            key=lambda item: item[0],
        )
        if shard_limit is not None:
            if shard_limit < 1:
                raise ValueError("shard_limit must be positive")
            indexed = indexed[:shard_limit]
        if not indexed:
            raise FileNotFoundError(f"No parquet files found in {self._dir}")
        indices = [idx for idx, _ in indexed]
        if len(indices) != len(set(indices)):
            raise ValueError("Multiple source files resolve to the same shard index")

        self._shard_files = [path for _, path in indexed]
        self._shard_index = None
        domain_list: List[np.ndarray] = []
        quality_list: List[np.ndarray] = []
        char_count_list: List[np.ndarray] = []
        self._per_shard_info = []
        global_start = 0
        required = ["text", "category_name", "category_score", *QUALITY_COLUMNS]

        print(f"[ShardMetadataManager] Direct STEM mode: {len(indexed)} shards")
        for shard_idx, shard_path in indexed:
            parquet_file = pq.ParquetFile(shard_path)
            missing = [name for name in required if name not in parquet_file.schema_arrow.names]
            if missing:
                raise ValueError(f"{shard_path}: missing required columns {missing}")

            shard_domains: List[np.ndarray] = []
            shard_quality: List[np.ndarray] = []
            shard_chars: List[np.ndarray] = []
            source_offset = 0
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=required):
                frame = batch.to_pandas()
                n = len(frame)

                text_valid = frame["text"].map(lambda value: isinstance(value, str)).to_numpy()
                domains = frame["category_name"].map(STEM_DOMAIN_TO_ID)
                domain_valid = domains.notna().to_numpy()
                category_score = pd.to_numeric(frame["category_score"], errors="coerce")
                score_values = category_score.to_numpy(dtype=np.float64, na_value=np.nan)
                score_valid = np.isfinite(score_values) & (score_values > 0.3)

                quality = np.empty((n, len(QUALITY_COLUMNS)), dtype=np.float64)
                quality_valid = np.ones(n, dtype=bool)
                for column_idx, name in enumerate(QUALITY_COLUMNS):
                    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(
                        dtype=np.float64, na_value=np.nan
                    )
                    quality[:, column_idx] = values
                    quality_valid &= np.isfinite(values)

                valid = text_valid & domain_valid & score_valid & quality_valid
                if not np.all(valid):
                    bad = int(np.flatnonzero(~valid)[0])
                    reasons = []
                    if not text_valid[bad]:
                        reasons.append("text is null or non-string")
                    if not domain_valid[bad]:
                        reasons.append(f"invalid category_name={frame.iloc[bad]['category_name']!r}")
                    if not score_valid[bad]:
                        reasons.append("category_score is not finite or <= 0.3")
                    if not quality_valid[bad]:
                        bad_columns = [
                            name for idx, name in enumerate(QUALITY_COLUMNS)
                            if not np.isfinite(quality[bad, idx])
                        ]
                        reasons.append(f"invalid quality columns={bad_columns}")
                    raise ValueError(
                        f"{shard_path}: invalid record at physical row "
                        f"{source_offset + bad}: {'; '.join(reasons)}. "
                        "Direct mode does not filter rows; fix parquet_filter first."
                    )

                shard_domains.append(domains.to_numpy(dtype=np.int64))
                shard_quality.append(quality)
                shard_chars.append(frame["text"].map(len).to_numpy(dtype=np.int64))
                source_offset += n

            n_docs = parquet_file.metadata.num_rows
            domain_list.append(
                np.concatenate(shard_domains) if shard_domains else np.empty(0, dtype=np.int64)
            )
            quality_list.append(
                np.concatenate(shard_quality)
                if shard_quality else np.empty((0, len(QUALITY_COLUMNS)), dtype=np.float64)
            )
            char_count_list.append(
                np.concatenate(shard_chars) if shard_chars else np.empty(0, dtype=np.int64)
            )
            self._per_shard_info.append({
                "shard_idx": shard_idx,
                "path": shard_path,
                "num_docs": n_docs,
                "start_idx": global_start,
                "end_idx": global_start + n_docs,
                "text_access": "physical_row",
            })
            global_start += n_docs

        self._domain_labels = np.concatenate(domain_list)
        self._quality_scores = np.concatenate(quality_list)
        self._doc_char_counts = np.concatenate(char_count_list)
        self._num_docs = global_start
        self._num_shards = len(self._shard_files)
        self._shard_starts = np.array(
            [item["start_idx"] for item in self._per_shard_info], dtype=np.int64
        )
        print(f"[ShardMetadataManager] Loaded {self._num_docs:,} direct STEM docs")
        print(f"[ShardMetadataManager] Quality scores: {self._quality_scores.shape}")

    # ── Properties ──

    @property
    def domain_labels(self) -> npt.NDArray[np.int64]:
        return self._domain_labels

    @property
    def quality_scores(self) -> npt.NDArray[np.float64]:
        return self._quality_scores

    @property
    def doc_char_counts(self) -> npt.NDArray[np.int64]:
        return self._doc_char_counts

    @property
    def num_docs(self) -> int:
        return self._num_docs

    @property
    def num_shards(self) -> int:
        return self._num_shards

    @property
    def shard_info(self) -> List[dict]:
        return list(self._per_shard_info)

    # ── Shared memory factory (avoids re-reading parquet in worker processes) ──

    @classmethod
    def from_shared(
        cls,
        domain_labels: npt.NDArray[np.int64],
        quality_scores: npt.NDArray[np.float64],
        doc_char_counts: npt.NDArray[np.int64],
        per_shard_info: List[dict],
        shard_starts: npt.NDArray[np.int64],
        preprocessed_dir: str = "",
    ) -> "ShardMetadataManager":
        """Create from pre-loaded arrays (shared memory or otherwise)."""
        mgr = cls.__new__(cls)
        mgr._dir = preprocessed_dir
        mgr.input_format = (
            "stem_raw"
            if per_shard_info and per_shard_info[0].get("text_access") == "physical_row"
            else "preprocessed"
        )
        mgr._domain_labels = domain_labels
        mgr._quality_scores = quality_scores
        mgr._doc_char_counts = doc_char_counts
        mgr._per_shard_info = per_shard_info
        mgr._shard_starts = shard_starts
        mgr._num_docs = len(domain_labels)
        mgr._num_shards = len(per_shard_info)
        mgr._shard_files = []
        mgr._shard_index = None
        return mgr

    # ── Token estimation ──

    def get_total_chars(self) -> int:
        """Total character count across all documents."""
        return int(np.sum(self._doc_char_counts))

    def get_total_tokens_estimate(self, chars_per_token: float = 4.0) -> int:
        total_chars = self.get_total_chars()
        return int(total_chars / chars_per_token)

    # ── Index resolution ──

    def global_to_shard_rows(
        self, global_indices: npt.NDArray[np.int64]
    ) -> Dict[int, Tuple[str, npt.NDArray[np.int64]]]:
        shard_ids = np.searchsorted(
            self._shard_starts, global_indices, side="right"
        ) - 1
        shard_ids = np.clip(shard_ids, 0, self._num_shards - 1)

        order = np.argsort(shard_ids)
        sorted_shard_ids = shard_ids[order]
        sorted_global_idx = global_indices[order]

        result: Dict[int, Tuple[str, npt.NDArray[np.int64]]] = {}
        unique_ids, starts, counts = np.unique(
            sorted_shard_ids, return_index=True, return_counts=True
        )

        for sid, start, cnt in zip(unique_ids, starts, counts):
            group_global = sorted_global_idx[start:start + cnt]
            local_rows = group_global - self._shard_starts[sid]
            shard_path = self._per_shard_info[sid]["path"]
            result[int(sid)] = (shard_path, local_rows)

        return result

    # ── Text loading ──

    def read_texts(
        self, global_indices: npt.NDArray[np.int64]
    ) -> List[str]:
        import pandas as pd
        if len(global_indices) == 0:
            return []

        shard_groups = self.global_to_shard_rows(global_indices)
        pos_map: Dict[int, List[int]] = {}
        for p, idx in enumerate(global_indices):
            pos_map.setdefault(int(idx), []).append(p)

        result = [""] * len(global_indices)

        for sid, (shard_path, local_rows) in shard_groups.items():
            parsed_rows, texts = read_parquet_text_rows(shard_path, local_rows)
            chunk_map = dict(zip(parsed_rows, texts))
            for local_row in local_rows:
                text = chunk_map.get(local_row, "")
                global_idx = self._shard_starts[sid] + local_row
                for pos in pos_map.get(int(global_idx), []):
                    result[pos] = text

        return result

    # ── Token count estimation (lightweight, based on real char counts) ──

    def estimate_token_counts(
        self,
    ) -> npt.NDArray[np.int64]:
        return np.maximum(self._doc_char_counts // 4, 1).astype(np.int64)
