"""
ShardMetadataManager — loads metadata (domain + quality signals) from
preprocessed multi-shard parquet files into memory, supporting
on-demand text loading for selected documents.

Now accepts a DatasetSchema to support arbitrary parquet schemas
beyond the hardcoded Essential-Web defaults.

Usage:
    from quadmix.data.dataset_schema import DatasetSchema

    # Essential-Web (default, backward compatible)
    mgr = ShardMetadataManager(preprocessed_dir)

    # Custom dataset via YAML
    schema = DatasetSchema.from_yaml("schema_stem.yaml")
    mgr = ShardMetadataManager(preprocessed_dir, schema=schema)

    dom = mgr.domain_labels          # [N] int64 (0..M-1)
    qs  = mgr.quality_scores         # [N, num_quality_criteria] float64
    tx  = mgr.read_texts(indices)    # List[str] for selected docs

    mgr.num_domains                  # M — detected from data
    mgr.num_quality_criteria         # N — len(schema.quality_cols)
    mgr.detected_domain_names        # ["数学", "化学", "生物学", "物理"]
    mgr.detected_quality_names       # ["category_score", "stem_relevance", ...]
"""

import json, os, glob, time

import multiprocessing as mp

from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from quadmix.utils.concurrency import ConcurrencyConfig

import numpy as np
import numpy.typing as npt

from quadmix.data.dataset_schema import DatasetSchema
from quadmix.data.shard_io import (
    _parse_shard_idx,
    _read_shard_metadata_pyarrow,
    read_one_shard_texts,
    read_one_shard_texts_with_rows,
    assemble_texts_array,
)
from quadmix.utils.json_utils import sanitize_for_json


_CACHE_FILENAME = "metadata_cache.npz"


class ShardMetadataManager:
    """
    Manages metadata from a directory of preprocessed parquet shards.

    Lazy loading:
      - __init__ reads **only** metadata columns from all shards
      - text is NEVER loaded upfront; call read_texts() to get text for specific docs

    Schema-driven:
      - Accepts DatasetSchema for column mapping
      - Default schema matches Essential-Web (backward compatible)
      - String domain columns auto-mapped to int 0..M-1
      - char_count computed from text if column missing
    """

    def __init__(
        self,
        preprocessed_dir: str,
        schema: DatasetSchema,
        index_file: Optional[str] = None,
        max_workers: Optional[int] = None,
    ):
        self._dir = preprocessed_dir
        self._schema = schema
        self._max_workers = max_workers

        self._init_paths_and_index(index_file)
        self._validate_first_shard()

        if not self._try_load_cache():
            self._load_from_shards()
            self._write_cache()

    # ── Init helpers (split from god constructor) ──

    def _init_paths_and_index(self, index_file: Optional[str]) -> None:
        """Discover shard files and load shard_index.json."""
        self._shard_files: List[str] = sorted(
            glob.glob(os.path.join(self._dir, "*.parquet"))
        )
        if not self._shard_files:
            raise FileNotFoundError(
                f"No .parquet files found in {self._dir}"
            )

        self._shard_index: Optional[dict] = None
        if index_file is None:
            index_candidate = os.path.join(self._dir, "shard_index.json")
            if os.path.exists(index_candidate):
                index_file = index_candidate
        if index_file:
            with open(index_file) as f:
                self._shard_index = json.load(f)

        if self._shard_index:
            expected_shards = self._shard_index.get("num_shards", 0)
            actual_shards = len(self._shard_files)
            if expected_shards != actual_shards:
                print(f"[ShardMetadataManager] WARNING: shard_index.json says {expected_shards} shards, "
                      f"but found {actual_shards} files. "
                      f"May need to re-run preprocessing.")

    def _try_load_cache(self) -> bool:
        """Check cache validity and load if valid. Returns True if loaded."""
        total_shards = len(self._shard_files)
        cache_path = os.path.join(self._dir, _CACHE_FILENAME)
        shard_info_path = os.path.join(self._dir, "metadata_shard_info.json")

        self._cache_path = cache_path
        self._shard_info_path = shard_info_path

        current_shard_stats = {
            os.path.basename(f): {"size": os.path.getsize(f), "mtime": os.path.getmtime(f)}
            for f in self._shard_files
        }
        current_basenames = sorted(current_shard_stats.keys())
        self._cache_shard_stats = current_shard_stats
        self._cache_basenames = current_basenames

        cache_valid = False
        if os.path.exists(cache_path) and os.path.exists(shard_info_path):
            try:
                with open(shard_info_path) as f:
                    cached_info = json.load(f)
                cached_basenames = cached_info.get("shard_basenames", [])
                cached_stats = cached_info.get("shard_stats", {})
                cached_schema_key = cached_info.get("schema_key", None)
                current_schema_key = (
                    f"{self._schema.domain_col}"
                    f":{','.join(self._schema.quality_cols)}"
                    f":{self._schema.text_col}"
                    f":{self._schema.char_count_col}"
                    f":{self._schema.row_in_shard_col}"
                    f":domain_names={','.join(self._schema.domain_names or [])}"
                    f":quality_directions={','.join(str(d) for d in self._schema.quality_directions)}"
                )

                if cached_basenames == current_basenames and cached_schema_key == current_schema_key:
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
                    print(f"[ShardMetadataManager] Cache invalid: schema or shard list changed")
            except Exception as e:
                print(f"[ShardMetadataManager] Cache read error: {e}")

        if not cache_valid:
            return False

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

        if "row_in_shard_cols_concat" in cached:
            concat = cached["row_in_shard_cols_concat"]
            boundaries = cached["row_in_shard_boundaries"]
            self._row_in_shard_cols = {}
            for i in range(len(self._per_shard_info)):
                start = int(boundaries[i])
                end = int(boundaries[i + 1]) if i + 1 < len(boundaries) else len(concat)
                self._row_in_shard_cols[i] = concat[start:end]
        else:
            self._row_in_shard_cols = {}

        self._is_row_col_sequential = self._check_row_col_sequential()
        self._row_in_shard_col_sorted_cache = {}
        self._row_in_shard_reverse_map_cache: Dict[int, dict] = {}

        unique_domains = np.unique(self._domain_labels)
        if self._schema.domain_names is not None:
            self._num_domains = len(self._schema.domain_names)
        else:
            self._num_domains = len(unique_domains)
        self._num_quality_criteria = len(self._schema.quality_cols)
        self._domain_cat_map = cached_info.get("domain_label_map")
        self._detected_domain_names = self._build_domain_names(unique_domains)
        self._detected_quality_names = (
            self._schema.quality_names
            if self._schema.quality_names is not None
            else list(self._schema.quality_cols)
        )
        valid_labels = self._domain_labels[self._domain_labels >= 0]
        self._domain_counts = np.bincount(valid_labels, minlength=self._num_domains)

        print(f"[ShardMetadataManager] Loaded {self._num_docs:,} docs "
              f"({self._num_shards} shards) from cache")
        print(f"[ShardMetadataManager] Quality scores: {self._quality_scores.shape}")
        return True

    def _load_from_shards(self) -> None:
        """Load metadata from all shards in parallel."""
        total_shards = len(self._shard_files)
        load_t0 = time.time()

        cfg = ConcurrencyConfig()
        n_workers = self._max_workers if self._max_workers is not None else min(cfg.max_io_workers, total_shards)
        print(f"[ShardMetadataManager] Discovered {total_shards} shards, "
              f"loading metadata with {n_workers} workers "
              f"(schema: domain_col='{self._schema.domain_col}', "
              f"quality_cols={self._schema.quality_cols})")

        self._per_shard_info: List[Optional[dict]] = [None] * total_shards
        shard_data: List[Optional[dict]] = [None] * total_shards
        _computed_char_count = False

        done = 0
        log_interval = max(1, total_shards // 20)

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            future_to_idx = {
                pool.submit(_read_shard_metadata_pyarrow, sf, self._schema): i
                for i, sf in enumerate(self._shard_files)
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                result = future.result()
                shard_data[i] = result
                if result["computed_char_count"]:
                    _computed_char_count = True
                done += 1
                if done % log_interval == 0 or done == total_shards:
                    elapsed = time.time() - load_t0
                    pct = done / total_shards * 100
                    docs_so_far = sum(r["num_docs"] for r in shard_data if r is not None)
                    eta = elapsed / done * (total_shards - done)
                    print(f"[ShardMetadataManager] {done}/{total_shards} "
                          f"({pct:.0f}%) — {docs_so_far:,} docs, "
                          f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

        if _computed_char_count:
            total_docs = sum(r["num_docs"] for r in shard_data)
            print(f"[ShardMetadataManager] WARNING: 从 {self._schema.text_col} 列计算 "
                  f"doc_char_count ({total_docs:,} docs)。"
                  f"建议预处理时直接生成该列以加速后续加载。")

        global_start = 0
        domain_list: List[np.ndarray] = []
        quality_list: List[np.ndarray] = []
        char_count_list: List[np.ndarray] = []
        row_col_list: List[np.ndarray] = []
        row_col_boundaries: List[int] = [0]
        domain_cat_maps: List[dict] = []

        for i, data in enumerate(shard_data):
            domain_list.append(data["domain"])
            quality_list.append(data["quality"])
            char_count_list.append(data["char_count"])
            row_col_list.append(data["row_in_shard_col"])
            if data["domain_cat_map"] is not None:
                domain_cat_maps.append(data["domain_cat_map"])
            self._per_shard_info[i] = {
                "shard_idx": data["shard_idx"] if data["shard_idx"] is not None else i,
                "path": data["path"],
                "num_docs": data["num_docs"],
                "start_idx": global_start,
                "end_idx": global_start + data["num_docs"],
            }
            row_col_boundaries.append(row_col_boundaries[-1] + data["num_docs"])
            global_start += data["num_docs"]

        self._domain_labels = np.concatenate(domain_list)
        self._quality_scores = np.concatenate(quality_list)
        self._doc_char_counts = np.concatenate(char_count_list)
        self._num_docs = global_start
        self._num_shards = total_shards

        self._shard_starts = np.array(
            [s["start_idx"] for s in self._per_shard_info], dtype=np.int64
        )

        self._row_in_shard_cols: Dict[int, np.ndarray] = {}
        for i, data in enumerate(shard_data):
            self._row_in_shard_cols[i] = data["row_in_shard_col"]
        self._is_row_col_sequential = self._check_row_col_sequential()
        self._row_in_shard_col_sorted_cache: Dict[int, bool] = {}
        self._row_in_shard_reverse_map_cache: Dict[int, dict] = {}

        unique_domains = np.unique(self._domain_labels)
        if self._schema.domain_names is not None:
            self._num_domains = len(self._schema.domain_names)
        else:
            self._num_domains = len(unique_domains)
        self._num_quality_criteria = len(self._schema.quality_cols)

        self._domain_cat_map: Optional[Dict[str, int]] = None
        if domain_cat_maps:
            merged: Dict[str, int] = {}
            for cat_map in domain_cat_maps:
                for label, code in cat_map.items():
                    if label not in merged:
                        merged[label] = code
            self._domain_cat_map = merged

        self._detected_domain_names = self._build_domain_names(unique_domains)
        self._detected_quality_names = (
            self._schema.quality_names
            if self._schema.quality_names is not None
            else list(self._schema.quality_cols)
        )

        valid_domain_labels = self._domain_labels[self._domain_labels >= 0]
        self._domain_counts = np.bincount(valid_domain_labels, minlength=self._num_domains)
        for m in range(self._num_domains):
            if self._domain_counts[m] < 100:
                pct = self._domain_counts[m] / self._num_docs * 100
                name = self._detected_domain_names[m] if m < len(self._detected_domain_names) else f"D{m}"
                print(f"[ShardMetadataManager] WARNING: domain '{name}' 仅 "
                      f"{self._domain_counts[m]} 条数据 ({pct:.1f}%)，"
                      f"该域的 percentile rank 和采样参数可能不稳定")

        total_time = time.time() - load_t0
        print(f"[ShardMetadataManager] Loaded {self._num_docs:,} docs "
              f"({self._num_shards} shards, {self._num_domains} domains, "
              f"{self._num_quality_criteria} quality criteria) "
              f"in {total_time:.1f}s")
        print(f"[ShardMetadataManager] Domain labels: {self._domain_labels.shape}, "
              f"Quality scores: {self._quality_scores.shape}")

    def _write_cache(self) -> None:
        """Save metadata cache to disk."""
        try:
            row_col_list = [
                self._row_in_shard_cols[i] for i in range(len(self._per_shard_info))
            ]
            row_in_shard_boundaries = [0]
            for arr in row_col_list:
                row_in_shard_boundaries.append(row_in_shard_boundaries[-1] + len(arr))

            row_in_shard_cols_concat = np.concatenate(row_col_list)
            row_in_shard_boundaries = np.array(row_in_shard_boundaries, dtype=np.int64)
            tmp_npz = self._cache_path + ".tmp"
            np.savez(tmp_npz,
                     domain_labels=self._domain_labels,
                     quality_scores=self._quality_scores,
                     doc_char_counts=self._doc_char_counts,
                     row_in_shard_cols_concat=row_in_shard_cols_concat,
                     row_in_shard_boundaries=row_in_shard_boundaries)
            os.replace(tmp_npz, self._cache_path)
            schema_key = (
                f"{self._schema.domain_col}"
                f":{','.join(self._schema.quality_cols)}"
                f":{self._schema.text_col}"
                f":{self._schema.char_count_col}"
                f":{self._schema.row_in_shard_col}"
                f":domain_names={','.join(self._schema.domain_names or [])}"
                f":quality_directions={','.join(str(d) for d in self._schema.quality_directions)}"
            )
            cache_meta = {
                "shard_basenames": self._cache_basenames,
                "shard_stats": self._cache_shard_stats,
                "per_shard_info": self._per_shard_info,
                "domain_label_map": self._domain_cat_map,
                "schema_key": schema_key,
            }
            tmp_json = self._shard_info_path + ".tmp"
            with open(tmp_json, "w") as f:
                json.dump(sanitize_for_json(cache_meta), f)
            os.replace(tmp_json, self._shard_info_path)
            cache_size = os.path.getsize(self._cache_path) / (1024 ** 3)
            print(f"[ShardMetadataManager] Saved metadata cache: {self._cache_path} "
                  f"({cache_size:.2f} GB)")
        except Exception as e:
            print(f"[ShardMetadataManager] Failed to save cache: {e}")

    # ── Validation ──

    def _validate_first_shard(self) -> None:
        """Validate schema against first and last shard's columns and dtypes."""
        import pyarrow.parquet as pq
        check_indices = [0, len(self._shard_files) - 1]
        for idx in check_indices:
            path = self._shard_files[idx]
            pf = pq.ParquetFile(path)
            schema_arrow = pf.schema_arrow
            columns = [f.name for f in schema_arrow]
            dtypes = {f.name: str(f.type) for f in schema_arrow}
            self._schema._validate(columns, dtypes)

        if self._schema.needs_text_for_char_count() and self._schema.text_col not in columns:
            raise ValueError(
                f"无法计算文档字符数: char_count_col 未指定且 "
                f"text_col '{self._schema.text_col}' 不存在于 parquet 中。\n"
                f"请在 YAML 中指定 char_count_col 或 text_col。\n"
                f"可用列: {columns}"
            )

        self._has_row_in_shard = (
            self._schema.row_in_shard_col is not None
            and self._schema.row_in_shard_col in columns
        )

    def _build_domain_names(self, unique_domains: np.ndarray) -> List[str]:
        """Build domain name list from detected data."""
        if self._schema.domain_names is not None:
            return list(self._schema.domain_names)

        if self._domain_cat_map is not None:
            code_to_name = {code: name for name, code in self._domain_cat_map.items()}
            return [code_to_name.get(int(m), f"D{m}") for m in range(self._num_domains)]

        return [f"D{m}" for m in range(self._num_domains)]

    def _check_row_col_sequential(self) -> bool:
        """Check if row_in_shard_col values equal sequential 0-based positions for ALL shards."""
        if not self._row_in_shard_cols:
            return True
        for sid, arr in self._row_in_shard_cols.items():
            expected = np.arange(len(arr), dtype=np.int64)
            if not np.array_equal(arr, expected):
                return False
        return True

    # ── Row-in-shard ID conversion ──

    def _is_row_col_sorted(self, sid: int) -> bool:
        if sid not in self._row_in_shard_col_sorted_cache:
            if sid not in self._row_in_shard_cols:
                self._row_in_shard_col_sorted_cache[sid] = True
            else:
                arr = self._row_in_shard_cols[sid]
                self._row_in_shard_col_sorted_cache[sid] = bool(np.all(arr[:-1] <= arr[1:]))
        return self._row_in_shard_col_sorted_cache[sid]

    def local_to_row_col(self, sid: int, local_positions: np.ndarray) -> np.ndarray:
        if self._is_row_col_sequential or sid not in self._row_in_shard_cols:
            return local_positions
        return self._row_in_shard_cols[sid][local_positions]

    def row_col_to_local(self, sid: int, row_col_values: np.ndarray) -> np.ndarray:
        if self._is_row_col_sequential or sid not in self._row_in_shard_cols:
            return row_col_values
        col_arr = self._row_in_shard_cols[sid]
        if self._is_row_col_sorted(sid):
            positions = np.searchsorted(col_arr, row_col_values)
            positions = np.clip(positions, 0, len(col_arr) - 1)
            matched = col_arr[positions] == row_col_values
            if not matched.all():
                n_miss = int((~matched).sum())
                raise ValueError(
                    f"[ShardMetadataManager] row_col_to_local: shard {sid} has "
                    f"{n_miss} unmatched row_in_shard_col values"
                )
            return positions.astype(np.int64)
        if sid not in self._row_in_shard_reverse_map_cache:
            self._row_in_shard_reverse_map_cache[sid] = {
                int(v): i for i, v in enumerate(col_arr)
            }
        reverse_map = self._row_in_shard_reverse_map_cache[sid]
        return np.array([reverse_map[int(v)] for v in row_col_values], dtype=np.int64)

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

    @property
    def num_domains(self) -> int:
        return self._num_domains

    @property
    def num_quality_criteria(self) -> int:
        return self._num_quality_criteria

    @property
    def detected_domain_names(self) -> List[str]:
        return self._detected_domain_names

    @property
    def detected_quality_names(self) -> List[str]:
        return self._detected_quality_names

    @property
    def quality_directions(self) -> List[bool]:
        return list(self._schema.quality_directions)

    @property
    def domain_label_map(self) -> Optional[Dict[str, int]]:
        return self._domain_cat_map

    @property
    def schema(self) -> DatasetSchema:
        return self._schema

    # ── Shared memory factory ──

    @classmethod
    def from_shared(
        cls,
        domain_labels: npt.NDArray[np.int64],
        quality_scores: npt.NDArray[np.float64],
        doc_char_counts: npt.NDArray[np.int64],
        per_shard_info: List[dict],
        shard_starts: npt.NDArray[np.int64],
        preprocessed_dir: str = "",
        schema: Optional[DatasetSchema] = None,
        num_domains: Optional[int] = None,
        num_quality_criteria: Optional[int] = None,
        detected_domain_names: Optional[List[str]] = None,
        detected_quality_names: Optional[List[str]] = None,
        quality_directions: Optional[List[bool]] = None,
        domain_label_map: Optional[Dict[str, int]] = None,
    ) -> "ShardMetadataManager":
        """Create from pre-loaded arrays (shared memory or otherwise)."""
        mgr = cls.__new__(cls)
        mgr._dir = preprocessed_dir
        mgr._domain_labels = domain_labels
        mgr._quality_scores = quality_scores
        mgr._doc_char_counts = doc_char_counts
        mgr._per_shard_info = per_shard_info
        mgr._shard_starts = shard_starts
        mgr._num_docs = len(domain_labels)
        mgr._num_shards = len(per_shard_info)
        mgr._shard_files = []
        mgr._shard_index = None
        mgr._schema = schema
        mgr._num_domains = num_domains if num_domains is not None else (
            len(mgr._schema.domain_names) if mgr._schema is not None and mgr._schema.domain_names is not None
            else len(np.unique(domain_labels[domain_labels >= 0]))
        )
        mgr._num_quality_criteria = num_quality_criteria if num_quality_criteria is not None else (
            len(mgr._schema.quality_cols) if mgr._schema is not None
            else quality_scores.shape[1]
        )
        mgr._detected_domain_names = detected_domain_names if detected_domain_names is not None else (
            list(mgr._schema.domain_names) if mgr._schema is not None and mgr._schema.domain_names is not None
            else [f"D{m}" for m in range(mgr._num_domains)]
        )
        mgr._detected_quality_names = detected_quality_names if detected_quality_names is not None else (
            list(mgr._schema.quality_cols) if mgr._schema is not None
            else [f"q{n}" for n in range(mgr._num_quality_criteria)]
        )
        mgr._domain_cat_map = domain_label_map
        valid_labels = domain_labels[domain_labels >= 0]
        mgr._domain_counts = np.bincount(valid_labels, minlength=mgr._num_domains)
        mgr._has_row_in_shard = (schema is not None and schema.row_in_shard_col is not None)
        mgr._row_in_shard_cols = {}
        mgr._is_row_col_sequential = True
        mgr._row_in_shard_col_sorted_cache = {}
        mgr._row_in_shard_reverse_map_cache = {}
        return mgr

    # ── Token estimation ──

    def get_total_chars(self) -> int:
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
        self,
        global_indices: npt.NDArray[np.int64],
        verbose: bool = True,
    ) -> List[str]:
        if len(global_indices) == 0:
            return []

        shard_groups = self.global_to_shard_rows(global_indices)

        text_col = self._schema.text_col
        row_col = self._schema.row_in_shard_col if self._has_row_in_shard else None
        is_row_col_sequential = self._is_row_col_sequential

        cfg = ConcurrencyConfig()
        n_shards = len(shard_groups)
        n_workers = min(cfg.max_io_workers, n_shards)

        if n_shards <= 1:
            n_workers = 1

        if verbose:
            print(f"[read_texts] {len(global_indices):,} texts from {n_shards} shards, "
                  f"{n_workers} I/O processes (spawn)")

        t0 = time.time()

        shard_task_args: Dict[int, Tuple] = {}
        for sid, (shard_path, local_rows) in shard_groups.items():
            shard_total_rows = self._per_shard_info[sid]["num_docs"]
            if row_col is not None:
                rcv = self.local_to_row_col(sid, local_rows)
            else:
                rcv = None
            shard_task_args[sid] = (
                shard_path, text_col, row_col, rcv,
                local_rows, shard_total_rows, is_row_col_sequential,
            )

        shard_results: Dict[int, List[str]] = {}

        if n_workers <= 1:
            for sid in shard_groups:
                args = shard_task_args[sid]
                shard_results[sid] = read_one_shard_texts(*args)
        else:
            log_interval = max(1, n_shards // 20)
            done = 0
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                future_map = {
                    pool.submit(read_one_shard_texts, *shard_task_args[sid]): sid
                    for sid in shard_groups
                }
                for future in as_completed(future_map):
                    sid = future_map[future]
                    shard_results[sid] = future.result()
                    done += 1
                    if verbose and (done % log_interval == 0 or done == n_shards):
                        elapsed = time.time() - t0
                        pct = done / n_shards * 100
                        eta = elapsed / done * (n_shards - done)
                        print(f"[read_texts] {done}/{n_shards} shards "
                              f"({pct:.0f}%) — elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

        result = assemble_texts_array(
            global_indices, shard_groups, shard_results, self._shard_starts
        )

        elapsed = time.time() - t0
        if verbose:
            print(f"[read_texts] Done: {len(global_indices):,} texts in {elapsed:.1f}s "
                  f"({n_shards} shards, {n_workers} processes)")

        return result

    # ── Token count estimation ──

    def estimate_token_counts(
        self,
    ) -> npt.NDArray[np.int64]:
        return np.maximum(self._doc_char_counts // 4, 1).astype(np.int64)
