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

import json, os, glob, time, re
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import numpy.typing as npt

from quadmix.constants import QUALITY_COLUMNS
from quadmix.data.dataset_schema import DatasetSchema, _parse_quality_cols

_DEFAULT_SCHEMA = DatasetSchema()


def _parse_shard_idx(basename: str) -> Optional[int]:
    m = re.search(r'(\d+)', basename)
    if m:
        return int(m.group(1))
    return None


def _read_shard_metadata_pyarrow(shard_path: str, schema: DatasetSchema) -> dict:
    import pyarrow.parquet as pq
    basename = os.path.basename(shard_path)
    parsed_idx = _parse_shard_idx(basename)

    read_cols = schema.metadata_read_columns()
    pf = pq.ParquetFile(shard_path)
    table = pf.read(columns=read_cols, use_threads=False)
    n = len(table)

    domain_col_data = table.column(schema.domain_col).to_numpy(zero_copy_only=False)
    if hasattr(domain_col_data.dtype, 'categories') or domain_col_data.dtype == object:
        import pandas as pd
        series = pd.Series(domain_col_data)
        cat_series = series.astype("category")
        domain_arr = cat_series.cat.codes.to_numpy(dtype=np.int64)
        cat_map = dict(zip(
            cat_series.cat.categories,
            range(len(cat_series.cat.categories)),
        ))
    elif domain_col_data.dtype.kind in ('i', 'u'):
        domain_arr = domain_col_data.astype(np.int64)
        unique_vals = np.unique(domain_arr)
        if len(unique_vals) > 0 and (unique_vals.min() != 0 or
            unique_vals.max() != len(unique_vals) - 1 or
            not np.all(unique_vals == np.arange(len(unique_vals)))):
            remap = {int(v): i for i, v in enumerate(unique_vals)}
            domain_arr = np.array([remap[v] for v in domain_arr], dtype=np.int64)
            cat_map = {str(v): i for i, v in enumerate(unique_vals)}
        else:
            cat_map = None
    else:
        raise ValueError(
            f"domain_col '{schema.domain_col}' has unsupported dtype "
            f"'{domain_col_data.dtype}'. Expected string/object or integer."
        )

    quality_arr = np.column_stack([
        table.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
        for c in schema.quality_cols
    ])
    nan_count = np.isnan(quality_arr).sum()
    if nan_count > 0:
        pct = nan_count / quality_arr.size * 100
        print(f"[ShardMetadataManager] WARNING: quality scores have {nan_count} "
              f"NaN values ({pct:.1f}%), filling with 0.0. "
              f"建议在预处理时处理缺失值。")
        quality_arr = np.nan_to_num(quality_arr, nan=0.0)

    if schema.char_count_col is not None:
        char_count_arr = table.column(schema.char_count_col).to_numpy(
            zero_copy_only=False).astype(np.int64)
    elif schema.needs_text_for_char_count():
        import pandas as pd
        text_series = pd.Series(table.column(schema.text_col).to_numpy(zero_copy_only=False))
        char_count_arr = text_series.apply(
            lambda t: len(str(t)) if t is not None else 0
        ).to_numpy(dtype=np.int64)
    else:
        char_count_arr = np.zeros(n, dtype=np.int64)

    return {
        "shard_idx": parsed_idx,
        "path": shard_path,
        "num_docs": n,
        "domain": domain_arr,
        "quality": quality_arr,
        "char_count": char_count_arr,
        "domain_cat_map": cat_map,
        "computed_char_count": schema.needs_text_for_char_count(),
    }


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
        schema: Optional[DatasetSchema] = None,
        index_file: Optional[str] = None,
        max_workers: Optional[int] = None,
    ):
        self._dir = preprocessed_dir
        self._schema = schema if schema is not None else _DEFAULT_SCHEMA

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
            with open(index_file) as f:
                self._shard_index = json.load(f)

        if self._shard_index:
            expected_shards = self._shard_index.get("num_shards", 0)
            actual_shards = len(self._shard_files)
            if expected_shards != actual_shards:
                print(f"[ShardMetadataManager] WARNING: shard_index.json says {expected_shards} shards, "
                      f"but found {actual_shards} files. "
                      f"May need to re-run preprocessing.")

        self._validate_first_shard()

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
                cached_schema_key = cached_info.get("schema_key", None)
                current_schema_key = f"{self._schema.domain_col}:{','.join(self._schema.quality_cols)}:{self._schema.text_col}:{self._schema.char_count_col}"

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

            unique_domains = np.unique(self._domain_labels)
            self._num_domains = len(unique_domains)
            self._num_quality_criteria = len(self._schema.quality_cols)
            self._domain_cat_map = cached_info.get("domain_label_map")
            self._detected_domain_names = self._build_domain_names(unique_domains)
            self._detected_quality_names = (
                self._schema.quality_names
                if self._schema.quality_names is not None
                else list(self._schema.quality_cols)
            )
            self._domain_counts = np.bincount(self._domain_labels, minlength=self._num_domains)

            total_time = time.time() - load_t0
            print(f"[ShardMetadataManager] Loaded {self._num_docs:,} docs "
                  f"({self._num_shards} shards) from cache in {total_time:.1f}s")
            print(f"[ShardMetadataManager] Quality scores: {self._quality_scores.shape}")
            return

        n_workers = max_workers if max_workers is not None else min(32, total_shards)
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
        domain_cat_maps: List[dict] = []

        for i, data in enumerate(shard_data):
            domain_list.append(data["domain"])
            quality_list.append(data["quality"])
            char_count_list.append(data["char_count"])
            if data["domain_cat_map"] is not None:
                domain_cat_maps.append(data["domain_cat_map"])
            self._per_shard_info[i] = {
                "shard_idx": data["shard_idx"] if data["shard_idx"] is not None else i,
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

        unique_domains = np.unique(self._domain_labels)
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

        self._domain_counts = np.bincount(self._domain_labels, minlength=self._num_domains)
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

        try:
            np.savez(cache_path,
                     domain_labels=self._domain_labels,
                     quality_scores=self._quality_scores,
                     doc_char_counts=self._doc_char_counts)
            schema_key = f"{self._schema.domain_col}:{','.join(self._schema.quality_cols)}:{self._schema.text_col}:{self._schema.char_count_col}"
            cache_meta = {
                "shard_basenames": current_basenames,
                "shard_stats": current_shard_stats,
                "per_shard_info": self._per_shard_info,
                "domain_label_map": self._domain_cat_map,
                "schema_key": schema_key,
            }
            with open(shard_info_path, "w") as f:
                json.dump(cache_meta, f)
            cache_size = os.path.getsize(cache_path) / (1024 ** 3)
            print(f"[ShardMetadataManager] Saved metadata cache: {cache_path} "
                  f"({cache_size:.2f} GB)")
        except Exception as e:
            print(f"[ShardMetadataManager] Failed to save cache: {e}")

    def _validate_first_shard(self) -> None:
        """Validate schema against first shard's columns and dtypes."""
        import pyarrow.parquet as pq
        first_path = self._shard_files[0]
        pf = pq.ParquetFile(first_path)
        schema_arrow = pf.schema_arrow
        columns = [f.name for f in schema_arrow]
        dtypes = {}
        import pandas as pd
        sample_df = pd.read_parquet(first_path, columns=[])
        # Use pyarrow schema for dtypes
        for f in schema_arrow:
            dtypes[f.name] = str(f.type)

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
        mgr._schema = schema if schema is not None else _DEFAULT_SCHEMA
        mgr._num_domains = num_domains if num_domains is not None else len(np.unique(domain_labels))
        mgr._num_quality_criteria = num_quality_criteria if num_quality_criteria is not None else len(mgr._schema.quality_cols)
        mgr._detected_domain_names = detected_domain_names if detected_domain_names is not None else [f"D{m}" for m in range(mgr._num_domains)]
        mgr._detected_quality_names = detected_quality_names if detected_quality_names is not None else list(mgr._schema.quality_cols)
        mgr._domain_cat_map = domain_label_map
        mgr._domain_counts = np.bincount(domain_labels, minlength=mgr._num_domains)
        mgr._has_row_in_shard = False
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
        text_col = self._schema.text_col
        row_col = self._schema.row_in_shard_col if self._has_row_in_shard else None

        for sid, (shard_path, local_rows) in shard_groups.items():
            if row_col is not None:
                df_chunk = pd.read_parquet(
                    shard_path,
                    columns=[row_col, text_col],
                    filters=[(row_col, "in", local_rows.tolist())],
                )
                chunk_map = dict(zip(df_chunk[row_col], df_chunk[text_col]))
                for local_row in local_rows:
                    text = chunk_map.get(local_row, "")
                    global_idx = self._shard_starts[sid] + local_row
                    for pos in pos_map.get(int(global_idx), []):
                        result[pos] = text
            else:
                df_chunk = pd.read_parquet(shard_path, columns=[text_col])
                for local_row in local_rows:
                    if local_row < len(df_chunk):
                        text = df_chunk.iloc[local_row][text_col]
                        global_idx = self._shard_starts[sid] + local_row
                        for pos in pos_map.get(int(global_idx), []):
                            result[pos] = str(text) if text is not None else ""

        return result

    # ── Token count estimation ──

    def estimate_token_counts(
        self,
    ) -> npt.NDArray[np.int64]:
        return np.maximum(self._doc_char_counts // 4, 1).astype(np.int64)
