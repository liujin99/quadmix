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

import json, os, glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import numpy.typing as npt

QUALITY_COLUMNS = [
    "qs_dclm", "qs_fineweb_edu_approx", "qs_english",
    "qs_eai_general_math", "qs_eai_open_web_math",
]
CHAR_COUNT_COL = "doc_char_count"


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
    ):
        self._dir = preprocessed_dir

        # Discover shards (sorted by filename for deterministic ordering)
        self._shard_files: List[str] = sorted(
            glob.glob(os.path.join(preprocessed_dir, "preprocessed_*.parquet"))
        )
        if not self._shard_files:
            raise FileNotFoundError(
                f"No preprocessed_*.parquet files found in {preprocessed_dir}"
            )

        # Load index for validation
        self._shard_index: Optional[dict] = None
        if index_file is None:
            index_candidate = os.path.join(preprocessed_dir, "shard_index.json")
            if os.path.exists(index_candidate):
                index_file = index_candidate
        if index_file:
            with open(index_file) as f:
                self._shard_index = json.load(f)

        # Validate: check if shard_index matches discovered files
        if self._shard_index:
            expected_shards = self._shard_index.get("num_shards", 0)
            actual_shards = len(self._shard_files)
            if expected_shards != actual_shards:
                print(f"[ShardMetadataManager] WARNING: shard_index.json says {expected_shards} shards, "
                      f"but found {actual_shards} files. "
                      f"May need to re-run preprocessing.")

        print(f"[ShardMetadataManager] Discovered {len(self._shard_files)} shards")

        # ── Load metadata (domain + quality columns + char count, skip text) ──
        domain_list: List[np.ndarray] = []
        quality_list: List[np.ndarray] = []
        char_count_list: List[np.ndarray] = []
        self._per_shard_info: List[dict] = []

        global_start = 0
        for sf in self._shard_files:
            df_meta = pd.read_parquet(
                sf,
                columns=["domain", *QUALITY_COLUMNS, CHAR_COUNT_COL],
            )
            n = len(df_meta)
            domain_list.append(df_meta["domain"].to_numpy(dtype=np.int64))
            quality_list.append(df_meta[QUALITY_COLUMNS].to_numpy(dtype=np.float64))
            char_count_list.append(df_meta[CHAR_COUNT_COL].to_numpy(dtype=np.int64))

            # Parse shard_idx from filename
            basename = os.path.basename(sf)
            idx_str = basename.replace("preprocessed_", "").replace(".parquet", "")
            parsed_idx = int(idx_str)

            self._per_shard_info.append({
                "shard_idx": parsed_idx,
                "path": sf,
                "num_docs": n,
                "start_idx": global_start,
                "end_idx": global_start + n,
            })
            global_start += n

        self._domain_labels = np.concatenate(domain_list)
        self._quality_scores = np.concatenate(quality_list)
        self._doc_char_counts = np.concatenate(char_count_list)
        self._num_docs = global_start
        self._num_shards = len(self._shard_files)

        self._shard_starts = np.array(
            [s["start_idx"] for s in self._per_shard_info], dtype=np.int64
        )

        print(f"[ShardMetadataManager] Loaded {self._num_docs:,} docs "
              f"({self._num_shards} shards)")
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

    # ── Token estimation ──

    def get_total_chars(self) -> int:
        """Total character count across all documents."""
        return int(np.sum(self._doc_char_counts))

    def get_total_tokens_estimate(self, chars_per_token: float = 4.0) -> int:
        """
        Estimate total tokens from character count.
        
        For English text, typical ratio is ~4 chars per token (GPT-NeoX tokenizer).
        
        Args:
            chars_per_token: Ratio for estimation. Default 4.0 for English.
        
        Returns:
            Estimated total tokens.
        """
        total_chars = self.get_total_chars()
        return int(total_chars / chars_per_token)

    # ── Index resolution ──

    def global_to_shard_rows(
        self, global_indices: npt.NDArray[np.int64]
    ) -> Dict[int, Tuple[str, npt.NDArray[np.int64]]]:
        """
        Convert global document indices to per-shard lookup instructions.

        Returns:
            Dict[shard_idx, (shard_path, local_row_indices)]
        """
        shard_ids = np.searchsorted(
            self._shard_starts, global_indices, side="right"
        ) - 1
        shard_ids = np.clip(shard_ids, 0, self._num_shards - 1)

        # Sort by shard_id for grouping
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
        """
        Read text for selected global indices, preserving input order.

        Groups by shard, reads only needed rows from each shard via
        parquet row filters.
        """
        if len(global_indices) == 0:
            return []

        shard_groups = self.global_to_shard_rows(global_indices)
        # global_idx → position in result array
        pos_map = {int(idx): p for p, idx in enumerate(global_indices)}

        result = [""] * len(global_indices)

        for sid, (shard_path, local_rows) in shard_groups.items():
            df_chunk = pd.read_parquet(
                shard_path,
                columns=["row_in_shard", "text"],
                filters=[("row_in_shard", "in", local_rows.tolist())],
            )
            chunk_map = dict(zip(df_chunk["row_in_shard"], df_chunk["text"]))
            for local_row in local_rows:
                text = chunk_map.get(local_row, "")
                global_idx = self._shard_starts[sid] + local_row
                pos = pos_map.get(int(global_idx))
                if pos is not None:
                    result[pos] = text

        return result

    # ── Token count estimation (lightweight, based on real char counts) ──

    def estimate_token_counts(
        self,
    ) -> npt.NDArray[np.int64]:
        """Estimate token count per doc: char_count // 4 (same formula as single-file mode)."""
        return np.maximum(self._doc_char_counts // 4, 1).astype(np.int64)
