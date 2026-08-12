"""TokenCache — memory + disk cache for tokenized document shards.

Manages LRU in-memory cache and on-disk npz cache per shard.
Extracted from EssentialWebProxyRunner to isolate the caching concern.
"""

import os
import time
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class TokenCache:
    """Two-tier (memory + disk) cache for tokenized shard data.

    Memory tier: per-shard {rows, tokens} with LRU eviction by byte budget.
    Disk tier: per-shard .npz files with fcntl file locking.
    """

    def __init__(
        self,
        token_cache_dir: str,
        block_size: int,
        memory_cache_max_gb: float = 8.0,
    ):
        self.token_cache_dir = token_cache_dir
        self.block_size = block_size
        self.memory_cache_max_gb = memory_cache_max_gb

        self._memory_cache: Dict[int, dict] = {}
        self._memory_cache_bytes = 0
        self._memory_cache_lru: List[int] = []
        self._memory_cache_lock = threading.Lock()

    # ── Memory cache ──

    def get_rows(self, sid: int) -> set:
        """Return set of row_in_shard already in memory cache for this shard."""
        with self._memory_cache_lock:
            if sid not in self._memory_cache:
                return set()
            if sid in self._memory_cache_lru:
                self._memory_cache_lru.remove(sid)
                self._memory_cache_lru.append(sid)
            return set(int(r) for r in self._memory_cache[sid]["rows"])

    def add_rows(
        self,
        sid: int,
        new_rows: np.ndarray,
        new_tokens: np.ndarray,
        skip_eviction: bool = False,
    ):
        """Add new rows to memory cache. LRU eviction when over limit."""
        with self._memory_cache_lock:
            old_bytes = 0
            if sid in self._memory_cache:
                old_data = self._memory_cache[sid]
                old_bytes = old_data["rows"].nbytes + old_data["tokens"].nbytes

            if sid not in self._memory_cache:
                self._memory_cache[sid] = {
                    "rows": np.array([], dtype=np.int64),
                    "tokens": np.zeros((0, new_tokens.shape[1]), dtype=np.int32),
                }

            old = self._memory_cache[sid]
            old_rows = old["rows"]
            old_tokens = old["tokens"]

            combined_rows = np.concatenate([old_rows, new_rows])
            combined_tokens = np.concatenate([old_tokens, new_tokens])

            reversed_rows = combined_rows[::-1]
            _, inverse_rev = np.unique(reversed_rows, return_index=True)
            keep_indices = len(combined_rows) - 1 - inverse_rev
            keep_rows = combined_rows[keep_indices]
            keep_tokens = combined_tokens[keep_indices]

            sort_order = np.argsort(keep_rows)
            unique_rows = keep_rows[sort_order].astype(np.int64)
            final_tokens = keep_tokens[sort_order]

            new_bytes = unique_rows.nbytes + final_tokens.nbytes
            self._memory_cache[sid] = {"rows": unique_rows, "tokens": final_tokens}

            self._memory_cache_bytes += new_bytes - old_bytes
            if sid in self._memory_cache_lru:
                self._memory_cache_lru.remove(sid)
            self._memory_cache_lru.append(sid)

            if skip_eviction:
                return

            max_bytes = int(self.memory_cache_max_gb * 1024 ** 3)
            while self._memory_cache_bytes > max_bytes and self._memory_cache_lru:
                victim_sid = self._memory_cache_lru.pop(0)
                if victim_sid in self._memory_cache:
                    victim = self._memory_cache.pop(victim_sid)
                    self._memory_cache_bytes -= (victim["rows"].nbytes + victim["tokens"].nbytes)

    def query(self, sid: int, requested_rows: List[int]) -> Tuple[np.ndarray, List[int], List[int]]:
        """Query memory cache for requested rows."""
        with self._memory_cache_lock:
            if sid not in self._memory_cache:
                return np.zeros((0, self.block_size), dtype=np.int32), [], requested_rows
            cache_data = self._memory_cache[sid]
            cache_rows_set = set(int(r) for r in cache_data["rows"])
            hit_rows_set = [r for r in requested_rows if int(r) in cache_rows_set]
            miss_rows = [r for r in requested_rows if int(r) not in cache_rows_set]

            if not hit_rows_set:
                return np.zeros((0, self.block_size), dtype=np.int32), [], miss_rows

            cache_rows_arr = cache_data["rows"]
            cache_tokens = cache_data["tokens"]

            sorted_hit_rows = sorted(hit_rows_set)
            positions = np.searchsorted(cache_rows_arr, sorted_hit_rows)

            valid_mask = positions < len(cache_rows_arr)
            assert valid_mask.all(), f"Some hit rows not in cache: {sorted_hit_rows}"

            tokens = cache_tokens[positions].copy()
            return tokens, sorted_hit_rows, miss_rows

    def get_shard_tokens(self, sid: int, row_col_vals: np.ndarray) -> np.ndarray:
        """Look up tokens for row_col_vals from memory cache (no miss handling).

        Raises RuntimeError if shard or rows not in cache.
        """
        with self._memory_cache_lock:
            cache = self._memory_cache.get(sid)
            if cache is None:
                raise RuntimeError(
                    f"Shard {sid} not in memory cache. "
                    f"This should not happen after tokenize_all_needed."
                )
            cache_rows = cache["rows"]
            cache_tokens = cache["tokens"]
            positions = np.searchsorted(cache_rows, row_col_vals)
            positions = np.clip(positions, 0, len(cache_rows) - 1)
            matched = cache_rows[positions] == row_col_vals
            if not matched.all():
                n_missing = int((~matched).sum())
                raise RuntimeError(
                    f"Shard {sid}: {n_missing}/{len(row_col_vals)} "
                    f"documents not found in tokenized cache. "
                    f"Check for shard tokenization failures."
                )
            return cache_tokens[positions]

    def bulk_add(self, sid: int, parsed_rows: np.ndarray, miss_tokens: np.ndarray):
        """Bulk merge tokenized results into memory cache (no dedup, sorted input assumed)."""
        with self._memory_cache_lock:
            if sid in self._memory_cache:
                old = self._memory_cache[sid]
                combined_rows = np.concatenate([old["rows"], parsed_rows])
                combined_tokens = np.concatenate([old["tokens"], miss_tokens])
                sort_order = np.argsort(combined_rows)
                self._memory_cache[sid] = {
                    "rows": combined_rows[sort_order],
                    "tokens": combined_tokens[sort_order],
                }
                self._memory_cache_bytes += parsed_rows.nbytes + miss_tokens.nbytes
            else:
                self._memory_cache[sid] = {"rows": parsed_rows, "tokens": miss_tokens}
                self._memory_cache_bytes += parsed_rows.nbytes + miss_tokens.nbytes
            if sid in self._memory_cache_lru:
                self._memory_cache_lru.remove(sid)
            self._memory_cache_lru.append(sid)

    @property
    def cache_bytes(self) -> int:
        return self._memory_cache_bytes

    @property
    def cache_gb(self) -> float:
        return self._memory_cache_bytes / (1024 ** 3)

    # ── Disk cache ──

    def get_shard_token_path(self, shard_idx: int) -> str:
        """Path to disk cache for a shard's selected tokens (npz, mmap-compatible)."""
        return os.path.join(
            self.token_cache_dir,
            f"shard_{shard_idx:05d}_bs{self.block_size}.npz",
        )

    def cached_shard_rows(self, sid: int) -> set:
        """Return set of row_in_shard already cached on disk for this shard."""
        cache_path = self.get_shard_token_path(sid)
        if not os.path.exists(cache_path):
            return set()
        with np.load(cache_path) as data:
            rows = set(data['rows'].tolist())
        return rows

    def add_to_disk(self, sid: int, new_rows: np.ndarray, new_tokens: torch.Tensor):
        """Add new rows to shard disk cache (immediate write with file lock)."""
        import fcntl

        cache_path = self.get_shard_token_path(sid)
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)

        new_np = new_tokens.numpy().astype(np.int32)

        cache_no_ext = cache_path[:-4]
        temp_path = cache_no_ext + f".tmp.{int(time.time() * 1000000)}"
        actual_temp = temp_path + ".npz"

        lock_path = cache_path + ".lock"
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)

        with open(lock_path, 'w') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                if os.path.exists(cache_path):
                    with np.load(cache_path) as old:
                        old_rows = old['rows'].copy()
                        old_tokens = old['tokens'].copy()
                else:
                    old_rows = np.array([], dtype=np.int64)
                    old_tokens = np.zeros((0, new_np.shape[1]), dtype=np.int32)

                combined_rows = np.concatenate([old_rows, new_rows])
                combined_tokens = np.concatenate([old_tokens, new_np])

                row_to_idx = {int(r): i for i, r in enumerate(combined_rows)}
                unique_rows = np.array(sorted(row_to_idx.keys()), dtype=np.int64)
                final_tokens = combined_tokens[[row_to_idx[int(r)] for r in unique_rows]]

                np.savez(temp_path, tokens=final_tokens, rows=unique_rows)
                os.replace(actual_temp, cache_path)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                if os.path.exists(actual_temp):
                    try:
                        os.remove(actual_temp)
                    except OSError:
                        pass

    def get_exp_token_path(self, exp_id: int) -> str:
        """Path to temporary token file for a single experiment."""
        return os.path.join(
            self.token_cache_dir,
            f"exp_{exp_id:04d}_tokens.npy"
        )
