#!/usr/bin/env python3
"""
EssentialWebProxyRunner — Real proxy training on essential-web-v1 data.

Shard-aware mode (recommended):
  Uses ShardMetadataManager → loads only metadata (domain+quality) upfront,
  reads text on-demand per experiment. Per-shard disk cache for tokens.

Legacy mode (single-file):
  Uses data_path → loads all text upfront, tokenizes all.

Multi-NPU Parallelism:
  Dynamic task queue mode (run_batch_parallel):
    - Workers fetch tasks from shared queue, no batch boundaries
    - Fast workers naturally do more experiments
    - Tokenize thread runs ahead, independent of NPU training
    - Each worker binds to NPU device by worker_id

Aligned with RegMix:
  - GPT-NeoX-20B BPE tokenizer (same as GPT-NeoX)
  - On-demand tokenization with per-shard disk cache
  - Validation on openhermes-10k with assistant-only loss
  - RegMix training loop: gradient accumulation, cosine LR, AdamW
"""

import os, math, time, json, glob
import multiprocessing as mp
import multiprocessing.shared_memory  # for SharedMemory in ndarray_to_shared
from functools import partial
from typing import List, Optional, Dict, Tuple
from contextlib import contextmanager
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F

from quadmix.core.types import ParameterSet, ProxyResult, QuaDMixConfig
from quadmix.core.quality_merger import compute_merged_quality_scores
from quadmix.core.quality_rank import compute_quality_ranks
from quadmix.core.sampler import compute_sampling_values
from quadmix.pipeline.proxy_runner import BaseProxyRunner


class PerfTimer:
    """Lightweight performance timer with nesting support."""
    _timings: Dict[str, List[float]] = {}
    _stack: List[Tuple[str, float]] = []
    _enabled: bool = os.environ.get("QUADMIX_PERF_TIMER", "0") == "1"

    @classmethod
    def enable(cls, enabled: bool = True):
        cls._enabled = enabled

    @classmethod
    @contextmanager
    def section(cls, name: str, prefix: str = ""):
        """Context manager for timing a section."""
        if not cls._enabled:
            yield
            return

        full_name = f"{prefix}.{name}" if prefix else name
        start = time.perf_counter()
        cls._stack.append((full_name, start))
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if full_name not in cls._timings:
                cls._timings[full_name] = []
            cls._timings[full_name].append(elapsed)
            cls._stack.pop()

    @classmethod
    def report(cls, top_n: int = 20) -> str:
        """Generate performance report."""
        if not cls._timings:
            return "[PerfTimer] No timings recorded"

        lines = ["\n" + "=" * 70, "PERFORMANCE REPORT", "=" * 70]

        sorted_items = sorted(
            cls._timings.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )[:top_n]

        for name, times in sorted_items:
            total = sum(times)
            count = len(times)
            avg = total / count
            lines.append(f"{name:50s} | total: {total:7.2f}s | count: {count:4d} | avg: {avg:.3f}s")

        lines.append("=" * 70)
        return "\n".join(lines)

    @classmethod
    def reset(cls):
        """Reset all timings."""
        cls._timings.clear()
        cls._stack.clear()

# Default cache/temp directory
# Override via QUADMIX_TEMP_DIR env var, defaults to ~/.cache/QuaDMix/temp/
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPTS_DIR)
DEFAULT_TEMP_DIR = os.environ.get(
    "QUADMIX_TEMP_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "QuaDMix", "temp"),
)
DEFAULT_TOKEN_CACHE_DIR = os.path.join(DEFAULT_TEMP_DIR, "token_cache")


# ── Shared memory helpers (avoids re-reading 15GB+ metadata per worker) ──────

class SharedArrayInfo:
    """Descriptor for a numpy array in shared memory — pickle-safe for mp."""

    def __init__(self, name: str, shape: tuple, dtype: str, nbytes: int):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes


def ndarray_to_shared(arr: np.ndarray, prefix: str) -> SharedArrayInfo:
    """Copy numpy array into shared memory, return descriptor."""
    shm = mp.shared_memory.SharedMemory(create=True, size=arr.nbytes, name=f"{prefix}_shm")
    shared = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    np.copyto(shared, arr)
    return SharedArrayInfo(name=shm.name, shape=arr.shape, dtype=str(arr.dtype), nbytes=arr.nbytes)


def shared_to_ndarray(info: SharedArrayInfo) -> np.ndarray:
    """Map shared memory back to numpy array.

    IMPORTANT: Return a COPY to avoid segfault when numpy operations
    (slicing, normalization) access shared memory buffer in spawn children.
    """
    shm = mp.shared_memory.SharedMemory(name=info.name)
    arr = np.ndarray(shape=info.shape, dtype=np.dtype(info.dtype), buffer=shm.buf)
    # Copy data to local memory to avoid shared memory segfault
    return arr.copy()


class EssentialWebProxyRunner(BaseProxyRunner):
    """
    Proxy runner with:
      - GPT-NeoX-20B BPE tokenizer (matching RegMix)
      - Training data: essential-web-v1 via ShardMetadataManager (on-demand)
      - Validation: openhermes-10k (pre-tokenized with assistant loss mask)
      - Per-shard disk cache for tokenized training data
      - Loss: assistant-only for validation, full LM for training
    """

    def __init__(
            self,
            config: QuaDMixConfig,
            val_data_path: str,
            metadata_manager: Optional[object] = None,
            # Legacy mode: if metadata_manager is None, fall back to data_path
            data_path: Optional[str] = None,
            output_dir: str = "./proxy_validation",
            device_type: str = "cpu",
            npu_device_id: int = 0,  # ← 新增
            # RegMix training params
            model_variant: str = "tinyllama_1M",
            global_batch_size: int = 64,
            micro_batch_size: int = 8,
            max_step: int = 25000,
            warmup_fraction: float = 0.04,  # warmup as fraction of actual steps (default 4% = RegMix default)
            learning_rate: float = 4e-4,
            weight_decay: float = 0.1,
            grad_clip: float = 1.0,
            # "just runnable" flags
            tiny_steps: int = 10,
            doc_limit: Optional[int] = None,
            test_block_size: Optional[int] = None,
            rank_ref_size: int = 10000,
            token_cache_dir: str = DEFAULT_TOKEN_CACHE_DIR,
            memory_cache_max_gb: float = 16.0,  # LRU eviction threshold
            checkpoint_interval: int = 1000,  # Record val_loss every N steps
    ):
        self.config = config
        self.metadata_manager = metadata_manager
        self.legacy_data_path = data_path  # only used when no metadata_manager
        self.val_data_path = val_data_path
        self.output_dir = output_dir
        self.model_variant = model_variant
        self.device_type = device_type
        self.npu_device_id = npu_device_id  # ← 新增：指定 NPU 卡號
        self.memory_cache_max_gb = memory_cache_max_gb
        self.tiny_steps = tiny_steps
        self.doc_limit = doc_limit
        self.rank_ref_size = rank_ref_size
        self.token_cache_dir = token_cache_dir
        self.checkpoint_interval = checkpoint_interval

        # RegMix training config
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.max_step = max_step
        self.warmup_fraction = warmup_fraction
        self.warmup_steps = 0  # computed at runtime from actual steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        # Model config
        from quadmix.core.proxy_model import ProxyConfig
        self.model_config = ProxyConfig.from_name(
            model_variant, block_size=test_block_size
        )
        self.block_size = self.model_config.block_size
        self.batch_size = global_batch_size // 1
        self.gradient_accumulation_steps = max(1, self.batch_size // micro_batch_size)

        # Tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        assert self.tokenizer.vocab_size <= self.model_config.vocab_size, \
            f"Tokenizer vocab ({self.tokenizer.vocab_size}) > model vocab ({self.model_config.vocab_size})"
        print(f"[ProxyRunner] GPT-NeoX tokenizer: vocab={self.tokenizer.vocab_size}")
        print(f"[ProxyRunner] Model config: {model_variant}, "
              f"block={self.block_size}, model_vocab={self.model_config.vocab_size}")
        print(f"[ProxyRunner] Training: batch={self.batch_size}, "
              f"grad_acc={self.gradient_accumulation_steps}")

        # Load modes
        if metadata_manager is not None:
            self._mode = "sharded"
            self._load_metadata_only()
        elif data_path is not None:
            self._mode = "legacy"
            self._legacy_load_and_tokenize()
        else:
            raise ValueError("Either metadata_manager or data_path must be provided")

        # ---- Validation data (same for both modes) ----
        print(f"[ProxyRunner] Loading validation set: {self.val_data_path}")
        val_data = torch.load(self.val_data_path, map_location="cpu", weights_only=False)
        self._val_token_ids = val_data["token_ids"]
        self._val_loss_mask = val_data["loss_mask"]
        print(f"[ProxyRunner] Val tokens: {self._val_token_ids.shape}, "
              f"assistant tokens: {self._val_loss_mask.sum().item()}/"
              f"{self._val_loss_mask.numel()}")

    # ═══════════════════════════════════════════════════════════
    # Mode: sharded (metadata only + on-demand text loading)
    # ═══════════════════════════════════════════════════════════

    def _load_metadata_only(self):
        """Load domain labels + quality scores from metadata manager (no text).

        Pre-compute normalized quality scores to avoid repeated normalization
        in Eq.1 (major performance bottleneck).
        """
        t0 = time.time()
        mgr = self.metadata_manager
        self._domain_labels = mgr.domain_labels
        self._quality_scores = mgr.quality_scores
        self._token_counts = mgr.estimate_token_counts()
        self._num_docs = mgr.num_docs
        self._train_idx = np.arange(self._num_docs)

        print(f"[ProxyRunner] Sharded mode: {self._num_docs:,} docs "
              f"(metadata only, {mgr.num_shards} shards) ({time.time() - t0:.0f}s)")

        # ── Pre-compute normalized quality (Eq.1 optimization) ───────────────
        # Each experiment repeats normalize_fn(quality_scores[:, n]) - expensive!
        # Pre-compute once, reuse in all experiments (only weighted sum needed)
        from quadmix.utils.normalization import get_normalizer
        normalize_fn = get_normalizer("rank")

        t1 = time.time()
        num_criteria = self._quality_scores.shape[1]
        self._normalized_quality = np.zeros_like(self._quality_scores)
        for n in range(num_criteria):
            self._normalized_quality[:, n] = normalize_fn(self._quality_scores[:, n])
        print(f"[ProxyRunner] Pre-normalized {num_criteria} quality criteria "
              f"({time.time() - t1:.1f}s) — Eq.1 now ~5x faster per experiment")

        # ── Pre-compute domain indices (optimization: avoid mask creation per experiment) ─
        t2 = time.time()
        unique_domains = np.unique(self._domain_labels)
        self._domain_indices: Dict[int, np.ndarray] = {}
        for m in unique_domains:
            self._domain_indices[int(m)] = np.where(self._domain_labels == m)[0]
        print(f"[ProxyRunner] Pre-computed domain indices for {len(self._domain_indices)} domains "
              f"({time.time() - t2:.1f}s) — Eq.1 mask elimination")

        # Per-shard token cache: dict[int, dict[int, torch.Tensor]]
        # cache[shard_idx] = {row_idx: token_ids_tensor}
        # OR saved to disk: cache_dir/tokens_shard_{sid:05d}_bs{bs}.npz
        os.makedirs(self.token_cache_dir, exist_ok=True)
        self._cache_hits = 0
        self._cache_misses = 0

        # ── Memory Cache (NPU Parallel Mode Optimization) ───────────────────
        # Dict[sid -> {rows: np.ndarray, tokens: np.ndarray}]
        # Used during batch tokenize to avoid repeated parquet reads
        # LRU eviction when total exceeds memory_cache_max_gb (default 16 GB)
        self._memory_cache: Dict[int, dict] = {}  # sid -> {"rows": ndarray, "tokens": ndarray}
        self._memory_cache_bytes: int = 0  # total bytes across all cached shards
        self._memory_cache_lru: List[int] = []  # sids in access order (head = LRU)

    def _tokenize_texts(self, texts: List[str]) -> torch.Tensor:
        """Tokenize a list of texts into [M, block_size] int64 tensor.

        Single-threaded batch mode — called from within worker processes
        in _tokenize_shard_parallel, so don't spawn subprocesses here.
        """
        B = 500
        all_ids = []
        for i in range(0, len(texts), B):
            batch = texts[i:i + B]
            enc = self.tokenizer(
                batch, max_length=self.block_size,
                truncation=True, padding="max_length",
                return_tensors="pt",
            )
            all_ids.append(enc["input_ids"])
        return torch.cat(all_ids, dim=0)

    # ═══════════════════════════════════════════════════════════
    # Memory Cache (NPU Parallel Mode Optimization)
    # ═══════════════════════════════════════════════════════════

    def _memory_cache_get_rows(self, sid: int) -> set:
        """Return set of row_in_shard already in memory cache for this shard."""
        if sid not in self._memory_cache:
            return set()
        # Mark as recently used (move to tail of LRU)
        if sid in self._memory_cache_lru:
            self._memory_cache_lru.remove(sid)
            self._memory_cache_lru.append(sid)
        return set(int(r) for r in self._memory_cache[sid]["rows"])

    def _memory_cache_add_rows(self, sid: int, new_rows: np.ndarray, new_tokens: np.ndarray):
        """Add new rows to memory cache. LRU eviction when over limit.

        Args:
            sid: Shard index
            new_rows: Array of row_in_shard indices
            new_tokens: Array of tokens [M, block_size]
        """
        # Track old byte count if sid already exists
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

        # Merge: new rows appended, dedupe by row_in_shard (new overwrites old)
        combined_rows = np.concatenate([old_rows, new_rows])
        combined_tokens = np.concatenate([old_tokens, new_tokens])

        # Deduplicate: keep latest (new overwrites old)
        row_to_idx = {}
        for i, r in enumerate(combined_rows):
            row_to_idx[int(r)] = i  # Later entry overwrites earlier

        unique_rows = np.array(sorted(row_to_idx.keys()), dtype=np.int64)
        final_tokens = combined_tokens[[row_to_idx[int(r)] for r in unique_rows]]

        new_bytes = unique_rows.nbytes + final_tokens.nbytes
        self._memory_cache[sid] = {"rows": unique_rows, "tokens": final_tokens}

        # Update byte tracking and LRU
        self._memory_cache_bytes += new_bytes - old_bytes
        if sid in self._memory_cache_lru:
            self._memory_cache_lru.remove(sid)
        self._memory_cache_lru.append(sid)  # Most recently used

        # Evict LRU entries if over limit
        max_bytes = int(self.memory_cache_max_gb * 1024 ** 3)
        while self._memory_cache_bytes > max_bytes and self._memory_cache_lru:
            victim_sid = self._memory_cache_lru.pop(0)  # Least recently used
            if victim_sid in self._memory_cache:
                victim = self._memory_cache.pop(victim_sid)
                self._memory_cache_bytes -= (victim["rows"].nbytes + victim["tokens"].nbytes)

    def _memory_cache_query(self, sid: int, requested_rows: List[int]) -> Tuple[np.ndarray, List[int], List[int]]:
        """Query memory cache for requested rows.

        Args:
            sid: Shard index
            requested_rows: List of row_in_shard to query

        Returns:
            (tokens, sorted_hit_rows, miss_rows)
            - tokens: Array of tokens for hit rows, sorted by sorted_hit_rows
            - sorted_hit_rows: List of hit rows in SORTED order (matches tokens order)
            - miss_rows: List of rows not in cache (in requested_rows order)
        """
        cached_rows = self._memory_cache_get_rows(sid)
        hit_rows_set = [r for r in requested_rows if int(r) in cached_rows]
        miss_rows = [r for r in requested_rows if int(r) not in cached_rows]

        if not hit_rows_set:
            return np.zeros((0, self.block_size), dtype=np.int32), [], miss_rows

        # Use np.searchsorted for efficient batch query
        cache_data = self._memory_cache[sid]
        cache_rows = cache_data["rows"]
        cache_tokens = cache_data["tokens"]

        # Sort hit_rows to match cache_rows order (cache_rows is sorted)
        sorted_hit_rows = sorted(hit_rows_set)
        positions = np.searchsorted(cache_rows, sorted_hit_rows)

        # Verify all positions are valid (searchsorted returns len if not found)
        valid_mask = positions < len(cache_rows)
        assert valid_mask.all(), f"Some hit rows not in cache: {sorted_hit_rows}"

        tokens = cache_tokens[positions]

        # Return sorted_hit_rows so caller knows tokens order
        return tokens, sorted_hit_rows, miss_rows

    # ═══════════════════════════════════════════════════════════
    # Incremental token cache (append new rows to existing npz)
    # ═══════════════════════════════════════════════════════════

    def _get_shard_token_path(self, shard_idx: int) -> str:
        """Path to disk cache for a shard's selected tokens (npz, mmap-compatible)."""
        return os.path.join(
            self.token_cache_dir,
            f"shard_{shard_idx:05d}_bs{self.block_size}.npz",
        )

    def _cached_shard_rows(self, sid: int) -> set:
        """Return set of row_in_shard already cached for this shard."""
        cache_path = self._get_shard_token_path(sid)
        if not os.path.exists(cache_path):
            return set()
        data = np.load(cache_path)
        rows = set(data['rows'].tolist())
        return rows

    def _cache_add_rows(self, sid: int, new_rows: np.ndarray, new_tokens: torch.Tensor):
        """Add new rows to shard cache (immediate write with file lock).

        Uses fcntl.flock for atomic merge. np.savez writes to temp file then
        os.replace for atomic replacement — no partial reads possible.
        """
        import fcntl

        cache_path = self._get_shard_token_path(sid)
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)

        new_np = new_tokens.numpy().astype(np.int32)  # [M, block_size]

        # Unique temp file per call (np.savez auto-appends .npz)
        cache_no_ext = cache_path[:-4]
        temp_path = cache_no_ext + f".tmp.{int(time.time() * 1000000)}"
        actual_temp = temp_path + ".npz"

        lock_path = cache_path + ".lock"
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)

        with open(lock_path, 'w') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                # Read existing cache
                if os.path.exists(cache_path):
                    old = np.load(cache_path)
                    old_rows = old['rows']
                    old_tokens = old['tokens']
                    del old
                else:
                    old_rows = np.array([], dtype=np.int64)
                    old_tokens = np.zeros((0, new_np.shape[1]), dtype=np.int32)

                # Merge + deduplicate (new overwrites old on conflict)
                combined_rows = np.concatenate([old_rows, new_rows])
                combined_tokens = np.concatenate([old_tokens, new_np])

                # Keep latest row for each key
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

    # ═══════════════════════════════════════════════════════════
    # Batch-level tokenization (CPU, called from main process)
    # ═══════════════════════════════════════════════════════════

    def _get_exp_token_path(self, exp_id: int) -> str:
        """Path to temporary token file for a single experiment."""
        return os.path.join(
            self.token_cache_dir,
            f"exp_{exp_id:04d}_tokens.pt"
        )

    def _tokenize_batch_union(
            self,
            batch_selected: List[np.ndarray],
            batch_exp_ids: List[int],
            async_write_queue: Optional["Queue"] = None,
    ) -> Dict[int, str]:
        """NPU Parallel Mode: Batch tokenize union miss rows across all experiments.

        Key optimization:
          1. Compute union of all rows needed by this batch
          2. For each shard, read parquet ONCE (not 8 times)
          3. Tokenize all miss rows in batch
          4. Store in memory_cache for subsequent batches
          5. Pack each exp's tokens to temp file
          6. AsyncWrite thread writes to disk cache in background

        Args:
            batch_selected: List of selected_indices for each exp
            batch_exp_ids: List of experiment IDs
            async_write_queue: Queue for AsyncWrite thread (optional)

        Returns:
            Dict[exp_id -> exp_token_path] for each experiment
        """
        mgr = self.metadata_manager
        t0 = time.time()

        # ── Step 1: Collect all shard rows needed by this batch ───────────
        # shard_to_exp_rows[sid] = {exp_id: [row_in_shard, ...]}
        shard_to_exp_rows: Dict[int, Dict[int, List[int]]] = {}
        exp_to_shard_groups: Dict[int, Dict] = {}  # Cache for Step 4

        for exp_id, selected_idx in zip(batch_exp_ids, batch_selected):
            shard_groups = mgr.global_to_shard_rows(selected_idx)
            exp_to_shard_groups[exp_id] = shard_groups  # Cache for reuse
            for sid, (shard_path, local_rows) in shard_groups.items():
                if sid not in shard_to_exp_rows:
                    shard_to_exp_rows[sid] = {}
                shard_to_exp_rows[sid][exp_id] = list(int(r) for r in local_rows)

        n_shards = len(shard_to_exp_rows)
        print(f"[BatchTokenize] {len(batch_exp_ids)} exps, {n_shards} shards")

        # ── Step 2: For each shard, compute union miss rows ───────────────
        # Check memory_cache first, then disk cache
        total_miss_rows = 0
        shard_miss_info: Dict[int, Tuple[str, np.ndarray]] = {}  # sid -> (shard_path, miss_rows)

        for sid, exp_rows_dict in shard_to_exp_rows.items():
            # Union of all rows needed by this batch for this shard
            all_needed_rows = set()
            for exp_id, rows in exp_rows_dict.items():
                all_needed_rows.update(rows)

            # Check memory_cache
            memory_cached = self._memory_cache_get_rows(sid)
            hit_in_memory = [r for r in all_needed_rows if r in memory_cached]

            # Check disk cache for remaining rows
            remaining_rows = [r for r in all_needed_rows if r not in memory_cached]
            disk_cached = self._cached_shard_rows(sid)
            hit_in_disk = [r for r in remaining_rows if r in disk_cached]

            # True miss: not in memory_cache or disk_cache
            miss_rows = [r for r in remaining_rows if r not in disk_cached]

            if miss_rows:
                shard_path = mgr._per_shard_info[sid]["path"]  # Get shard path
                miss_rows_arr = np.array(sorted(miss_rows), dtype=np.int64)
                shard_miss_info[sid] = (shard_path, miss_rows_arr)
                total_miss_rows += len(miss_rows)

            # Load disk cache hits into memory_cache (for subsequent batches)
            if hit_in_disk:
                cache_path = self._get_shard_token_path(sid)
                data = np.load(cache_path)
                disk_rows = data['rows']
                disk_tokens = data['tokens']

                # Find positions for hit rows
                row_to_pos = {int(r): i for i, r in enumerate(disk_rows)}
                positions = np.array([row_to_pos[int(r)] for r in hit_in_disk], dtype=np.int64)
                hit_tokens = disk_tokens[positions]

                # Add to memory_cache
                self._memory_cache_add_rows(sid, np.array(hit_in_disk, dtype=np.int64), hit_tokens)

        if total_miss_rows == 0:
            print(f"[BatchTokenize] All {n_shards} shards fully cached, 0 miss rows")
        else:
            print(f"[BatchTokenize] {total_miss_rows:,} miss rows across {len(shard_miss_info)} shards")

        # ── Step 3: Batch read parquet + tokenize (ALL shards in parallel) ──
        if shard_miss_info:
            read_t0 = time.time()

            # Build shard tasks list
            shard_tasks = [
                (sid, shard_path, miss_rows_arr.tolist())
                for sid, (shard_path, miss_rows_arr) in shard_miss_info.items()
            ]

            # Parallel tokenize across ALL shards
            parallel_results = _tokenize_shard_parallel(
                shard_tasks, self.tokenizer.name_or_path, self.block_size
            )

            # Process results
            for sid, parsed_rows, miss_tokens, io_time, tokenize_time, total_time in parallel_results:
                # Add to memory_cache
                self._memory_cache_add_rows(sid, parsed_rows, miss_tokens)

                # Schedule async write to disk cache (if queue provided)
                if async_write_queue is not None:
                    async_write_queue.put((sid, parsed_rows, miss_tokens))
                else:
                    # Fallback: immediate write (blocking)
                    self._cache_add_rows(sid, parsed_rows, torch.from_numpy(miss_tokens))

                print(
                    f"  [Shard {sid}] {len(parsed_rows):,} docs (IO {io_time:.1f}s, tok {tokenize_time:.1f}s, total {total_time:.1f}s)")

            read_time = time.time() - read_t0
            print(f"[BatchTokenize] Parquet read + tokenize (parallel): {read_time:.1f}s")

        # ── Step 4: Pack each exp's tokens from memory_cache ───────────────
        exp_token_paths: Dict[int, str] = {}
        pack_t0 = time.time()

        for exp_id, selected_idx in zip(batch_exp_ids, batch_selected):
            shard_groups = exp_to_shard_groups[exp_id]  # Reuse cached result
            all_tokens = []

            for sid, (shard_path, local_rows) in shard_groups.items():
                # Query memory_cache for all needed rows (should be 100% hit now)
                requested_rows = [int(r) for r in local_rows]
                tokens, sorted_hit_rows, miss_rows = self._memory_cache_query(sid, requested_rows)

                if miss_rows:
                    print(f"  WARNING: exp {exp_id} shard {sid} has {len(miss_rows)} miss rows after batch tokenize!")
                    # Fallback: read from disk cache (should not happen in normal flow)
                    cache_path = self._get_shard_token_path(sid)
                    if os.path.exists(cache_path):
                        data = np.load(cache_path)
                        disk_rows = data['rows']
                        disk_tokens = data['tokens']
                        row_to_pos = {int(r): i for i, r in enumerate(disk_rows)}
                        for r in miss_rows:
                            if r in row_to_pos:
                                pass  # TODO: handle this edge case

                # Reorder tokens to match requested_rows order
                # tokens[i] corresponds to sorted_hit_rows[i]
                # Create mapping: row -> position in tokens array
                row_to_token_pos = {int(r): i for i, r in enumerate(sorted_hit_rows)}
                positions = np.array([row_to_token_pos[int(r)] for r in requested_rows], dtype=np.int64)
                shard_tokens = torch.from_numpy(tokens[positions].astype(np.int64))
                all_tokens.append(shard_tokens)

            # Pack into temporary file
            result = torch.cat(all_tokens, dim=0)
            exp_token_path = self._get_exp_token_path(exp_id)
            torch.save(result, exp_token_path)
            exp_token_paths[exp_id] = exp_token_path

        pack_time = time.time() - pack_t0
        print(f"[BatchTokenize] Pack {len(batch_exp_ids)} exps: {pack_time:.1f}s")

        elapsed = time.time() - t0
        print(f"[BatchTokenize] Total: {elapsed:.1f}s for {len(batch_exp_ids)} experiments")

        return exp_token_paths

    def _load_tokens_for_experiment(
            self, selected_idx: np.ndarray, exp_id: int = None
    ) -> torch.Tensor:
        """Load tokens for selected document indices.

        In sharded mode (parallel):
          1. If exp_id provided and temporary file exists → load from temp file
          2. Otherwise fallback to shard cache logic

        In legacy mode:
          Directly index into self._token_ids (already loaded)
        """
        if self._mode == "legacy":
            return self._token_ids[selected_idx]

        # Parallel mode: check for temporary file
        if exp_id is not None:
            exp_token_path = self._get_exp_token_path(exp_id)
            if os.path.exists(exp_token_path):
                result = torch.load(exp_token_path, map_location="cpu", weights_only=True)
                print(f"  [TokenLoad] exp {exp_id:04d}: {len(result):,} docs from temp file")
                return result

        # Fallback: load from shard cache (used in sequential mode)
        t0 = time.time()
        mgr = self.metadata_manager
        shard_groups = mgr.global_to_shard_rows(selected_idx)

        all_tokens = []
        for sid, (shard_path, local_rows) in shard_groups.items():
            cache_path = self._get_shard_token_path(sid)

            if os.path.exists(cache_path):
                data = np.load(cache_path)
                token_data = data['tokens']
                row_index = data['rows']
                row_to_pos = {int(r): i for i, r in enumerate(row_index)}

                hit_rows = [r for r in local_rows if int(r) in row_to_pos]
                miss_rows = [r for r in local_rows if int(r) not in row_to_pos]

                hit_tokens = None
                miss_tokens = None

                if hit_rows:
                    positions = np.array(
                        [row_to_pos[int(r)] for r in hit_rows], dtype=np.int64
                    )
                    hit_tokens = torch.from_numpy(
                        token_data[positions].astype(np.int64)
                    )
                    self._cache_hits += len(hit_rows)

                del data, token_data, row_index

                if miss_rows:
                    self._cache_misses += len(miss_rows)
                    miss_rows_arr = np.array(miss_rows, dtype=np.int64)

                    df_shard = pd.read_parquet(
                        shard_path,
                        columns=["row_in_shard", "text"],
                        filters=[("row_in_shard", "in", miss_rows_arr.tolist())],
                    )
                    df_shard = df_shard.sort_values("row_in_shard")
                    selected_texts = df_shard["text"].astype(str).tolist()
                    parsed_rows = df_shard["row_in_shard"].to_numpy(dtype=np.int64)

                    print(f"    [Partial miss] shard {sid}: "
                          f"tokenizing {len(miss_rows):,} docs "
                          f"(hit {len(hit_rows):,} from cache)...")

                    miss_tokens_full = self._tokenize_texts(selected_texts)
                    self._cache_add_rows(sid, parsed_rows, miss_tokens_full)

                    row_to_pos_new = {int(r): i for i, r in enumerate(parsed_rows)}
                    positions = np.array(
                        [row_to_pos_new[int(r)] for r in miss_rows], dtype=np.int64
                    )
                    miss_tokens = torch.from_numpy(
                        miss_tokens_full.numpy().astype(np.int64)[positions]
                    )

                if miss_rows:
                    row_to_token = {}
                    if hit_tokens is not None:
                        for i, r in enumerate(hit_rows):
                            row_to_token[int(r)] = hit_tokens[i]
                    if miss_tokens is not None:
                        for i, r in enumerate(miss_rows):
                            row_to_token[int(r)] = miss_tokens[i]
                    shard_tokens = torch.stack([
                        row_to_token[int(r)] for r in local_rows
                    ])
                else:
                    shard_tokens = hit_tokens
            else:
                self._cache_misses += len(local_rows)
                df_shard = pd.read_parquet(
                    shard_path,
                    columns=["row_in_shard", "text"],
                    filters=[("row_in_shard", "in", local_rows.tolist())],
                )
                df_shard = df_shard.sort_values("row_in_shard")
                selected_texts = df_shard["text"].astype(str).tolist()
                parsed_rows = df_shard["row_in_shard"].to_numpy(dtype=np.int64)

                print(f"    [Cache miss] shard {sid}: "
                      f"tokenizing {len(selected_texts):,} docs...")

                tokenized = self._tokenize_texts(selected_texts)
                self._cache_add_rows(sid, parsed_rows, tokenized)

                row_to_pos = {int(r): i for i, r in enumerate(parsed_rows)}
                positions = np.array(
                    [row_to_pos[int(r)] for r in local_rows], dtype=np.int64
                )
                shard_tokens = torch.from_numpy(
                    tokenized.numpy().astype(np.int64)[positions]
                )

            all_tokens.append(shard_tokens)

        result = torch.cat(all_tokens, dim=0)
        elapsed = time.time() - t0
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(1, total) * 100
        print(f"  [TokenLoad] {len(result)} docs loaded, "
              f"cache: {self._cache_hits}/{total} hits ({hit_rate:.0f}%) "
              f"({elapsed:.1f}s)")
        return result

    # ═══════════════════════════════════════════════════════════
    # Mode: legacy (single-file, all data loaded upfront)
    # ═══════════════════════════════════════════════════════════

    def _legacy_load_and_tokenize(self):
        """Legacy: load all text from single parquet, tokenize all upfront."""
        import pandas as pd
        t0 = time.time()
        print(f"[ProxyRunner] (legacy) Loading training data: {self.legacy_data_path}")
        df = pd.read_parquet(self.legacy_data_path)
        texts = df["text"].astype(str).tolist()

        if self.doc_limit and self.doc_limit < len(texts):
            texts = texts[:self.doc_limit]

        self._domain_labels = df["domain"].to_numpy(dtype=np.int64)[:len(texts)]
        self._quality_scores = df[[
            "qs_dclm", "qs_fineweb_edu_approx", "qs_english",
            "qs_eai_general_math", "qs_eai_open_web_math",
        ]].to_numpy(dtype=np.float64)[:len(texts)]
        self._num_docs = len(texts)
        print(f"[ProxyRunner] (legacy) {self._num_docs:,} docs ({time.time() - t0:.0f}s)")

        # Tokenize with caching
        dl = self.doc_limit if self.doc_limit else "all"
        cache = os.path.join(
            os.path.dirname(self.token_cache_dir),
            f"legacy_neox_{self.block_size}_dl{dl}.pt",
        )
        if os.path.exists(cache):
            print(f"[ProxyRunner] (legacy) Loading cached tokens: {cache}")
            cached = torch.load(cache, map_location="cpu", weights_only=False)
            self._token_ids = cached["token_ids"]
            self._token_counts = cached["token_counts"].numpy()
            print(f"[ProxyRunner] (legacy) Cached: {self._token_ids.shape}")
        else:
            print(f"[ProxyRunner] (legacy) Tokenizing {self._num_docs:,} docs...")
            self._token_ids = self._tokenize_texts(texts)
            token_counts = (self._token_ids != self.tokenizer.pad_token_id).sum(dim=1).numpy()
            torch.save({
                "token_ids": self._token_ids,
                "token_counts": torch.from_numpy(token_counts),
            }, cache)
            print(f"[ProxyRunner] (legacy) Tokenized: {self._token_ids.shape} cached")

        self._train_idx = np.arange(self._num_docs)

        # ── Pre-compute normalized quality (Eq.1 optimization) ───────────────
        from quadmix.utils.normalization import get_normalizer
        normalize_fn = get_normalizer("rank")

        t1 = time.time()
        num_criteria = self._quality_scores.shape[1]
        self._normalized_quality = np.zeros_like(self._quality_scores)
        for n in range(num_criteria):
            self._normalized_quality[:, n] = normalize_fn(self._quality_scores[:, n])
        print(f"[ProxyRunner] (legacy) Pre-normalized {num_criteria} criteria "
              f"({time.time() - t1:.1f}s) — Eq.1 now ~5x faster")

        # ── Pre-compute domain indices ───────────────────────────
        t2 = time.time()
        unique_domains = np.unique(self._domain_labels)
        self._domain_indices: Dict[int, np.ndarray] = {}
        for m in unique_domains:
            self._domain_indices[int(m)] = np.where(self._domain_labels == m)[0]
        print(f"[ProxyRunner] (legacy) Pre-computed domain indices for {len(self._domain_indices)} domains "
              f"({time.time() - t2:.1f}s)")

    # ═══════════════════════════════════════════════════════════
    # Experiment execution
    # ═══════════════════════════════════════════════════════════

    def _compute_ranks_for_params(
            self, params: ParameterSet, experiment_id: int,
    ) -> np.ndarray:
        """Per-experiment Eq.1 + Eq.2 (optimized with pre-normalized quality + pre-computed domain indices).

        Optimizations:
          - Pre-normalized quality scores (no re-normalization per experiment)
          - Pre-computed _domain_indices (no 275M bool mask creation × 3000 exps)
        """
        M = self.config.num_domains
        rng = np.random.default_rng(experiment_id + 1729)

        # Eq.1 (optimized): Weighted sum using pre-normalized quality + pre-computed indices
        # OLD: compute_merged_quality_scores() normalized every time (~20s)
        # MID: boolean mask per domain (~4s, still creates 275M mask × M)
        # NEW: direct indexing via pre-computed _domain_indices (~1s)
        merged_scores = np.zeros(self._num_docs, dtype=np.float64)

        for m in range(M):
            indices = self._domain_indices.get(m)
            if indices is None or len(indices) == 0:
                continue
            alpha_m = params.merge_config.get_final_weights(m)
            merged_scores[indices] = self._normalized_quality[indices] @ alpha_m

        # Eq.2: Subset-based rank estimation per domain (token-weighted)
        # Paper: "calculate the size of the set by adding up the number of
        # tokens for all samples within the set"
        ranks = np.zeros(self._num_docs, dtype=np.float64)
        has_tokens = hasattr(self, '_token_counts') and self._token_counts is not None
        for m in range(M):
            indices = self._domain_indices.get(m)
            if indices is None:
                continue
            n_domain = len(indices)
            if n_domain == 0:
                continue
            domain_scores = merged_scores[indices]

            k = min(self.rank_ref_size, n_domain)
            ref_idx = rng.choice(n_domain, k, replace=False)
            ref_scores_unsorted = domain_scores[ref_idx]
            sort_order = np.argsort(ref_scores_unsorted)
            ref_scores = ref_scores_unsorted[sort_order]

            positions = np.searchsorted(ref_scores, domain_scores, side='right')

            if has_tokens:
                ref_tokens = self._token_counts[indices[ref_idx]][sort_order].astype(np.float64)
                cum_tokens = np.concatenate(([0.0], np.cumsum(ref_tokens)))
                total_ref_tokens = cum_tokens[-1]
                if total_ref_tokens > 0:
                    ranks[indices] = cum_tokens[positions] / total_ref_tokens
                else:
                    ranks[indices] = positions.astype(np.float64) / k
            else:
                ranks[indices] = positions.astype(np.float64) / k

        return ranks

    def run_experiment(
            self,
            params: ParameterSet,
            experiment_id: int = 0,
            selected_idx: Optional[np.ndarray] = None,
            checkpoint_interval: Optional[int] = None,
    ) -> ProxyResult:
        """Train one proxy model. Validates on openhermes-10k.
        """
        from quadmix.core.proxy_model import ProxyModel
        from quadmix.npu.device import DeviceManager, DeviceType

        # Use instance-level checkpoint_interval if not explicitly provided
        if checkpoint_interval is None:
            checkpoint_interval = self.checkpoint_interval

        os.makedirs(self.output_dir, exist_ok=True)
        exp_dir = os.path.join(self.output_dir, f"exp_{experiment_id:04d}")
        os.makedirs(exp_dir, exist_ok=True)

        device_mgr = DeviceManager(
            device_type=DeviceType(self.device_type),
            npu_device_id=self.npu_device_id,
        )
        device = device_mgr.get_device()

        # ---- 0. Compute quality ranks (Eq.1+Eq.2) + sample ----
        selected_idx: Optional[np.ndarray] = None
        if selected_idx is None:
            # Original flow: Eq.1-3 + Bernoulli inside run_experiment
            quality_ranks = self._compute_ranks_for_params(params, experiment_id)
            sv = compute_sampling_values(quality_ranks, self._domain_labels, params)
            train_sv = sv[self._train_idx]

            # Proper fractional sampling (copies _select_documents_vectorized logic)
            int_part = np.floor(train_sv).astype(np.int64)
            frac_part = train_sv - int_part
            rng = np.random.default_rng(experiment_id + 42)
            random_mask = rng.uniform(size=len(train_sv)) < frac_part
            repeats = int_part + random_mask.astype(np.int64)
            selected_idx = self._train_idx[
                np.repeat(np.arange(len(self._train_idx)), repeats)
            ]
            if len(selected_idx) < 10:
                rng2 = np.random.default_rng(experiment_id + 42)
                selected_idx = rng2.choice(self._train_idx, 100, replace=False)
        else:
            # Pre-computed mode: verify correct dtype
            selected_idx = np.asarray(selected_idx, dtype=np.int64)

        print(f"  [Exp {experiment_id:04d}] QuaDMix sampled {len(selected_idx)} docs "
              f"(from {len(self._train_idx):,})")

        _timer_prefix = f"exp{experiment_id:04d}"

        # ---- 1. Load / tokenize training data on demand ----
        with PerfTimer.section("load_tokens", _timer_prefix):
            train_tokens = self._load_tokens_for_experiment(selected_idx, exp_id=experiment_id)
        # Keep on CPU to avoid HBM pressure — move to device per-batch

        # ---- 2. Create model ----
        with PerfTimer.section("create_model", _timer_prefix):
            model = ProxyModel(config=self.model_config).to(device)

        # Compile model for kernel fusion (CPU/CUDA only, not NPU)
        if self.device_type != "npu" and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='max-autotune')
                print(f"  [Exp {experiment_id:04d}] Model compiled with torch.compile")
            except Exception as e:
                print(f"  [Exp {experiment_id:04d}] torch.compile failed: {e}, using eager mode")
        elif self.device_type == "npu":
            print(f"  [Exp {experiment_id:04d}] Skipping torch.compile (NPU not supported)")

        non_emb = model.count_params(non_embedding_only=True)
        print(f"  [Exp {experiment_id:04d}] Model on {device}: "
              f"{model.count_params():,} total, {non_emb:,} non-emb")

        # ---- 3. Optimizer (fused only on CUDA) ----
        use_fused = self.device_type == "cuda"
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate,
            betas=(0.9, 0.95), weight_decay=self.weight_decay, fused=use_fused,
        )

        # ---- 4. Training data preparation (RegMix-style: permutation shuffle) ----
        with PerfTimer.section("data_prep", _timer_prefix):
            pad_id = self.tokenizer.pad_token_id
            eos_id = self.tokenizer.eos_token_id
            real_mask = train_tokens != pad_id
            non_empty = real_mask.any(dim=1)
            real_tokens_list = []
            eos_buf = torch.tensor([eos_id], dtype=train_tokens.dtype)
            for doc in train_tokens[non_empty]:
                real_tokens_list.append(doc[doc != pad_id])
                real_tokens_list.append(eos_buf)
            flat_train = torch.cat(real_tokens_list)  # Keep on CPU to save NPU memory
            del train_tokens, real_tokens_list, real_mask, non_empty, eos_buf
        num_steps = self.tiny_steps if self.tiny_steps > 0 else self.max_step
        grad_acc = self.gradient_accumulation_steps
        max_iters = num_steps * grad_acc
        tok_per_step = self.micro_batch_size * self.block_size

        # Compute warmup_steps from fraction of actual training steps
        warmup_steps = max(1, int(num_steps * self.warmup_fraction))
        self.warmup_steps = warmup_steps  # for _lr_schedule

        # Compute total blocks available (each block = block_size + 1 tokens)
        total_blocks = max(1, flat_train.size(0) - self.block_size)

        # Build a permutation of all block indices (like RegMix PackedDataset)
        # Use numpy on CPU to avoid huge GPU tensor for randperm
        epoch_rng = np.random.default_rng(experiment_id + 42)

        def get_epoch_permutation():
            return epoch_rng.permutation(total_blocks)

        perm = get_epoch_permutation()
        epoch_pos = 0  # current position in permutation
        epoch = 0

        # Pre-allocate: NPU buffers for batched transfer
        # Transfer grad_acc micro-batches at once per optimizer step to reduce host->device latency
        accum_bs = self.micro_batch_size * grad_acc  # e.g. 8*8=64
        batch_buf = torch.empty(accum_bs, self.block_size + 1, dtype=torch.long, device=device)
        block_starts_buf = torch.empty(accum_bs, dtype=torch.long, device=device)  # NPU staging
        arange_buf = torch.arange(self.block_size + 1, dtype=torch.long, device=device)  # Pre-compute on device
        arange_cpu = torch.arange(self.block_size + 1, dtype=torch.long)  # CPU version for indexing

        # Clear NPU memory fragmentation before training loop
        if device.type == "npu":
            torch.npu.empty_cache()

        # ---- 5. Training loop (RegMix-style: permutation shuffle, no replacement) ----
        _train_t0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        self._ckpt_results = {}  # step -> val_loss
        iter_ct = 0
        step_ct = 0
        log_int = max(1, num_steps // 5)
        t_start = time.time()

        print(f"  [Exp {experiment_id:04d}] Training {num_steps} steps "
              f"(grad_acc={grad_acc}, warmup={warmup_steps} steps ({self.warmup_fraction * 100:.0f}%), "
              f"micro_batch={self.micro_batch_size}, "
              f"blocks={total_blocks}, epochs~{math.ceil(max_iters / total_blocks)})"
              f"{', checkpoint every ' + str(checkpoint_interval) + ' steps' if checkpoint_interval > 0 else ''})...")

        while iter_ct < max_iters:
            micro_in_step = iter_ct % grad_acc
            mb_start = micro_in_step * self.micro_batch_size
            mb_end = mb_start + self.micro_batch_size

            # Batch transfer at step boundary: collect all grad_acc micro-batches
            if micro_in_step == 0:
                remaining = total_blocks - epoch_pos
                if remaining >= accum_bs:
                    block_starts_cpu = perm[epoch_pos:epoch_pos + accum_bs].copy()
                    epoch_pos += accum_bs
                else:
                    block_starts_cpu = np.empty(accum_bs, dtype=np.int64)
                    block_starts_cpu[:remaining] = perm[epoch_pos:epoch_pos + remaining]
                    filled = remaining
                    while filled < accum_bs:
                        perm = get_epoch_permutation()
                        epoch_pos = 0
                        epoch += 1
                        chunk = min(accum_bs - filled, total_blocks)
                        block_starts_cpu[filled:filled + chunk] = perm[:chunk]
                        filled += chunk
                        epoch_pos = chunk

                # Index on CPU, then transfer to device
                block_starts_cpu_tensor = torch.from_numpy(block_starts_cpu)
                idx_cpu = block_starts_cpu_tensor.unsqueeze(1) + arange_cpu.unsqueeze(0)
                batch_cpu = flat_train[idx_cpu]
                batch_buf.copy_(batch_cpu.to(device))

            # Extract this micro-batch slice on device
            batch = batch_buf[mb_start:mb_end]
            inp = batch[:, :self.block_size].contiguous()
            tgt = batch[:, 1:self.block_size + 1].contiguous()

            logits = model(inp)
            loss = F.cross_entropy(
                logits.view(-1, self.model_config.vocab_size), tgt.view(-1)
            )

            is_acc = (iter_ct + 1) % grad_acc != 0
            (loss / grad_acc).backward()

            if not is_acc:
                lr = self._lr_schedule(iter_ct, max_iters)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                step_ct += 1

                # Checkpoint: record val_loss every checkpoint_interval steps
                # Skip if this is the final step — validation will run after training loop
                if checkpoint_interval > 0 and step_ct % checkpoint_interval == 0 and step_ct < num_steps:
                    if device.type == "npu":
                        torch.npu.empty_cache()
                    elif device.type == "cuda":
                        torch.cuda.empty_cache()
                    ckpt_val = self._run_validation(model, device)
                    self._ckpt_results[step_ct] = ckpt_val
                    elapsed_ckpt = time.time() - t_start
                    print(f"    [CHECKPOINT step={step_ct}] val_loss={ckpt_val:.4f} ({elapsed_ckpt:.0f}s)")

            total_loss += loss.item()
            iter_ct += 1

            if not is_acc and (step_ct % log_int == 0 or step_ct == 1):
                avg = total_loss / step_ct
                elapsed = time.time() - t_start
                rem = (num_steps - step_ct) * elapsed / max(1, step_ct)
                print(f"    Step {step_ct}/{num_steps}, loss={avg:.4f}, "
                      f"lr={lr:.2e}, {tok_per_step * step_ct / elapsed:.0f} tok/s, "
                      f"ETA: {rem:.0f}s")

        _train_elapsed = time.perf_counter() - _train_t0
        PerfTimer._timings.setdefault(f"{_timer_prefix}.training_loop", []).append(_train_elapsed)

        # ---- 6. Free training resources before validation ----
        with PerfTimer.section("free_resources", _timer_prefix):
            del flat_train, batch_buf, block_starts_buf, arange_buf, optimizer, perm
            if device.type == "npu":
                import gc as _gc
                _gc.collect()
                torch.npu.empty_cache()
            elif device.type == "cuda":
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()

        # ---- 7. Validation: assistant-only loss (full validation set) ----
        with PerfTimer.section("validation", _timer_prefix):
            val_loss = self._run_validation(model, device)

        # ---- 8. Save metadata ----
        with PerfTimer.section("save_metadata", _timer_prefix):
            avg_train = total_loss / step_ct if step_ct > 0 else 0

            meta = {
                "experiment_id": experiment_id,
                "variant": self.model_variant,
                "train_loss": avg_train,
                "val_loss": val_loss,
                "val_ppl": float(np.exp(val_loss)),
                "num_steps": step_ct,
                "sampled_docs": len(selected_idx),
                "val_docs": len(self._val_token_ids),
                "assistant_loss": True,
                "params_lambda": [sc.lambda_ for sc in params.sampling_configs],
                "params_omega": [sc.omega for sc in params.sampling_configs],
                "params_eta": [sc.eta for sc in params.sampling_configs],
                "params_epsilon": [sc.epsilon for sc in params.sampling_configs],
                "global_weights": params.merge_config.global_weights.tolist(),
                "domain_weights": params.merge_config.domain_weights.tolist(),
                "checkpoint_steps": dict(self._ckpt_results) if hasattr(self, '_ckpt_results') else {},
            }
            np.save(os.path.join(exp_dir, "selected_indices.npy"), selected_idx)
            with open(os.path.join(exp_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            ckpt_results = dict(self._ckpt_results) if hasattr(self, '_ckpt_results') else {}
            if ckpt_results:
                ckpt_trajectory = {
                    "experiment_id": experiment_id,
                    "checkpoint_interval": checkpoint_interval,
                    "num_steps": step_ct,
                    "final_val_loss": val_loss,
                    "checkpoints": {str(k): v for k, v in sorted(ckpt_results.items())},
                }
                ckpt_path = os.path.join(exp_dir, "checkpoint_trajectory.json")
                with open(ckpt_path, "w") as f:
                    json.dump(ckpt_trajectory, f, indent=2)

        print(f"  [Exp {experiment_id:04d}] Done. train_loss={avg_train:.4f}, "
              f"val_loss={val_loss:.4f} (ppl={np.exp(val_loss):.1f})")
        # ── Free NPU memory (torch-npu doesn't release HBM on del) ──
        with PerfTimer.section("free_npu", _timer_prefix):
            del model, optimizer
            if device.type == "npu":
                import gc
                gc.collect()
                torch.npu.empty_cache()

        return ProxyResult(parameters=params, validation_loss=val_loss, metadata=meta)

    def _run_validation(self, model, device) -> float:
        """Run validation on full validation set, return val_loss."""
        import torch.nn.functional as F
        model.eval()
        bs = self.block_size
        val_n = len(self._val_token_ids)
        val_tokens = self._val_token_ids[:val_n, :bs].to(device)
        val_mask = self._val_loss_mask[:val_n, :bs].to(device)
        with torch.no_grad():
            val_bs = min(64, val_n)
            per_doc_losses = []
            for start in range(0, len(val_tokens), val_bs):
                end = min(start + val_bs, len(val_tokens))
                ids_in = val_tokens[start:end, :-1]
                ids_tgt = val_tokens[start:end, 1:]
                mask_tgt = val_mask[start:end, 1:]
                logits = model(ids_in)
                loss = F.cross_entropy(
                    logits.view(-1, self.model_config.vocab_size),
                    ids_tgt.reshape(-1),
                    reduction="none",
                )
                loss = loss.view(ids_tgt.shape)
                assistant_count = mask_tgt.float().sum(dim=1).clamp(min=1)
                per_doc = (loss * mask_tgt.float()).sum(dim=1) / assistant_count
                per_doc_losses.append(per_doc)
            val_loss = float(torch.cat(per_doc_losses).mean())
        model.train()
        return val_loss

    def _lr_schedule(self, it: int, max_iters: int) -> float:
        """Cosine LR with linear warmup."""
        warm = self.warmup_steps * self.gradient_accumulation_steps
        if it < warm:
            return self.learning_rate * it / max(1, warm)
        if it > max_iters:
            return self.learning_rate * 0.025
        dr = (it - warm) / max(1, max_iters - warm)
        coeff = 0.5 * (1.0 + math.cos(math.pi * dr))
        return self.learning_rate * 0.025 + coeff * (
                self.learning_rate - self.learning_rate * 0.025
        )

    def save_summary(self, results: List[ProxyResult], path: str):
        summary = {
            "num_experiments": len(results),
            "experiments": [{
                "id": r.metadata["experiment_id"],
                "val_loss": r.validation_loss,
                "params_flattened": r.parameters.flatten().tolist(),
                **r.metadata,
            } for r in results],
            "mean_val_loss": float(np.mean([r.validation_loss for r in results])),
            "std_val_loss": float(np.std([r.validation_loss for r in results])),
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[ProxyRunner] Summary: {path}")

    # ═══════════════════════════════════════════════════════════
    # Pre-compute sampling (Phase 0) — pure numpy, CPU only
    # ═══════════════════════════════════════════════════════════
    def precompute_samples(
            self, all_params: List[ParameterSet]
    ) -> List[np.ndarray]:
        """Run Eq.1-3 + fractional sampling for all experiments.

        Returns list of selected_indices (one per experiment).
        Pure numpy, no tokenization or GPU involved.
        """
        all_selected: List[np.ndarray] = []
        t0 = time.time()
        n = len(all_params)

        print(f"[PreSample] Pre-sampling {n} experiments (Eq.1-3)...")

        with PerfTimer.section("eq123_sampling", "precompute"):
            for i, params in enumerate(all_params):
                quality_ranks = self._compute_ranks_for_params(params, i)
                sv = compute_sampling_values(
                    quality_ranks, self._domain_labels, params
                )
                train_sv = sv[self._train_idx]

                # Proper fractional sampling (same as _select_documents_vectorized)
                int_part = np.floor(train_sv).astype(np.int64)
                frac_part = train_sv - int_part
                rng = np.random.default_rng(i + 42)
                random_mask = rng.uniform(size=len(train_sv)) < frac_part
                repeats = int_part + random_mask.astype(np.int64)
                selected = self._train_idx[
                    np.repeat(np.arange(len(self._train_idx)), repeats)
                ]
                if len(selected) < 10:
                    rng2 = np.random.default_rng(i + 42)
                    selected = rng2.choice(self._train_idx, 100, replace=False)

                all_selected.append(selected)

                # Progress: every 5 experiments or last one
                if (i + 1) % 5 == 0 or (i + 1) == n:
                    elapsed = time.time() - t0
                    eta = elapsed / (i + 1) * (n - i - 1)
                    print(f"[PreSample] {i + 1}/{n} done ({elapsed:.1f}s, ETA: {eta:.0f}s)")

        elapsed = time.time() - t0
        total_docs = sum(len(s) for s in all_selected)
        print(f"[PreSample] {n} experiments pre-sampled in {elapsed:.1f}s")
        print(f"[PreSample] Total selected docs: {total_docs:,} "
              f"(avg {total_docs // max(1, n):,}/exp)")

        # Collect unique docs → per-shard rows (union across all experiments)
        with PerfTimer.section("collect_unique", "precompute"):
            if self.metadata_manager is not None:
                all_unique = np.unique(np.concatenate(all_selected))
                shard_groups = self.metadata_manager.global_to_shard_rows(all_unique)
                # Store per-shard needed rows for precise cache miss tokenization
                self._per_shard_needed_rows = {
                    sid: rows for sid, (_, rows) in shard_groups.items()
                }
                unique_ratio = len(all_unique) / max(1, self._num_docs) * 100
                print(f"[PreSample] Unique docs: {len(all_unique):,} "
                      f"({unique_ratio:.1f}% of pool) "
                      f"across {len(shard_groups)} shards")
                avg_per_shard = len(all_unique) // max(1, len(shard_groups))
                print(f"[PreSample] Avg {avg_per_shard:,} docs/shard to tokenize "
                      f"(vs {self.metadata_manager._per_shard_info[0]['num_docs']:,} full)")

        return all_selected

    def tokenize_all_needed(self, all_selected: List[np.ndarray]):
        """Tokenize all documents needed by all experiments (union).

        One-shot parallel tokenize → all subsequent exps get cache hits.

        Process:
          1. Collect union of all needed docs across all experiments
          2. Group by shard
          3. For each shard:
             - Check existing cache (memory + disk)
             - Tokenize missing rows (parallel)
             - Write to shard cache
          4. No temporary files (exps read directly from shard cache)

        Args:
            all_selected: List of selected_indices from precompute_samples()
        """
        if self._mode != "sharded":
            print("[TokenizeAll] Legacy mode: tokens already loaded")
            return

        mgr = self.metadata_manager
        t0 = time.time()

        with PerfTimer.section("collect_union", "tokenize_all"):
            all_unique = np.unique(np.concatenate(all_selected))
            shard_groups = mgr.global_to_shard_rows(all_unique)

        total_tokenized = 0
        total_cached = 0

        print(f"\n[TokenizeAll] Union: {len(all_unique):,} unique docs across {len(shard_groups)} shards")

        shard_miss_info = []
        shard_miss_meta = {}

        with PerfTimer.section("check_cache", "tokenize_all"):
            for sid, (shard_path, local_rows) in shard_groups.items():
                cached_rows = self._cached_shard_rows(sid)
                memory_cached = self._memory_cache_get_rows(sid)

                local_rows_int = [int(r) for r in local_rows]
                hit_rows = [r for r in local_rows_int if r in cached_rows or r in memory_cached]
                miss_rows = [r for r in local_rows_int if r not in cached_rows and r not in memory_cached]

                total_cached += len(hit_rows)

                if miss_rows:
                    miss_rows_arr = np.array(sorted(miss_rows), dtype=np.int64)
                    shard_miss_info.append((sid, shard_path, miss_rows_arr.tolist()))
                    shard_miss_meta[sid] = len(miss_rows)

        if shard_miss_info:
            print(f"[TokenizeAll] {sum(shard_miss_meta.values()):,} miss rows across {len(shard_miss_info)} shards, parallel tokenizing...")

            with PerfTimer.section("parallel_tokenize", "tokenize_all"):
                parallel_results = _tokenize_shard_parallel(
                    shard_miss_info, self.tokenizer.name_or_path, self.block_size
                )

            with PerfTimer.section("cache_results", "tokenize_all"):
                for sid, parsed_rows, miss_tokens, io_time, tokenize_time, total_time in parallel_results:
                    self._memory_cache_add_rows(sid, parsed_rows, miss_tokens)
                    self._cache_add_rows(sid, parsed_rows, torch.from_numpy(miss_tokens))
                    total_tokenized += len(parsed_rows)
                    print(f"  [Shard {sid}] tokenized {len(parsed_rows):,} docs (IO {io_time:.1f}s, tok {tokenize_time:.1f}s)")
        else:
            print(f"[TokenizeAll] All {len(shard_groups)} shards fully cached, 0 miss rows")

        elapsed = time.time() - t0
        print(f"[TokenizeAll] Done: {total_tokenized:,} new docs tokenized, "
              f"{total_cached:,} from cache ({elapsed:.1f}s)")
        print(f"[TokenizeAll] All {len(all_selected)} experiments ready (0 cache miss expected)")

    # ═══════════════════════════════════════════════════════════
    # Parallel run across multiple NPU devices
    # ═══════════════════════════════════════════════════════════

    def _serialize_config(self, shared_metadata: Optional[Dict[str, "SharedArrayInfo"]] = None) -> dict:
        """Pickle-safe config for worker processes.

        Args:
            shared_metadata: If provided, workers use shared memory for metadata arrays
                             instead of re-reading parquet. Dict with keys:
                             'domain_labels', 'quality_scores', 'doc_char_counts',
                             'normalized_quality'.
        """
        cfg = {
            "config": self.config,
            "val_data_path": self.val_data_path,
            "preprocessed_dir": (
                self.metadata_manager._dir if self.metadata_manager else None
            ),
            "output_dir": self.output_dir,
            "device_type": self.device_type,
            "model_variant": self.model_variant,
            "global_batch_size": self.global_batch_size,
            "micro_batch_size": self.micro_batch_size,
            "max_step": self.max_step,
            "warmup_fraction": self.warmup_fraction,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "tiny_steps": self.tiny_steps,
            "doc_limit": self.doc_limit,
            "test_block_size": self.block_size,
            "rank_ref_size": self.rank_ref_size,
            "checkpoint_interval": self.checkpoint_interval,
            "token_cache_dir": self.token_cache_dir,
            # Shared memory descriptors (None if not using shared memory)
            "shared_domain_labels": shared_metadata.get("domain_labels") if shared_metadata else None,
            "shared_quality_scores": shared_metadata.get("quality_scores") if shared_metadata else None,
            "shared_doc_char_counts": shared_metadata.get("doc_char_counts") if shared_metadata else None,
            "shared_normalized_quality": shared_metadata.get("normalized_quality") if shared_metadata else None,
            # Per-shard info (pickle-safe, small)
            "per_shard_info": self.metadata_manager.shard_info if self.metadata_manager else None,
        }
        return cfg

    def run_batch_parallel(
            self,
            params_list: List[ParameterSet],
            all_selected: List[np.ndarray],
            num_workers: int = 1,
            device_type: str = "npu",
            tokenize_lookahead: int = None,
    ) -> List:
        """Run proxy experiments in parallel using dynamic task queue.

        Workers pull tasks on-demand from a shared queue. This ensures:
          - No batch boundary: fast workers do more experiments
          - Better load balance: slow experiments don't block others
          - CPU tokenize thread runs independently

        Args:
            params_list: Parameter configurations for all experiments.
            all_selected: Pre-computed selected_indices (from precompute_samples).
            num_workers: Number of parallel workers (= NPU devices to use).
            device_type: Device type for training.
            tokenize_lookahead: How many experiments to pre-tokenize per batch.
                                Default: same as num_workers (cache batch = NPU batch).

        Returns:
            List of ProxyResult in experiment order.
        """
        n_exp = len(params_list)
        assert len(all_selected) == n_exp, (
            f"all_selected length {len(all_selected)} != "
            f"params_list length {n_exp}"
        )

        # Default: tokenize batch size = NPU count (cache完一批正好NPU开跑)
        if tokenize_lookahead is None:
            tokenize_lookahead = num_workers

        if num_workers <= 1:
            # Sequential fallback
            results = []
            for i, (params, sel) in enumerate(zip(params_list, all_selected)):
                r = self.run_experiment(
                    params, experiment_id=i, selected_idx=sel
                )
                results.append(r)
            return results

        # ── Dynamic task queue mode ───────────────────────────
        return self._run_batch_dynamic(
            params_list, all_selected, num_workers, device_type, tokenize_lookahead
        )

    # ═══════════════════════════════════════════════════════════
    # Dynamic task queue implementation
    # ═══════════════════════════════════════════════════════════

    def _run_batch_dynamic(
            self,
            params_list: List[ParameterSet],
            all_selected: List[np.ndarray],
            num_workers: int,
            device_type: str,
            tokenize_lookahead: int = None,
    ) -> List:
        """Dynamic task queue mode: workers pull tasks on-demand.

        NPU Parallel Mode Optimization:
          Main Process:
            ├─ Tokenize Thread (CPU) → batch tokenize union miss rows
            ├─ AsyncWrite Thread → background write to disk cache
            ├─ Dispatcher Thread → push ready tasks (no batch boundary)
            └─ Collector Thread → receive results

          Worker 0-7 (NPU):
            └─ pull task → run → push result → pull next task

        Key optimizations:
          - Batch tokenize: each shard read ONCE per batch (not 8 times)
          - Memory cache: subsequent batches reuse tokens from memory
          - AsyncWrite: disk cache write in background, non-blocking
          - No batch boundary: single exp ready → immediate dispatch

        Args:
            params_list: All parameter configurations
            all_selected: Pre-computed selected indices
            num_workers: Number of NPU workers
            device_type: Device type (cpu/cuda/npu)
            tokenize_lookahead: How many experiments to pre-tokenize per batch

        Returns:
            List of ProxyResult in experiment order
        """
        import threading
        import queue as thread_queue  # For AsyncWrite thread

        # Default: tokenize batch size = NPU count
        if tokenize_lookahead is None:
            tokenize_lookahead = num_workers

        n_exp = len(params_list)
        all_results = [None] * n_exp
        # ── Shared memory for metadata (avoids 15GB+ reload per worker) ─
        shared_meta: Optional[Dict[str, "SharedArrayInfo"]] = None
        if self._mode == "sharded" and self.metadata_manager is not None:
            try:
                shared_meta = {}
                mgr = self.metadata_manager
                shared_meta["domain_labels"] = ndarray_to_shared(
                    mgr.domain_labels, f"dl_{os.getpid()}")
                shared_meta["quality_scores"] = ndarray_to_shared(
                    mgr.quality_scores, f"qs_{os.getpid()}")
                shared_meta["doc_char_counts"] = ndarray_to_shared(
                    mgr.doc_char_counts, f"cc_{os.getpid()}")
                # Pre-computed in _load_metadata_only
                if hasattr(self, '_normalized_quality') and self._normalized_quality is not None:
                    shared_meta["normalized_quality"] = ndarray_to_shared(
                        self._normalized_quality, f"nq_{os.getpid()}")
                total_gb = sum(v.nbytes for v in shared_meta.values()) / (1024 ** 3)
                print(f"[SharedMem] Packed {total_gb:.1f} GB metadata for {num_workers} workers")
            except Exception as e:
                print(f"[SharedMem] WARNING: shared memory setup failed ({e}), "
                      f"workers will reload from disk")
                shared_meta = None

        config_ser = self._serialize_config(shared_meta)
        t_start = time.time()

        print(f"\n[DynamicParallel] {n_exp} experiments, {num_workers} workers")
        print(f"[DynamicParallel] tokenize_lookahead={tokenize_lookahead} (batch union mode)")
        print(f"[DynamicParallel] NPU Parallel Mode: memory_cache + AsyncWrite enabled")

        # ── Create queues ───────────────────────────────────────
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue(maxsize=num_workers * 2)  # Limit backlog
        result_queue = ctx.Queue()

        # AsyncWrite queue (thread queue, not mp.Queue)
        async_write_queue = thread_queue.Queue()

        # Track which experiments are tokenized
        ready_events: Dict[int, bool] = {}
        ready_cond = threading.Condition()  # replaces busy-wait polling
        completed_count = 0

        # ── 0. AsyncWrite Thread (background write to disk cache) ──────
        def async_write_thread_func():
            """Write shard cache to disk in background, non-blocking."""
            write_count = 0
            while True:
                try:
                    item = async_write_queue.get(timeout=1.0)
                    if item is None:
                        # Termination signal
                        break
                    sid, rows, tokens = item
                    # Write to disk cache (blocking I/O but in background thread)
                    self._cache_add_rows(sid, rows, torch.from_numpy(tokens))
                    write_count += 1
                except thread_queue.Empty:
                    continue  # Timeout, check if tokenize thread finished

            print(f"[AsyncWrite] Wrote {write_count} shard caches to disk")

        async_write_thread = threading.Thread(target=async_write_thread_func, daemon=True)
        async_write_thread.start()

        # ── 1. Tokenize Thread (CPU, batch union mode) ───────────
        def tokenize_thread_func():
            """Continuously pre-tokenize experiments in BATCH UNION mode."""
            print(f"[TokenizeThread] STARTED at {time.time():.0f}")
            pos = 0
            batch_count = 0
            while pos < n_exp:
                # Tokenize a batch of experiments using UNION optimization
                end_pos = min(pos + tokenize_lookahead, n_exp)
                batch_ids = list(range(pos, end_pos))
                batch_selected = [all_selected[i] for i in batch_ids]

                try:
                    # Batch tokenize: union miss rows, each shard read ONCE
                    exp_token_paths = self._tokenize_batch_union(
                        batch_selected, batch_ids, async_write_queue
                    )
                    batch_count += 1

                    # Mark all batch exps as ready in ONE lock acquisition
                    with ready_cond:
                        for exp_id in batch_ids:
                            ready_events[exp_id] = True
                        ready_cond.notify_all()  # wake dispatcher + main wait loop

                    print(f"[TokenizeThread] Batch {batch_count}: {len(batch_ids)} exps ready ({pos}-{end_pos - 1})")

                except Exception as e:
                    print(f"[TokenizeThread] ERROR batch {batch_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Mark all exps in this batch as failed
                    with ready_cond:
                        for exp_id in batch_ids:
                            ready_events[exp_id] = False
                        ready_cond.notify_all()

                pos = end_pos
                time.sleep(0.05)  # Small pause to let workers catch up

            # Signal AsyncWrite thread to finish
            async_write_queue.put(None)
            print(
                f"[TokenizeThread] All {n_exp} experiments tokenized in {batch_count} batches ({time.time() - t_start:.1f}s)")

        tokenize_thread = threading.Thread(target=tokenize_thread_func, daemon=True)
        tokenize_thread.start()

        # ── Wait for first batch to be tokenized ────────────────
        first_batch_end = min(tokenize_lookahead, n_exp)
        print(f"[DynamicParallel] Waiting for first batch (exp 0-{first_batch_end - 1}) to be tokenized...")
        wait_start = time.time()
        last_progress_time = 0
        while True:
            with ready_cond:
                ready_count = sum(1 for i in range(first_batch_end) if ready_events.get(i, False))
            if ready_count == first_batch_end:
                break
            if time.time() - wait_start > 900:  # 15 min timeout
                with ready_cond:
                    for i in range(first_batch_end):
                        status = ready_events.get(i, None)
                        print(f"[DynamicParallel] Exp {i} status: {status}")
                print("[DynamicParallel] TIMEOUT waiting for first batch - check tokenize errors above")
                break
            now = time.time()
            if now - last_progress_time > 5:
                print(f"[DynamicParallel] tokenizing... {ready_count}/{first_batch_end} ready")
                last_progress_time = now
            with ready_cond:
                ready_cond.wait(timeout=5.0)  # signal-based wait, not polling
        print(f"[DynamicParallel] First batch tokenized ({first_batch_end} exps), starting workers")

        # ── 2. Dispatcher Thread (signal-driven, no busy-wait) ────────
        def dispatcher_thread_func():
            """Push ready tasks when signaled (no busy-wait polling)."""
            pos = 0
            failed_exps = []
            while pos < n_exp:
                with ready_cond:
                    is_ready = ready_events.get(pos, None)

                    if is_ready is True:
                        # Immediate dispatch: single exp ready
                        task_queue.put((pos, params_list[pos], all_selected[pos]))
                        pos += 1
                        continue
                    elif is_ready is False:
                        print(f"[Dispatcher] Skipping exp {pos} (tokenize failed)")
                        failed_exps.append(pos)
                        result_queue.put(ProxyResult(
                            parameters=params_list[pos],
                            validation_loss=float('inf'),
                            metadata={"experiment_id": pos, "error": "tokenize_failed"}
                        ))
                        pos += 1
                        continue
                    else:
                        # Not ready yet — wait for tokenize thread signal
                        ready_cond.wait(timeout=1.0)  # timeout for safety, but normally woken by notify

            for _ in range(num_workers):
                task_queue.put(None)

            if failed_exps:
                print(f"[Dispatcher] {len(failed_exps)} experiments failed tokenization")
            print(f"[Dispatcher] All {n_exp} tasks dispatched")

        dispatcher_thread = threading.Thread(target=dispatcher_thread_func, daemon=True)
        dispatcher_thread.start()

        # ── 3. Collector Thread ────────────────
        alive_workers = {"count": num_workers}

        def collector_thread_func():
            """Collect results from result queue."""
            nonlocal completed_count
            while completed_count < n_exp:
                try:
                    result = result_queue.get(timeout=1.0)
                    if result is None:
                        # Worker finished (normal shutdown or crash)
                        alive_workers["count"] -= 1
                        if alive_workers["count"] <= 0 and completed_count < n_exp:
                            # All workers done but not all results received
                            missing = [i for i in range(n_exp) if all_results[i] is None]
                            print(f"[Collector] Workers exited; {completed_count}/{n_exp} results, "
                                  f"{len(missing)} missing (exps: {missing})")
                            # Fill missing with inf loss so pipeline can continue
                            for eid in missing:
                                all_results[eid] = ProxyResult(
                                    parameters=params_list[eid],
                                    validation_loss=float('inf'),
                                    metadata={"experiment_id": eid, "error": "worker_crash"}
                                )
                                completed_count += 1
                            break
                        continue
                    eid = result.metadata["experiment_id"]
                    all_results[eid] = result
                    completed_count += 1

                    elapsed = time.time() - t_start
                    eta = (n_exp - completed_count) * elapsed / max(1, completed_count)
                    if completed_count % 50 == 0 or completed_count == n_exp:
                        print(f"[Collector] {completed_count}/{n_exp} done ({elapsed:.0f}s, ETA: {eta:.0f}s)")
                except:
                    pass

        collector_thread = threading.Thread(target=collector_thread_func, daemon=True)
        collector_thread.start()

        # ── 4. Launch Worker Processes ───────────────────────────
        worker_processes = []
        for wid in range(num_workers):
            p = ctx.Process(
                target=_worker_dynamic_loop,
                args=(wid, device_type, config_ser, task_queue, result_queue),
            )
            p.start()
            worker_processes.append(p)

        # ── 5. Wait for completion ───────────────────────────────
        collector_thread.join()
        dispatcher_thread.join()
        tokenize_thread.join()
        async_write_thread.join()

        for p in worker_processes:
            p.join(timeout=5.0)
            print(f"[DynamicParallel] Worker {p.pid} exitcode={p.exitcode}")

        elapsed = time.time() - t_start
        print(f"\n[DynamicParallel] All {n_exp} experiments complete ({elapsed:.0f}s ≈ {elapsed / 60:.1f}min)")

        # ── Cleanup shared memory ─────────────────────────────
        if shared_meta:
            import gc
            for name, info in list(shared_meta.items()):
                try:
                    shm = mp.shared_memory.SharedMemory(name=info.name)
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
            print(f"[SharedMem] Cleaned up {len(shared_meta)} shared memory blocks")

        return all_results


# Module-level tokenizer cache to avoid repeated loading
_tokenizer_cache: Dict[str, "Tokenizer"] = {}


def _get_tokenizer(tokenizer_path: str) -> "Tokenizer":
    """Get or create a cached tokenizer instance (per-process cache)."""
    if tokenizer_path not in _tokenizer_cache:
        from tokenizers import Tokenizer
        _tokenizer_cache[tokenizer_path] = Tokenizer.from_pretrained(tokenizer_path)
    return _tokenizer_cache[tokenizer_path]


def _io_read_shard(
        sid: int,
        shard_path: str,
        miss_rows: List[int],
) -> Tuple[int, np.ndarray, List[str], float]:
    """Stage 1 worker: read one shard's parquet, return (sid, rows, texts, io_time)."""
    import pandas as pd
    io_t0 = time.time()
    df_shard = pd.read_parquet(
        shard_path,
        columns=["row_in_shard", "text"],
        filters=[("row_in_shard", "in", miss_rows)],
    )
    df_shard = df_shard.sort_values("row_in_shard")
    texts = df_shard["text"].astype(str).tolist()
    parsed_rows = df_shard["row_in_shard"].to_numpy(dtype=np.int64)
    io_time = time.time() - io_t0
    return (sid, parsed_rows, texts, io_time)


def _process_shard_full(
        sid: int,
        shard_path: str,
        miss_rows: List[int],
        tokenizer_path: str,
        block_size: int,
) -> Tuple[int, np.ndarray, np.ndarray, float, float, float]:
    """Process one shard: IO + tokenize in sequence.
    
    This enables pipelining: as soon as one shard's IO completes, its tokenize starts
    immediately without waiting for other shards.
    
    Returns (sid, parsed_rows, tokens_array, io_time, tok_time, total_time).
    """
    # Stage 1: IO
    io_t0 = time.time()
    import pandas as pd
    df_shard = pd.read_parquet(
        shard_path,
        columns=["row_in_shard", "text"],
        filters=[("row_in_shard", "in", miss_rows)],
    )
    df_shard = df_shard.sort_values("row_in_shard")
    texts = df_shard["text"].astype(str).tolist()
    parsed_rows = df_shard["row_in_shard"].to_numpy(dtype=np.int64)
    io_time = time.time() - io_t0
    
    # Stage 2: Tokenize
    tok_t0 = time.time()
    chunk = [(sid, idx, text) for idx, text in enumerate(texts)]
    meta, tokens_array = _tokenize_chunk_to_array(chunk, tokenizer_path, block_size, 1)
    tok_time = time.time() - tok_t0
    
    return (sid, parsed_rows, tokens_array, io_time, tok_time, io_time + tok_time)


def _tokenize_shard_parallel(
        shard_tasks: List[Tuple[int, str, List[int]]],
        tokenizer_path: str,
        block_size: int,
) -> List[Tuple[int, np.ndarray, np.ndarray, float, float, float]]:
    """Parallel tokenize with separate IO and tokenize thread pools.
    
    This avoids GIL serialization:
    - IO pool: 100 threads read parquet files in parallel (pyarrow releases GIL)
    - Tokenize pool: 96 threads tokenize in parallel (Rust releases GIL)
    - Queue connects them: IO results flow to tokenize workers immediately
    
    Returns list of (sid, rows, tokens_int32, io_time, tok_time, total_time).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from queue import Queue
    import threading

    n_shards = len(shard_tasks)
    num_cpus = mp.cpu_count() or 8

    # Set thread limits for tokenize workers
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    env_workers = int(os.environ.get("TOKENIZE_WORKERS", "0"))
    if env_workers >= 1:
        n_tokenize_workers = env_workers
    else:
        n_tokenize_workers = min(96, num_cpus)
        n_tokenize_workers = max(4, n_tokenize_workers)

    n_io_workers = n_shards  # One thread per shard for IO

    print(f"  [ParallelTokenize] {n_shards} shards, {n_io_workers} IO threads + {n_tokenize_workers} tokenize threads")

    # Queue for IO → Tokenize communication
    io_queue = Queue()
    
    # Results storage
    results = []
    results_lock = threading.Lock()
    
    # Progress tracking
    io_completed = [0]
    tok_completed = [0]

    def io_worker(sid, shard_path, miss_rows):
        """Read parquet and put result in queue."""
        try:
            result = _io_read_shard(sid, shard_path, miss_rows)
            io_queue.put(result)
            with results_lock:
                io_completed[0] += 1
                # Print progress every 10 shards
                if io_completed[0] % 10 == 0 or io_completed[0] == n_shards:
                    elapsed = time.time() - t0
                    speed = io_completed[0] / elapsed if elapsed > 0 else 0
                    eta = (n_shards - io_completed[0]) / speed if speed > 0 else 0
                    print(f"  [IO Progress] {io_completed[0]}/{n_shards} shards "
                          f"({io_completed[0]*100//n_shards}%), "
                          f"{speed:.1f} shards/s, ETA {eta:.0f}s")
        except Exception as e:
            print(f"  [IO Error] shard {sid}: {e}")
            io_queue.put(None)  # Signal error

    def tokenize_worker():
        """Take IO results from queue and tokenize."""
        while True:
            item = io_queue.get()
            
            # Check for termination signal
            if item is None:
                io_queue.put(None)  # Pass signal to next worker
                break
            
            sid, parsed_rows, texts, io_time = item
            
            # Tokenize
            tok_t0 = time.time()
            chunk = [(sid, idx, text) for idx, text in enumerate(texts)]
            meta, tokens_array = _tokenize_chunk_to_array(chunk, tokenizer_path, block_size, 1)
            tok_time = time.time() - tok_t0
            
            # Store result
            with results_lock:
                results.append((sid, parsed_rows, tokens_array, io_time, tok_time, io_time + tok_time))
                tok_completed[0] += 1
                
                # Print progress every 10 shards
                if tok_completed[0] % 10 == 0 or tok_completed[0] == n_shards:
                    elapsed = time.time() - t0
                    speed = tok_completed[0] / elapsed if elapsed > 0 else 0
                    eta = (n_shards - tok_completed[0]) / speed if speed > 0 else 0
                    print(f"  [Tokenize Progress] {tok_completed[0]}/{n_shards} shards "
                          f"({tok_completed[0]*100//n_shards}%), "
                          f"{speed:.1f} shards/s, ETA {eta:.0f}s")

    t0 = time.time()
    
    with PerfTimer.section("parallel_tokenize", "parallel_tokenize"):
        # Start IO pool
        io_executor = ThreadPoolExecutor(max_workers=n_io_workers)
        io_futures = [
            io_executor.submit(io_worker, sid, shard_path, miss_rows)
            for sid, shard_path, miss_rows in shard_tasks
        ]
        
        # Start tokenize pool
        tok_executor = ThreadPoolExecutor(max_workers=n_tokenize_workers)
        tok_futures = [
            tok_executor.submit(tokenize_worker)
            for _ in range(n_tokenize_workers)
        ]
        
        # Wait for all IO to complete
        for fut in io_futures:
            fut.result()
        
        # Send termination signals (one per tokenize worker)
        for _ in range(n_tokenize_workers):
            io_queue.put(None)
        
        # Wait for all tokenize to complete
        for fut in tok_futures:
            fut.result()
        
        io_executor.shutdown(wait=False)
        tok_executor.shutdown(wait=False)
    
    total_time = time.time() - t0
    total_docs = sum(len(r[1]) for r in results)
    print(f"  [ParallelTokenize] {total_docs:,} docs in {total_time:.1f}s "
          f"({total_docs / total_time:.0f} docs/s)")

    return results


def _tokenize_chunk_with_meta(
        chunk: List[Tuple[int, int, str]],
        tokenizer_path: str,
        block_size: int,
) -> List[Tuple[int, int, List[int]]]:
    """Tokenize a chunk of (sid, idx, text) items. Returns (sid, idx, token_ids).

    Uses the Rust-based tokenizers library directly for max throughput.
    """
    tok = _get_tokenizer(tokenizer_path)

    # Extract texts in order
    texts = [item[2] for item in chunk]

    # encode_batch is internally multi-threaded
    encodings = tok.encode_batch(texts)

    # GPT-NeoX uses EOS token 50256 as pad
    PAD_TOKEN = 50256

    # Pair back with (sid, idx), truncate/pad to block_size
    results = []
    for (sid, idx, _), enc in zip(chunk, encodings):
        ids = list(enc.ids)
        # Truncate
        if len(ids) > block_size:
            ids = ids[:block_size]
        # Pad (avoid slow per-element append)
        if len(ids) < block_size:
            ids = ids + [PAD_TOKEN] * (block_size - len(ids))
        results.append((sid, idx, ids))
    return results


def _tokenize_chunk_to_array(
        chunk: List[Tuple[int, int, str]],
        tokenizer_path: str,
        block_size: int,
        threads_per_worker: int = 4,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Tokenize a chunk, returning compact numpy array instead of per-doc lists.

    Returns ((sid, idx) pairs, np.array[N x block_size, dtype=int32]).
    This reduces IPC overhead from ~1.6GB of Python objects to ~1.6GB of contiguous
    shared memory that gets pickled once.
    """
    import os as _os
    _os.environ["RAYON_NUM_THREADS"] = str(threads_per_worker)
    _os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)

    tok = _get_tokenizer(tokenizer_path)
    texts = [item[2] for item in chunk]
    encodings = tok.encode_batch(texts)

    PAD_TOKEN = 50256
    N = len(chunk)
    # Pre-allocate with padding
    tokens_array = np.full((N, block_size), PAD_TOKEN, dtype=np.int32)

    meta = []
    for (i, (sid, idx, _)), enc in zip(enumerate(chunk), encodings):
        ids = list(enc.ids)
        n = min(len(ids), block_size)
        tokens_array[i, :n] = ids[:n]
        meta.append((sid, idx))

    return meta, tokens_array


def _worker_dynamic_loop(
        worker_id: int,
        device_type: str,
        config_dict: dict,
        task_queue,
        result_queue,
):
    """Worker loop for dynamic mode: pull task → run → push result → repeat.

    Uses shared memory for metadata arrays when available, avoiding
    per-worker 15GB+ parquet reload. Falls back to disk loading gracefully.
    """
    # Write marker to stderr on function entry
    with open(f"/tmp/worker_{worker_id}_entry.log", "w") as ef:
        ef.write(f"[Worker {worker_id}] FUNCTION ENTERED at {time.time()}")
    try:
        from quadmix.data.metadata_manager import ShardMetadataManager

        # ── Metadata loading: shared memory or disk ──────────────────
        shared_dl = config_dict.get("shared_domain_labels")
        shared_qs = config_dict.get("shared_quality_scores")
        shared_cc = config_dict.get("shared_doc_char_counts")
        shared_nq = config_dict.get("shared_normalized_quality")
        per_shard_info = config_dict.get("per_shard_info")

        use_shared = (shared_dl is not None and shared_qs is not None
                      and shared_cc is not None and per_shard_info is not None)

        if use_shared:
            t0 = time.time()
            # Map shared memory arrays (zero-copy, no disk read)
            domain_labels = shared_to_ndarray(shared_dl)
            quality_scores = shared_to_ndarray(shared_qs)
            doc_char_counts = shared_to_ndarray(shared_cc)

            # Reconstruct shard_starts from per_shard_info
            shard_starts_arr = np.array(
                [s["start_idx"] for s in per_shard_info], dtype=np.int64
            )

            mgr = ShardMetadataManager.from_shared(
                domain_labels=domain_labels,
                quality_scores=quality_scores,
                doc_char_counts=doc_char_counts,
                per_shard_info=per_shard_info,
                shard_starts=shard_starts_arr,
                preprocessed_dir=config_dict.get("preprocessed_dir", ""),
            )
            print(f"[Worker {worker_id}] Mapped {len(domain_labels):,} docs from shared memory "
                  f"({time.time() - t0:.1f}s vs ~60s disk reload)")
        else:
            # Fallback: reload from disk (backward compatible)
            if config_dict.get("preprocessed_dir"):
                mgr = ShardMetadataManager(config_dict["preprocessed_dir"])
            else:
                mgr = None  # Legacy mode, no sharded metadata needed

        runner = EssentialWebProxyRunner(
            config=config_dict["config"],
            metadata_manager=mgr,
            val_data_path=config_dict["val_data_path"],
            output_dir=config_dict["output_dir"],
            device_type=device_type,
            npu_device_id=worker_id,  # Bind to specific NPU card
            model_variant=config_dict["model_variant"],
            global_batch_size=config_dict["global_batch_size"],
            micro_batch_size=config_dict["micro_batch_size"],
            max_step=config_dict["max_step"],
            warmup_fraction=config_dict.get("warmup_fraction", 0.04),
            learning_rate=config_dict["learning_rate"],
            weight_decay=config_dict["weight_decay"],
            grad_clip=config_dict["grad_clip"],
            tiny_steps=config_dict["tiny_steps"],
            doc_limit=config_dict["doc_limit"],
            test_block_size=config_dict["test_block_size"],
            rank_ref_size=config_dict["rank_ref_size"],
            token_cache_dir=config_dict["token_cache_dir"],
            checkpoint_interval=config_dict.get("checkpoint_interval", 1000),
        )

        completed = 0
        while True:
            task = task_queue.get()  # Blocking pull

            if task is None:
                # Termination signal
                print(f"[Worker {worker_id}] Shutdown, completed {completed} experiments")
                break

            exp_id, params, selected_idx = task
            print(f"[Worker {worker_id}] Running exp {exp_id}")

            r = runner.run_experiment(params, experiment_id=exp_id, selected_idx=selected_idx,
                                      checkpoint_interval=config_dict.get("checkpoint_interval", 1000))
            result_queue.put(r)
            completed += 1

            # Free NPU HBM between experiments (torch-npu lazy allocation)
            import gc
            gc.collect()
            if device_type == "npu":
                torch.npu.empty_cache()

            # Clean up temporary token file
            exp_token_path = runner._get_exp_token_path(exp_id)
            if os.path.exists(exp_token_path):
                os.remove(exp_token_path)
                print(f"[Worker {worker_id}] Cleaned temp file for exp {exp_id}")

        result_queue.put(None)  # Signal completion to collector


    except Exception as top_err:
        import sys, traceback
        err_path = f"/tmp/worker_{worker_id}_error.log"
        try:
            with open(err_path, "w") as ef:
                ef.write(f"[Worker {worker_id}] CRASH: {top_err}\n")
                ef.write(traceback.format_exc())
            print(f"[Worker {worker_id}] ERROR -> {err_path}", flush=True)
        except:
            pass
        # Put error results for any tasks that were not completed,
        # so collector doesn't hang waiting forever.
        # We don't know which specific exp failed (could be any pending),
        # but at minimum signal completion to unblock the collector.
        try:
            result_queue.put(None)
        except:
            pass
        sys.exit(1)


def test_runner_sharded():
    """Quick test: 2 experiments using sharded metadata manager."""
    from quadmix import QuaDMixConfig
    from quadmix.pipeline.param_sampler import ParameterSampler
    from quadmix.data.metadata_manager import ShardMetadataManager

    mgr = ShardMetadataManager(
        os.path.join(DEFAULT_TEMP_DIR, "preprocessed")
    )

    config = QuaDMixConfig(
        num_domains=10, num_quality_criteria=5, num_proxy_experiments=2
    )

    runner = EssentialWebProxyRunner(
        config=config,
        metadata_manager=mgr,
        val_data_path=os.path.join(_PROJECT_DIR, "data/openhermes_10k_assistant_tokenized.pt"),
        output_dir=os.path.join(DEFAULT_TEMP_DIR, "outputs/test_sharded"),
        device_type="cpu",
        micro_batch_size=2,
        global_batch_size=8,
        tiny_steps=5,
        doc_limit=5000,
        test_block_size=64,
        rank_ref_size=500,
    )

    params = ParameterSampler(config).sample_batch(2)
    print(f"\nRunning {len(params)} experiments (sharded)...")
    t0 = time.time()
    results = runner.run_batch(params)
    print(f"\n{'=' * 60}")
    print(f"  Test Complete! ({time.time() - t0:.1f}s)")
    for r in results:
        print(f"  Exp {r.metadata['experiment_id']:04d}: "
              f"val_loss={r.validation_loss:.4f}")
    runner.save_summary(results, os.path.join(runner.output_dir, "test_summary.json"))
    print("=" * 60)


if __name__ == "__main__":
    test_runner_sharded()