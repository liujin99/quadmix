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
from functools import partial
from typing import List, Optional, Dict, Tuple
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F

from quadmix.core.types import ParameterSet, ProxyResult, QuaDMixConfig
from quadmix.core.quality_merger import compute_merged_quality_scores
from quadmix.core.quality_rank import compute_quality_ranks
from quadmix.core.sampler import compute_sampling_values
from quadmix.pipeline.proxy_runner import BaseProxyRunner

# Default cache directory for per-shard token cache
# Override via token_cache_dir param in constructor
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPTS_DIR)
DEFAULT_TOKEN_CACHE_DIR = os.path.join(_PROJECT_DIR, "temp/token_cache")


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
        global_batch_size: int = 512,
        micro_batch_size: int = 4,
        max_step: int = 25000,
        warmup_steps: int = 1000,
        learning_rate: float = 4e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        # "just runnable" flags
        tiny_steps: int = 10,
        doc_limit: Optional[int] = None,
        test_block_size: Optional[int] = None,
        rank_ref_size: int = 10000,
        val_doc_limit: int = 500,
        token_cache_dir: str = DEFAULT_TOKEN_CACHE_DIR,
    ):
        self.config = config
        self.metadata_manager = metadata_manager
        self.legacy_data_path = data_path  # only used when no metadata_manager
        self.val_data_path = val_data_path
        self.output_dir = output_dir
        self.model_variant = model_variant
        self.device_type = device_type
        self.npu_device_id = npu_device_id  # ← 新增：指定 NPU 卡號
        self.tiny_steps = tiny_steps
        self.doc_limit = doc_limit
        self.rank_ref_size = rank_ref_size
        self.val_doc_limit = val_doc_limit
        self.token_cache_dir = token_cache_dir

        # RegMix training config
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.max_step = max_step
        self.warmup_steps = warmup_steps
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
        """Load domain labels + quality scores from metadata manager (no text)."""
        t0 = time.time()
        mgr = self.metadata_manager
        self._domain_labels = mgr.domain_labels
        self._quality_scores = mgr.quality_scores
        self._num_docs = mgr.num_docs
        self._train_idx = np.arange(self._num_docs)

        print(f"[ProxyRunner] Sharded mode: {self._num_docs:,} docs "
              f"(metadata only, {mgr.num_shards} shards) ({time.time()-t0:.0f}s)")

        # Per-shard token cache: dict[int, dict[int, torch.Tensor]]
        # cache[shard_idx] = {row_idx: token_ids_tensor}
        # OR saved to disk: cache_dir/tokens_shard_{sid:05d}_bs{bs}.npy
        os.makedirs(self.token_cache_dir, exist_ok=True)
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_shard_token_path(self, shard_idx: int) -> str:
        """Path to disk cache for a shard's tokens."""
        return os.path.join(
            self.token_cache_dir,
            f"shard_{shard_idx:05d}_bs{self.block_size}.npy",
        )

    def _tokenize_texts(self, texts: List[str]) -> torch.Tensor:
        """Tokenize a list of texts into [M, block_size] int64 tensor."""
        B = 500  # batch size for tokenizer
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
        data = np.load(cache_path, mmap_mode='r')
        rows = set(int(r) for r in data['rows'])
        return rows

    def _cache_add_rows(self, sid: int, new_rows: np.ndarray,
                        new_tokens: torch.Tensor):
        """Append rows to an existing cache npz (rewrite if needed).

        Reads old cache, merges new rows (sorted by row_in_shard),
        rewrites single npz. Bound by total unique docs per shard (~25K).
        
        Race condition handling: re-read cache right before write to ensure
        we don't lose rows added by other concurrent exps.
        """
        cache_path = self._get_shard_token_path(sid)
        new_np = new_tokens.numpy().astype(np.int32)  # [M, block_size]

        # First read: get baseline
        if os.path.exists(cache_path):
            old = np.load(cache_path)
            old_rows: np.ndarray = old['rows']
            old_tokens: np.ndarray = old['tokens']
            del old
        else:
            old_rows = np.array([], dtype=np.int64)
            old_tokens = np.zeros((0, new_np.shape[1]), dtype=np.int32)

        # Merge with new rows (dedupe by row_in_shard)
        combined_rows = np.concatenate([old_rows, new_rows])
        combined_tokens = np.concatenate([old_tokens, new_np])
        
        # Deduplicate: keep latest (new overwrites old if duplicate)
        unique_rows, inverse = np.unique(combined_rows, return_index=True)
        # Sort by row value for deterministic order
        order = np.argsort(unique_rows)
        unique_rows = unique_rows[order]
        
        # For tokens: need to handle duplicates - new tokens overwrite old
        # Use a dict to track which token to use for each row
        row_to_token_idx = {}
        for i, r in enumerate(combined_rows):
            row_to_token_idx[int(r)] = i  # Later entry overwrites earlier
        final_tokens = combined_tokens[[row_to_token_idx[int(r)] for r in unique_rows]]
        
        # Atomic write: write to temp file first, then rename
        temp_path = cache_path + ".tmp"
        np.savez(temp_path, tokens=final_tokens, rows=unique_rows)
        
        # Rename is atomic on POSIX systems
        if os.path.exists(cache_path):
            os.replace(temp_path, cache_path)
        else:
            os.rename(temp_path, cache_path)

    # ═══════════════════════════════════════════════════════════
    # Batch-level tokenization (CPU, called from main process)
    # ═══════════════════════════════════════════════════════════

    def _get_exp_token_path(self, exp_id: int) -> str:
        """Path to temporary token file for a single experiment."""
        return os.path.join(
            self.token_cache_dir,
            f"exp_{exp_id:04d}_tokens.pt"
        )

    def _pack_exp_tokens(
        self,
        exp_id: int,
        selected_idx: np.ndarray,
    ) -> str:
        """Pack tokens for a single experiment into a temporary file.

        Process:
          1. For each shard needed by this exp:
             - Check shard cache for existing rows
             - Tokenize missing rows → write to shard cache
          2. Collect all tokens for this exp
          3. Save to temporary file (exp_{id}_tokens.pt)

        Returns:
            Path to the temporary token file.
        """
        mgr = self.metadata_manager
        shard_groups = mgr.global_to_shard_rows(selected_idx)

        all_tokens = []
        total_tokenized = 0
        t0 = time.time()

        for sid, (shard_path, local_rows) in shard_groups.items():
            cache_path = self._get_shard_token_path(sid)
            cached_rows = self._cached_shard_rows(sid)

            # Split into hit and miss
            hit_rows = [r for r in local_rows if int(r) in cached_rows]
            miss_rows = [r for r in local_rows if int(r) not in cached_rows]

            hit_tokens = None
            miss_tokens = None

            # Hit: read from cache
            if hit_rows:
                data = np.load(cache_path, mmap_mode='r')
                token_mmap = data['tokens']
                row_index = data['rows']
                row_to_pos = {int(r): i for i, r in enumerate(row_index)}
                positions = np.array([row_to_pos[int(r)] for r in hit_rows], dtype=np.int64)
                hit_tokens = torch.from_numpy(token_mmap[positions].astype(np.int64))
                del data

            # Miss: tokenize and add to cache
            if miss_rows:
                miss_rows_arr = np.array(miss_rows, dtype=np.int64)
                df_shard = pd.read_parquet(
                    shard_path,
                    columns=["row_in_shard", "text"],
                    filters=[("row_in_shard", "in", miss_rows_arr.tolist())],
                )
                df_shard = df_shard.sort_values("row_in_shard")
                texts = df_shard["text"].astype(str).tolist()
                parsed_rows = df_shard["row_in_shard"].to_numpy(dtype=np.int64)

                miss_tokens_full = self._tokenize_texts(texts)
                self._cache_add_rows(sid, parsed_rows, miss_tokens_full)
                total_tokenized += len(texts)

                # Extract requested rows
                row_to_pos_new = {int(r): i for i, r in enumerate(parsed_rows)}
                positions = np.array([row_to_pos_new[int(r)] for r in miss_rows], dtype=np.int64)
                miss_tokens = torch.from_numpy(miss_tokens_full.numpy().astype(np.int64)[positions])

            # Merge hit and miss in original order
            if miss_rows:
                row_to_token = {}
                if hit_tokens is not None:
                    for i, r in enumerate(hit_rows):
                        row_to_token[int(r)] = hit_tokens[i]
                if miss_tokens is not None:
                    for i, r in enumerate(miss_rows):
                        row_to_token[int(r)] = miss_tokens[i]
                shard_tokens = torch.stack([row_to_token[int(r)] for r in local_rows])
            else:
                shard_tokens = hit_tokens

            all_tokens.append(shard_tokens)

        # Pack into temporary file
        result = torch.cat(all_tokens, dim=0)
        exp_token_path = self._get_exp_token_path(exp_id)
        torch.save(result, exp_token_path)

        elapsed = time.time() - t0
        print(f"  [PackExp {exp_id:04d}] {len(result):,} docs, "
              f"tokenized {total_tokenized:,} new, "
              f"{len(shard_groups)} shards ({elapsed:.1f}s)")

        return exp_token_path

    def tokenize_batch_delta(
        self,
        batch_selected: List[np.ndarray],
        batch_exp_ids: List[int],
    ):
        """Tokenize and pack tokens for each experiment in the batch.

        For each exp:
          1. Check shard cache, tokenize missing rows
          2. Pack into temporary file (exp_{id}_tokens.pt)
        """
        for exp_id, selected_idx in zip(batch_exp_ids, batch_selected):
            self._pack_exp_tokens(exp_id, selected_idx)

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
                data = np.load(cache_path, mmap_mode='r')
                token_mmap = data['tokens']
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
                        token_mmap[positions].astype(np.int64)
                    )
                    self._cache_hits += len(hit_rows)

                del data, token_mmap

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
        print(f"[ProxyRunner] (legacy) {self._num_docs:,} docs ({time.time()-t0:.0f}s)")

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

    # ═══════════════════════════════════════════════════════════
    # Experiment execution
    # ═══════════════════════════════════════════════════════════

    def _compute_ranks_for_params(
        self, params: ParameterSet, experiment_id: int,
    ) -> np.ndarray:
        """Per-experiment Eq.1 + Eq.2 (unchanged from original)."""
        N = self.config.num_quality_criteria
        M = self.config.num_domains
        rng = np.random.default_rng(experiment_id + 1729)

        # Eq.1: Merge with domain-specific α_m weights
        merged = compute_merged_quality_scores(
            self._quality_scores, self._domain_labels,
            params.merge_config,
        )

        # Eq.2: Subset-based rank estimation per domain
        ranks = np.zeros(self._num_docs, dtype=np.float64)
        for m in range(M):
            mask = self._domain_labels == m
            if mask.sum() == 0:
                continue
            indices = np.where(mask)[0]
            n_domain = len(indices)
            domain_scores = merged[indices]

            k = min(self.rank_ref_size, n_domain)
            ref_idx = rng.choice(n_domain, k, replace=False)
            ref_scores = np.sort(domain_scores[ref_idx])

            positions = np.searchsorted(ref_scores, domain_scores, side='right')
            ranks[indices] = positions.astype(np.float64) / k

        return ranks

    def run_experiment(
        self, params: ParameterSet, experiment_id: int = 0,
        selected_idx: Optional[np.ndarray] = None,  # ← 新增：外部傳入選中文檔索引
    ) -> ProxyResult:
        """Train one proxy model. Validates on openhermes-10k.

        If selected_idx is provided, skips Eq.1-3 sampling (pre-computed mode).
        """
        from quadmix.core.proxy_model import ProxyModel
        from quadmix.npu.device import DeviceManager, DeviceType

        os.makedirs(self.output_dir, exist_ok=True)
        exp_dir = os.path.join(self.output_dir, f"exp_{experiment_id:04d}")
        os.makedirs(exp_dir, exist_ok=True)

        device_mgr = DeviceManager(
            device_type=DeviceType(self.device_type),
            npu_device_id=self.npu_device_id,
        )
        device = device_mgr.get_device()

        # ---- 0. Compute quality ranks (Eq.1+Eq.2) + sample ----
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

        # ---- 1. Load / tokenize training data on demand ----
        train_tokens = self._load_tokens_for_experiment(selected_idx, exp_id=experiment_id).to(device)

        # ---- 2. Create model ----
        model = ProxyModel(config=self.model_config).to(device)
        non_emb = model.count_params(non_embedding_only=True)
        print(f"  [Exp {experiment_id:04d}] Model on {device}: "
              f"{model.count_params():,} total, {non_emb:,} non-emb")

        # ---- 3. Optimizer ----
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate,
            betas=(0.9, 0.95), weight_decay=self.weight_decay, fused=False,
        )

        # ---- 4. Training data preparation ----
        flat_train = train_tokens.reshape(-1)
        num_steps = self.tiny_steps if self.tiny_steps > 0 else self.max_step
        grad_acc = self.gradient_accumulation_steps
        max_iters = num_steps * grad_acc
        tok_per_step = self.micro_batch_size * self.block_size

        # ---- 5. Training loop (RegMix-style) ----
        model.train()
        total_loss = 0.0
        iter_ct = 0
        step_ct = 0
        log_int = max(1, num_steps // 5)
        t_start = time.time()

        print(f"  [Exp {experiment_id:04d}] Training {num_steps} steps "
              f"(grad_acc={grad_acc}, micro_batch={self.micro_batch_size})...")

        while iter_ct < max_iters:
            max_st = max(1, flat_train.size(0) - self.block_size - 1)
            st = torch.randint(0, max_st, (self.micro_batch_size,))
            batch = torch.stack([flat_train[s:s + self.block_size + 1] for s in st])
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

            total_loss += loss.item()
            iter_ct += 1

            if not is_acc and (step_ct % log_int == 0 or step_ct == 1):
                avg = total_loss / step_ct
                elapsed = time.time() - t_start
                rem = (num_steps - step_ct) * elapsed / max(1, step_ct)
                print(f"    Step {step_ct}/{num_steps}, loss={avg:.4f}, "
                      f"lr={lr:.2e}, {tok_per_step*step_ct/elapsed:.0f} tok/s, "
                      f"ETA: {rem:.0f}s")

        # ---- 6. Validation: assistant-only loss ----
        model.eval()
        bs = self.block_size
        val_n = min(self.val_doc_limit, len(self._val_token_ids))
        val_tokens = self._val_token_ids[:val_n, :bs].to(device)
        val_mask = self._val_loss_mask[:val_n, :bs].to(device)

        with torch.no_grad():
            val_bs = 200
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

        # ---- 7. Save metadata ----
        np.save(os.path.join(exp_dir, "selected_indices.npy"), selected_idx)
        avg_train = total_loss / step_ct if step_ct > 0 else 0

        meta = {
            "experiment_id": experiment_id,
            "variant": self.model_variant,
            "train_loss": avg_train,
            "val_loss": val_loss,
            "val_ppl": float(np.exp(val_loss)),
            "num_steps": step_ct,
            "sampled_docs": len(selected_idx),
            "val_docs": val_n,
            "assistant_loss": True,
            "params_lambda": [sc.lambda_ for sc in params.sampling_configs],
            "params_omega": [sc.omega for sc in params.sampling_configs],
            "params_eta": [sc.eta for sc in params.sampling_configs],
            "params_epsilon": [sc.epsilon for sc in params.sampling_configs],
            "global_weights": params.merge_config.global_weights.tolist(),
            "domain_weights": params.merge_config.domain_weights.tolist(),
        }
        with open(os.path.join(exp_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [Exp {experiment_id:04d}] Done. train_loss={avg_train:.4f}, "
              f"val_loss={val_loss:.4f} (ppl={np.exp(val_loss):.1f})")

        return ProxyResult(parameters=params, validation_loss=val_loss, metadata=meta)

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

        elapsed = time.time() - t0
        total_docs = sum(len(s) for s in all_selected)
        n = len(all_params)
        print(f"\n[PreSample] {n} experiments pre-sampled in {elapsed:.1f}s")
        print(f"[PreSample] Total selected docs: {total_docs:,} "
              f"(avg {total_docs//max(1,n):,}/exp)")

        # Collect unique docs → per-shard rows (union across all experiments)
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

        For CPU/sequential mode: one-shot tokenize → all subsequent exps get cache hits.

        Process:
          1. Collect union of all needed docs across all experiments
          2. Group by shard
          3. For each shard:
             - Check existing cache
             - Tokenize missing rows
             - Write to shard cache
          4. No temporary files (exps read directly from shard cache)

        Args:
            all_selected: List of selected_indices from precompute_samples()
        """
        if self._mode != "sharded":
            # Legacy mode: already tokenized upfront
            print("[TokenizeAll] Legacy mode: tokens already loaded")
            return

        mgr = self.metadata_manager
        t0 = time.time()

        # Collect union of all needed docs
        all_unique = np.unique(np.concatenate(all_selected))
        shard_groups = mgr.global_to_shard_rows(all_unique)

        total_tokenized = 0
        total_cached = 0

        print(f"\n[TokenizeAll] Union: {len(all_unique):,} unique docs across {len(shard_groups)} shards")

        for sid, (shard_path, local_rows) in shard_groups.items():
            cache_path = self._get_shard_token_path(sid)
            cached_rows = self._cached_shard_rows(sid)

            # Split into hit and miss
            hit_rows = [r for r in local_rows if int(r) in cached_rows]
            miss_rows = [r for r in local_rows if int(r) not in cached_rows]

            total_cached += len(hit_rows)

            if miss_rows:
                miss_rows_arr = np.array(miss_rows, dtype=np.int64)
                df_shard = pd.read_parquet(
                    shard_path,
                    columns=["row_in_shard", "text"],
                    filters=[("row_in_shard", "in", miss_rows_arr.tolist())],
                )
                df_shard = df_shard.sort_values("row_in_shard")
                texts = df_shard["text"].astype(str).tolist()
                parsed_rows = df_shard["row_in_shard"].to_numpy(dtype=np.int64)

                miss_tokens = self._tokenize_texts(texts)
                self._cache_add_rows(sid, parsed_rows, miss_tokens)
                total_tokenized += len(texts)

                print(f"  [Shard {sid}] tokenized {len(texts):,} docs, hit {len(hit_rows):,} from cache")

        elapsed = time.time() - t0
        print(f"[TokenizeAll] Done: {total_tokenized:,} new docs tokenized, "
              f"{total_cached:,} from cache ({elapsed:.1f}s)")
        print(f"[TokenizeAll] All {len(all_selected)} experiments ready (0 cache miss expected)")

    # ═══════════════════════════════════════════════════════════
    # Parallel run across multiple NPU devices
    # ═══════════════════════════════════════════════════════════

    def _serialize_config(self) -> dict:
        """Pickle-safe config for worker processes."""
        return {
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
            "warmup_steps": self.warmup_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "tiny_steps": self.tiny_steps,
            "doc_limit": self.doc_limit,
            "test_block_size": self.block_size,
            "rank_ref_size": self.rank_ref_size,
            "val_doc_limit": self.val_doc_limit,
            "token_cache_dir": self.token_cache_dir,
        }

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

        Architecture:
          Main Process:
            ├─ Tokenize Thread (CPU) → continuously pre-tokenize
            ├─ Dispatcher Thread → push ready tasks to queue
            └─ Collector Thread → receive results

          Worker 0-7 (NPU):
            └─ pull task → run → push result → pull next task

        Key: Wait for first batch to be tokenized before starting workers.
             This ensures workers get cache hits instead of re-tokenizing.

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
        from multiprocessing import Queue

        # Default: tokenize batch size = NPU count
        if tokenize_lookahead is None:
            tokenize_lookahead = num_workers

        n_exp = len(params_list)
        all_results = [None] * n_exp
        config_ser = self._serialize_config()
        t_start = time.time()

        print(f"\n[DynamicParallel] {n_exp} experiments, {num_workers} workers")
        print(f"[DynamicParallel] tokenize_lookahead={tokenize_lookahead}")

        # ── Create queues ───────────────────────────────────────
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue(maxsize=num_workers * 2)  # Limit backlog
        result_queue = ctx.Queue()

        # Track which experiments are tokenized
        ready_events: Dict[int, bool] = {}
        ready_lock = threading.Lock()
        completed_count = 0

        # ── 1. Tokenize Thread (CPU, continuously run) ───────────
        def tokenize_thread_func():
            """Continuously pre-tokenize experiments ahead of workers."""
            pos = 0
            while pos < n_exp:
                # Tokenize a batch of experiments
                end_pos = min(pos + tokenize_lookahead, n_exp)
                batch_ids = list(range(pos, end_pos))
                batch_selected = [all_selected[i] for i in batch_ids]

                # Tokenize one-by-one, mark each as ready only when successful
                for exp_id, selected_idx in zip(batch_ids, batch_selected):
                    try:
                        self._pack_exp_tokens(exp_id, selected_idx)
                        # Only mark as ready if successful
                        with ready_lock:
                            ready_events[exp_id] = True
                    except Exception as e:
                        print(f"[TokenizeThread] ERROR exp {exp_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Mark as failed - Worker will see no temp file and fail
                        with ready_lock:
                            ready_events[exp_id] = False  # Explicitly mark as failed

                pos = end_pos
                time.sleep(0.05)  # Small pause to let workers catch up

            print(f"[TokenizeThread] All {n_exp} experiments processed")

        tokenize_thread = threading.Thread(target=tokenize_thread_func, daemon=True)
        tokenize_thread.start()

        # ── Wait for first batch to be tokenized ────────────────
        # Key fix: Workers must wait for cache to be ready
        first_batch_end = min(tokenize_lookahead, n_exp)
        print(f"[DynamicParallel] Waiting for first batch (exp 0-{first_batch_end-1}) to be tokenized...")
        while True:
            with ready_lock:
                first_ready = ready_events.get(first_batch_end - 1, False)
            if first_ready:
                break
            time.sleep(0.5)
        print(f"[DynamicParallel] First batch tokenized, starting workers")

        # ── 2. Dispatcher Thread (push ready tasks) ───────────────
        def dispatcher_thread_func():
            """Push tokenized experiments to task queue."""
            pos = 0
            failed_exps = []
            while pos < n_exp:
                with ready_lock:
                    is_ready = ready_events.get(pos, None)  # None = not yet processed

                if is_ready is True:
                    task_queue.put((pos, params_list[pos], all_selected[pos]))
                    pos += 1
                elif is_ready is False:
                    # Tokenize failed for this exp - skip and record
                    print(f"[Dispatcher] Skipping exp {pos} (tokenize failed)")
                    failed_exps.append(pos)
                    # Push a failed result placeholder directly to result queue
                    result_queue.put(ProxyResult(
                        parameters=params_list[pos],
                        validation_loss=float('inf'),
                        metadata={"experiment_id": pos, "error": "tokenize_failed"}
                    ))
                    pos += 1
                else:
                    # Not yet processed, wait
                    time.sleep(0.02)

            # Send termination signals
            for _ in range(num_workers):
                task_queue.put(None)

            if failed_exps:
                print(f"[Dispatcher] {len(failed_exps)} experiments failed tokenization")
            print(f"[Dispatcher] All {n_exp} tasks dispatched")

        dispatcher_thread = threading.Thread(target=dispatcher_thread_func, daemon=True)
        dispatcher_thread.start()

        # ── 3. Collector Thread (receive results) ────────────────
        def collector_thread_func():
            """Collect results from result queue."""
            nonlocal completed_count
            while completed_count < n_exp:
                try:
                    result = result_queue.get(timeout=1.0)
                    if result is not None:
                        eid = result.metadata["experiment_id"]
                        all_results[eid] = result
                        completed_count += 1

                        elapsed = time.time() - t_start
                        eta = (n_exp - completed_count) * elapsed / max(1, completed_count)
                        if completed_count % 50 == 0 or completed_count == n_exp:
                            print(f"[Collector] {completed_count}/{n_exp} done "
                                  f"({elapsed:.0f}s, ETA: {eta:.0f}s)")
                except:
                    pass  # Timeout, continue waiting

        collector_thread = threading.Thread(target=collector_thread_func, daemon=True)
        collector_thread.start()

        # ── 4. Launch Worker Processes ───────────────────────────
        worker_processes = []
        for wid in range(num_workers):
            p = ctx.Process(
                target=_worker_dynamic_loop,
                args=(
                    wid,
                    device_type,
                    config_ser,
                    task_queue,
                    result_queue,
                ),
            )
            p.start()
            worker_processes.append(p)

        # ── 5. Wait for completion ───────────────────────────────
        collector_thread.join()
        dispatcher_thread.join()
        tokenize_thread.join()

        for p in worker_processes:
            p.join(timeout=5.0)

        elapsed = time.time() - t_start
        print(f"\n[DynamicParallel] All {n_exp} experiments complete "
              f"({elapsed:.0f}s ≈ {elapsed/60:.1f}min)")
        return all_results


def _worker_dynamic_loop(
    worker_id: int,
    device_type: str,
    config_dict: dict,
    task_queue,
    result_queue,
):
    """Worker loop for dynamic mode: pull task → run → push result → repeat."""
    from quadmix.data.metadata_manager import ShardMetadataManager

    # Re-create runner in worker process
    mgr = ShardMetadataManager(config_dict["preprocessed_dir"])
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
        warmup_steps=config_dict["warmup_steps"],
        learning_rate=config_dict["learning_rate"],
        weight_decay=config_dict["weight_decay"],
        grad_clip=config_dict["grad_clip"],
        tiny_steps=config_dict["tiny_steps"],
        doc_limit=config_dict["doc_limit"],
        test_block_size=config_dict["test_block_size"],
        rank_ref_size=config_dict["rank_ref_size"],
        val_doc_limit=config_dict["val_doc_limit"],
        token_cache_dir=config_dict["token_cache_dir"],
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

        r = runner.run_experiment(params, experiment_id=exp_id, selected_idx=selected_idx)
        result_queue.put(r)
        completed += 1

        # Clean up temporary token file
        exp_token_path = runner._get_exp_token_path(exp_id)
        if os.path.exists(exp_token_path):
            os.remove(exp_token_path)
            print(f"[Worker {worker_id}] Cleaned temp file for exp {exp_id}")

    result_queue.put(None)  # Signal completion to collector


def test_runner_sharded():
    """Quick test: 2 experiments using sharded metadata manager."""
    from quadmix import QuaDMixConfig
    from quadmix.pipeline.param_sampler import ParameterSampler
    from quadmix.data.metadata_manager import ShardMetadataManager

    mgr = ShardMetadataManager(
        os.path.join(_PROJECT_DIR, "temp/preprocessed")
    )

    config = QuaDMixConfig(
        num_domains=10, num_quality_criteria=5, num_proxy_experiments=2
    )

    runner = EssentialWebProxyRunner(
        config=config,
        metadata_manager=mgr,
        val_data_path=os.path.join(_PROJECT_DIR, "data/openhermes_10k_assistant_tokenized.pt"),
        output_dir=os.path.join(_PROJECT_DIR, "temp/outputs/test_sharded"),
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
    print(f"  Test Complete! ({time.time()-t0:.1f}s)")
    for r in results:
        print(f"  Exp {r.metadata['experiment_id']:04d}: "
              f"val_loss={r.validation_loss:.4f}")
    runner.save_summary(results, os.path.join(runner.output_dir, "test_summary.json"))
    print("=" * 60)


if __name__ == "__main__":
    test_runner_sharded()
