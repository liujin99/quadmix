#!/usr/bin/env python3
"""
EssentialWebProxyRunner — Real proxy training on essential-web-v1 data.

Shard-aware mode (recommended):
  Uses ShardMetadataManager → loads only metadata (domain+quality) upfront,
  reads text on-demand per experiment. Per-shard disk cache for tokens.

Legacy mode (single-file):
  Uses data_path → loads all text upfront, tokenizes all.

Aligned with RegMix:
  - GPT-NeoX-20B BPE tokenizer (same as GPT-NeoX)
  - On-demand tokenization with per-shard disk cache
  - Validation on openhermes-10k with assistant-only loss
  - RegMix training loop: gradient accumulation, cosine LR, AdamW
"""

import os, math, time, json, glob
from typing import List, Optional, Dict, Any

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
DEFAULT_TOKEN_CACHE_DIR="/home/liujin99/quadmix/temp/token_cache"


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

        # Apply doc_limit (truncate metadata + training indices)
        if self.doc_limit and self.doc_limit < self._num_docs:
            self._domain_labels = self._domain_labels[:self.doc_limit]
            self._quality_scores = self._quality_scores[:self.doc_limit]
            self._num_docs = self.doc_limit
            self._train_idx = np.arange(self._num_docs)
            print(f"[ProxyRunner] Limited to {self.doc_limit} docs")

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

    def _load_tokens_for_experiment(
        self, selected_idx: np.ndarray
    ) -> torch.Tensor:
        """
        Load or tokenize tokens for selected document indices.

        In sharded mode:
          1. Use metadata_manager to get (shard_idx, row) per selected index
          2. Group by shard, check disk cache for each shard
          3. For cache misses: read text from parquet, tokenize, save to cache
          4. Concatenate all token tensors

        In legacy mode:
          Directly index into self._token_ids (already loaded)
        """
        if self._mode == "legacy":
            return self._token_ids[selected_idx]

        # Sharded mode
        t0 = time.time()
        mgr = self.metadata_manager

        # Get per-shard groups
        shard_groups = mgr.global_to_shard_rows(selected_idx)

        all_tokens = []
        for sid, (shard_path, local_rows) in shard_groups.items():
            cache_path = self._get_shard_token_path(sid)

            if os.path.exists(cache_path):
                # Memory-mapped load: only the pages containing local_rows
                # are actually read from disk (no full file load)
                disk_tokens = np.load(cache_path, mmap_mode='r')  # [N, block_size] int32
                shard_tokens = torch.from_numpy(
                    disk_tokens[local_rows].astype(np.int64)
                )
                # Release mmap by deleting the reference
                del disk_tokens
                self._cache_hits += len(local_rows)
            else:
                # Cache miss: read text + tokenize entire shard
                self._cache_misses += len(local_rows)
                # Read all text from this shard
                import pandas as pd
                df_shard = pd.read_parquet(shard_path, columns=["text"])
                shard_texts = df_shard["text"].astype(str).tolist()

                print(f"    [Cache miss] shard {sid}: "
                      f"tokenizing {len(shard_texts):,} docs "
                      f"(need {len(local_rows)} for this exp)...")

                # Tokenize ALL docs in this shard
                shard_all = self._tokenize_texts(shard_texts)  # [N, block_size]

                # Save to disk cache (int32 for safety: vocab may exceed int16 max)
                np.save(cache_path, shard_all.numpy().astype(np.int32))
                print(f"    [Cache miss] shard {sid}: cached → {cache_path}")

                # Select the rows we need
                shard_tokens = shard_all[local_rows]

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
        self, params: ParameterSet, experiment_id: int = 0
    ) -> ProxyResult:
        """Train one proxy model. Validates on openhermes-10k."""
        from quadmix.core.proxy_model import ProxyModel
        from quadmix.npu.device import DeviceManager, DeviceType

        os.makedirs(self.output_dir, exist_ok=True)
        exp_dir = os.path.join(self.output_dir, f"exp_{experiment_id:04d}")
        os.makedirs(exp_dir, exist_ok=True)

        device_mgr = DeviceManager(device_type=DeviceType(self.device_type))
        device = device_mgr.get_device()

        # ---- 0. Compute quality ranks (Eq.1+Eq.2) ----
        quality_ranks = self._compute_ranks_for_params(params, experiment_id)
        sv = compute_sampling_values(quality_ranks, self._domain_labels, params)
        train_sv = sv[self._train_idx]
        probs = np.clip(train_sv, 0, 1)
        rng = np.random.default_rng(experiment_id + 42)
        selected = rng.uniform(size=len(self._train_idx)) < probs
        selected_idx = self._train_idx[selected]
        if len(selected_idx) < 10:
            selected_idx = rng.choice(self._train_idx, 100, replace=False)
        print(f"  [Exp {experiment_id:04d}] QuaDMix sampled {len(selected_idx)} docs "
              f"(from {len(self._train_idx):,}, {probs.mean():.3f} avg prob)")

        # ---- 1. Load / tokenize training data on demand ----
        train_tokens = self._load_tokens_for_experiment(selected_idx).to(device)

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


def test_runner_sharded():
    """Quick test: 2 experiments using sharded metadata manager."""
    from quadmix import QuaDMixConfig
    from quadmix.pipeline.param_sampler import ParameterSampler
    from quadmix.data.metadata_manager import ShardMetadataManager

    mgr = ShardMetadataManager(
        "/home/liujin99/quadmix/temp/preprocessed"
    )

    config = QuaDMixConfig(
        num_domains=10, num_quality_criteria=5, num_proxy_experiments=2
    )

    runner = EssentialWebProxyRunner(
        config=config,
        metadata_manager=mgr,
        val_data_path="/home/liujin99/data/openhermes_10k_tokenized.pt",
        output_dir="/home/liujin99/quadmix/temp/outputs/test_sharded",
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
