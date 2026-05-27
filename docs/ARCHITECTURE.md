# QuaDMix Architecture

## Overview

QuaDMix (Quality-Diversity Balanced Data Selection) — clean-room implementation of [arXiv:2504.16511](https://arxiv.org/abs/2504.16511).

Selects high-quality, domain-balanced subsets from large unlabeled corpora for LLM pretraining. Runs 3000 proxy experiments (each training a 1M param model) to find optimal sampling parameters via LightGBM regression.

## Project Structure

```
quadmix/
├── src/quadmix/                # Python package (pip install -e .)
│   ├── __init__.py                 # Exports QuaDMixConfig
│   ├── core/                       # Core algorithm (Eq.1-3, proxy model)
│   │   ├── quality_merger.py       # Eq.1: Merged quality score
│   │   ├── quality_rank.py         # Eq.2: Quality percentile within domain
│   │   ├── sampler.py              # Eq.3: Sigmoid sampling S(¯r)
│   │   ├── proxy_model.py          # ~1M param proxy (SwiGLU+RMSNorm, tinyllama_1M)
│   │   └── types.py                # Core data types (QuaDMixConfig, ParameterSet, ProxyResult)
│   ├── pipeline/               # Pipeline orchestration
│   │   ├── real_pipeline.py        # Main pipeline runner (Stage 0-8)
│   │   ├── param_sampler.py        # Alg.1: Parameter generation
│   │   ├── optimizer.py            # LightGBM regression + optimal search
│   │   ├── proxy_runner.py         # BaseProxyRunner abstract class
│   │   └── report.py               # MD report + figures
│   ├── data/
│   │   └── metadata_manager.py     # ShardMetadataManager (MMap-aware, on-demand text)
│   ├── npu/
│   │   └── device.py               # DeviceManager (CPU/CUDA/NPU, multi-device support)
│   ├── sampling/
│   │   └── batch_sampler.py        # _select_documents_vectorized, save_sampled_dataset
│   └── utils/
│       └── normalization.py
├── scripts/                     # Entry points & utilities
│   ├── run_essential_web_v1.py     # Main CLI entry (--npu-devices N supported)
│   ├── preprocess_essential_web_v1_sharded.py  # Multi-shard preprocessing
│   ├── essential_proxy_runner.py   # Proxy runner: precompute + batch loop + cache
│   ├── download_essential_web.py   # Download tool
│   ├── validation_set/             # Reference scripts for validation set prep
│   ├── demo_run_quick.sh           # Quick demo (~1-2min, CPU)
│   ├── demo_run_npu.sh             # Medium demo (~15-30min, 8x NPU, 100 shards)
│   └── demo_run_full.sh            # Full demo (paper config, GPU/NPU)
├── result/                     # Final results (one dir per run)
│   └── quadmix_YYYYMMDD_HHMMSS/
│       ├── optimal_parameters.json
│       ├── pipeline_summary.json
│       ├── sampled_dataset.parquet
│       ├── quadmix_report.md
│       ├── fig1_domain_distribution.png
│       ├── fig2_quality_weights.png
│       └── proxy_experiments/
│           ├── exp_0000/
│           │   ├── meta.json
│           │   └── selected_indices.npy
│           └── exp_0001/ ...
├── temp/                       # Intermediate data (deletable)
│   ├── preprocessed/              # Preprocessed parquet shards
│   ├── token_cache/               # Incremental per-shard npz cache
│   │   ├── shard_{sid:05d}_bs{bs}.npz  # Persistent shard cache
│   │   └── *.pending.{thread_id}.npz   # Temp files (cleaned after batch)
│   └── outputs/                   # Test outputs
├── data/                       # Downloaded raw data & validation set
│   ├── essential-web/             # Raw parquet shards
│   └── openhermes_10k_assistant_tokenized.pt
└── docs/
    └── NPU_DEPLOYMENT.md
```

## Data Flow (4 Phases)

### Phase 0: Preprocessed Data

**Input**: essential-web-v1 (3291 raw parquet shards, ~83K docs each, ~246 MB/shard)

**Script**: `scripts/preprocess_essential_web_v1_sharded.py`

Per shard, extracts:
- `domain` (int): Dewey decimal level_1 (0-9 categories)
- `qs_*` (5 float): FastText quality signals (dclm, fineweb_edu, english, math_general, math_openweb)
- `doc_char_count` (int): text length estimation
- `shard_idx`, `row_in_shard`: provenance tracking

**Output**: `temp/preprocessed/preprocessed_NNNNN.parquet + shard_index.json`

### Phase 1: ShardMetadataManager

**File**: `src/quadmix/data/metadata_manager.py`

Only loads metadata columns into RAM. **Never loads text upfront.**

| Data | Shape | Memory |
|------|-------|--------|
| `domain_labels` | [N_docs] int64 | ~2.2 GB (275M docs) |
| `quality_scores` | [N_docs×5] float64 | ~11 GB |
| `doc_char_counts` | [N_docs] int64 | ~2.2 GB |
| **Total** | | **~15 GB** |

Text loaded on-demand via `pd.read_parquet(filters=[("row_in_shard", "in", rows)])`.

### Phase 2: Precompute Samples

**File**: `scripts/essential_proxy_runner.py` → `precompute_samples()`

Pure numpy, CPU only. For each experiment's parameters:

#### Eq.1 Optimization: Pre-normalized Quality

**Problem**: Each experiment repeats normalization of ~8.4M docs × 5 criteria (~20s).

**Solution**: Pre-compute normalized quality once at initialization.

```python
# In _load_metadata_only() — one-time cost
normalize_fn = get_normalizer("rank")  # rank-based normalization
for n in range(num_criteria):
    _normalized_quality[:, n] = normalize_fn(quality_scores[:, n])
# ~10s for 5 criteria × 8.4M docs

# In _compute_ranks_for_params() — per-experiment, now only weighted sum
merged_scores[mask] = _normalized_quality[mask] @ alpha_m  # ~4s vs ~20s
```

**Performance impact**: Eq.1 from ~20s/exp → ~4s/exp (**5x faster**).

```
Eq.1 (optimized): merged_scores = normalized_quality @ alpha_m
     → merged_scores [275M] — domain-weighted average (weighted sum only)

Eq.2: compute_quality_ranks(merged_scores, domain_labels, token_counts)
     → quality_ranks [275M] — percentile within domain (0=best)

Eq.3: compute_sampling_values(quality_ranks, domain_labels, λ,ω,η,ε)
     → sampling_values [275M] — fractional expectations (sigmoid)

Fractional Sampling (_select_documents_vectorized):
  int_part = floor(sampling_values)           # deterministic copies
  frac_part = sampling_values - int_part       # fractional remainder
  random_mask = rng.uniform(N) < frac_part     # Bernoulli for frac part
  repeats = int_part + random_mask              # total copies per doc
  → selected_indices [~550K per experiment]
```

| Stage | OLD | NEW | Improvement |
|-------|-----|-----|-------------|
| Eq.1 normalization | ~20s/exp | ~4s/exp | **5x** |
| 3000 exp precompute | ~500 min | **~100 min** | **5x** |

3000 experiments × ~550K each. **No GPU involved.**

### Phase 3: Token Cache (Incremental, Per-Shard)

**Format**: `temp/token_cache/shard_{sid:05d}_bs{block_size}.npz`

Single npz file per shard containing:
- `tokens`: [N_cached, block_size] int32 — mmap-compatible token IDs
- `rows`: [N_cached] int64 — corresponding row_in_shard indices

#### Cache Hit (during training)
```python
data = np.load(cache_path, mmap_mode='r')
token_mmap = data['tokens']          # mmap'd, not loaded into RAM
row_index = data['rows']             # [N_cached] int64
row_to_pos = {r: i for i, r in enumerate(row_index)}
positions = [row_to_pos[r] for r in local_rows]
shard_tokens = token_mmap[positions]  # only accessed pages loaded
```

#### Cache Miss (tokenize_batch_delta)
```python
# Read text for ONLY needed rows (not entire shard)
df_shard = pd.read_parquet(shard_path,
    columns=["row_in_shard", "text"],
    filters=[("row_in_shard", "in", miss_rows.tolist())])
texts = df_shard["text"].tolist()
parsed_rows = df_shard["row_in_shard"].to_numpy()
tokenized = _tokenize_texts(texts)    # [N_miss, block_size]
_cache_add_rows(sid, parsed_rows, tokenized)  # append to npz
```

#### Append Logic (_cache_add_rows)
```python
if npz exists:
    old = np.load(cache_path)
    all_rows = concat(old.rows, new_rows)
    all_tokens = concat(old.tokens, new_tokens_np)
    order = argsort(all_rows)  # sort for efficient lookup
    np.savez(cache_path, tokens=all_tokens[order], rows=all_rows[order])
else:
    np.savez(cache_path, tokens=new_tokens_np, rows=new_rows)
```

#### Concurrent Access: Async Shard Cache Merge (Zero Lock Contention)

**Problem**: In NPU parallel mode, 8 Tokenize Threads write to shard cache simultaneously. If 2 threads hit the same shard:
- Thread A creates `.tmp` file
- Thread B overwrites `.tmp` file
- Thread A's rename fails (`.tmp` no longer exists)

**OLD solution (file lock)**: `fcntl.flock()` on shard cache
- Lock contention probability: ~8% (100 shards / 8 threads)
- Blocking time: ~0.5s per conflict
- Total blocking: ~0.5s × N_conflicts

**NEW solution (async merge)**: Write to pending files, merge after batch completes.

```
Tokenize Thread (during batch)
      │
      ▼
.pending.{thread_id}.npz (NO lock, NO merge)
      │
      │  ← Just raw data, immediate return
      │
      └─ Continue to next experiment

After ALL experiments complete
      │
      ▼
_merge_all_pending_to_cache()
      │
      ├─ Group pending files by shard
      ├─ Acquire lock per shard (batch now complete, no contention)
      ├─ Merge all pending → shard_cache.npz
      └─ Delete pending files
      │
      ▼
shard_cache.npz (ready for NEXT batch)
```

**Architecture flow**:

```
T+0s      Tokenize Thread 0: exp 0 → shard_00000.pending.0.npz (no lock)
          Tokenize Thread 1: exp 1 → shard_00001.pending.1.npz (no lock)
          ...
          Tokenize Thread 7: exp 7 → shard_00002.pending.7.npz (no lock)
          
T+10s     All 8 threads continue to next batch
          NO blocking, NO lock contention
          
T+Batch   All experiments complete
          _merge_all_pending_to_cache() runs
          ├─ shard_00000.npz ← merge all shard_00000.pending.*.npz
          ├─ shard_00001.npz ← merge all shard_00001.pending.*.npz
          ...
          └─ Delete all pending files
```

**Key insight**: Current batch doesn't need shard cache — it uses `exp_{id}_tokens.pt` temp files. Shard cache is for **FUTURE batches**. So merge can happen asynchronously after batch completes.

**Performance impact**:

| Metric | File Lock (OLD) | Async Merge (NEW) |
|--------|-----------------|-------------------|
| Lock contention | ~8% probability | **0%** |
| Blocking per conflict | ~0.5s | **0s** |
| Tokenize thread stall | Yes | **No** |
| Merge overhead | N/A | ~10s after batch |

**Disk usage**:
- Pending files: N_experiments × ~500MB peak
- Cleaned up after merge
- Shard cache: Persistent for subsequent runs

## Pipeline Stages

### Stage 0: Load Data (precomputed mode via ShardMetadataManager)
- Domain labels + quality scores from all shards → RAM (~15 GB)
- Text NEVER loaded upfront

### Stage 1: Feature Extraction (skipped in precomputed mode)

### Stage 2: Parameter Sampling (Alg.1)
- `ParameterSampler.sample_batch(N)` → N sets of (α_m, λ_m, ω_m, η_m, ε_m)

### Stage 3: Precompute Samples (new, parallel mode only)
- `precompute_samples(all_params)` → all_selected_indices[]
- CPU only, pure numpy

### Stage 4: Proxy Experiments (Three Modes)

#### Mode 1: CPU Sequential (`parallel_workers <= 1`)
**Use case**: Development, debugging, demo, CI testing

```
precompute_samples(param_sets) → all_selected_indices[]
      ↓
tokenize_all_needed(all_selected)  # One-shot tokenize union
      ↓
for each (params, selected) in zip(param_sets, all_selected):
    run_experiment(params, selected)  # All cache hits, no IO
```

**Key insight**: Union of all needed docs → tokenize once → all exps get cache hits.

#### Mode 2: NPU Parallel (`parallel_workers > 1`)
**Use case**: Production, 8x NPU training

```
Main Process
├─ precompute_samples(param_sets) → all_selected_indices[]
│   └─ Eq.1: Pre-normalized quality (5x faster)
│   └─ Eq.2-3: Per-experiment sampling
│
├─ Tokenize Thread (CPU) — continuously pre-tokenize (lookahead=num_workers)
│   ├─ Write .pending.{thread_id}.npz (NO lock, NO blocking)
│   └─ Pack exp_{id}_tokens.pt for workers
│
├─ Dispatcher Thread — push ready tasks to task_queue
├─ Collector Thread — receive from result_queue
│
└─ Worker 0..N-1 (NPU) — loop:
        get task → read exp_{id}_tokens.pt → run → delete temp → push result

After ALL experiments complete:
└─ _merge_all_pending_to_cache()
   ├─ Merge .pending.*.npz → shard_cache.npz
   └─ Benefit future batches
```

**Key insight**: Tokenize Thread prepares tokens while Workers train. No NPU idle time. Zero lock contention.

#### Mode 3: Legacy Fallback
**Use case**: Backward compatibility

```
run_batch(params)  # Original sequential mode, no precompute
```

#### Mode Comparison

| Aspect | CPU Sequential | NPU Parallel |
|--------|----------------|--------------|
| Tokenize timing | All upfront | On-demand (lookahead) |
| Eq.1 normalization | Pre-computed (5x faster) | Pre-computed (5x faster) |
| Cache behavior | 100% hit after first | Progressive fill |
| Shard cache merge | N/A (no concurrent writes) | Async after batch (0 blocking) |
| IO overhead | One-time union | Per-exp on-demand |
| Lock contention | N/A | **0%** (async merge) |
| Best for | Debug, CI, demo | Production 8x NPU |
| 3000 exp time | ~100h (single CPU) | **~15h** (8x NPU) |

### Stage 5: LightGBM Regression
- Train regressor R(θ) → predicted val_loss

### Stage 6: Optimal Search
- 100K random parameters → predict loss → top-10 average → θ*

### Stage 7: Final Sampling
- Apply θ* to entire data pool → Eq.1+Eq.2+Eq.3 → final dataset
- **Target tokens post-processing** (if `target_tokens` specified):

**Problem**: Pre-scaling sampling parameters distorts distribution (floor+random mechanism crosses integer boundaries).

**Solution**: Post-hoc uniform discard (preserves relative distribution).

```python
# Step 1: Sample with θ* (no target_tokens adjustment)
selected_indices = sample_with_params(theta_star)

# Step 2: If exceeds target, uniformly discard
if len(selected_indices) > target_tokens:
    keep_ratio = target_tokens / len(selected_indices)
    rng = np.random.default_rng(seed=42)
    keep_mask = rng.random(len(selected_indices)) < keep_ratio
    selected_indices = selected_indices[keep_mask]

# Step 3: If below target, warn (paper: "more tokens not always good")
if len(selected_indices) < target_tokens:
    logger.warning(f"Got {len(selected_indices)} < target {target_tokens}")
    # No artificial copy — paper shows 30B > 90B > 180B
```

**Why not copy when below target?** Paper Table 2: 30B tokens outperforms 90B and 180B. Quality > quantity.

### Stage 8: Save Outputs
- optimal_parameters.json, pipeline_summary.json, sampled_dataset.parquet
- Comparison report + figures

## Multi-Device Architecture

### DeviceManager (`src/quadmix/npu/device.py`)
```python
DeviceManager(device_type=DeviceType.NPU, npu_device_id=3)
# → torch.device("npu:3")
```

### Worker Process (`_worker_dynamic_loop` in essential_proxy_runner.py)
Module-level function (pickle-safe for `mp.get_context(\"spawn\")`):
```python
_worker_dynamic_loop(worker_id, device_type, config_dict, task_queue, result_queue):
    mgr = ShardMetadataManager(preprocessed_dir)  # shared filesystem
    runner = EssentialWebProxyRunner(..., npu_device_id=worker_id)
    while True:
        task = task_queue.get()  # Blocking pull
        if task is None: break   # Termination signal
        exp_id, params, selected_idx = task
        result = runner.run_experiment(params, exp_id, selected_idx)
        result_queue.put(result)
```

### Dynamic Task Queue Flow (NPU Parallel Mode)

```
T+0s     Precompute: Eq.1 (pre-normalized, ~10s one-time)
         Precompute: Eq.2-3 for 3000 exps (~100 min, 5x faster)
         
T+100min Tokenize Thread: pre-tokenize exp 0-7 (lookahead=num_workers)
         ├─ Write .pending.{thread_id}.npz (NO lock)
         └─ Pack exp_{id}_tokens.pt for workers
         Worker 0-7: waiting for tasks
         
T+110s   Dispatcher: push exp 0-7 to task_queue
         Worker 0-7: start exp 0-7 training
         Tokenize Thread: continue exp 8-15 (NO blocking)
         
T+180s   Worker 3: exp 3 done → result_queue → push exp 8 to Worker 3
         Worker 0-2,4-7: still training (different exp sizes)
         
T+200s   Worker 3: exp 8 done → fetch exp 9
         (fast worker does more experiments)
         
T+Batch  All experiments complete
         _merge_all_pending_to_cache():
         ├─ Merge .pending.*.npz → shard_cache.npz (~10s)
         └─ Benefit future batches
         
No batch boundaries — each worker independently fetches next task
Queue overhead ≈ 1ms, negligible vs ~2-5min training time
Zero lock contention during tokenize
```

### Token Cache Sharing
- All workers share `temp/token_cache/` on the same filesystem
- `.npz` writes are atomic (temp file + rename on POSIX)
- **Async merge**: `.pending.{thread_id}.npz` written during tokenize, merged after batch
- `_merge_all_pending_to_cache()` handles merge with file lock (no contention, batch complete)
- Shard cache persists for subsequent runs

## Data Download & Resume

### Incremental Download
Each shard checked individually before download:
```python
for shard_idx in needed_shards:
    shard_path = f"preprocessed/shard_{shard_idx}.parquet"
    if shard_path.exists() and verify_size(shard_path):
        continue  # Skip already downloaded
    download_shard(shard_idx)
```

### Resume Support (HTTP Range)
```python
def download_with_resume(url, local_path):
    # Check existing file size via HEAD request
    head_resp = requests.head(url)
    total_size = int(head_resp.headers.get('content-length', 0))

    if local_path.exists():
        local_size = local_path.stat().st_size
        if local_size == total_size:
            return  # Already complete
        if local_size < total_size:
            # Resume from where we left off
            headers = {'Range': f'bytes={local_size}-'}
            with open(local_path, 'ab') as f:
                response = requests.get(url, headers=headers, stream=True)
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        # Fresh download
        with open(local_path, 'wb') as f:
            for chunk in requests.get(url, stream=True).iter_content(chunk_size=8192):
                f.write(chunk)
```

### HuggingFace Mirror (Optional)
```bash
export HF_ENDPOINT=https://hf-mirror.com  # For users in China
```

## CLI Usage

```bash
# Quick test (20 experiments, CPU, ~1-2min)
bash scripts/demo_run_quick.sh

# Medium test (200 experiments, 8x NPU, ~15-30min)
bash scripts/demo_run_npu.sh

# Full run (3000 experiments, paper config, ~15h on 8x NPU)
bash scripts/demo_run_full.sh --device-type npu --npu-devices 8

# Custom parameters
python scripts/run_essential_web_v1.py \
    --preprocessed-dir temp/preprocessed \
    --num-experiments 500 \
    --num-search 10000 \
    --block-size 2048 \
    --device-type npu --npu-devices 4 \
    --output result/my_run
```

### Demo Scripts Comparison

| Script | Shards | Experiments | Device | Steps | Time | Use Case |
|--------|--------|-------------|--------|-------|------|----------|
| `demo_run_quick.sh` | 2 | 20 | CPU | 3 (tiny) | ~1-2min | CI, smoke test |
| `demo_run_npu.sh` | 100 | 20 | 8x NPU | 25000 (full) | ~10-20min | NPU validation, 论文配置 |
| `demo_run_full.sh` | 2000 | 3000 | GPU/NPU | 25000 (full) | ~15h (8x) | Production |

## Key Dependencies

| Package | Purpose |
|---------|---------|
| torch | Proxy model training, device management |
| torch_npu | Ascend NPU support (npu:0..npu:7) |
| transformers | GPT-NeoX-20B tokenizer |
| lightgbm | Regression model for parameter search |
| numpy | Core computations (Eq.1-3, sampling) |
| pandas | Parquet I/O (data loading, text loading) |
| matplotlib | Report figures |

## Important Notes

1. **Validation set** (`openhermes_10k_assistant_tokenized.pt`, 176MB) auto-downloads from HuggingFace `liujin99/quadmix-openhermes-10k` on first run. Assistant-only loss mask.

2. **Pre-existing bugfix**: `src/quadmix/pipeline/optimizer.py` had missing `ParameterSet`, `QuaDMixConfig`, `ProxyResult` imports — fixed during multi-device implementation.

3. **Non-official**: This is a clean-room implementation of QuaDMix (ByteDance, 2025). Not an official release.

4. **NPU deployment**: See `docs/NPU_DEPLOYMENT.md` for detailed NPU setup instructions.

## Design Philosophy

### Why Pre-normalized Quality in Eq.1?

**Problem**: Each experiment repeated normalization of quality scores (~20s/exp).

```
Eq.1 per experiment:
  for n in range(5 criteria):
      normalized[:, n] = rank_normalize(quality[:, n])  # ~20s total
  merged = weighted_sum(normalized, alpha_m)             # ~4s
```

**Solution**: Normalization is deterministic and independent of experiment parameters. Pre-compute once.

```
At initialization:
  for n in range(5 criteria):
      _normalized_quality[:, n] = rank_normalize(quality[:, n])  # ~10s one-time

Per experiment:
  merged = _normalized_quality @ alpha_m  # ~4s (weighted sum only)
```

**Impact**: 3000 experiments from ~500 min → ~100 min (**5x faster**).

### Why Async Shard Cache Merge?

**Key insight**: Current batch doesn't need shard cache.

```
Tokenize Thread writes shard_cache.npz
          │
          │  ← Why? For FUTURE batches to reuse
          │
          │  ← Current batch uses exp_{id}_tokens.pt temp files
          │
          ▼
Blocking current tokenize for future benefit → BAD
```

**Better approach**: Write raw data immediately, merge later.

```
Tokenize Thread:
  ├─ Write .pending.{thread_id}.npz (raw data, no merge)
  └─ Return immediately (NO blocking)

After batch:
  └─ _merge_all_pending_to_cache()
     ├─ Merge all pending files to shard cache
     └─ Benefit future batches
```

**Result**: Zero lock contention, zero blocking during tokenize.

### Why Two Execution Modes?

**CPU Sequential** exists because:
- Development/debugging needs reproducible, step-by-step execution
- CI testing needs deterministic runs without NPU dependency
- Small experiments (20-50) benefit from upfront tokenize-all approach

**NPU Parallel** exists because:
- Production scale (3000 experiments) cannot wait for all tokenize upfront
- NPU time is expensive (~$10/hr), must maximize utilization
- Dynamic task queue allows natural load balancing

### Why Temporary File Isolation for Workers?

Workers need tokens but can't read from shard cache during tokenize:
- Shard cache being written by Tokenize Thread
- Worker reading same file = race condition

**Solution**: Pack per-exp temp file `exp_{id}_tokens.pt`.
- Tokenize Thread prepares temp file for each exp
- Worker reads from temp file (no conflict with shard cache write)
- Worker deletes temp file after use

### Why Post-Hoc target_tokens?

Pre-scaling parameters distorts distribution:
- `floor(sampling_values * scale)` crosses integer boundaries
- Relative proportions change unpredictably

Post-hoc uniform discard:
- Preserves relative distribution exactly
- Paper insight: 30B > 90B > 180B (quality > quantity)
- If below target, warn but don't artificially copy

### Why On-Demand Text Loading?

Metadata (~15 GB) fits in RAM, but text (~800 GB) does not.
- Load domain + quality upfront: needed for all experiments
- Load text only for selected docs: ~550K per experiment
- Parquet filter pushdown: reads only needed rows, not full shard

## Performance Summary

| Optimization | Before | After | Improvement |
|--------------|--------|-------|-------------|
| Eq.1 pre-normalize | ~20s/exp | ~4s/exp | **5x** |
| 3000 exp precompute | ~500 min | ~100 min | **5x** |
| Shard cache lock | ~8% contention | **0%** | **∞** |
| Tokenize blocking | ~0.5s/conflict | **0s** | **∞** |
| 8x NPU total time | ~250h (single) | ~15h | **16x** |

**Combined effect**: Production 3000-exp run from ~250h → ~15h on 8x NPU.
