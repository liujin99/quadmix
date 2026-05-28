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
│   │   └── shard_{sid}_bs{bs}.lock    # Lock files (tiny)
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

#### Concurrent Access: File Lock Protection (Immediate Write)

**Problem**: Tokenize Thread writes to shard cache. If multiple writes happen to same shard:
- Thread A creates `.tmp` file
- Thread B overwrites `.tmp` file
- Thread A's rename fails (`.tmp` no longer exists)

**Solution**: File lock (`fcntl.flock()`) + unique temp file name.

```
Tokenize Thread
      │
      ├─ _cache_add_rows(sid, rows, tokens)
      │     │
      │     ├─ Acquire fcntl.flock(lock_path, LOCK_EX)
      │     │
      │     ├─ Read existing shard cache (if any)
      │     │
      │     ├─ Merge new rows → deduplicate
      │     │
      │     ├─ Write to .tmp.{timestamp} (unique name)
      │     │
      │     ├─ os.replace(.tmp, cache.npz) — atomic
      │     │
      │     └─ Release lock
      │
      ▼
shard_cache.npz (immediately visible to next exp)
```

**Key insight**: Current batch experiments need to see each other's tokenize results.
- Exp 0 tokenizes docs → shard cache updated immediately
- Exp 1 needs same docs → `_cached_shard_rows()` finds them → cache hit
- Async merge would break this: pending files not visible to same batch

**Architecture flow**:

```
T+0s      Exp 0: tokenize 500 docs from shard_00000
          ├─ _cache_add_rows() → shard_00000.npz (immediate)
          └─ Pack exp_0000_tokens.pt
          
T+10s     Exp 1: needs 300 docs from shard_00000 (200 overlap with Exp 0)
          ├─ _cached_shard_rows() → finds 500 cached rows
          ├─ Cache hit: 200 docs read from shard_00000.npz
          ├─ Cache miss: 100 new docs → tokenize + _cache_add_rows()
          └─ Pack exp_0001_tokens.pt
          
T+20s     shard_00000.npz now has 600 cached rows
          Future exps benefit from accumulated cache
```

**Performance impact**:

| Metric | Value |
|--------|-------|
| Lock contention | Rare (single Tokenize Thread) |
| Lock wait time | ~0.5s if contention happens |
| vs Tokenization time | ~100s per exp (negligible) |

**Disk usage**:
- Shard cache: Persistent, grows as experiments run
- Temp files: `.tmp.{timestamp}` deleted after atomic replace
- Lock files: `.lock` text files (tiny)

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
| Shard cache write | Immediate (no lock needed) | Immediate (file lock protection) |
| IO overhead | One-time union | Per-exp on-demand |
| Lock contention | N/A | Rare (single thread) |
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
         ├─ For each exp: tokenize miss rows → _cache_add_rows() (immediate)
         ├─ shard_cache.npz visible to subsequent exps
         └─ Pack exp_{id}_tokens.pt for workers
         Worker 0-7: waiting for tasks
         
T+110s   Dispatcher: push exp 0-7 to task_queue
         Worker 0-7: start exp 0-7 training
         Tokenize Thread: continue exp 8-15
         ├─ Exp 8 may have cache hits from Exp 0-7
         └─ Same docs already tokenized → reuse
         
T+180s   Worker 3: exp 3 done → result_queue → push exp 8 to Worker 3
         Worker 0-2,4-7: still training (different exp sizes)
         
T+200s   Worker 3: exp 8 done → fetch exp 9
         (fast worker does more experiments)
         
T+Batch  All experiments complete
         shard_cache accumulated for future runs
         
No batch boundaries — each worker independently fetches next task
Shard cache progressively filled — later exps benefit from earlier
File lock ensures atomic writes (minimal contention)
```

### Token Cache Sharing
- All workers share `temp/token_cache/` on the same filesystem
- `.npz` writes are atomic (temp file + rename on POSIX)
- **Immediate write**: `_cache_add_rows()` updates shard cache directly
- **File lock**: `fcntl.flock()` ensures atomic merge when concurrent writes happen
- Shard cache progressively filled — later experiments benefit from earlier
- Cache persists for subsequent runs

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

### Why File Lock for Shard Cache Write?

**Key insight**: Current batch experiments need to see each other's tokenize results.

```
Exp 0 tokenizes docs → shard_cache.npz updated
Exp 1 needs same docs → _cached_shard_rows() finds them → cache hit
```

If we used async merge (pending files):
- Exp 0 writes to `.pending.0.npz`
- Exp 1 checks shard cache → not found → cache miss
- Exp 1 re-tokenizes same docs (wasted work)

**Solution**: Immediate write with file lock.
- `fcntl.flock()` ensures atomic merge
- Unique temp file `.tmp.{timestamp}` prevents collision
- `os.replace()` atomic rename

**Lock contention is minimal**:
- Single Tokenize Thread (no concurrent writes in current architecture)
- Only relevant for future multi-thread tokenize
- Lock wait (~0.5s) is negligible vs tokenization time (~100s)

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

## NPU Parallel Mode Optimization (Implemented)

### Current Architecture (Problem) — RESOLVED

```
Time    Tokenize Thread                Workers (NPU)
─────────────────────────────────────────────────────
T+0s    开始处理 batch 0-7              等待中...
        ├─ tokenize shard_0 miss rows
        ├─ 写 cache npz ← 磁盘IO ~2s    ← NPU 空转!
        ├─ tokenize shard_1 miss rows
        ├─ 写 cache npz ← 磁盘IO ~2s    ← NPU 空转!
        ├─ ... (重复 100 shards)
        ├─ 打包 exp_0_tokens.pt
        ├─ 打包 exp_1_tokens.pt
        ├─ ...
        ├─ 打包 exp_7_tokens.pt
        └─ 8 个 exp 全部 ready
T+120s  
        Dispatcher 推送任务              Workers 启动训练
                                        ├─ Worker 0: exp 0
                                        ├─ Worker 1: exp 1
                                        └─ ...

T+180s  开始处理 batch 8-15             Workers 还在训练 exp 0-7
        ├─ tokenize + 写 cache
        ├─ 打包 exp_8_tokens.pt
        ├─ ... 等 batch 8-15 全部完成
T+300s  
        8 个 exp 全部 ready              Workers 可能已完成 exp 0-7
                                        但没有新任务 ← NPU 空转!
        Dispatcher 推送 batch 8-15      Workers 继续训练
```

**Problems**:
1. **Cache write blocks tokenize**: Each `_cache_add_rows()` ~2s disk IO, NPU idle
2. **Batch boundary blocks dispatch**: Workers wait for entire batch to tokenize
3. **Repeated IO**: Same shard read multiple times across experiments

### Optimized Architecture — NOW ACTIVE

```
Time    Tokenize Thread                Workers (NPU)         AsyncWrite Thread
──────────────────────────────────────────────────────────────────────────────
T+0s    开始处理 batch 0-7              等待中...
        ├─ 计算 union miss rows
        ├─ 批量读取 parquet (每 shard 一次)
        ├─ 批量 tokenize union
        ├─ 存入 memory_cache
        ├─ 批量打包 8 个 exp_tokens.pt
        └─ 8 个 exp 全部 ready
T+60s  
        Dispatcher 推送任务              Workers 启动训练      后台启动
                                        ├─ Worker 0: exp 0    ├─ 写 cache npz
                                        ├─ Worker 1: exp 1    ├─ shard_0...
                                        └─ ...                └─ (不阻塞)

T+90s   开始处理 batch 8-15             Workers 训练中        写 cache 继续...
        ├─ 计算 union miss rows
        ├─ 检查 memory_cache → 已有部分
        ├─ 只读取真正 miss (减少 IO)
        ├─ 批量 tokenize 新 miss
        ├─ 合并入 memory_cache
        ├─ 批量打包 8 个 exp_tokens.pt
        └─ batch 8-15 全部 ready
        
        Dispatcher 推送 batch 8-15     Workers 继续          写 cache 继续...
                                        (单个 exp done 就拿新任务)

T+120s  开始处理 batch 16-23            Workers 继续          写 cache 完成
        ...                             ← 无 batch 边界限制!
```

**Key improvements**:
1. **Batch tokenize → no repeated IO**: Each shard read once per batch
2. **Memory cache → no disk block**: Tokenize from RAM, cache written async
3. **AsyncWrite → NPU not idle**: Workers start before cache write finishes
4. **No batch boundary**: Single exp ready → dispatch immediately

### Memory Cache Design

```python
# Structure: Dict[shard_id -> {rows: Array, tokens: Array}]
memory_cache: Dict[int, dict] = {
    shard_id: {
        'rows': np.ndarray[int64],    # [N] row_in_shard, sorted
        'tokens': np.ndarray[int32],  # [N, block_size]
    }
}

# Query: use np.searchsorted for O(log N) lookup
def get_tokens_from_memory(sid, rows_needed):
    shard_cache = memory_cache.get(sid)
    if not shard_cache:
        return None
    positions = np.searchsorted(shard_cache['rows'], rows_needed)
    return shard_cache['tokens'][positions]

# Merge: concatenate + dedupe + sort
def merge_to_memory_cache(sid, new_rows, new_tokens):
    if sid not in memory_cache:
        memory_cache[sid] = {'rows': new_rows, 'tokens': new_tokens}
    else:
        combined_rows = np.concatenate([memory_cache[sid]['rows'], new_rows])
        combined_tokens = np.concatenate([memory_cache[sid]['tokens'], new_tokens])
        unique_rows, idx = np.unique(combined_rows, return_index=True)
        memory_cache[sid] = {'rows': unique_rows, 'tokens': combined_tokens[idx]}
```

**Why this structure**:
- `np.searchsorted` faster than dict lookup for batch queries
- Contiguous array → no memory fragmentation
- Same structure as disk cache → code reuse

### Batch Processing Flow

```python
def tokenize_batch_optimized(batch_ids, batch_selected):
    # Step 1: Compute union miss rows
    union_miss = compute_union_miss(batch_selected, memory_cache, disk_cache)
    
    # Step 2: Batch read + tokenize (each shard once)
    for sid, miss_rows in union_miss.items():
        texts = read_parquet_rows(sid, miss_rows)
        tokens = tokenize_texts(texts)
        merge_to_memory_cache(sid, miss_rows, tokens)
    
    # Step 3: Batch pack exp_tokens.pt
    for exp_id, selected in zip(batch_ids, batch_selected):
        pack_exp_from_memory(exp_id, selected)
        ready_events[exp_id] = True
    
    # Step 4: Queue async write (non-blocking)
    async_write_queue.put(union_miss)
    
    # Step 5: Immediately start next batch tokenize
    # (Workers already running from previous batch)
```

### AsyncWrite Thread

```python
def async_write_thread_func():
    while True:
        union_miss = async_write_queue.get()
        if union_miss is None:
            break
        for sid, miss_rows in union_miss.items():
            rows = memory_cache[sid]['rows']
            tokens = memory_cache[sid]['tokens']
            _cache_add_rows(sid, rows, torch.from_numpy(tokens))
        # After write, can optionally clear memory_cache entries
```

**Key insight**: Cache write benefits future batches/runs, not current batch.
- Current batch uses memory_cache (already tokenized)
- Disk cache written in background (parallel with training)
- No blocking, no NPU idle time

### Dispatcher: No Batch Boundary

```python
# Current: wait for batch boundary
while pos < batch_end:
    if ready_events[pos]:
        task_queue.put(pos)
        pos += 1
    else:
        wait()

# Optimized: dispatch immediately when single exp ready
while pos < n_exp:
    if ready_events[pos]:
        task_queue.put(pos)
        pos += 1
    else:
        wait(0.02)  # Short poll
```

**Effect**: Worker finishes exp 3 → immediately gets exp 8 (no wait for batch 8-15)

### Expected Performance Improvement

| Metric | Current | Optimized | Improvement |
|--------|---------|-----------|-------------|
| Tokenize time per batch | ~120s (8× disk IO) | ~60s (no disk IO) | **2x** |
| NPU idle time | ~120s per batch | ~0s | **∞** |
| Parquet reads | 8× per shard | 1× per shard | **8x** |
| Total batch time | ~180s | ~90s | **2x** |

**Estimated**: 3000 exp from ~15h → **~8h** on 8x NPU

| Optimization | Before | After | Improvement |
|--------------|--------|-------|-------------|
| Eq.1 pre-normalize | ~20s/exp | ~4s/exp | **5x** |
| 3000 exp precompute | ~500 min | ~100 min | **5x** |
| Shard cache write | N/A | Immediate + file lock | Safe concurrent |
| 8x NPU total time | ~250h (single) | ~15h | **16x** |

**Combined effect**: Production 3000-exp run from ~250h → ~15h on 8x NPU.
