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
│   ├── demo_run_quick.sh           # Quick demo (5-15s, CPU)
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
│   │   └── shard_{sid:05d}_bs{bs}.npz
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

```
Eq.1: compute_merged_quality_scores(quality_scores, domain_labels, α_m)
     → merged_scores [275M] — domain-weighted average of 5 quality signals

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

3000 experiments × ~550K each ≈ 5-10 min CPU. **No GPU involved.**

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

### Stage 4: Proxy Experiments (Dynamic Task Queue)
**Sequential mode** (--npu-devices 1):
- `proxy_runner.run_batch(params)` → sequential experiments

**Parallel mode** (--npu-devices N):
- `proxy_runner.run_batch_parallel(params, selected, N)` — dynamic task queue:
  ```
  Main Process
  ├─ Tokenize Thread (CPU) — continuously pre-tokenize (lookahead=32)
  ├─ Dispatcher Thread — push ready tasks to task_queue
  ├─ Collector Thread — receive from result_queue
  └─ Worker 0..N-1 (NPU) — loop: get task → run → push result → repeat
  ```
- No batch boundaries — workers fetch tasks on-demand
- Fast workers naturally do more experiments, slow ones only affect themselves
- CPU tokenize thread independent of NPU training (overlap)

### Stage 5: LightGBM Regression
- Train regressor R(θ) → predicted val_loss

### Stage 6: Optimal Search
- 100K random parameters → predict loss → top-10 average → θ*

### Stage 7: Final Sampling
- Apply θ* to entire data pool → Eq.1+Eq.2+Eq.3 → final dataset
- Optional target token post-processing:
  - If θ* produces > target: uniform random discard (preserves distribution)
  - If θ* produces < target: warn, no copy (paper: "more tokens not always good")

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

### Dynamic Task Queue Flow

```
T+0s     Tokenize Thread: pre-tokenize exp 0-31 (lookahead=32)
         Worker 0-7: waiting for tasks
T+10s    Dispatcher: push exp 0-7 to task_queue
         Worker 0-7: start exp 0-7 training
         Tokenize Thread: continue exp 8-31
T+180s   Worker 3: exp 3 done → result_queue → push exp 8 to Worker 3
         Worker 0-2,4-7: still training (different exp sizes)
T+200s   Worker 3: exp 8 done → fetch exp 9
         (fast worker does more experiments)
...
No batch boundaries — each worker independently fetches next task
Queue overhead ≈ 1ms, negligible vs ~2-5min training time
```

### Token Cache Sharing
- All workers share `temp/token_cache/` on the same filesystem
- `.npz` writes are atomic (temp file + rename on POSIX)
- No file locking needed: different workers almost never miss the same shard simultaneously
- `_cache_add_rows` handles both first write and subsequent appends

## CLI Usage

```bash
# Single device (backward compatible)
bash scripts/demo_run_full.sh

# 8 NPU cards parallel
bash scripts/demo_run_full.sh --device-type npu --npu-devices 8

# 4 CUDA GPUs parallel
bash scripts/demo_run_full.sh --device-type cuda --npu-devices 4

# Quick test (2 experiments, CPU)
bash scripts/demo_run_quick.sh
```

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
