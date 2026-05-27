# QuaDMix — Quality-Diversity Balanced Data Selection

> **⚠️ Disclaimer**: This repository is a personal clean-room implementation of the QuaDMix paper,
> not an official ByteDance release. For educational and research purposes only.

> **Paper**: Fengze Liu, Weidong Zhou, Binbin Liu, et al. (ByteDance, 2025)
>
> [arXiv:2504.16511v2](https://arxiv.org/abs/2504.16511)

Selects high-quality, domain-balanced subsets from large unlabeled corpora for LLM pretraining.

## Algorithm Pipeline

```
Raw Shards (parquet) → Preprocess → Metadata (domain + quality) in memory
  ↓
Alg.1: Sample θ = (α₁…αₘ, β₁…βₘ)
  ↓
For each θ:
  Eq.1: Merge quality signals with αₘ → ¯q
  Eq.2: Rank ¯q within domain (subsample reference) → ¯r
  Eq.3: S(¯r) = sigmoid(λ(ω-¯r))^η + ε → Bernoulli sample
  Train 1M param proxy model → val_loss (assistant-only)
  ↓
LightGBM: R(θ) → predicted loss
  ↓
Search: Alg.1 × 100K → predict → top-10 average → θ*
  ↓
Eq.1+Eq.2(θ*.αₘ) → Eq.3(θ*.βₘ) → Sampled Dataset
```

## Project Structure

```
quadmix/
├── src/quadmix/                # Python package (pip install -e .)
│   ├── core/                   # Core algorithm (Eq.1-3, proxy model)
│   │   ├── quality_merger.py       # Eq.1: Merged quality score
│   │   ├── quality_rank.py         # Eq.2: Quality percentile within domain
│   │   ├── sampler.py              # Eq.3: Sigmoid sampling
│   │   ├── proxy_model.py          # ~1M param proxy (SwiGLU+RMSNorm)
│   │   └── types.py                # Core data types
│   ├── pipeline/               # Pipeline orchestration
│   │   ├── real_pipeline.py        # Main pipeline runner
│   │   ├── param_sampler.py        # Alg.1: Parameter generation
│   │   ├── optimizer.py            # LightGBM regression + optimal search
│   │   └── report.py               # MD report + figures
│   ├── data/
│   │   └── metadata_manager.py     # ShardMetadataManager (MMap-aware)
│   ├── npu/
│   │   └── device.py               # DeviceManager (CPU/CUDA/NPU)
│   └── utils/
│       └── normalization.py
├── scripts/
│   ├── run_essential_web_v1.py     # Main entry (essential-web-v1)
│   ├── preprocess_essential_web_v1_sharded.py  # Multi-shard preprocessing
│   ├── essential_proxy_runner.py   # Shard-aware proxy experiments
│   ├── download_essential_web.py   # Download tool
│   ├── validation_set/             # Validation set prep script (reference only)
│   ├── demo_run_quick.sh           # Quick demo (~1-2min, CPU)
│   ├── demo_run_npu.sh             # Medium demo (~15-30min, 8x NPU)
│   └── demo_run_full.sh            # Full demo (paper config, GPU)
├── result/                     # Final results (one dir per run)
├── temp/                       # Intermediate data (deletable)
│   ├── preprocessed/               # Preprocessed shards
│   └── token_cache/                # mmap-mapped tokenization cache
├── docs/
│   └── NPU_DEPLOYMENT.md          # NPU 部署指南
└── data/
    └── essential-web/              # Downloaded raw data
```

## Quick Start

```bash
# Install
pip install -e .

# Quick demo (20 experiments, ~1-2min, CPU) — auto-downloads validation set
bash scripts/demo_run_quick.sh

# NPU demo (200 experiments, ~15-30min, 8x NPU)
bash scripts/demo_run_npu.sh

# Full run (paper config, needs GPU/NPU)
bash scripts/demo_run_full.sh

# Custom run
python scripts/run_essential_web_v1.py \
    --preprocessed-dir temp/preprocessed \
    --num-experiments 200 \
    --num-search 100000 \
    --block-size 2048 \
    --output result/my_run
```

> **Note**: The validation set (`openhermes_10k_assistant_tokenized.pt`)
> is automatically downloaded from [HuggingFace](https://huggingface.co/datasets/liujin99/quadmix-openhermes-10k)
> on first run. No manual data preparation required.

## Architecture Highlights

### Multi-Shard Scalability
|- **Metadata in memory**: only domain labels + quality scores (12 GB for 275M docs)
|- **Text on demand**: per-shard parquet, loaded only for selected documents
|- **mmap token cache**: `np.load(path, mmap_mode='r')` — pages loaded lazily, not full file

### Multi-NPU Parallelism
|- **Dynamic task queue**: Workers fetch tasks on-demand, no batch boundaries
|- **Auto load-balancing**: Fast workers naturally do more experiments
|- **CPU-NPU overlap**: Tokenize thread runs independently, ahead of training
|- **~8x speedup**: 3000 experiments reduced from ~100h to ~15h with 8 NPUs

### Directory Layout
| Path | Content | Persistence |
|------|---------|-------------|
| `result/<experiment>/` | optimal_parameters.json, sampled_dataset.parquet, report, figures | Permanent |
| `temp/preprocessed/` | Domain labels + quality scores per shard | Keep until data changes |
| `temp/token_cache/` | mmap-able .npy files (one per shard) | Auto-regenerated |

### Data Flow
```
Raw: 3291 shards × 246 MB = ~808 GB total
  │
  ▼ [preprocess: extract domain + quality signals]
Preprocessed: 1 shard → 1 parquet (~180 MB each)
  │
  ▼ [ShardMetadataManager: in-memory metdata]
In-memory: domain(275M int64) + quality_scores(275M×5 float64)
  │
  ▼ [EssentitalWebProxyRunner: per-experiment]
Token Cache: shard_{idx}_bs{bs}.npz (mmap-mode, int32 + row_index)
  │
  ▼ [output]
Final: optimal_parameters.json + sampled_dataset.parquet
```

## Output

```
result/<experiment_name>/
├── optimal_parameters.json        # Optimal θ (α_m + β_m)
├── pipeline_summary.json          # Config + metrics + statistics
├── sampled_dataset.parquet        # Final optimally sampled dataset
├── quadmix_report.md              # Comparison report
├── fig1_domain_distribution.png   # Original vs optimal domain distribution
├── fig2_quality_weights.png       # Quality signal weights
└── proxy_experiments/             # Per-experiment results
    ├── exp_0000/
    │   ├── meta.json                # Full parameters + val_loss
    │   └── selected_indices.npy     # Selected document indices
    └── exp_0001/ ...
```

## License

Apache 2.0
