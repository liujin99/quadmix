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
  (RegMix-style: permutation shuffle, warmup_fraction=4%)
  ↓
LightGBM: R(θ) → predicted loss (inf/nan filtered)
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
│   ├── demo_run_cpu.sh            # CPU 快速验证 (~1-2min)
│   ├── demo_run_quick.sh          # NPU 快速验证 (~3-5min, 8x NPU)
│   └── demo_run_full.sh           # 中等规模验证 (~2-4h, GPU/NPU)
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

# Quick demo (CPU, ~1-2min) — auto-downloads validation set
bash scripts/demo_run_cpu.sh

# NPU quick demo (~3-5min, 8x NPU)
bash scripts/demo_run_quick.sh

# Medium demo (~2-4h, GPU/NPU)
bash scripts/demo_run_full.sh

# Custom run
python scripts/runners/run_essential_web_v1.py \
    --preprocessed-dir temp/preprocessed \
    --num-experiments 200 \
    --num-search 100000 \
    --block-size 2048 \
    --output result/my_run
```

> **Note**: The validation set (`openhermes_10k_assistant_tokenized.pt`)
> is automatically downloaded from [HuggingFace](https://huggingface.co/datasets/liujin99/quadmix-openhermes-10k)
> on first run. No manual data preparation required.

## Validation Status

**已验证环境：**

| 配置项 | 规格 |
|-------|------|
| NPU | 8x Ascend 910B3, 64GB VRAM each |
| 内存 | 1500 GB |
| CPU | ARM 192 vCPUs |
| 验证脚本 | `demo_run_quick.sh` (8 experiments, 5000 steps) |
| 验证结果 | ✓ RegMix 风格训练 + 全量验证集 + checkpoint trajectory |

验证日志：
- 并行 tokenize (Stage 1 IO + Stage 2 tokenize) 正常执行
- 8 workers 动态任务队列调度正常
- val_loss 计算正常（tinyllama_1M proxy model）
- Permutation shuffle 训练循环正常（无放回 epoch 遍历）
- Checkpoint trajectory 每 1000 步记录 val_loss

**验证通过的修复点：**
- `shared_to_ndarray()` 返回 `.copy()` — 解决 spawn 子进程 shared memory segfault
- `notify_all()` 移入 `with ready_cond:` — 解决信号丢失竞态
- val batch size → 16 — 避免 NPU OOM（全量 10k 验证集）
- NPU HBM 显式释放 — 实验间 `gc.collect()` + `torch.npu.empty_cache()`
- Optimizer inf/nan 过滤 — 防止非有限 val_loss 污染 LightGBM

## Demo Scripts

| 脚本 | 设备 | 数据量 | 实验 | 步数 | Batch | 耗时 | 用途 |
|------|------|--------|------|------|-----|------|------|
| `demo_run_cpu.sh` | CPU | 3 shards | 20 | 3 | 8/2 | ~1-2min | CI / 流程验证 |
| `demo_run_quick.sh` | 8x NPU | 20 shards | 8 | 5000 | 64/4 | ~1.5h | **快速测试** |
| `demo_run_full.sh` | 8x NPU | 20 shards | 96 | 5000 | 64/4 | ~2h | 中等规模验证 |

所有 demo 使用全量验证集（10k docs）、warmup_fraction=4%、checkpoint_interval=1000。

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
    │   ├── meta.json                # Full parameters + val_loss + checkpoint_steps
    │   ├── selected_indices.npy     # Selected document indices
    │   └── checkpoint_trajectory.json  # val_loss at checkpoint intervals
    └── exp_0001/ ...
```

## License

Apache 2.0
