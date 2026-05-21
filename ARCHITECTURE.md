# QuaDMix — 正式版架构设计

## 整体架构

```
Raw Parquet Shards (N × ~246 MB)
        │
        ▼
┌──────────────────────────────┐
│  Sharded Preprocessing        │  ← 逐 shard 提取 domain + quality
│  (per-shard independent)      │      输出: preprocessed_XXXXX.parquet
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  ShardMetadataManager         │  ← 元数据常驻内存
│  ├── domain_labels [N_docs]   │      (12 GB for 275M docs)
│  └── quality_scores [N_docs×5]│
│  └── text: on-demand (per shard)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Token Cache (mmap)           │  ← 按需分词 + 全 shard 缓存
│  shard_{idx}_bs{bs}.npy       │      np.load(mmap_mode='r')
│  (page-level lazy loading)    │      仅选中行才从磁盘读取
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Quality Merger (Eq.1)     │
│     Quality Rank   (Eq.2)     │
│     Sigmoid Sample (Eq.3)     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    Parameter Optimization     │
│  ┌──────────────┐            │
│  │ ParamSampler │─┐          │  ← Algorithm 1
│  └──────────────┘ │          │
│                   ▼          │
│  ┌──────────────────────┐    │
│  │ Proxy Experiments     │    │  ← 真实 1M 模型训练
│  │ (tinyllama_1M,        │    │     assistant-only loss
│  │  cache-aware loader)  │    │
│  └──────────┬────────────┘    │
│             ▼                 │
│  ┌──────────────────────┐    │
│  │ LightGBM Regression   │    │  ← 预测 loss vs 95维θ
│  └──────────┬────────────┘    │
│             ▼                 │
│  ┌──────────────────────┐    │
│  │ Optimal Search        │    │  ← 100K Alg.1 → top-10 平均
│  └──────────────────────┘    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Final Sampling           │
│  (metadata → Eq.1-3 → select)│
│  (text loaded on-demand)      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      输出                     │
│  result/<experiment>/         │
│  ├── optimal_parameters.json  │
│  ├── sampled_dataset.parquet  │
│  ├── quadmix_report.md        │
│  ├── fig*.png                 │
│  └── proxy_experiments/       │
└──────────────────────────────┘
```

## 模块设计

### 1. 数据预处理 `scripts/preprocess_essential_web_v1_sharded.py`
- 输入：原始 essential-web-v1 parquet shard（.parquet 格式）
- 输出：`temp/preprocessed/preprocessed_NNNNN.parquet` + `shard_index.json`
- 每个 shard 独立处理，提取：
  - **domain label**: 从 `eai_taxonomy.free_decimal_correspondence.level_1` 提取
  - **quality signals**: 从 `quality_signals.fasttext.*` 提取（5 个 FastText 信号）
  - 保留 `shard_idx`, `row_in_shard` 追溯原始位置
- 10 类杜威十进制域标签（0-9）
- 可并行处理所有 shard

### 2. 元数据管理 `src/quadmix/data/metadata_manager.py`
- `ShardMetadataManager`: 从 `shard_index.json` + preprocessed shards 加载
- 只读 `domain_labels` + `quality_scores` 列，跳过 `text` 列
- 提供 `global_to_shard_rows()` 将全局索引映射到 `(shard_idx, local_row)`
- 提供 `read_texts()` 按需读取指定 shard 的文本
- 全量 275M docs 元数据约 12 GB（可常驻内存）

### 3. Core Algorithm `src/quadmix/core/`
| 文件 | 功能 | 公式 |
|------|------|------|
| `quality_merger.py` | 质量融合 | Eq.1: ¯q = Σσ(qₙ)·αₙₘ |
| `quality_rank.py` | 域内百分位排名 | Eq.2: ¯r = percentile(¯q within domain) |
| `sampler.py` | Sigmoid 采样值 | Eq.3: S(¯r) = (2/(1+e^{-λ(ω-¯r)}))^η + ε |
| `proxy_model.py` | 代理模型 | tinyllama_1M (1.31M non-emb, RMSNorm+SwiGLU) |
| `types.py` | 数据类型 | SamplingConfig, ParameterSet, QuaDMixConfig |

### 4. Pipeline 编排 `src/quadmix/pipeline/`
| 文件 | 功能 |
|------|------|
| `real_pipeline.py` | 完整端到端流程（9 个 Stage） |
| `param_sampler.py` | Algorithm 1：参数采样 (95维θ) |
| `optimizer.py` | LightGBM 回归 + 100K 搜索 + top-10 平均 |
| `report.py` | MD 报告 + 分布对比图 + 质量权重图 |

### 5. Proxy 实验引擎 `scripts/essential_proxy_runner.py`
- 支持 `legacy` 和 `sharded` 两种模式
- sharded 模式：通过 `ShardMetadataManager` 加载元数据
- 每实验重新 Eq.1+Eq.2（用对应 αₘ）
- 训练 tinyllama_1M → assistant-only validation loss → 保存结果
- **磁盘缓存**: `np.load(cache_path, mmap_mode='r')` — 页面级延迟加载
- 缓存键: `shard_{idx}_bs{block_size}.npy`，格式 int32 [N_docs, block_size]

### 6. 文件布局
| 路径 | 内容 | 生命周期 |
|------|------|----------|
| `result/<experiment>/` | 最终输出（参数、数据集、报告、图表） | 永久保存 |
| `temp/preprocessed/` | 预处理的 shard 元数据 | 数据变更后重新生成 |
| `temp/token_cache/` | 分词缓存 npy 文件 | 自动重建，可删除 |
| `data/essential-web/` | 下载的原始 shard | 下载后缓存 |

### 7. Shell 包装脚本
| 脚本 | 用途 | 运行时间 |
|------|------|----------|
| `demo_run_quick.sh` | 快速验证 (2 exp, 5 search) | ~15s (CPU) |
| `demo_run_full.sh` | 论文配置 (3000 exp, 100K search) | 需 GPU |
