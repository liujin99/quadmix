# QuaDMix 完整流程与论文对齐分析

## 一、数据预处理

**文件**: `scripts/preprocess_essential_web_v1_sharded.py`

**输入**: 原始 essential-web-v1 parquet shard（`train-NNNNN-of-03291.parquet`）

**输出**: `temp/preprocessed/preprocessed_NNNNN.parquet` + `shard_index.json`

每 shard 独立输出 9 列：

```
text | domain | shard_idx | row_in_shard | qs_dclm | qs_fineweb_edu_approx | qs_english | qs_eai_general_math | qs_eai_open_web_math
```

**关键细节**:

- 域标签从 `eai_taxonomy.free_decimal_correspondence.level_1` 提取（10 类杜威十进制，0-9 映射）
- 质量信号从**顶层列** `quality_signals.fasttext.*` 提取（5 个 FastText 分类器输出）
- 记录 `shard_idx` 和 `row_in_shard` 以便追溯回原始 shard
- 不可分类（domain=-1）的数据保留但不参与域内排名
- 支持任意数量 shard 并行处理

### ShardMetadataManager

**文件**: `src/quadmix/data/metadata_manager.py`

```
ShardMetadataManager:
  读 shard_index.json → 发现所有 preprocessed shard
  只读 domain_labels + quality_scores 列 (跳过 text)
  提供方法:
    global_to_shard_rows(global_indices) → {shard_idx: (path, local_rows)}
    read_texts(shard_idx) → [text1, text2, ...]
  ---
  内存: (275M × 8 bytes) + (275M × 5 × 8 bytes) ≈ 12 GB
```

---

## 二、Pipeline 运行流程

入口：

```bash
python scripts/run_essential_web_v1.py --preprocessed-dir temp/preprocessed --quick
# => pipeline.run(..., metadata_manager=manager)
```

### Stage 0: 加载数据

**文件**: `real_pipeline.py` — sharded 分支

```python
load_precomputed(metadata_manager=manager):
    self._domain_labels = manager.domain_labels     # [N_docs]
    self._quality_scores = manager.quality_scores   # [N_docs, 5]
    self._texts = None                               # 按需加载
    self.text_source = "sharded"                     # 标记 shard 模式
```

### Stage 1: 跳过（precomputed mode）

---

### Stage 3: 参数采样 — Alg.1

**文件**: `real_pipeline.py` → `param_sampler.py`

```python
param_sets = self._param_sampler.sample_batch(n_exp)
```

对应论文 Alg.1：

```
Alg.1 Step                        代码位置
─────────────────────────────────────────────────────────
a₁…aₙ ~ U(0,1)                    a = rng.uniform(0, 1, N=5)
ãₙ = aₙ / Σaᵢ                     a_norm = a / a.sum()

for m in 0..M-1:
    b₁…bₙ ~ U(0,1)                b = rng.uniform(0, 1, N)
    b̃ₙ = ãₙ·bₙ / Σ(ãᵢ·bᵢ)       b_norm = b/(b.sum())
    αₘ = (b̃₀, …, b̃ₙ₋₁)           domain_weights

    λ, ω, η, ε ~ U(0,1)           rng.uniform(min, max)
    λ̃ = λ × 1000
    ω̃ = ω × 0.1
    η̃ = η × 1.0
    ε̃ = ε / 1000
    βₘ = (λ̃, ω̃, η̃, ε̃)          SamplingConfig

θᵢ = (α₁…αₘ, β₁…βₘ)              ParameterSet
```

**flatten 布局**（95 维）：

```
[global_w(5) | domain_w(5×10=50) | λ₀..λ₉(10) | ω₀..ω₉(10) | η₀..η₉(10) | ε₀..ε₉(10)]
                                                                          = 95
```

---

### Stage 4: Proxy 实验

**文件**: `essential_proxy_runner.py`

两种模式：
- `--mode legacy` — 单 parquet 文件加载（旧方式）
- `--mode sharded`（默认）— 通过 ShardMetadataManager 按需加载

#### 初始化（`__init__`）

sharded 模式：
```python
# 元数据从 ShardMetadataManager 读取
self._domain_labels = manager.domain_labels
self._quality_scores = manager.quality_scores
self._token_ids = None  # 不预加载所有 token

# 额外初始化
self._mode = "sharded"
self.metadata_manager = manager
self._cache_hits = 0
self._cache_misses = 0
```

#### 每实验流程（`run_experiment`）

```
Input: params = (αₘ, βₘ)

Step 0 — 计算排位（Eq.1 + Eq.2）:
  quality_ranks = _compute_ranks_for_params(params, experiment_id)
    │
    ├── Eq.1: merged = Σ σ(qₙ) · αₙₘ    # 用本实验 αₘ 融合
    └── Eq.2: 子集参考排名法               # seed=exp_id+1729

Step 1 — 采样文档（Eq.3）:
  sv = compute_sampling_values(quality_ranks, labels, params)
  S(¯r) = sigmoid(λ(ω-¯r))^η + ε          # 用 βₘ

Step 2 — 加载 token（MMap Cache）:
  _load_tokens_for_experiment(selected_idx):
    按 (shard_idx, local_rows) 分组
    对每个 shard:
      cache_path = shard_{sid:05d}_bs{block_size}.npz
      if cache exists:
        data = np.load(cache_path, mmap_mode='r')
        token_mmap = data['tokens']  # mmap'd, not loaded into RAM
        row_index = data['rows']
        positions = [row_to_pos[r] for r in local_rows]
        selected = token_mmap[positions]  # only accessed pages loaded
      else:
        读 shard text → tokenize → np.savez(cache_path, tokens=..., rows=...)

Step 3 — 创建模型:
  ProxyModel(2层, 8头, 256d, 50432vocab, RMSNorm+SwiGLU)
  14,222,592 total / 1,312,000 non-emb params

Step 4 — 训练 (RegMix 风格 permutation shuffle):
  AdamW(β=(0.9,0.95), lr=4e-4, wd=0.1)
  cosine LR + linear warmup (warmup_fraction=4% of actual steps) + grad clip 1.0

  训练循环改进:
    - 数据保留 CPU，per-batch 移到 device（减少 HBM 压力）
    - Epoch-level permutation shuffle（无放回遍历所有 block）
    - warmup_steps = max(1, int(num_steps × 0.04))
    - checkpoint_interval（默认 1000 步）记录中间 val_loss

  ```
  total_blocks = flat_train.size(0) - block_size
  perm = epoch_rng.permutation(total_blocks)  # numpy on CPU
  epoch_pos = 0, epoch = 0

  while iter_ct < max_iters:
      if epoch_pos >= total_blocks:
          perm = get_epoch_permutation()  # reshuffle
          epoch_pos = 0
          epoch += 1

      block_starts_buf[i] = perm[epoch_pos]
      epoch_pos += 1

      # CPU advanced indexing → move to device
      idx_cpu = block_starts_buf.cpu().unsqueeze(1) + arange(block_size+1)
      batch = flat_train[idx_cpu].to(device)

      # ... forward/backward/step ...

      # Checkpoint: record val_loss every N steps
      if checkpoint_interval > 0 and step_ct % checkpoint_interval == 0:
          ckpt_val = _run_validation(model, device)
          _ckpt_results[step_ct] = ckpt_val
  ```

Step 5 — 验证 (assistant-only loss, 全量 10k):
  _run_validation(model, device):
    val_n = len(val_token_ids)  # 全量 10k，不再限制
    val_bs = min(16, val_n)     # NPU OOM 防护
    per_doc = Σ(loss × mask) / Σ(mask)
    val_loss = mean(per_doc)
    model.train()  # restore training mode
    return val_loss

Step 6 — 保存:
   save selected_indices.npy
   save meta.json (λ,ω,η,ε, αₘ, val_loss, checkpoint_steps)
   save checkpoint_trajectory.json (如果 checkpoint_interval > 0)
   del model, optimizer + gc.collect() + torch.npu.empty_cache()
```

#### mmap Cache 原理

```python
# 文件格式：
# token_cache/shard_{sid:05d}_bs{block_size}.npz
# 包含两个数组：
#   tokens: [N_cached, block_size] int32 — mmap-compatible token IDs
#   rows: [N_cached] int64 — corresponding row_in_shard indices

# 读取（延迟加载）：
data = np.load(cache_path, mmap_mode='r')
token_mmap = data['tokens']  # 没有实际磁盘 I/O 发生
row_index = data['rows']

# 通过 row_index 找到对应的 cache 位置
row_to_pos = {r: i for i, r in enumerate(row_index)}
positions = [row_to_pos[r] for r in local_rows]
selected = token_mmap[positions]  # 只有 positions 对应的 pages 被读取

del data  # 释放 mmap 引用
```

**性能对比**（单个 shard, 增量缓存）:
| 方式 | 读 100 行 | 读 10000 行 |
|------|-----------|-------------|
| `np.load`（全文件） | 全量 I/O + RAM | 全量 I/O + RAM |
| `np.load(mmap_mode='r')` | ~80KB I/O + RAM | ~8MB I/O + RAM |
| **增量 npz** | 只缓存需要的行 | 已缓存行零 I/O |

---

### Stage 5: LightGBM 回归

**文件**: `optimizer.py`

```python
train_regressor():
    X = [p.flatten() for p in proxy_results]   # [n_exp, 95]
    y = [r.validation_loss for r in results]   # [n_exp]

    # 过滤 inf/nan（防止非有限 val_loss 污染 LightGBM）
    valid_mask = np.isfinite(y)
    if not np.all(valid_mask):
        print(f"WARNING: filtering {(~valid_mask).sum()} non-finite val_loss")
        X = [x for x, v in zip(X, valid_mask) if v]
        y = y[valid_mask]

    # 划分
    train_idx, val_idx = random_split()

    # LightGBM 训练
    model = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05,
        num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
    )
    model.fit(X[train], y[train])

    train_r2 = model.score(X[train], y[train])
    val_r2   = model.score(X[val], y[val])
```

---

### Stage 6: 最优参数搜索

**文件**: `optimizer.py`

对应论文 Section 3.3：

```python
def search_optimal(self, n_search_points=100000, top_k=10):
    # Alg.1 sample 100,000 候选参数
    sampler = ParameterSampler(self.config, seed=9999)
    candidates = sampler.sample_batch(100000)

    # 用 LightGBM 预测 loss
    predicted_losses = self._regressor.predict(
        [c.flatten() for c in candidates]
    )

    # 取 loss 最小的 10 个平均
    top_indices = np.argsort(predicted_losses)[:10]
    avg_arr = np.mean([candidates[i].flatten() for i in top_indices], axis=0)
    optimal_params = ParameterSet.from_flattened(avg_arr, M=10, N=5)
```

**论文逐句对照**：

| 论文 | 代码 |
|------|------|
| "Once the regressor is trained" | `self._regressor` is fitted |
| "search the input space" | `ParameterSampler.sample_batch(100000)` |
| "sample 100,000 data points using Alg.1" | 用 Alg.1 采样，非 random uniform |
| "mitigate the influence of outliers" | Alg.1 的归一化+rescale 流程排除极值 |
| "sort based on predicted target values" | `np.argsort(predicted_losses)` |
| "average of the top 10 data points" | `np.mean(top_10.flatten()) → from_flattened()` |

---

### Stage 7: 最终采样

**文件**: `real_pipeline.py`

```python
# 用最优 αₘ 重算 Eq.1 + Eq.2（全量排序，非子集）
final_ranks = self.compute_quality_ranks(
    quality_scores, domain_labels, optimal_params, token_counts,
)

# Eq.3 + Bernoulli 采样
selected_indices, sampling_values, _ = sample_with_optimal_params(
    final_ranks, domain_labels, optimal_params,
)
```

**Target Token 事后处理**：

如果设置了 `--target-tokens`：

```python
# θ* 产生最优分布，只均匀丢弃，不 scale（避免整数边界扭曲分布）
actual_tokens = np.sum(token_counts[selected_indices])

if actual_tokens > target_tokens:
    # 均匀随机丢弃（每个 copy 丢弃概率相同，保持相对分布）
    keep_prob = target_tokens / actual_tokens
    keep_mask = np.random.random(len(selected_indices)) < keep_prob
    selected_indices = selected_indices[keep_mask]
elif actual_tokens < target_tokens * 0.95:
    # 不复制（论文: "more tokens not always good"）
    # θ* 产生更少数据但 loss 更优
    print("[WARN] θ* produces less than target")
```

**原理**：
- 论文 Table 2: 30B > 90B > 180B（数据量少反而 loss 更好）
- 复制会破坏最优分布的"稀有度"设计
- `scale = target/expected` 在 `floor + random` 采样时会在整数边界非线性扭曲分布

### Stage 8: 保存输出

sharded 模式：

```python
# 读取选中文档的 text
selected_texts = []
for sid, local_rows in shard_groups.items():
    texts = metadata_manager.read_texts(sid)
    selected_texts.extend(texts[local_rows])

# 保存
save_sampled_dataset(
    output_dir / "sampled_dataset.parquet",
    texts=selected_texts,         # 已过滤
    domains=selected_domains,
)
```

**sharded 模式下特别注意**：`texts` 已预过滤，不能再对 `texts` 做 `[selected_indices]` 二次索引。

---

### Stage 9: 报告生成

**文件**: `pipeline/report.py`

```python
# sharded 模式需要注入 domain_labels_override
# 避免 report.py 尝试 pd.read_parquet(data_path)（data_path 是目录）
report = generate_report(
    ...,
    domain_labels=self._domain_labels,  # 直接传 numpy 数组
)
```

报告包含：
- `fig1_domain_distribution.png` — 原始 vs 最优域分布柱状图
- `fig2_quality_weights.png` — 各域质量信号权重堆叠图
- `quadmix_report.md` — 包含配置、结果、对比表格

---

## 三、输出目录结构

```
result/<experiment_name>/
├── optimal_parameters.json        (3.5K)  # 最优 θ
├── pipeline_summary.json          (500B)  # 配置 + 指标
├── sampled_dataset.parquet        (14K+)  # 采样数据集
├── quadmix_report.md              (1K+)   # 报告
├── fig1_domain_distribution.png   (60K+)  # 域分布对比图
├── fig2_quality_weights.png       (70K+)  # 质量权重图
└── proxy_experiments/
    ├── exp_0000/
    │   ├── meta.json              (1K+)   # 含完整参数 + val_loss + checkpoint_steps
    │   ├── selected_indices.npy   (var)   # 采样索引
    │   └── checkpoint_trajectory.json     # 训练中 val_loss 变化轨迹
    └── exp_0001/ ...
```

### 中间数据（可删除）

```
temp/
├── preprocessed/                   # 预处理输出
│   ├── preprocessed_00000.parquet  (~180MB × N shards)
│   ├── preprocessed_00001.parquet
│   └── shard_index.json
└── token_cache/                    # 分词缓存 (增量 npz)
    └── shard_00000_bs64.npz        (tokens + rows, 增量追加)
```

---

## 四、核心公式 → 代码映射

| 公式 | 文件 | 函数 | 关键实现 |
|------|------|------|----------|
| Eq.1: ¯q = Σσ(qₙ)·αₙₘ | `core/quality_merger.py` | `compute_merged_quality_scores` | rank 归一化→域内加权和→¯q |
| Eq.2: ¯r = percentile(¯q) | `core/quality_rank.py` | `compute_quality_ranks` | token-weighted cumsum → ¯r ∈ [0,1] |
| Eq.3: S(¯r) | `core/sampler.py` | `compute_sampling_values` | sigmoid(λ(ω-¯r))^η + ε, ω-threshold |
| Alg.1: θ 采样 | `pipeline/param_sampler.py` | `sample_one` / `sample_batch` | U(0,1)→归一化→rescale |
| 子集 Eq.2 (proxy) | `essential_proxy_runner.py` | `_compute_ranks_for_params` | 每域抽 k 个文档建参考分布 → searchsorted |
| LightGBM | `pipeline/optimizer.py` | `train_regressor` | 95 维 θ → predicted loss |
| 最优搜索 | `pipeline/optimizer.py` | `search_optimal` | 100K Alg.1 样本 + top-10 平均 |
| Proxy 训练 | `essential_proxy_runner.py` | `run_experiment` | tinyllama_1M 训练 + assistant-only 验证 |
| MMap Cache | `essential_proxy_runner.py` | `_load_tokens_for_experiment` | `np.load(path, mmap_mode='r')` |
| Shard 元数据 | `data/metadata_manager.py` | `ShardMetadataManager` | 只读 metadata 列，text 按需 |

---

## 五、数据流图

```
                        ┌───────────────────────────┐
                        │  essential-web-v1 raw      │
                        │  3291 shards × 246 MB      │
                        │  (≈808 GB total)           │
                        └─────────────┬─────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  preprocess_essential_web_v1_     │
                    │  sharded.py                       │
                    │  (per shard, parallelizable)       │
                    │  ├ domain: eai_taxonomy.level_1   │
                    │  ├ quality: fasttext (5 signals)  │
                    │  └ shard_idx/row_in_shard         │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────▼─────────────────┐
                    │  temp/preprocessed/               │
                    │  ├ preprocessed_00000.parquet     │
                    │  ├ preprocessed_00001.parquet     │
                    │  └ shard_index.json               │
                    └─────────────────┬─────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │    ShardMetadataManager                        │
              │    (只读 domain + quality 列)                  │
              │    domain_labels [N_docs]                      │
              │    quality_scores [N_docs, 5]                  │
              │    text → 按需加载                             │
              └───────────────────────┼───────────────────────┘
                                      │
              ┌───────────────────────▼───────────────────────┐
              │         QuaDMixPipeline.run()                  │
              │                                                │
              │  Stage 0: 加载元数据                            │
              │  Stage 2: Alg.1 × n_exp → param_sets           │
              │                                                │
              │  Stage 4: Proxy Experiments                    │
              │    for each θₘ:                                │
              │      Eq.1: ¯q = Σσ(qₙ)·αₙₘ                    │
              │      Eq.2: ¯r = 子集参考排名                    │
              │      Eq.3: S(¯r) = sigmoid^η + ε              │
              │      Bernoulli采样                              │
              │        │                                       │
              │        ▼   _load_tokens_for_experiment()       │
              │      ┌──────────────────┐                      │
              │      │ MMap Cache (.npz) │                    │
              │      │ shard_0_bs64.npz  │── hit → np.load    │
              │      │ shard_1_bs64.npz  │     (mmap_mode=r)  │
              │      │  tokens + rows    │                    │
              │      └──────────────────┘                      │
              │        │                                       │
              │        ▼                                       │
              │      Train tinyllama_1M → val_loss              │
              │      → meta.json + selected_indices.npy        │
              │                                                │
              │  Stage 5: LightGBM R(θ) → predicted loss       │
              │  Stage 6: Alg.1 × 100K → predict → top-10 avg  │
              │  Stage 7: Eq.1+Eq.2(θ*) → Eq.3 → 采样          │
              │  Stage 8: 保存                                  │
              │                                                │
              │  Stage 9: 报告 + 图表                           │
              └───────────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │  result/<experiment_name>/               │
                    │  ├ optimal_parameters.json               │
                    │  ├ pipeline_summary.json                 │
                    │  ├ sampled_dataset.parquet               │
                    │  ├ quadmix_report.md                     │
                    │  ├ fig1_domain_distribution.png          │
                    │  ├ fig2_quality_weights.png              │
                    │  └ proxy_experiments/exp_*/              │
                    └─────────────────────────────────────────┘
```

---

## 六、Shell 包装脚本

### demo_run_cpu.sh

```
参数: 20 experiments, 200 search, block_size=64 (CPU 快速验证)
时间: ~1-2分钟 (CPU)
用途: 快速验证架构是否正确
注: 无 doc_limit，proxy experiments 使用完整数据池
```

### demo_run_quick.sh

```
参数: 8 experiments, 1000 search, block_size=2048, tiny_steps=5000
      global_batch_size=64, micro_batch_size=4 (grad_acc=16)
      warmup_fraction=4%, 验证集全量 10k, checkpoint_interval=1000
时间: ~1.5小时 (8x NPU)
用途: 快速测试，端到端流程验证
```

### demo_run_full.sh

```
参数: 96 experiments, 5000 search, block_size=2048, tiny_steps=5000
      global_batch_size=64, micro_batch_size=4 (grad_acc=16)
      warmup_fraction=4%, 验证集全量 10k, checkpoint_interval=1000
时间: ~2h (8x NPU)
用途: 中等规模验证

多卡并行 (--npu-devices N):
  动态任务队列架构：
    - Worker 完成任务后立即领取下一个，无批次边界
    - 快的 Worker 自然做更多实验
    - CPU tokenize 线程独立运行，与 NPU 训练重叠
```

三个脚本都会自动：
1. 检查数据是否存在（`data/essential-web/`）
2. 检查预处理是否完成（`temp/preprocessed/shard_index.json`）
3. 缺失则自动下载 + 预处理
4. 运行完整 pipeline

---

## 七、验证集

| 项目 | 值 |
|------|-----|
| 来源 | OpenHermes-2.5-1M (assistant-only 子集) |
| 文件 | `data/openhermes_10k_assistant_tokenized.pt` (176MB, PyTorch tensor) |
| HF 数据集 | `liujin99/quadmix-openhermes-10k` |
| 文档数 | 10,000 |
| 预处理脚本 | `scripts/validation_set/prepare_openhermes_assistant_10k.py` (僅供參考，不需手動執行) |
| 预分词缓存 | `openhermes_10k_assistant_tokenized.pt` (自動從 HF 下載) |
| 验证方式 | assistant-only loss mask (所有非 padding token) |

> **注意**: 使用者不需手動下載 openhermes 原始數據或執行準備腳本。
> `run_essential_web_v1.py` 和 `demo_run_*.sh` 會在首次運行時自動從
> [liujin99/quadmix-openhermes-10k](https://huggingface.co/datasets/liujin99/quadmix-openhermes-10k) 下載。
> 此腳本僅作為數據處理流程的說明文檔保留。

---

## 八、与 QuaDMix 论文对齐总结

| 论文要点 | 实现状态 | 文件位置 |
|----------|----------|----------|
| Eq.1 域内质量融合 αₘ | ✅ 每实验用对应 αₘ 重新融合 | `quality_merger.py` |
| Eq.2 token-weighted 百分位 | ✅ 全量排序法（core）和子集参考法（proxy） | `quality_rank.py`, `proxy_runner` |
| Eq.3 sigmoid 采样 | ✅ S(¯r) = (2/(1+e^{-λ(ω-¯r)}))^η + ε | `sampler.py` |
| Alg.1 参数采样 | ✅ U(0,1)→归一化→rescale→θ | `param_sampler.py` |
| proxy 模型 1M params | ✅ tinyllama_1M (1.31M non-emb) | `proxy_model.py` |
| RegMix 训练 | ✅ cosine LR, AdamW, grad clip 1.0, permutation shuffle, warmup_fraction=4% | `proxy_runner` |
| assistant-only loss | ✅ loss_mask 过滤 user/system token, 全量 10k | `proxy_runner` |
| LightGBM 回归 | ✅ 1000 trees, 31 leaves | `optimizer.py` |
| 100K 搜索 Alg.1 | ✅ `ParameterSampler(seed=9999)` | `optimizer.py` |
| top-10 平均降方差 | ✅ `np.mean(top_10_flattened)` | `optimizer.py` |
| 质量信号 | ✅ 5 FastText 信号 | `preprocess_*.py` |
| 域分类 | ✅ 10 类杜威十进制 (level_1) | `preprocess_*.py` |
| 大规模 shard 支持 | ✅ ShardMetadataManager + mmap cache | `metadata_manager.py` |
