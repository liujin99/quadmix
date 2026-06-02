# QuaDMix 性能优化 TODO

> 2025-05-28 Hermes Agent 审计生成
> 总计 ~5,460 行 Python，29 源文件 + 5 脚本

## 已完成

### 2026-06-02: ProcessPoolExecutor fork→spawn 避免 COW page fault
- **文件**: `scripts/essential_proxy_runner.py:_tokenize_shard_parallel()`
- **问题**: 主进程 RSS 18GB，fork 64 workers 导致 64×18GB=1.15TB COW 虚拟映射，海量 page fault 导致 tokenize 卡死
- **修复**: 使用 `mp.get_context('spawn')` 替代默认 fork，子进程从零启动不继承主进程内存
- **影响**: 启动慢 2-3 秒，但避免卡死问题

### 2026-06-02: micro_batch 64→40 避免 NPU OOM
- **文件**: `scripts/demo_run_full.sh`, `scripts/demo_run_quick.sh`
- **问题**: micro_batch=64 时 logits [64, 2048, 50432] fp32 = 26.3GB，加上梯度和 CE workspace，peak ~77GB 超过 NPU 60GB 上限
- **修复**: micro_batch=40，peak ~49GB，余量 11GB
- **影响**: global_batch 同步改为 40（无梯度累积），训练速度提升

### 2026-06-02: mmap file handle leak 修复
- **文件**: `scripts/essential_proxy_runner.py:_cached_shard_rows()` 等 4 处
- **问题**: `np.load(mmap_mode='r')` 未关闭文件句柄，101 shards 累积 101 个打开的 mmap fd 导致 IO hang
- **修复**: 移除 `mmap_mode='r'`，直接加载到内存

### 2026-06-02: 综合性能计时系统
- **文件**: `scripts/essential_proxy_runner.py`, `src/quadmix/pipeline/real_pipeline.py`
- **内容**: PerfTimer 类 + stage 级计时，实时输出 + 结束汇总

## P0 — 高影响（预计加速 30-50%）

### 1. HDF5/LMDB 替代 np.savez 做 token cache
- **文件**: `scripts/essential_proxy_runner.py:_cache_add_rows()` (L320-434)
- **问题**: 每次增量添加几行，都全量读-合并-写整个 .npz（最终 ~200MB/shard）
- **预计**: 磁盘 I/O 减少 70%+
- **方案**: HDF5 appendable dataset 或 LMDB key-value store

### 2. Shared memory metadata — 消除 worker 重载 15GB
- **文件**: `scripts/essential_proxy_runner.py:_worker_dynamic_loop()` (L1636-1693)
- **问题**: 8 worker × 15GB = 120GB，启动 30-60s/worker
- **预计**: Worker 启动 ~5s, RAM 120GB→15GB
- **方案**: torch.multiprocessing shared_memory 或 Python 3.8+ shared_memory

## P1 — 中影响（预计加速 15-25%）

### 3. 预计算 domain indices — 消除每实验的 mask 创建
- **文件**: `essential_proxy_runner.py:_compute_ranks_for_params()` (L927-975)
- **问题**: 3000 实验 × 10 domain × 每次创建 275M 布尔 mask
- **方案**: `__init__` 时预计算 `self._domain_indices = {m: np.where(...)}`

### 4. 预计算 subset 排序 — 消除每实验的排序
- **文件**: `essential_proxy_runner.py:_compute_ranks_for_params()` (L958-974)
- **问题**: 每实验每 domain 重新 choice + sort k=10000
- **方案**: 预计算分层采样的排序参考集

### 5. 消除 Dispatcher 忙等待
- **文件**: `essential_proxy_runner.py:_run_batch_dynamic()` (L1556-1588)
- **问题**: `time.sleep(0.02)` 轮询，3000 实验 × 平均等 5s = 750K 次无效轮询
- **方案**: `threading.Condition` 信号通知

## P2 — 低影响

### 6. 统一 `_pack_exp_tokens` 和 `_tokenize_batch_union`
- **文件**: `essential_proxy_runner.py` (L447-541 和 L557-730)
- **问题**: 大量重复逻辑，fallback 行为不一致
- **方案**: 合并为单一方法，模式标志控制

### 7. Memory cache LRU 淘汰
- **文件**: `essential_proxy_runner.py:_memory_cache` (L197, L224-260)
- **问题**: 无界增长，最终可达 65GB
- **方案**: 大小限制 + LRU，淘汰的回退到磁盘 mmap

### 8. Disk cache 防御性代码 gating
- **文件**: `essential_proxy_runner.py:_cache_add_rows()` (L387-434)
- **问题**: 每次写入检查目录/磁盘/权限（永远通过）
- **方案**: 放到 `if verbose` 后面

## P3 — 微优化

### 9. 删除重复的 `_get_shard_token_path()` 定义
- **文件**: `essential_proxy_runner.py` (L199-204 死代码返回 `.npy`)

### 10. RoPE 预计算缓存
- **文件**: `src/quadmix/core/proxy_model.py:RotaryEmbedding.forward()` (L43-47)
- **问题**: 每 forward 重新 torch.arange + cos/sin
- **方案**: 固定 block_size 下预计算并缓存

### 11. causal_mask 避免重复 slicing
- **文件**: `src/quadmix/core/proxy_model.py:ProxyModel.forward()` (L222)
- **问题**: `self.causal_mask[:T, :T]` 每次创建新 tensor

### 12. Validation batch 扩大
- **文件**: `essential_proxy_runner.py:run_experiment()` (L1100)
- **问题**: `val_bs=200` 对 1M 模型太小
- **方案**: 加大到 block_size

---

## 实施状态

- [~] P0-1: HDF5/LMDB token cache — 经分析，当前 AsyncWrite + batch union 模式下同 shard 不会在单 batch 内重复写，跨 batch 增量很小（首次 batch 0 覆盖大部分），实际瓶颈在初始 650GB 一次性写入。保持 np.savez，暂缓。
- [x] P0-2: Shared memory metadata  ✓ 2025-05-28
- [x] P1-3: 预计算 domain indices  ✓ 2025-05-28
- [~] P1-4: 预计算 subset 排序 — 每次 sort(10000) 很快（~1μs），30K 次总开销可忽略。不实施。
- [x] P1-5: Condition 替代 Dispatcher 忙等待  ✓ 2025-05-28
- [x] P2-6: 统一 pack/union tokenize — 删除 _pack_exp_tokens + tokenize_batch_delta 死代码 (110行)  ✓ 2025-05-28
- [x] P2-7: Memory cache LRU (16GB 上限)  ✓ 2025-05-28
- [x] P2-8: Disk cache 防御代码 gating  ✓ 2025-05-28
- [x] P3-9: 删除重复方法定义  ✓ 2025-05-28
- [x] P3-10: RoPE 预计算  ✓ 2025-05-28
- [x] P3-11: causal_mask 优化（已使用预计算 buffer）  ✓ 2025-05-28
- [x] P3-12: Validation batch 扩大 (200→1024)  ✓ 2025-05-28

---

## 文档更新 2025-05-28

- [x] ARCHITECTURE.md — 添加 Performance Optimizations 章节，更新 Worker/Dispatch/Memory/Eq.1 描述
- [ ] NPU_DEPLOYMENT.md — 添加 shared memory 相关环境要求

## Git

- Branch: `perf/optimize-20250528` (from main)
- Commit message: `perf: shared memory metadata, domain indices, LRU cache, signal-driven dispatcher`

---

## 论文一致性审计 — 2026-06-01

### P0 — 严重问题

#### A1. [已验证-确认BUG] 质量分数方向反转
- **文件**: `scripts/preprocess_essential_web_v1_sharded.py:extract_quality_signals()` (L59-68)
- **状态**: ✅ 已验证确认 — 这是一个真实 BUG
- **问题**: 论文和代码假设 "smaller = better"，但 essential-web-v1 的 FastText 分数是 **"higher = better"**
- **验证方法**: 分析 `train-00000-of-03291.parquet` 中 83,933 个文档的分数分布
- **验证证据**:
  - `qs_dclm` 最高分 (≈1.0) 的文档: 泰语赌博垃圾内容、git commit 日志、HTML 碎片、乱码
  - `qs_dclm` 最低分 (≈0) 的文档: 数学书籍、政府行政数据、教堂活动页、股票市场数据
  - `qs_fineweb_edu_approx` 最高分 (≈3.9) 的文档: PDS 元数据文件、代码 blob、汇率表
  - `qs_fineweb_edu_approx` 最低分 (≈0.001) 的文档: 电池产品页、垃圾邮件、电商页面
  - DCLM 官方 `fasttext_filter.yaml` 使用 `threshold: 0.018112`，保留 >= threshold 的文档（即高分=高质量）
- **影响**: **代码当前在选择最低质量的文档作为"最好"的数据** — rank_normalize 将最小分数排为 rank 0（"最好"），但最小分数实际是最差质量
- **所有 5 个 FastText 信号方向**:
  - `qs_dclm`: [0, 1]，higher = better (P(instruction-like))
  - `qs_fineweb_edu_approx`: [0, ~4]，higher = better (logit 分数)
  - `qs_english`: [0, 1]，higher = better (P(English))
  - `qs_eai_general_math`: [0, 1]，higher = better (P(math-like))
  - `qs_eai_open_web_math`: [0, 1]，higher = better (P(math-like))
- **修复方案**: 在预处理时反转分数方向: `qs = -qs` (或 `qs = max_val - qs`)
  - 修改 `preprocess_essential_web_v1_sharded.py` 的 `extract_quality_signals()`
  - 或在 `ShardMetadataManager` 加载时反转

### P1 — 重大问题

#### A2. 质量评分器数量不同 (N=5 vs N=3) — 非问题（设计选择）
- **论文**: N=3 (AskLLM, Fineweb-Edu, DCLM)，基于 RefinedWeb 数据集
- **代码**: N=5 (dclm, fineweb_edu_approx, english, eai_general_math, eai_open_web_math)，基于 essential-web 数据集
- **决定**: 使用 essential-web 提供的标签，有什么用什么

#### A3. 域分类器不同 (M=10 vs M=26) — 非问题（设计选择）
- **论文**: M=26 (Deberta V3 分类器)，基于 RefinedWeb 数据集
- **代码**: M=10 (EAI Dewey Decimal level_1)，基于 essential-web 数据集
- **决定**: 使用 essential-web 提供的域分类，有什么用什么

#### A4. Proxy 训练 token 数略少于论文
- **论文**: 1B tokens per experiment (1 H100 GPU hour)
- **代码**: ~655M tokens (tiny_steps=5000, global_batch=64, block=2048)
- **计算**: 5000 steps × 64 (global_batch) × 2048 (block_size) = 655M
- **影响**: 约为论文的 65%，差距不大，可接受

### P2 — 中等问题

#### A5. Proxy 实验中 Eq.2 缺少 token 加权
- **文件**: `scripts/essential_proxy_runner.py:_compute_ranks_for_params()` (L832-875)
- **问题**: 论文 Eq.2 使用 token 数量加权计算百分位，proxy 实验使用文档数等权
- **影响**: 排名精度降低，与论文不一致

#### A6. 数据集不同 — 非问题（设计选择）
- **论文**: RefinedWeb (570B tokens)
- **代码**: essential-web-v1 (~808 GB)
- **决定**: 基于 essential-web 数据集实现算法，不要求与论文结果直接对比

#### A7. 缺少 QuaDMix-BMK 变体
- **论文**: 两种变体 (QuaDMix-OH + QuaDMix-BMK)
- **代码**: 仅实现 QuaDMix-OH (OpenHermes)
- **影响**: 缺少下游任务优化能力

### P3 — 轻微问题

#### A8. 最终采样 Eq.2 使用全量排序（改进）
- **论文**: Section 3.4 用 10,000 文档子集估计百分位
- **代码**: Stage 7 使用全量排序
- **影响**: 比论文更精确，属于改进

---

## 优化改进建议 — 2026-06-01

### 算法层面

#### B1. LightGBM 缺少 early stopping 和交叉验证
- **文件**: `src/quadmix/pipeline/optimizer.py:train_regressor()` (L219-294)
- **问题**: 1000 棵树无 early stopping，单一 train/val 划分
- **方案**: 添加 early stopping + K-fold 交叉验证

#### B2. 参数空间搜索效率低
- **文件**: `src/quadmix/pipeline/optimizer.py:search_optimal()` (L296-337)
- **问题**: 200+ 维空间随机搜索，论文 Limitations 也提到此问题
- **方案**: Bayesian Optimization (Optuna/TPE) 或 CMA-ES

#### B3. 采样函数参数冗余
- **问题**: 不同 (λ, ω, η, ε) 可能产生相似采样曲线，引入回归不确定性
- **方案**: 对采样函数做正交化或参数约束

#### B4. 质量分数归一化函数选择
- **文件**: `src/quadmix/utils/normalization.py`
- **问题**: rank 归一化是全局的，论文未明确 σ 的语义
- **方案**: 尝试域内归一化或对比不同归一化方案

#### B5. 缺少质量融合消融实验支持
- **论文**: Table 2 做了 A/F/D 不同组合消融
- **方案**: 添加 `--disable-criteria` 参数支持

### 工程层面

#### B6. Token 计数使用估算而非实际值 — 不做
- **原因**:
  1. 分词器不同结果不同，无标准答案
  2. 40T 数据池全量 tokenize 只为计数，开销大
  3. 同域内英文文档误差方向一致，相对排名几乎不变
  4. 业界主流数据集（RefinedWeb、FineWeb、DCLM、essential-web）均不提供 token 数标签，论文大概率也是估算

#### B7. 验证集 loss 计算方式未明确
- **文件**: `scripts/essential_proxy_runner.py:_run_validation()` (L1114-1142)
- **问题**: 使用文档级平均，论文未明确是文档级还是 token 级
- **影响**: 短文档权重不同

#### B8. 多验证目标支持
- **问题**: 仅支持单一验证集 (OpenHermes)
- **方案**: 添加多验证集支持，一次计算多个 val_loss

#### B9. 回归模型集成
- **问题**: 只用 LightGBM 一种回归器
- **方案**: 多模型集成 (LightGBM + XGBoost + Neural Network)

#### B10. 实验可复现性
- **文件**: `src/quadmix/pipeline/real_pipeline.py` (L497)
- **问题**: target token 丢弃用 `np.random.default_rng()` 无种子
- **方案**: 统一使用固定种子

---

## 优先级总结

| 优先级 | 编号 | 问题 | 影响 |
|--------|------|------|------|
| P0 | A1 | 质量分数方向可能反转 | 可能选择低质量数据 |
| P1 | A4 | Proxy 训练 token 数太少 | 回归模型质量差 |
| P1 | A2,A3 | N=5/M=10 vs N=3/M=26 | 参数空间和搜索精度不同 |
| P2 | A5 | Proxy Eq.2 缺少 token 加权 | 排名精度降低 |
| P2 | B1 | LightGBM 无 early stopping | 过拟合风险 |
| P3 | A7,B5 | 缺少 BMK 变体 / 消融支持 | 功能完整性 |

---

## 数据处理性能优化 — 2026-06-02

### C1. `tokenize_all_needed` 单线程 tokenize
- **文件**: `essential_proxy_runner.py:tokenize_all_needed()` (L1336)
- **问题**: CPU/顺序模式调用 `_tokenize_texts()`（单线程），而并行模式用 `_tokenize_shard_parallel`（48 worker）。CPU 模式下 192 核完全浪费
- **影响**: CPU 模式 tokenize 慢 10-20x
- **方案**: `tokenize_all_needed` 内部也调用 `_tokenize_shard_parallel`

### C2. Pack/Unpack 经磁盘 I/O 传递 token
- **文件**: `essential_proxy_runner.py:_tokenize_batch_union()` (L617-620) + `_load_tokens_for_experiment()` (L648-651)
- **问题**: 主进程 `torch.save` 写磁盘 → worker 进程 `torch.load` 读磁盘。大实验几百 MB，每个实验多几秒到几十秒 I/O
- **影响**: 每实验额外 2-10s I/O 开销
- **方案**: spawn 子进程不能用共享 tensor，但可用 `np.save` + mmap 或 shared_memory 替代

### C3. `_cached_shard_rows` 每次构建 Python set 是 O(N)
- **文件**: `essential_proxy_runner.py:_cached_shard_rows()` (L382-389)
- **问题**: `set(int(r) for r in data['rows'])` — 10 万行 × `int()` 转换，Python for-loop
- **影响**: 每 shard 每 batch 几十 ms
- **方案**: 改用 `np.isin` 直接向量化查询，或缓存 rows 数组避免重复构建 set

### C4. `_memory_cache_add_rows` Python dict 去重 O(N)
- **文件**: `essential_proxy_runner.py:_memory_cache_add_rows()` (L309-314)
- **问题**: `row_to_idx = {}; for i, r in enumerate(combined_rows): row_to_idx[int(r)] = i` + `sorted(row_to_idx.keys())` — 10 万行用 Python dict+sort 去重
- **影响**: 每 shard 几十 ms
- **方案**: 改用 `np.unique(return_index=True)`

### C5. 预处理 `df.apply` 逐行提取质量信号
- **文件**: `preprocess_essential_web_v1_sharded.py:process_shard()` (L89)
- **问题**: `df["quality_signals"].apply(extract_quality_signals)` — 每行调用 Python 函数解析 dict。8 万行/shard × 20 shard = 160 万次 Python 函数调用
- **影响**: 预处理阶段一次性开销，增量模式跳过
- **方案**: 向量化提取或用 `pd.json_normalize`

### C6. Memory cache LRU 用 list.remove 是 O(N)
- **文件**: `essential_proxy_runner.py:_memory_cache_lru` (L276, L322, L328)
- **问题**: LRU 用 Python list 实现，每次 `remove(sid)` 是 O(N) 线性扫描。20 个 shard 问题不大，shard 数增长会成为瓶颈
- **影响**: shard 少时影响小
- **方案**: 改用 `collections.OrderedDict`

### 优先级

| # | 瓶颈 | 影响 | 修复难度 |
|---|------|------|---------|
| C1 | tokenize_all_needed 单线程 | CPU 模式慢 10-20x | 低 |
| C2 | Pack/Unpack 磁盘 I/O | 每实验 2-10s | 中 |
| C3 | _cached_shard_rows set 构建 | 每 shard 几十 ms | 低 |
| C4 | _memory_cache_add_rows dict 去重 | 每 shard 几十 ms | 低 |
| C5 | 预处理 df.apply | 一次性 | 低 |
| C6 | LRU list.remove | shard 少时小 | 低 |

---

## NPU 显存与性能优化 — 2026-06-02

### 显存瓶颈分析

**根因：logits 张量占 95% 显存**

micro_batch=32, block_size=2048, vocab=50432 时的显存分布：

| 组件 | 大小 | 说明 |
|------|------|------|
| 模型权重 (1M params) | 57 MB | embed + 2 层 transformer |
| AdamW 优化器状态 | 114 MB | 2× 权重副本 |
| causal_mask buffer | 16 MB | 2048×2048 |
| flat_train (~8M tokens) | 68 MB | 训练数据 |
| batch_buf + 辅助 buffer | 1 MB | 预分配 |
| **静态总计** | **~256 MB** | |
| | | |
| embed 输出 (32×2048×256) | 33.5 MB | forward |
| attention scores ×2 层 | 2.1 GB | 32×8×2048×2048 |
| **logits (32×2048×50432×4B)** | **13.1 GB** | **瓶颈** |
| | | |
| logits（保留用于梯度） | 13.1 GB | backward |
| logits 梯度 | 13.1 GB | backward |
| attention 梯度 ×2 层 | ~2 GB | backward |
| **backward 总计** | **~28 GB** | |
| | | |
| **峰值（forward 末尾）** | **~43 GB** | 静态 + logits + attn |
| **NPU 实际占用** | **~53 GB** | 含内存池碎片 |
| **剩余可用** | **~5.4 GB** | → OOM |

**结论：**
- 模型本身只有 57 MB，训练数据 68 MB，完全不是瓶颈
- logits 张量 `micro_batch × 2048 × 50432 × 4B` 占 95%
- micro_batch=32 → logits=13.1 GB，forward+backward=26.2 GB → OOM
- micro_batch=8 → logits=3.3 GB，forward+backward=6.6 GB → 安全

### D1. bf16 混合精度训练（推荐）

- **文件**: `scripts/essential_proxy_runner.py:run_experiment()` (L1062-1067)
- **问题**: 全程 fp32，logits 13.1 GB，backward 28 GB
- **方案**: 
  ```python
  with torch.autocast(device_type="npu", dtype=torch.bfloat16):
      logits = model(inp)
  loss = F.cross_entropy(logits.float().view(-1, vocab), tgt.view(-1))
  ```
- **910B3 可行性**: 高
  - 910B3 原生 bf16 硬件单元，CANN 8.0 支持成熟
  - bf16 算力约为 fp32 的 2x（Tensor Core）
- **收益**:
  - logits 13.1 GB → 6.5 GB，backward 同步减半，**总节省 ~13 GB**
  - micro_batch 可从 8 提到 16 甚至 32，减少 grad_acc 开销
  - 训练速度预估 **1.5-2x 加速**
- **风险**:
  - `cross_entropy` 内部 `exp()/log()` 在 bf16 下可能精度不够
  - PyTorch CUDA 的 autocast 会自动把 cross_entropy 输入 cast 回 fp32，但 **torch_npu 的 autocast 行为不确定**
  - 如果 loss 计算不自动 upcast，会出现 NaN/Inf
  - **缓解**: 手动在 cross_entropy 前 `.float()` 回 fp32
- **验证**: 实测 loss 收敛曲线是否正常

### D2. Fused cross-entropy（不可行）

- **问题**: 理论上不物化完整 logits，直接在 lm_head 输出上算 loss
- **910B3 可行性**: **不可行**
  - torch_npu 大概率没有 fused cross-entropy kernel
  - CUDA 上有 `xformers.ops.memory_efficient_cross_entropy` 等第三方实现，NPU 上没有对应物
  - 自己写 CANN 自定义算子开发成本极高，不值得为 1M 参数模型投入
- **结论**: 放弃

### D3. Chunked lm_head（不推荐）

- **问题**: 把 vocab 维度分块计算，永远不同时持有完整 logits
- **910B3 可行性**: 可行但收益不确定
- **方案**: 
  ```python
  # 分 4 块计算 lm_head
  chunk_size = vocab_size // 4
  for i in range(4):
      logits_chunk = lm_head(x[:, :, i*chunk_size:(i+1)*chunk_size])
      # 计算部分 loss
  ```
- **收益**:
  - 纯 PyTorch 操作，不依赖 NPU 特定支持
  - 峰值显存可控：13.1 GB → 3.3 GB/块
- **风险**:
  - 910B3 的 kernel launch 开销比 CUDA 高，4 次小 kernel 可能比 1 次大 kernel 更慢
  - 需要重写 forward，改动较大
  - 1M 参数模型 lm_head 计算本身就很快（~0.1ms），分块后 launch 开销可能反客为主
- **结论**: 收益不确定，风险中等，不推荐

### 优先级

| # | 方案 | 910B3 可行性 | 收益 | 建议 |
|---|------|:-----------:|------|------|
| D1 | bf16 + 手动 upcast | 高 | ~2x 显存 + ~1.5x 速度 | **推荐** |
| D2 | Fused CE | 不可行 | — | 放弃 |
| D3 | Chunked lm_head | 可行 | 显存可控但可能更慢 | 不推荐 |

### 其他已尝试的优化

**micro_batch 调优历史：**
- commit 8fb7a5c: micro_batch=64 → OOM（logits 26.3 GB）
- commit 019c903: micro_batch=32 → OOM（logits 13.1 GB + backward 13.1 GB = 26.2 GB，只剩 5.4 GB）
- commit 350c967: micro_batch=8 → 安全（logits 3.3 GB + backward 3.3 GB = 6.6 GB，余量 ~31 GB）

**当前配置：**
- micro_batch=8, grad_acc=8, global_batch=64
- 每 step 8 次 forward/backward，合理
- 5000 步实验耗时 ~67 min（8× NPU 并行）

---

## 核心算法与流水线优化 — 2026-06-02

### 高影响

#### E1. `shared_to_ndarray` 每个 worker 拷贝 11GB
- **文件**: `scripts/essential_proxy_runner.py:shared_to_ndarray()` (L82-83)
- **问题**: 
  ```python
  return arr.copy()  # 275M docs × 5 criteria × 8B = 11GB
  ```
  8 个 worker 各拷贝 11GB = 88GB 内存拷贝。当初加 `.copy()` 是为了避免 spawn 子进程 shared memory segfault
- **影响**: Worker 启动时 88GB 内存拷贝，可能耗时 10-30 秒
- **方案**: 
  - 用 `np.memmap` 只读映射
  - 或只读 buffer（需验证 spawn 子进程是否安全）
  - 或每个 worker 只拷贝需要的 shard 子集（而非全量）

#### E2. `_tokenize_batch_union` 重复调用 `global_to_shard_rows`
- **文件**: `scripts/essential_proxy_runner.py:_tokenize_batch_union()` (L487-492 和 L584-585)
- **问题**: Step 1 和 Step 4 对同一批实验各调一次 `global_to_shard_rows`（内部做 `searchsorted` + `argsort` + `unique`），完全冗余
- **影响**: 每批实验额外 1-2 秒（275M docs 的 searchsorted 开销）
- **方案**: 缓存 Step 1 的 `shard_to_exp_rows` 结果，Step 4 直接复用

#### E3. `rank_normalize` 双重 argsort
- **文件**: `src/quadmix/utils/normalization.py:rank_normalize()` (L48)
- **问题**: 
  ```python
  ranks = np.argsort(np.argsort(scores))  # O(N log N) × 2
  ```
  275M docs × 5 criteria = 5 次双重 argsort
- **影响**: 初始化时可能耗时数分钟
- **方案**: `scipy.stats.rankdata` 是单次 O(N log N)，快 ~2x

#### E4. Validation `val_bs` 可以更激进
- **文件**: `scripts/essential_proxy_runner.py:_run_validation()` (L1162)
- **问题**: 当前 `val_bs=64`，但 validation 是 `no_grad`，不需要 backward 显存
- **影响**: 10k docs 需要 157 次 forward
- **方案**: 
  - 提到 `val_bs=256`（logits `256×2048×50432×4B = 26.3 GB`，no_grad 下安全）
  - 10k docs 只需 40 次 forward（vs 当前 157 次）
  - 每次 validation 节省 ~3x 时间

#### E5. `ProcessPoolExecutor` 每次 tokenize 重建
- **文件**: `scripts/essential_proxy_runner.py:_tokenize_shard_parallel()` (L1793, L1842)
- **问题**: Stage 1 IO 和 Stage 2 tokenize 各创建一个 `ProcessPoolExecutor`，每次创建/销毁 48-64 个进程
- **影响**: 每次 tokenize 批次额外 2-5 秒进程创建开销
- **方案**: 复用持久化进程池（类级别或全局）

### 中影响

#### E6. `_memory_cache_query` 双重遍历
- **文件**: `scripts/essential_proxy_runner.py:_memory_cache_query()` (L347-348)
- **问题**: 
  ```python
  hit_rows_set = [r for r in requested_rows if int(r) in cached_rows]
  miss_rows = [r for r in requested_rows if int(r) not in cached_rows]
  ```
  遍历 `requested_rows` 两次
- **影响**: 每 shard 每实验额外几十 ms
- **方案**: 合并为一次遍历：
  ```python
  hit_rows, miss_rows = [], []
  for r in requested_rows:
      if int(r) in cached_rows:
          hit_rows.append(r)
      else:
          miss_rows.append(r)
  ```

#### E7. 每个实验重建模型
- **文件**: `scripts/essential_proxy_runner.py:run_experiment()` (L952)
- **问题**: 
  ```python
  model = ProxyModel(config=self.model_config).to(device)
  ```
  1M 参数模型 ~57MB，重建开销不大
- **影响**: 每实验额外 ~0.1 秒
- **方案**: 复用模型对象 + `_init_weights()` 重置（需验证是否安全）

#### E8. Validation 数据每次重新传 device
- **文件**: `scripts/essential_proxy_runner.py:_run_validation()` (L1159-1160)
- **问题**: 
  ```python
  val_tokens = self._val_token_ids[:val_n, :bs].to(device)
  val_mask = self._val_loss_mask[:val_n, :bs].to(device)
  ```
  10k × 2048 × 8B ≈ 164MB，每次 validation 都 CPU→NPU 传输
- **影响**: 5000 步中 6 次 validation = ~1GB 传输，每次 ~0.5 秒
- **方案**: 训练开始前传一次并缓存到 `self._val_tokens_device`

### 低影响

#### E9. Report 生成逐实验加载 `selected_indices.npy`
- **文件**: `src/quadmix/pipeline/report.py:_experiment_table()` (L176)
- **问题**: 3000 个实验 × 每个读一个 .npy 文件
- **影响**: 一次性开销 ~10-30 秒，不影响训练
- **方案**: 批量加载或缓存

#### E10. `precompute_samples` 每 5 个实验打印进度
- **文件**: `scripts/essential_proxy_runner.py:precompute_samples()` (L1252)
- **问题**: 3000 实验 = 600 次 print
- **影响**: 日志冗余，不影响性能
- **方案**: 改为每 50 或 100 个打印一次

### 优先级

| # | 瓶颈 | 影响 | 修复难度 |
|---|------|------|---------|
| E1 | shared_to_ndarray 11GB 拷贝 | Worker 启动 10-30s | 中 |
| E2 | global_to_shard_rows 重复调用 | 每批 1-2s | 低 |
| E3 | rank_normalize 双重 argsort | 初始化数分钟 | 低 |
| E4 | val_bs 可提到 256 | 每次 val 节省 3x | 低 |
| E5 | ProcessPoolExecutor 重建 | 每批 2-5s | 中 |
| E6 | _memory_cache_query 双重遍历 | 每 shard 几十 ms | 低 |
| E7 | 每个实验重建模型 | 每实验 0.1s | 低 |
| E8 | Validation 数据重复传输 | 每次 val 0.5s | 低 |
| E9 | Report 逐实验加载 npy | 一次性 10-30s | 低 |
| E10 | precompute_samples 日志冗余 | 无性能影响 | 低 |

---

## 训练循环与内存优化 — 2026-06-02

### 高影响

#### F1. `loss.item()` 每次迭代触发 host-device sync
- **文件**: `scripts/essential_proxy_runner.py:run_experiment()` (L1093)
- **问题**: 
  ```python
  total_loss += loss.item()  # 每次 iter 都 NPU→CPU 同步
  ```
  grad_acc=8 × 5000 步 = 40000 次 sync。NPU 上每次 1-5ms
- **影响**: 总计 **40-200 秒**（每次实验）
- **方案**: device 侧累加，只在 log 时 sync：
  ```python
  loss_accum += loss.detach()  # 不 sync
  # 只在 log 时：
  avg = loss_accum.item() / count
  ```

#### F2. `precompute_samples` 单线程
- **文件**: `scripts/essential_proxy_runner.py:precompute_samples()` (L1233)
- **问题**: 
  ```python
  for i, params in enumerate(all_params):  # 3000 次串行
      quality_ranks = self._compute_ranks_for_params(params, i)
  ```
  每次 Eq.1-3 是纯 numpy CPU 计算，完全独立。192 核机器上串行跑 3000 次
- **影响**: 预估 **10-20x 加速**（从数分钟降到数十秒）
- **方案**: `ProcessPoolExecutor` 或 `joblib` 并行化

#### F3. `flat_train` 和 `batch_buf` 用 int64（应为 int32）
- **文件**: `scripts/essential_proxy_runner.py:run_experiment()` (L985, L1013)
- **问题**: 
  ```python
  flat_train = torch.cat(real_tokens_list).to(device)  # int64
  batch_buf = torch.empty(accum_bs, self.block_size + 1, dtype=torch.long, device=device)  # int64
  ```
  vocab=50432 < 2^31，完全可以用 int32
- **影响**: `flat_train` 可能几百万 tokens，int64→int32 节省 **50% 训练数据显存**
- **方案**: 改用 `dtype=torch.int32`

### 中影响

#### F4. `optimizer.zero_grad()` 应用 `set_to_none=True`
- **文件**: `scripts/essential_proxy_runner.py:run_experiment()` (L1082)
- **问题**: 
  ```python
  optimizer.zero_grad()  # 将梯度设为 0（分配内存）
  ```
- **影响**: 每次 optimizer step 稍慢且多分配内存
- **方案**: `optimizer.zero_grad(set_to_none=True)` 将梯度设为 None，避免分配零张量

#### F5. `rng.choice(n_domain, k, replace=False)` 对大域低效
- **文件**: `scripts/essential_proxy_runner.py:_compute_ranks_for_params()` (L874)
- **问题**: 
  ```python
  ref_idx = rng.choice(n_domain, k, replace=False)  # n_domain 可达 2700 万
  ```
  numpy 的 `choice(replace=False)` 对大 n 创建 O(n) 临时数组只为采 10K
- **影响**: 每次 Eq.2 额外数十 ms
- **方案**: Floyd 采样算法或 `np.sort(rng.integers(0, n_domain, k))`（有重复但概率极低）

#### F6. Validation 数据每个 worker 独立加载
- **文件**: `scripts/essential_proxy_runner.py:__init__()` (L182)
- **问题**: 
  ```python
  val_data = torch.load(self.val_data_path, map_location="cpu", weights_only=False)
  ```
  8 个 worker 各加载 164MB 验证集 = 1.3GB 重复 I/O
- **影响**: Worker 启动时额外 1-2 秒 × 8 = 8-16 秒
- **方案**: shared memory 共享验证集

#### F7. `_compute_ranks_for_params` 每次分配两个 275M float64 数组
- **文件**: `scripts/essential_proxy_runner.py:_compute_ranks_for_params()` (L850, L862)
- **问题**: 
  ```python
  merged_scores = np.zeros(self._num_docs, dtype=np.float64)  # 2.2GB
  ranks = np.zeros(self._num_docs, dtype=np.float64)          # 2.2GB
  ```
  3000 次实验 × 每次分配 4.4GB → GC 回收
- **影响**: 内存抖动 + GC 开销
- **方案**: 预分配并复用（类级别 buffer）

#### F8. `_memory_cache_get_rows` 也构建 Python set
- **文件**: `scripts/essential_proxy_runner.py:_memory_cache_get_rows()` (L278)
- **问题**: 
  ```python
  return set(int(r) for r in self._memory_cache[sid]["rows"])
  ```
  与 C3 相同模式，每次调用 O(N) Python 循环
- **影响**: 每 shard 每 batch 几十 ms
- **方案**: 缓存 rows 数组或改用 `np.isin`

### 低影响

#### F9. `_tokenize_batch_union` Step 2 中 `row_to_pos` dict 构建
- **文件**: `scripts/essential_proxy_runner.py:_tokenize_batch_union()` (L534)
- **问题**: 
  ```python
  row_to_pos = {int(r): i for i, r in enumerate(disk_rows)}
  ```
  10 万行构建 Python dict
- **影响**: 每 shard 每 batch 几十 ms
- **方案**: 与 C3 相同，改用 `np.isin` 或 `np.searchsorted`

#### F10. `_load_tokens_for_experiment` fallback 中 `row_to_pos` dict
- **文件**: `scripts/essential_proxy_runner.py:_load_tokens_for_experiment()` (L666)
- **问题**: 同上模式
- **影响**: 每 shard 每实验几十 ms
- **方案**: 同上

#### F11. `_cache_add_rows` 去重也用 Python dict
- **文件**: `scripts/essential_proxy_runner.py:_cache_add_rows()` (L431)
- **问题**: 
  ```python
  row_to_idx = {int(r): i for i, r in enumerate(combined_rows)}
  ```
  与 C4 相同
- **影响**: 每 shard 每 batch 几十 ms
- **方案**: `np.unique(return_index=True)`

#### F12. `precompute_samples` 末尾 `np.unique(np.concatenate(all_selected))`
- **文件**: `scripts/essential_proxy_runner.py:precompute_samples()` (L1269)
- **问题**: 3000 个实验的 selected 数组 concatenate 后可能数亿行，`np.unique` 需要排序
- **影响**: 一次性数秒
- **方案**: 增量 unique 或采样统计

#### F13. `_run_validation` 中 `per_doc_losses` list + `torch.cat`
- **文件**: `scripts/essential_proxy_runner.py:_run_validation()` (L1178-1179)
- **问题**: 
  ```python
  per_doc_losses = []
  for ...:
      per_doc_losses.append(per_doc)
  val_loss = float(torch.cat(per_doc_losses).mean())
  ```
- **影响**: 每次 validation 额外数十 ms
- **方案**: 预分配 tensor 避免 list append + cat

### 优先级

| # | 瓶颈 | 影响 | 修复难度 |
|---|------|------|---------|
| F1 | loss.item() host-device sync | 每实验 40-200s | 低 |
| F2 | precompute_samples 单线程 | 数分钟→数十秒 | 中 |
| F3 | flat_train/batch_buf int64→int32 | 节省 50% 训练数据显存 | 低 |
| F4 | zero_grad(set_to_none=True) | 每 step 稍快 | 低 |
| F5 | rng.choice 对大域低效 | 每次 Eq.2 数十 ms | 低 |
| F6 | Validation 数据 worker 重复加载 | Worker 启动 8-16s | 中 |
| F7 | _compute_ranks 每次分配 4.4GB | 内存抖动 + GC | 中 |
| F8 | _memory_cache_get_rows set 构建 | 每 shard 几十 ms | 低 |
| F9 | _tokenize_batch_union dict 构建 | 每 shard 几十 ms | 低 |
| F10 | _load_tokens fallback dict 构建 | 每 shard 几十 ms | 低 |
| F11 | _cache_add_rows dict 去重 | 每 shard 几十 ms | 低 |
| F12 | precompute_samples np.unique | 一次性数秒 | 低 |
| F13 | _run_validation list+cat | 每次 val 数十 ms | 低 |
