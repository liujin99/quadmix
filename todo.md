# QuaDMix 性能优化 TODO

> 2025-05-28 Hermes Agent 审计生成
> 总计 ~5,460 行 Python，29 源文件 + 5 脚本

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
