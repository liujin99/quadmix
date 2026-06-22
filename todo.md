# QuaDMix 性能优化 TODO

> 2025-05-28 Hermes Agent 审计生成
> 总计 ~5,460 行 Python，29 源文件 + 5 脚本

## 已完成

### 2026-06-03: NPU 显存碎片化修复
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py`
- **问题**: Worker 0 在 backward 时 OOM，其他 worker 正常。原因是每迭代调用 `empty_cache()` 释放完整块，导致 forward/backward 分配位置漂移，碎片累积
- **修复**: 移除训练循环中的 per-iteration `empty_cache()`，保留低频调用（训练开始、checkpoint、实验结束）
- **影响**: 允许 PyTorch 分配器自然复用缓存块，解决碎片化 OOM

### 2026-06-03: precompute_samples 并行化 + 内存优化
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:precompute_samples()`
- **问题**: 3000 实验串行处理，每次 Eq.1-3 分配 13GB（275M docs）
- **修复**: 
  1. 新增 `_sample_one_experiment` 方法，按 domain 逐域处理，内存 O(N/M) 而非 O(N)
  2. ThreadPoolExecutor 并行（numpy 释放 GIL，真正并行）
  3. 动态限制线程数防止 OOM
- **影响**: 10-20x 加速，内存从 13GB → 1.3GB/experiment

### 2026-06-03: NPU device context 修复
- **文件**: `src/quadmix/npu/device.py`
- **问题**: Worker 1-7 默认使用 NPU 0 context，导致 `.to(device)` 时 ACL stream synchronize 失败
- **修复**: 每个 worker 进程启动时调用 `torch.npu.set_device(dev_id)`
- **影响**: 8 NPU 并行训练正常工作

### 2026-06-03: bf16 混合精度训练
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py`, `scripts/demo_run_full.sh`
- **问题**: 全程 fp32，logits 13.1GB，backward 28GB，OOM
- **修复**: 
  ```python
  with torch.autocast(device_type="npu", dtype=torch.bfloat16):
      logits = model(inp)
  loss = F.cross_entropy(logits.float().view(-1, vocab), tgt.view(-1))
  ```
- **影响**: logits 13.1→6.5GB，backward 28→14GB，训练速度 1.5-2x

### 2026-06-03: Flash Attention + Pack 优化
- **文件**: `src/quadmix/core/proxy_model.py`, `src/quadmix/pipeline/essential_proxy_runner.py`
- **修复**:
  1. Flash Attention: `F.scaled_dot_product_attention` 替代显式 2048×2048 注意力矩阵，节省 8GB
  2. Pack: `np.searchsorted` 替代 Python dict，`np.save` 替代 `torch.save`，`np.load(mmap_mode='r')` 近零拷贝
- **影响**: 显存节省 8GB，Pack 时间 148s → 10-30s

### 2026-06-03: ProcessPoolExecutor tokenize (GIL bypass)
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_tokenize_shard_parallel()`
- **问题**: 48 Python threads 共享 1 GIL + 4 全局 Rust threads，实际只有 4 线程做 CPU 工作（90% 空闲）
- **修复**: ProcessPoolExecutor(48) 每个进程独立 GIL + 4 Rust threads = 192 线程真并行
- **影响**: 7 min → ~40s（10x 加速）

### 2026-06-02: ProcessPoolExecutor fork→spawn 避免 COW page fault
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_tokenize_shard_parallel()`
- **问题**: 主进程 RSS 18GB，fork 64 workers 导致 64×18GB=1.15TB COW 虚拟映射，海量 page fault 导致 tokenize 卡死
- **修复**: 使用 `mp.get_context('spawn')` 替代默认 fork，子进程从零启动不继承主进程内存
- **影响**: 启动慢 2-3 秒，但避免卡死问题

### 2026-06-02: micro_batch 64→40 避免 NPU OOM
- **文件**: `scripts/demo_run_full.sh`, `scripts/demo_run_quick.sh`
- **问题**: micro_batch=64 时 logits [64, 2048, 50432] fp32 = 26.3GB，加上梯度和 CE workspace，peak ~77GB 超过 NPU 60GB 上限
- **修复**: micro_batch=40，peak ~49GB，余量 11GB
- **影响**: global_batch 同步改为 40（无梯度累积），训练速度提升

### 2026-06-02: mmap file handle leak 修复
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_cached_shard_rows()` 等 4 处
- **问题**: `np.load(mmap_mode='r')` 未关闭文件句柄，101 shards 累积 101 个打开的 mmap fd 导致 IO hang
- **修复**: 移除 `mmap_mode='r'`，直接加载到内存

### 2026-06-02: 综合性能计时系统
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py`, `src/quadmix/pipeline/real_pipeline.py`
- **内容**: PerfTimer 类 + stage 级计时，实时输出 + 结束汇总

### 2026-06-01: 质量分数方向反转修复 (A1)
- **文件**: `scripts/preprocess/preprocess_essential_web_v1_sharded.py:extract_quality_signals()`
- **问题**: FastText 分数 higher=better，但代码按 smaller=better 处理，选择最低质量文档
- **修复**: 在 `extract_quality_signals()` 中取反 `qs = -qs`
- **影响**: 正确选择高质量文档

### 2026-06-01: Token-weighted Eq.2 (A5) + LightGBM early stopping (B1)
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py`, `src/quadmix/pipeline/optimizer.py`
- **修复**:
  1. A5: Proxy Eq.2 使用 token 数加权计算百分位，匹配论文
  2. B1: LightGBM 添加 early stopping (patience=50)，防止过拟合

## P0 — 高影响（预计加速 30-50%）

### 1. HDF5/LMDB 替代 np.savez 做 token cache
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_cache_add_rows()` (L320-434)
- **问题**: 每次增量添加几行，都全量读-合并-写整个 .npz（最终 ~200MB/shard）
- **预计**: 磁盘 I/O 减少 70%+
- **方案**: HDF5 appendable dataset 或 LMDB key-value store

### 2. Shared memory metadata — 消除 worker 重载 15GB
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_worker_dynamic_loop()` (L1636-1693)
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

### 2026-06-01 ~ 06-03 新增完成项

- [x] A1: 质量分数方向反转修复 ✓ 2026-06-01 (e852787)
- [x] A5: Proxy Eq.2 token-weighted ✓ 2026-06-01 (2269457)
- [x] B1: LightGBM early stopping ✓ 2026-06-01 (2269457)
- [x] C1: tokenize_all_needed 并行化 ✓ 2026-06-02 (55a4248)
- [x] D1: bf16 混合精度训练 ✓ 2026-06-03 (23d022b)
- [x] F2: precompute_samples 并行化 + 内存优化 ✓ 2026-06-03 (dbd1bf6)
- [x] Flash Attention ✓ 2026-06-03 (601337f)
- [x] Pack 优化 (np.searchsorted + np.save) ✓ 2026-06-03 (601337f)
- [x] NPU device context 修复 ✓ 2026-06-03 (4cb7177)
- [x] NPU 显存碎片化修复 ✓ 2026-06-03 (d450d4a)
- [x] ProcessPoolExecutor tokenize (GIL bypass) ✓ 2026-06-03 (55a4248)
- [x] mmap file handle leak 修复 ✓ 2026-06-02 (c948f91)
- [x] PerfTimer 综合计时系统 ✓ 2026-06-02 (c948f91)
- [x] micro_batch 64→40 (NPU OOM) ✓ 2026-06-02 (1099020)
- [x] ProcessPoolExecutor spawn mode ✓ 2026-06-02 (bacba24)
- [x] val_bs 16→64 ✓ 2026-06-02 (019c903)

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

#### ~~A1. 质量分数方向反转~~ ✅ 已修复
- **文件**: `scripts/preprocess/preprocess_essential_web_v1_sharded.py:extract_quality_signals()`
- **完成**: 2026-06-01 (commit e852787)
- **问题**: FastText 分数 higher=better，但代码按 smaller=better 处理
- **修复**: 在 `extract_quality_signals()` 中取反 `qs = -qs`
- **影响**: 正确选择高质量文档

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

#### ~~A5. Proxy 实验中 Eq.2 缺少 token 加权~~ ✅ 已修复
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_compute_ranks_for_params()`
- **完成**: 2026-06-01 (commit 2269457)
- **修复**: Proxy Eq.2 使用 token 数加权计算百分位，匹配论文

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

#### ~~B1. LightGBM 缺少 early stopping~~ ✅ 已修复
- **文件**: `src/quadmix/pipeline/optimizer.py:train_regressor()`
- **完成**: 2026-06-01 (commit 2269457)
- **修复**: 添加 early stopping (patience=50)，防止过拟合
- **注**: K-fold 交叉验证未实施，当前使用单一 train/val 划分

#### B2. 参数空间搜索效率低
- **文件**: `src/quadmix/pipeline/optimizer.py:search_optimal()` (L296-337)
- **问题**: 200+ 维空间随机搜索，论文 Limitations 也提到此问题
- **方案**: Bayesian Optimization (Optuna/TPE) 或 CMA-ES

#### B3. 采样函数参数冗余
- **问题**: 不同 (λ, ω, η, ε) 可能产生相似采样曲线，引入回归不确定性
- **方案**: 对采样函数做正交化或参数约束

#### B4. 质量分数归一化函数选择 ✅ 已确认 rank 最优
- **文件**: `src/quadmix/utils/normalization.py`
- **完成**: 2026-06-04
- **实验结果**:

  | Normalizer | 实验数 | Val R² (single) | Bootstrap mean Val R² | val_loss std | 结论 |
  |------------|--------|-----------------|-----------------------|--------------|------|
  | rank | 64 | 0.538 | 0.779 | 0.137 | 信号最强 |
  | rank | 200 | 0.880 | 0.808 | — | 稳定，最佳 |
  | zscore | 64 | 0.137 | 0.576 | 0.096 | 信号中等 |
  | log1p_z | 64 | 0.006 | — | 0.094 | 信号最弱，不可用 |

- **rank 最优的原因**:
  - 每个准则强制展开到 [0,1] 均匀分布，α 对不同准则有同等话语权
  - merged_score 差异大 → 不同 α 组合产生显著不同的采样分布 → val_loss 方差大（0.137）
  - LightGBM 能清晰学到 α 的作用
- **log1p_z/minmax 失败的原因**:
  - 保留原始分布形状，偏斜准则（dclm IQR=0.008）的值域仍然极小
  - merged_score 被均匀分布的准则（fineweb_edu IQR=0.645）主导
  - 其他准则的 α 几乎不起作用 → val_loss 方差小 → LightGBM 无信号可学
- **rank "扭曲分布" 无害的原因**:
  - merged_score 是 5 个准则的加权和，噪声层文档在其他准则上有区分
  - sigmoid 的平滑性抹平了噪声层内部的假排名差异
  - 信号层文档 merged_score 整体高于噪声层，α 变化有效影响采样概率
  - α 学到的本质是"哪些准则更重要"，噪声层的假排序不影响最终结果
- **当前状态**: 默认 normalizer 为 rank，R²=0.808

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
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_run_validation()` (L1114-1142)
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

#### B11. 归一化方法改进：阈值 + rank（两阶段归一化）
- **文件**: `src/quadmix/utils/normalization.py`
- **背景**: rank 是当前最优（R²=0.808），但在 ω 参数范围扩大时存在局限性
- **问题**:
  - rank 把噪声层（如 dclm 76% 近零文档）赋予了假排名
  - 当前 ω 范围（top 10-20%）只采高质量文档，噪声层进不来，没问题
  - 如果 ω 调大（top 50%），噪声层文档开始被采到，引入虚假信号
  - 后果：val_loss 方差混入噪声、LightGBM 学到虚假规律、R² 下降
- **方案**: 阈值 + rank（两阶段归一化）
  ```
  Step 1: 识别噪声层 vs 信号层（per criterion）
  Step 2: 噪声层统一赋 0，信号层做 rank
  ```
  以 dclm 为例：
  ```
  原始:  [0.001, 0.002, ..., 0.008, 0.1, 0.3, 0.5, 0.9]
          ←── 76% 噪声层 ──→      ←── 24% 信号层 ──→

  rank:    [0.01, 0.05, ..., 0.76,  0.77, 0.82, 0.88, 0.99]
           ← 假排名，噪声被展开 →  ← 信号被压缩 →

  阈值+rank: [0, 0, ..., 0,         0.04, 0.25, 0.58, 1.0]
             ← 噪声统一为 0 →       ← 信号层内部 rank →
  ```
- **效果**:
  - ω 小时（top 10%）：只采信号层文档，和纯 rank 一样
  - ω 大时（top 50%）：噪声层全部 merged_score=0，不会被采到。采样范围扩大到信号层内部更低的文档，但不会混入噪声
- **阈值确定方法**:
  1. 分位数阈值：取 p75-p80（dclm 76% 近零 → p76 附近是自然分界）
  2. 拐点检测：对排序后的分数做二阶差分，找斜率突变最大的位置
  3. 固定阈值：原始分数 > 0.05 才算信号（简单但需要 per-criterion 调参）
- **per-criterion 处理策略**:

  | 类型 | 标签 | 分布特征 | 推荐处理 |
  |------|------|---------|---------|
  | 右偏（噪声+信号） | dclm, eai_general_math | 76-77% 近 0, skew>+4.9 | 阈值 + rank |
  | 均匀 | fineweb_edu | 均匀 [0, 3.94], skew=+1.09 | 直接 rank |
  | 左偏（饱和） | english | 98% 在 [0.5, 1.0], skew=-3.63 | 反向阈值 + rank 或二值化 |
  | 中度偏斜 | eai_open_web_math | 中等分布, skew=+2.54 | 直接 rank |

- **通用判断框架**（遇到新质量标签时）:
  ```
  1. 画分布直方图，看偏度 skew 和 IQR
  2. 判断类型：
     - |skew| > 3 且有明确拐点 → 阈值 + rank
     - |skew| < 2 且分布均匀 → rank 或 minmax 均可
     - 2 < |skew| < 3 → 看拐点是否明显，不明显则直接 rank
  3. 阈值确定：
     - 优先用拐点检测（二阶差分最大突变点）
     - 备选：分位数（右偏取 p75-p80，左偏取 p20-p25）
  4. 验证：
     - 跑 64 组实验，对比 Val R² 和 val_loss std
     - val_loss std 越大越好（信号强）
     - Val R² 越高越好（LightGBM 能学到）
  ```
- **实施建议**:
  - 当前优先级：低（R²=0.808，ω 范围 top 10-20%，够用）
  - 触发条件：如果未来需要 ω 覆盖更大范围（top 50%），必须实施
  - 对 dclm/math 加阈值收益最大（这两个偏斜最严重）

---

## 优先级总结

| 优先级 | 编号 | 问题 | 状态 |
|--------|------|------|------|
| ~~P0~~ | ~~A1~~ | ~~质量分数方向反转~~ | ✅ 已修复 |
| P1 | A4 | Proxy 训练 token 数太少 | 待评估 |
| P1 | A2,A3 | N=5/M=10 vs N=3/M=26 | 设计选择，不改 |
| ~~P2~~ | ~~A5~~ | ~~Proxy Eq.2 缺少 token 加权~~ | ✅ 已修复 |
| ~~P2~~ | ~~B1~~ | ~~LightGBM 无 early stopping~~ | ✅ 已修复 |
| ~~P2~~ | ~~B4~~ | ~~归一化函数选择~~ | ✅ 已确认 rank 最优 |
| P2 | B2 | Bayesian Optimization | 待实施 |
| P3 | A7,B5 | 缺少 BMK 变体 / 消融支持 | 待实施 |
| P3 | B11 | 阈值 + rank 两阶段归一化 | 待实施（ω 扩大时） |

---

## 数据处理性能优化 — 2026-06-02

### ~~C1. `tokenize_all_needed` 单线程 tokenize~~ ✅ 已完成
- **完成**: 2026-06-02 (commit 55a4248)
- **修复**: ProcessPoolExecutor(48) 替代 ThreadPool，每个进程独立 GIL + 4 Rust threads = 192 线程真并行
- **效果**: 7 min → ~40s（10x 加速）

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

| # | 瓶颈 | 影响 | 状态 |
|---|------|------|------|
| ~~C1~~ | ~~tokenize_all_needed 单线程~~ | ~~CPU 模式慢 10-20x~~ | ✅ 已完成 |
| C2 | Pack/Unpack 磁盘 I/O | 每实验 2-10s | 部分完成 (np.save) |
| C3 | _cached_shard_rows set 构建 | 每 shard 几十 ms | 待实施 |
| C4 | _memory_cache_add_rows dict 去重 | 每 shard 几十 ms | 待实施 |
| C5 | 预处理 df.apply | 一次性 | 待实施 |
| C6 | LRU list.remove | shard 少时小 | 待实施 |

---

## NPU 显存与性能优化 — 2026-06-02

### 显存瓶颈分析

**根因：logits 张量占 95% 显存**

micro_batch=32, block_size=2048, vocab=50432 时的显存分布（bf16 后）：

| 组件 | 大小 | 说明 |
|------|------|------|
| 模型权重 (1M params) | 57 MB | embed + 2 层 transformer |
| AdamW 优化器状态 | 114 MB | 2× 权重副本 |
| ~~causal_mask buffer~~ | ~~16 MB~~ | Flash Attention 后不再需要 |
| flat_train (~8M tokens) | 68 MB | 训练数据 |
| batch_buf + 辅助 buffer | 1 MB | 预分配 |
| **静态总计** | **~240 MB** | |
| | | |
| embed 输出 (32×2048×256) | 16.8 MB | forward (bf16) |
| ~~attention scores ×2 层~~ | ~~2.1 GB~~ | Flash Attention 不物化 |
| **logits (32×2048×50432×2B)** | **6.5 GB** | **bf16 减半** |
| | | |
| logits（保留用于梯度） | 6.5 GB | backward |
| logits 梯度 | 6.5 GB | backward |
| ~~attention 梯度 ×2 层~~ | ~~2 GB~~ | Flash Attention 不物化 |
| **backward 总计** | **~13 GB** | |
| | | |
| **峰值（forward 末尾）** | **~20 GB** | 静态 + logits |
| **NPU 实际占用** | **~26 GB** | 含内存池碎片 |
| **剩余可用** | **~32 GB** | 安全 |

**结论：**
- bf16 + Flash Attention 后，micro_batch=32 安全运行
- logits 从 13.1 GB (fp32) → 6.5 GB (bf16)
- attention 从 2.1 GB → 0 (Flash Attention 不物化)
- backward 从 28 GB → 13 GB

### ~~D1. bf16 混合精度训练~~ ✅ 已完成
- **完成**: 2026-06-03 (commit 23d022b)
- **修复**: 
  ```python
  with torch.autocast(device_type="npu", dtype=torch.bfloat16):
      logits = model(inp)
  loss = F.cross_entropy(logits.float().view(-1, vocab), tgt.view(-1))
  ```
- **效果**: logits 13.1→6.5GB，backward 28→14GB，训练速度 1.5-2x

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

| # | 方案 | 910B3 可行性 | 收益 | 状态 |
|---|------|:-----------:|------|------|
| ~~D1~~ | ~~bf16 + 手动 upcast~~ | 高 | ~2x 显存 + ~1.5x 速度 | ✅ 已完成 |
| D2 | Fused CE | 不可行 | — | 放弃 |
| D3 | Chunked lm_head | 可行 | 显存可控但可能更慢 | 不推荐 |

### 其他已尝试的优化

**micro_batch 调优历史：**
- commit 8fb7a5c: micro_batch=64 → OOM（logits 26.3 GB fp32）
- commit 019c903: micro_batch=32 → OOM（logits 13.1 GB + backward 13.1 GB = 26.2 GB，只剩 5.4 GB）
- commit 350c967: micro_batch=8 → 安全（logits 3.3 GB + backward 3.3 GB = 6.6 GB，余量 ~31 GB）
- commit 1099020: micro_batch=40 → 安全（fp32，peak ~49GB）
- commit 23d022b: bf16 → micro_batch=32 安全（logits 6.5 GB bf16 + backward 6.5 GB）

**当前配置：**
- micro_batch=32, grad_acc=2, global_batch=64
- bf16 混合精度，每 step 2 次 forward/backward
- Flash Attention 节省 8GB 显存

---

## 核心算法与流水线优化 — 2026-06-02

### 高影响

#### E1. `shared_to_ndarray` 每个 worker 拷贝 11GB
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:shared_to_ndarray()` (L82-83)
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
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_tokenize_batch_union()` (L487-492 和 L584-585)
- **问题**: Step 1 和 Step 4 对同一批实验各调一次 `global_to_shard_rows`（内部做 `searchsorted` + `argsort` + `unique`），完全冗余
- **影响**: 每批实验额外 1-2 秒（275M docs 的 searchsorted 开销）
- **方案**: 缓存 Step 1 的 `shard_to_exp_rows` 结果，Step 4 直接复用

#### ~~E3. `rank_normalize` 双重 argsort~~ ✅ 已确认 rank 最优（不改）
- **文件**: `src/quadmix/utils/normalization.py`, `src/quadmix/pipeline/essential_proxy_runner.py`, `src/quadmix/core/quality_merger.py`
- **完成**: 2026-06-04（确认 rank 最优，回退到 rank）
- **原问题**: rank_normalize 用双重 argsort (O(N log N) × 2)，且丢失数值关系导致 α 权重失效
- **尝试**: 改为 zscore（Val R²=0.137）和 log1p_z（Val R²=0.006），均远差于 rank（Val R²=0.538-0.880）
- **结论**: rank 的"扭曲分布"在 Eq.1→Eq.2→Eq.3 链路中无害（sigmoid 平滑性 + 多准则加权和），且提供最大参数敏感度。保持 rank。

#### E4. Validation `val_bs` 可以更激进（部分完成）
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_run_validation()` (L1325)
- **当前**: `val_bs=64`（从 16 提升到 64，commit 019c903）
- **问题**: validation 是 `no_grad`，不需要 backward 显存，可以更激进
- **方案**: 
  - 提到 `val_bs=256`（logits `256×2048×50432×4B = 26.3 GB`，no_grad 下安全）
  - 10k docs 只需 40 次 forward（vs 当前 157 次）
  - 每次 validation 节省 ~3x 时间
- **状态**: 部分完成（16→64），可继续提升到 256

#### E5. `ProcessPoolExecutor` 每次 tokenize 重建
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_tokenize_shard_parallel()` (L1793, L1842)
- **问题**: Stage 1 IO 和 Stage 2 tokenize 各创建一个 `ProcessPoolExecutor`，每次创建/销毁 48-64 个进程
- **影响**: 每次 tokenize 批次额外 2-5 秒进程创建开销
- **方案**: 复用持久化进程池（类级别或全局）

### 中影响

#### E6. `_memory_cache_query` 双重遍历
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_memory_cache_query()` (L347-348)
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
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:run_experiment()` (L952)
- **问题**: 
  ```python
  model = ProxyModel(config=self.model_config).to(device)
  ```
  1M 参数模型 ~57MB，重建开销不大
- **影响**: 每实验额外 ~0.1 秒
- **方案**: 复用模型对象 + `_init_weights()` 重置（需验证是否安全）

#### E8. Validation 数据每次重新传 device
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_run_validation()` (L1159-1160)
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
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:precompute_samples()` (L1252)
- **问题**: 3000 实验 = 600 次 print
- **影响**: 日志冗余，不影响性能
- **方案**: 改为每 50 或 100 个打印一次

### 优先级

| # | 瓶颈 | 影响 | 状态 |
|---|------|------|------|
| E1 | shared_to_ndarray 11GB 拷贝 | Worker 启动 10-30s | 待实施 |
| E2 | global_to_shard_rows 重复调用 | 每批 1-2s | 待实施 |
| ~~E3~~ | ~~rank_normalize 双重 argsort~~ | ~~初始化数分钟~~ | ✅ 已确认 rank 最优（不改） |
| E4 | val_bs 可提到 256 | 每次 val 节省 3x | 部分完成 (16→64) |
| E5 | ProcessPoolExecutor 重建 | 每批 2-5s | 待实施 |
| E6 | _memory_cache_query 双重遍历 | 每 shard 几十 ms | 待实施 |
| E7 | 每个实验重建模型 | 每实验 0.1s | 待实施 |
| E8 | Validation 数据重复传输 | 每次 val 0.5s | 待实施 |
| E9 | Report 逐实验加载 npy | 一次性 10-30s | 待实施 |
| E10 | precompute_samples 日志冗余 | 无性能影响 | 待实施 |

---

## 训练循环与内存优化 — 2026-06-02

### 高影响

#### F1. `loss.item()` 每次迭代触发 host-device sync
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:run_experiment()` (L1093)
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

#### ~~F2. `precompute_samples` 单线程~~ ✅ 已完成
- **完成**: 2026-06-03 (commit dbd1bf6)
- **修复**:
  1. 新增 `_sample_one_experiment` 方法，按 domain 逐域处理，内存 O(N/M) 而非 O(N)
  2. ThreadPoolExecutor 并行（numpy 释放 GIL，真正并行）
  3. 动态限制线程数防止 OOM
- **效果**: 10-20x 加速，内存从 13GB → 1.3GB/experiment

#### F3. `flat_train` 和 `batch_buf` 用 int64（应为 int32）
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:run_experiment()` (L985, L1013)
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
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:run_experiment()` (L1082)
- **问题**: 
  ```python
  optimizer.zero_grad()  # 将梯度设为 0（分配内存）
  ```
- **影响**: 每次 optimizer step 稍慢且多分配内存
- **方案**: `optimizer.zero_grad(set_to_none=True)` 将梯度设为 None，避免分配零张量

#### F5. `rng.choice(n_domain, k, replace=False)` 对大域低效
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_compute_ranks_for_params()` (L874)
- **问题**: 
  ```python
  ref_idx = rng.choice(n_domain, k, replace=False)  # n_domain 可达 2700 万
  ```
  numpy 的 `choice(replace=False)` 对大 n 创建 O(n) 临时数组只为采 10K
- **影响**: 每次 Eq.2 额外数十 ms
- **方案**: Floyd 采样算法或 `np.sort(rng.integers(0, n_domain, k))`（有重复但概率极低）

#### F6. Validation 数据每个 worker 独立加载
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:__init__()` (L182)
- **问题**: 
  ```python
  val_data = torch.load(self.val_data_path, map_location="cpu", weights_only=False)
  ```
  8 个 worker 各加载 164MB 验证集 = 1.3GB 重复 I/O
- **影响**: Worker 启动时额外 1-2 秒 × 8 = 8-16 秒
- **方案**: shared memory 共享验证集

#### F7. `_compute_ranks_for_params` 每次分配两个 275M float64 数组
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_compute_ranks_for_params()` (L850, L862)
- **问题**: 
  ```python
  merged_scores = np.zeros(self._num_docs, dtype=np.float64)  # 2.2GB
  ranks = np.zeros(self._num_docs, dtype=np.float64)          # 2.2GB
  ```
  3000 次实验 × 每次分配 4.4GB → GC 回收
- **影响**: 内存抖动 + GC 开销
- **方案**: 预分配并复用（类级别 buffer）

#### F8. `_memory_cache_get_rows` 也构建 Python set
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_memory_cache_get_rows()` (L278)
- **问题**: 
  ```python
  return set(int(r) for r in self._memory_cache[sid]["rows"])
  ```
  与 C3 相同模式，每次调用 O(N) Python 循环
- **影响**: 每 shard 每 batch 几十 ms
- **方案**: 缓存 rows 数组或改用 `np.isin`

### 低影响

#### F9. `_tokenize_batch_union` Step 2 中 `row_to_pos` dict 构建
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_tokenize_batch_union()` (L534)
- **问题**: 
  ```python
  row_to_pos = {int(r): i for i, r in enumerate(disk_rows)}
  ```
  10 万行构建 Python dict
- **影响**: 每 shard 每 batch 几十 ms
- **方案**: 与 C3 相同，改用 `np.isin` 或 `np.searchsorted`

#### F10. `_load_tokens_for_experiment` fallback 中 `row_to_pos` dict
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_load_tokens_for_experiment()` (L666)
- **问题**: 同上模式
- **影响**: 每 shard 每实验几十 ms
- **方案**: 同上

#### F11. `_cache_add_rows` 去重也用 Python dict
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_cache_add_rows()` (L431)
- **问题**: 
  ```python
  row_to_idx = {int(r): i for i, r in enumerate(combined_rows)}
  ```
  与 C4 相同
- **影响**: 每 shard 每 batch 几十 ms
- **方案**: `np.unique(return_index=True)`

#### F12. `precompute_samples` 末尾 `np.unique(np.concatenate(all_selected))`
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:precompute_samples()` (L1269)
- **问题**: 3000 个实验的 selected 数组 concatenate 后可能数亿行，`np.unique` 需要排序
- **影响**: 一次性数秒
- **方案**: 增量 unique 或采样统计

#### F13. `_run_validation` 中 `per_doc_losses` list + `torch.cat`
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py:_run_validation()` (L1178-1179)
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

| # | 瓶颈 | 影响 | 状态 |
|---|------|------|------|
| F1 | loss.item() host-device sync | 每实验 40-200s | 待实施 |
| ~~F2~~ | ~~precompute_samples 单线程~~ | ~~数分钟→数十秒~~ | ✅ 已完成 |
| F3 | flat_train/batch_buf int64→int32 | 节省 50% 训练数据显存 | 待实施 |
| F4 | zero_grad(set_to_none=True) | 每 step 稍快 | 待实施 |
| F5 | rng.choice 对大域低效 | 每次 Eq.2 数十 ms | 待实施 |
| F6 | Validation 数据 worker 重复加载 | Worker 启动 8-16s | 待实施 |
| F7 | _compute_ranks 每次分配 4.4GB | 内存抖动 + GC | 待实施 |
| F8 | _memory_cache_get_rows set 构建 | 每 shard 几十 ms | 待实施 |
| F9 | _tokenize_batch_union dict 构建 | 每 shard 几十 ms | 待实施 |
| F10 | _load_tokens fallback dict 构建 | 每 shard 几十 ms | 待实施 |
| F11 | _cache_add_rows dict 去重 | 每 shard 几十 ms | 待实施 |
| F12 | precompute_samples np.unique | 一次性数秒 | 待实施 |
| F13 | _run_validation list+cat | 每次 val 数十 ms | 待实施 |

---

## 推理准确性与置信度提升 — 2026-06-22

### 搜索策略升级

#### G1. Bayesian Optimization 替代随机搜索
- **文件**: `src/quadmix/pipeline/optimizer.py:search_optimal()`
- **问题**: 100K 点在 90 维空间极度稀疏（每维平均 ~3 个点），随机搜索效率低
- **方案**:
  1. **TPE (Tree-structured Parzen Estimator)**: 用已评估点构建概率模型，迭代采样更有希望的区域
  2. **CMA-ES**: 协方差矩阵自适应进化策略，适合连续高维空间
  3. 在 `search_optimal()` 中增加 `bayesian` 模式，用 `optuna` 库
- **预期效果**: 同等搜索预算下，Search Lift 提升 2-5x，Spearman +0.1~0.2
- **实施成本**: 中（加 optuna 依赖，~200 行代码）
- **关联**: 扩展 B2

#### G2. 两阶段搜索（粗筛 + 精搜）
- **文件**: `src/quadmix/pipeline/optimizer.py:search_optimal()`
- **方案**:
  - Stage 1: 100K 随机点粗筛 → top-1000
  - Stage 2: 在 top-1000 邻域做局部精细搜索（小范围扰动 + BO）
- **预期效果**: 有效提高 Top-K Recall
- **实施成本**: 低（在现有随机搜索基础上增加第二阶段）

### 模型改进

#### G3. Conformal Prediction（严格预测区间）
- **文件**: `src/quadmix/pipeline/optimizer.py`
- **问题**: 当前只有 bootstrap CI，缺乏严格的预测区间
- **方案**:
  1. **Conformal Prediction**: 无需分布假设，保证覆盖率（如 90% 预测区间）
  2. **LightGBM Quantile Regression**: 直接预测 10%/90% 分位数
  3. 每个搜索结果附带置信区间，回答"这个配比有多可靠"
- **预期效果**: 提供有理论保证的预测区间，增强搜索结果可信度
- **实施成本**: 低（后处理，~100 行代码）

#### G4. 多模型家族集成
- **文件**: `src/quadmix/pipeline/optimizer.py`
- **问题**: 只用 LightGBM 一种回归器（B9 的扩展）
- **方案**:
  1. **XGBoost + CatBoost**: 不同树结构偏好，减少单模型偏差
  2. **Ridge/ElasticNet**: 线性 baseline，高维时可能更稳
  3. **Gaussian Process**: 自带不确定性估计，适合小数据
  4. 取 ensemble average 或 stacking
- **预期效果**: R² +0.05~0.1，降低单模型偏差
- **实施成本**: 中（需统一接口 + 集成策略）

#### G5. Multi-Task 学习 / Stacking
- **文件**: `src/quadmix/pipeline/optimizer.py:_train_per_task_models()`
- **问题**: 21 个独立 LightGBM 无法共享跨 task 信息
- **方案**:
  1. **Stacking**: 为每个 task 添加其他 task 的预测作为额外特征
  2. **共享特征工程**: 利用 task 间相关性提升低 R² task 的预测
- **预期效果**: 减少过拟合，提升低 R² task 的预测质量
- **实施成本**: 中

### 实验设计优化

#### G6. Active Learning 选择实验点
- **文件**: `src/quadmix/pipeline/param_sampler.py`, `src/quadmix/pipeline/optimizer.py`
- **问题**: 当前 ~900 个实验是随机采样的，信息密度不均
- **方案**:
  1. 用已有模型预测未评估区域的**不确定性**
  2. 优先在**高不确定性 + 高预测性能**区域跑新实验
  3. 迭代式：训练 → 选点 → 评估 → 重新训练
- **预期效果**: 用更少实验获得更高信息增益
- **实施成本**: 中

#### G7. 增加实验数量到 2000+
- **问题**: 当前 874 实验 vs 90 维参数 → ~10 samples/dim，偏少
- **方案**: 增加到 2000-3000 实验，配合 Active Learning 选择最有信息量的点
- **预期效果**: R² +0.1~0.2
- **实施成本**: 高（算力，每实验 ~30 分钟 NPU 训练）

#### G8. 参数空间正交化
- **文件**: `src/quadmix/pipeline/param_sampler.py`, `src/quadmix/core/sampler.py`
- **问题**: 不同 (λ, ω, η, ε) 组合产生相似采样曲线，引入回归不确定性（B3 扩展）
- **方案**:
  1. **PCA/特征变换**: 将 4 个采样参数映射到"采样曲线形状"的主成分空间
  2. 对 sigmoid 输出做 PCA，取前 2-3 个主成分作为新参数
- **预期效果**: 降低有效维度（90 → ~60），减少回归噪声
- **实施成本**: 中

#### G9. 多 Proxy 集成（减少标签噪声）
- **文件**: `src/quadmix/pipeline/essential_proxy_runner.py`
- **问题**: 单个 proxy model 训练有随机性，同一参数配置不同 seed 可能得到不同 loss
- **方案**:
  1. 对同一参数配置训练 3 个不同 seed 的 proxy model
  2. 取 loss 均值作为标签 → 减少训练随机性噪声
  3. 取 loss 方差 → 额外的不确定性信号
- **预期效果**: 标签噪声降低 ~√3 倍
- **实施成本**: 高（3x 训练算力）

### 验证与评估增强

#### G10. 扩大验证集（特别是低样本 task）
- **文件**: `scripts/validation_set/prepare_core_bmk_v5.py`
- **问题**: 部分 task 样本太少（repeat_copy_logic 32, operators 210, lsat_ar 230），loss 估计方差大 → 噪声标签
- **方案**: 扩充到 50K+ samples，特别是 <500 样本的 task 补充到 500+
- **预期效果**: 减少 label noise，提升 per-task R²
- **实施成本**: 中（需要找到可用数据源）

#### G11. Proxy 模型容量提升
- **文件**: `src/quadmix/core/proxy_model.py`
- **问题**: 1M 参数无法做推理任务（copa, commonsense_qa, lsat_ar），这些 task 的 loss 信号接近噪声
- **方案**:
  1. 提升到 5M-10M 参数（仍在可承受的训练成本内）
  2. 或改用 2-3 个不同大小的 proxy，用 scaling law 外推
- **预期效果**: 推理类 task 的 loss 信号从噪声变为有效信号
- **实施成本**: 高（算力 + 显存）

### 特征工程

#### G12. 添加非线性/交互特征
- **文件**: `src/quadmix/pipeline/optimizer.py`, `src/quadmix/core/types.py`
- **问题**: 当前 90 维是原始参数，LightGBM 需要自己学交互
- **方案**:
  1. **交互特征**: alpha_i × lambda_j（质量权重 × 采样参数的交互）
  2. **聚合特征**: 每个 domain 的"有效采样率"（sigmoid 输出积分）
  3. **统计特征**: 每个 domain 的文档数、平均质量分等
- **预期效果**: 帮助 LightGBM 更快学到关键 pattern
- **实施成本**: 低

#### G13. SHAP/fANOVA 参数敏感度分析
- **文件**: `src/quadmix/pipeline/optimizer.py`
- **方案**:
  1. 用 SHAP 或 fANOVA 分析哪些参数对 loss 影响最大
  2. 冻结不敏感参数 → 降低有效搜索维度
  3. 对敏感参数做更精细的搜索网格
- **预期效果**: 搜索效率提升 2-3x
- **实施成本**: 低（后处理分析）

### 优先级总结

| 优先级 | 编号 | 改进 | 预期收益 | 实施成本 | 状态 |
|:---|:---|:---|:---|:---|:---|
| **P0** | G1 | Bayesian Optimization | Spearman +0.1~0.2 | 中 | 待实施 |
| **P0** | G3 | Conformal Prediction | 置信区间保证 | 低 | 待实施 |
| **P1** | G7 | 增加实验到 2000+ | R² +0.1~0.2 | 高（算力） | 待实施 |
| **P1** | G4 | 多模型集成 | R² +0.05~0.1 | 中 | 待实施 |
| **P1** | G8 | 参数正交化 | 有效维度降低 | 中 | 待实施 |
| **P1** | G2 | 两阶段搜索 | Top-K Recall 提升 | 低 | 待实施 |
| **P2** | G6 | Active Learning | 信息增益最大化 | 中 | 待实施 |
| **P2** | G10 | 扩大验证集 | 标签噪声降低 | 中 | 待实施 |
| **P2** | G5 | Multi-Task / Stacking | 低 R² task 提升 | 中 | 待实施 |
| **P2** | G11 | Proxy 容量提升 | 推理 task 信号 | 高（算力） | 待实施 |
| **P2** | G9 | 多 Proxy 集成 | 标签噪声 √3 倍 | 高（算力） | 待实施 |
| **P3** | G13 | SHAP 敏感度分析 | 搜索效率 2-3x | 低 | 待实施 |
| **P3** | G12 | 非线性/交互特征 | 模型表达力 | 低 | 待实施 |

---

## LightGBM 拟合质量优化 — 2026-06-22

### ~~H1. 自适应超参数公式~~ ✅ 已完成
- **文件**: `src/quadmix/pipeline/optimizer.py:_build_model()`
- **问题**: 硬编码两档（n<500 和 n>=500），断崖跳变，无法适配不同样本量
- **修复**: 基于 `n_train` 的连续公式，自动适配：
  ```python
  num_leaves = min(31, max(7, int(sqrt(n))))
  max_depth = min(8, max(3, int(log2(n))))
  min_child = max(5, n // 20)
  lr = max(0.01, 0.5 / sqrt(n))
  reg = max(0.05, 1.0 * (200/n)^0.5)
  n_est = min(2000, max(300, 20*sqrt(n)))
  ```
- **效果**: 样本少时自动保守（小树+强正则），样本多时自动放开，零人工干预

### ~~H2. 目标变量 log 变换~~ ✅ 已完成
- **文件**: `src/quadmix/pipeline/optimizer.py:fit()/predict()/score()/save()/load()`
- **问题**: cross-entropy loss 正数且右偏，直接拟合效率低
- **修复**:
  - `fit()`: `y = log(loss + offset)`，offset 处理 loss<=0 的情况
  - `predict()`: `exp(raw_pred) - offset`，还原到原始 loss 空间
  - `score()`: 手动计算 R²（在原始空间），避免 sklearn 内部 score 用 log 空间
  - `save()/load()`: 持久化 `_log_transform` 和 `_log_offset`
- **效果**: 对数变换让分布更对称，LightGBM 更好拟合；预测和评估仍在原始空间

### ~~H3. 大样本加 max_depth~~ ✅ 已完成（包含在 H1 中）
- **问题**: 大样本 regime 无 `max_depth`，只靠 `num_leaves` 控制复杂度
- **修复**: H1 的连续公式对所有 n_train 都设置 `max_depth`

### ~~H4. Fold 余数修复~~ ✅ 已完成
- **文件**: `src/quadmix/pipeline/optimizer.py:_train_per_task_models()`
- **问题**: `fold_size = n_total // n_folds` + 顺序切片，`n_total % n_folds` 个样本丢失
- **修复**: `np.array_split(indices, n_folds)` 自动均分余数
- **效果**: 所有样本都参与 CV 评估，R² 估计更准确
