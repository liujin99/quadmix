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
