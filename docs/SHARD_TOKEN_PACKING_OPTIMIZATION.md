# Shard Token Packing 优化设计 — 消除全局索引的单线程 memcpy 瓶颈

> **版本**: v1.0
> **日期**: 2026-07-26
> **状态**: 已实现
> **分支**: dev/dataset-schema

---

## 1. 背景与动机

### 1.1 瓶颈现象

`demo_run_stem.sh` 在并行 tokenize 完成后（`[ParallelTokenize] 19,467,325 docs in 619.9s`）卡住，CPU 占用降至 4%，长时间无输出。卡点**不在**并行 tokenize 阶段，而在 `tokenize_all_needed` 返回后的**单进程聚合阶段**。

### 1.2 根因

`tokenize_all_needed` 构建全局 `_global_index`（152GB 连续数组），per-exp 用 `searchsorted` + fancy index 查询。在 19M docs / 152GB 规模下，单线程 memcpy 成为瓶颈：

| 位置 | 操作 | 开销 |
|------|------|------|
| `tokenize_all_needed` line 1792-1794 循环 | 逐 shard 调 `_memory_cache_add_rows`（concatenate + unique + argsort） | ~450GB memcpy（冗余） |
| `tokenize_all_needed` line 1820-1821 | `np.concatenate` 全局 ids + tokens | 152GB memcpy |
| `tokenize_all_needed` line 1831 | `np.argsort(all_global_ids)` | 19M int64 排序 |
| `_tokenize_batch_union` line 512 | 每 exp `all_tokens_flat[flat_positions]` fancy index | 152GB × N_exp |

总计 ~750GB+ 单线程 memcpy，5-10 分钟，且块缓冲无 flush 看似卡死。

### 1.3 目标

消除全局 `_global_index` 的构建与查询，改为按 shard 分块存储 + 分组查询，在保证结果正确性的前提下消除 ~99% 单线程 memcpy，使 200+ exp 规模下 tokenize 不再成为训练瓶颈。

---

## 2. 核心不变式（zero-copy 前提）

方案的安全前提是：`_read_one_shard_texts_with_rows`（`metadata_manager.py:226-300`）返回的 `parsed_rows` **已 sorted ascending + unique**，使得跳过 `_memory_cache_add_rows` 的去重排序不改变结果。

4 条读取路径均满足此不变式：

| 路径 | 代码 | 保证 |
|------|------|------|
| 无 row_col | line 254 `np.arange(len(texts))` | 自然有序唯一 |
| sequential | line 273 `row_col_values.astype(np.int64)` | 输入是 `np.sort` 的（line 1773） |
| 高选择率 (>0.3) | line 286 `row_col_values.astype(np.int64)` | 同上 |
| 低选择率 (≤0.3) | line 299 `row_arr[sort_idx]` | argsort 后有序；输入来自 `np.unique` 并集（line 1729） |

→ `_memory_cache_add_rows`（line 355-366）的 concatenate + reverse-unique + argsort 是**冗余的**，可跳过。

---

## 3. 方案设计（方案 D-lite）

### 3.1 核心思路

- **不构建**全局 `_global_index`（删除 152GB concatenate + argsort）
- **保留** `_memory_cache` 按 shard 分块存储（152GB 常驻，不 clear）
- **zero-copy 接管** `parallel_results` 的 `miss_tokens` 数组（因 parsed_rows 已 sorted+unique）
- per-exp 打包改为按 shard 分组查询 + stable sort 恢复原序

### 3.2 改动清单

#### 改动 1 — `tokenize_all_needed` line 1792-1794：zero-copy 接管

跳过 `_memory_cache_add_rows`，直接填充 `_memory_cache`。因 `parsed_rows` 已 sorted+unique，结果一致。

消除 ~450GB memcpy（冗余 concatenate+unique+argsort）。

#### 改动 2 — `tokenize_all_needed` line 1803-1841：删除全局索引构建

删除 `cache_snapshot2` → `np.concatenate` → `argsort` → `_global_index` 三元组 → `_memory_cache.clear()` 整段。保留 `_memory_cache`（152GB 常驻，不 clear）。

消除 152GB concatenate + argsort。`_memory_cache_bytes` 保持 152GB < `memory_cache_max_gb=500`（line 169），不触发 eviction。

#### 改动 3 — `_tokenize_batch_union` 快速路径 line 490-543

删除整个 `if self._global_index is not None` 分支，改为调用新 helper `_pack_exp_tokens_by_shard`。

#### 改动 4 — `_tokenize_batch_union` fallback 路径 line 639-691

删除 line 639-659 的全局索引构建（concatenate + argsort + 重排 token），per-exp 打包（line 663-674）改为调用 `_pack_exp_tokens_by_shard`。

#### 新增 helper — `_pack_exp_tokens_by_shard`

按 shard 分组查询 `_memory_cache`，用 stable sort 恢复 `selected_idx` 原序：

```python
def _pack_exp_tokens_by_shard(self, selected_idx, exp_id, mgr):
    # 1. shard 分组（stable sort 保留原序）
    shard_ids = np.searchsorted(mgr._shard_starts, selected_idx, side="right") - 1
    shard_ids = np.clip(shard_ids, 0, mgr._num_shards - 1)
    orig_order = np.argsort(shard_ids, kind="stable")  # 关键：stable

    result = np.empty((len(selected_idx), self.block_size), dtype=np.int32)

    # 2. 按 shard 分组查询 _memory_cache
    unique_sids, starts, counts = np.unique(
        shard_ids[orig_order], return_index=True, return_counts=True)
    for sid, start, cnt in zip(unique_sids, starts, counts):
        group_pos = orig_order[start:start + cnt]          # 原序位置
        group_global = selected_idx[group_pos]
        local_rows = group_global - mgr._shard_starts[sid]
        row_col_vals = mgr.local_to_row_col(sid, local_rows)

        with self._memory_cache_lock:
            cache = self._memory_cache[sid]
            positions = np.searchsorted(cache["rows"], row_col_vals)
            matched = cache["rows"][positions] == row_col_vals
            assert matched.all(), f"exp {exp_id} shard {sid}: miss"
            result[group_pos] = cache["tokens"][positions]   # 分片赋值

    return result
```

helper 返回后，shm 打包逻辑（line 514-529）不变。

### 3.3 `_global_index` 引用清理

6 处引用全部删除/改写，grep 确认零残留：

| 行号 | 原用途 | 改动 |
|------|--------|------|
| 490-491 | 快速路径判断 + 解包 | 删除，走 helper |
| 497 | 打印 | 删除 |
| 639-640 | fallback 全局索引构建 | 删除，走 helper |
| 1833 | `tokenize_all_needed` 构建 | 删除 |

---

## 4. 正确性分析

### 4.1 算法层面（零风险）

| 保证点 | 依据 |
|--------|------|
| `_memory_cache` 全覆盖 | `tokenize_all_needed` line 1729 `np.unique(np.concatenate(all_selected))` 取所有 exp 并集 |
| `_memory_cache` 不 evict | `memory_cache_max_gb=500`，token 数据 152GB < 500GB；改动 1 用 `skip_eviction=True` |
| per-exp 原序恢复 | stable argsort 保留同 shard 内原序 + `result[group_pos]` 按 orig_order 赋值恢复跨 shard 原序 |
| selected_idx 含重复 | line 1036-1038 `np.repeat` 产生重复；`searchsorted side='left'` 与当前 `_global_index` 行为一致 |
| `_memory_cache` 并发 | 只 `tokenize_thread` 单线程读，`_memory_cache_lock` 保护；worker 通过 shm 读，不访问 `_memory_cache` |

当前 `_global_index` 路径：`np.searchsorted(sorted_global_ids, selected_idx)` → `sort_idx[positions]` → `all_tokens_flat[flat_positions]`，取的是 global_id == selected_idx[i] 对应 doc 的 token。

改后：按 shard 分组 `np.searchsorted(cache_rows, row_col_vals)` → `cache_tokens[positions]`，取的是同一 doc 的 token。**数学等价**。

### 4.2 实现层面（中风险，可验证）

| 风险 | 缓解 |
|------|------|
| per-exp 原序恢复实现错误 | helper 内 `assert matched.all()` 命中检查（与现有 line 504-510 一致） |
| `_global_index` 6 处引用清理不彻底 | grep 确认零残留 + hash 验证 |

### 4.3 不随 exp 数退化

`tokenize_all_needed` 一次性预填充并集，200+ exp 的 `_tokenize_batch_union` cache **全命中**（零 miss），不重新 tokenize。正确性保证不随 exp 数量变化。

---

## 5. 性能分析

### 5.1 单次聚合阶段

| 维度 | 改前 | 改后 | 收益 |
|------|------|------|------|
| memcpy 量 | ~750GB+ | ~几 GB（per-exp 分片赋值） | 消除 ~99% |
| peak RAM | ~456GB（parallel_results + cache 双份 + concatenate 临时） | 152GB（共享引用） | 降 3 倍 |
| 聚合耗时 | 5-10 分钟 | ~10 秒 | 30-60 倍 |

### 5.2 200+ exp 规模

关键数字：`selected_idx ≈ 320k/exp`（5000 steps × 64 batch），`result ≈ 2.6GB/exp`（320k × 2048 × 4B）

| 阶段 | 改前 | 改后 | 提升 |
|------|------|------|------|
| `tokenize_all_needed` 聚合 | 5-10 分钟 | ~10 秒 | 30-60 倍 |
| per-exp 打包 | ~30s（152GB fancy index） | ~3-5s（按 shard searchsorted + 2.6GB 分片赋值） | 6-10 倍 |
| tokenize_thread 总耗时 | 200 × 30s = 100 分钟 | 200 × 5s = 17 分钟 | 6 倍 |
| 训练时间 | 200/8 × ~5-10 分钟 ≈ 125-250 分钟 | 同左 | — |

**tokenize 17 分钟 << 训练 125-250 分钟**，tokenize 在独立线程批量预准备（`tokenize_lookahead=num_workers*2`），不阻塞训练。改前 100 分钟接近训练时间，**可能成为瓶颈**；改后彻底消除此风险。

### 5.3 可扩展性

- 按 shard 分块存储，无全局 152GB 连续内存分配，不受数据规模限制
- shm 由 worker 用完立即 `close()+unlink()`（parallel_dispatch.py:389-390），同时最多 8 个 shm（×2.6GB=20.8GB），不泄漏

---

## 6. 已知局限

1. **Python for 循环开销**：`_pack_exp_tokens_by_shard` 内按 shard 循环 1000 次 × 200 exp = 200k 次迭代，~1s/exp 的 Python 开销。可后期优化为向量化批处理，但当前 ~5s/exp 可接受。

2. **CPU 串行模式不受益**：`_load_tokens_for_experiment` fallback（line 731+）走 disk npz，不读 `_memory_cache`。200+ exp CPU 模式不受益，但不影响正确性。NPU 模式走 shm 路径，不受影响。

---

## 7. 验证计划

1. grep `_global_index` 确认零残留
2. 跑 `demo_run_cpu.sh`，对比改前改后 `val_loss` + `sampled_dataset.parquet` hash
3. helper 内 `assert matched.all()` 兜底（与现有 line 504-510 一致）

---

## 8. 相关文件

| 文件 | 关键位置 |
|------|----------|
| `src/quadmix/pipeline/essential_proxy_runner.py` | `tokenize_all_needed` (line 1717+), `_tokenize_batch_union` (line 479+), `_memory_cache_add_rows` (line 336) |
| `src/quadmix/data/metadata_manager.py` | `global_to_shard_rows` (line 825), `_read_one_shard_texts_with_rows` (line 226) |
| `src/quadmix/pipeline/parallel_dispatch.py` | `_tokenize_shard_parallel`, worker shm 清理 (line 389) |
| `scripts/demo_run_stem.sh` | STEM demo（8 NPU, 8 exp, 5000 steps, block_size 2048） |
| `configs/schema_stem.yaml` | STEM schema 配置 |
