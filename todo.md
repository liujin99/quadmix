# QuaDMix 项目审计问题清单

> 2026-07-22 全面审计，二次分析确认。按优先级分类，标注已修复/已确认状态。

---

## 已修复的 bug（按 commit 时间顺序）

| Commit | Bug | 文件 | 描述 |
|--------|-----|------|------|
| `645fe5e` | SSL 绕过 | `_hf_remote_size()` | HF mirror 环境下载失败 |
| `8dddca9` | 预计算线程爆炸 | precompute | 未限制线程数 |
| `c39c3eb` | ConcurrencyConfig 中央化 | concurrency.py | 环境变量分散管理 |
| `5fe3f75` | GIL 修复 | parallel_dispatch.py | loky ProcessPoolExecutor + 共享内存 |
| `8db1d96` | 每进程内存估算修复 | parallel_dispatch.py | 共享内存被重复计入 |
| `2f00da7` | demo 脚本过时 env var | 3 个 demo_run_*.sh | 移除 TOKENIZE_WORKERS/THREADS |
| `df57be9` | 共享内存计入 + CANN 警告 | parallel_dispatch.py | 共享内存不计入每进程限制 |
| `f877f18` | row_in_shard_col ID 空间 | 3 文件 | STEM source_record_idx 不匹配 |
| `ab466ad` | CANN 警告模块级 | 3 文件 | suppress 在 import 之后才生效 |
| `43ee2be` | 自适应 val_bs + chunk_size | loss_utils.py, parallel_dispatch.py | 32GB NPU OOM |
| `44aeb8e` | micro_batch_size=32 | demo_run_stem.sh | 梯度累积=2，200实验 |
| `a6b0522` | worker crash + chunk_size + Stage8 卡住 | 4 文件 | ① except 块 import torch UnboundLocalError ② cs>seq_len 跳过 ③ read_texts 并行化 |
| `84bccdd` | `_stage9_report` NameError | real_pipeline.py | domain_names/quality_names/mm 未定义 → 加参数 |

---

## CRITICAL — 运行必崩或结果严重错误（未修复）

### C1. `optimizer.py:844` — `regression_params_nested` 未定义 ✅确认
- `_compute_reliability` 引用 `regression_params_nested`，但该变量只在 `_train_per_task_models` (line 616) 中定义
- bootstrap 验证时触发 `NameError`
- **修复**：在 `_compute_reliability` 内本地定义 `regression_params_nested = {**self.regression_params, "n_jobs": self._concurrency.model_n_jobs_nested}`

### C2. ✅已修复 (commit `84bccdd`)

### C3. `essential_proxy_runner.py:1270-1271` — `M/N` 使用前未定义 ✅确认
- `range(M)` (line 1270) 和 `range(N)` (line 1271) 在 fallback 分支中使用
- `M = params.num_domains` (line 1272) 和 `N = params.num_criteria` (line 1273) 在下一行才赋值
- 当 `self._domain_names` 或 `self._quality_names` 为 None 时触发 `NameError`
- **修复**：将 `M = params.num_domains` 和 `N = params.num_criteria` 移到 line 1270 之前

### C4. `report.py:129` — `QUALITY_SHORT` 未定义 ✅确认
- 引用 `QUALITY_SHORT[max_idx]`，但模块中只有 `_DEFAULT_QUALITY_SHORT` (line 33)
- 生成报告时触发 `NameError`
- **修复**：替换为 `_DEFAULT_QUALITY_SHORT[max_idx % len(_DEFAULT_QUALITY_SHORT)]`

### C5. `metadata_manager.py:403` — `np.bincount` 收到 -1 domain labels ✅确认
- 新鲜加载路径 (line 403) 直接 `np.bincount(self._domain_labels)` 无过滤 -1
- 缓存路径 (line 288-289) 有 `valid_labels = self._domain_labels[self._domain_labels >= 0]` 过滤
- STEM 数据有 unlabeled 文档时会崩
- **修复**：新鲜加载路径也加过滤 `valid_labels = self._domain_labels[self._domain_labels >= 0]`

### C6. `metadata_manager.py:388-394` — 跨 shard categorical code 不一致 ✅确认
- 无 `domain_names` 时，各 shard 独立做 categorical encoding，相同 label 在不同 shard 可得不同 code
- merge 用 "first-occurrence wins" (line 391-393)，不检查 code 冲突
- **后果**：不同 label 映射到同一 code → domain collision → 数据错
- **修复**：先收集全局 category names，再统一 remap 各 shard domain_arr

### C7. `sampler.py:60` — 非连续 domain labels 作数组索引 ✅确认
- `params.sampling_configs[m]` 用 domain label 值 `m` 直接作列表索引
- 若 domain labels 为 [0,3,7]，domain=3 会查 `sampling_configs[3]`，但数组只有 M=3 个元素 → IndexError 或查错
- **修复**：引入 domain label→index 映射，或在 pipeline 入口验证 domain labels 必须连续 0..M-1

### C8. 3 个 demo 脚本 — `$prev_arg` 未初始化 ⬇️降级为功能错
- 脚本没有 `set -u`，不会因未绑定变量崩溃
- 但行为 bug：首个 `--val-set` 参数值不会被捕获（第一次循环 `$prev_arg` 为空串）
- **修复**：循环前加 `prev_arg=""`

### C9. `reval_with_new_valset.py:636` — `_build_domain_dist_change` 中 `domain_names` NameError ✅确认
- 模块级函数 (line 632) 引用 `domain_names`，但它是 `main()` 的局部变量 (line 216)
- **修复**：将 `domain_names` 加为函数参数，更新调用处

### C10. `resample_with_optimal_params.py:111` — preprocess 脚本路径错误 ✅确认
- 拼接 `scripts/runners/preprocess_essential_web_v1_sharded.py`
- 实际路径为 `scripts/preprocess/preprocess_essential_web_v1_sharded.py`
- **修复**：改为 `os.path.join(_SCRIPT_DIR, "..", "preprocess", "preprocess_essential_web_v1_sharded.py")`

---

## HIGH — 严重行为错误或资源泄漏（未修复）

### H1. `shared_memory.py:19-35` — SharedMemory 不 close() → fd 泄漏 ✅确认
- `ndarray_to_shared` 创建 SharedMemory 后不调用 `shm.close()` (line 21-24)
- `shared_to_ndarray` 返回 `arr.copy()` 后不关闭 shm (line 33-35)
- **后果**：1000 shards × 16 workers → fd 泄漏累积
- **修复**：copy 后立即 `shm.close()`；`ndarray_to_shared` 也需 close

### H2. `essential_proxy_runner.py` — `exp_shm_info` dict 无锁并发读写 ✅确认
- tokenize 线程写 `exp_shm_info` (line 2014)，dispatcher 线程读 (line 2078)，无同步
- **修复**：将 `exp_shm_info` 写入移入 `ready_cond` lock 内，或加独立锁

### H3. `essential_proxy_runner.py:293-365` — `_memory_cache/_memory_cache_lru` 无锁 ✅确认
- dict/list/int 三种结构被 tokenize 线程和主线程并发访问 (38 处引用)
- `_memory_cache_lru.remove()` + `.append()` 不是原子操作
- **修复**：加 `threading.Lock`，或用 `collections.OrderedDict` 替代 LRU list

### H4. `essential_proxy_runner.py` 5处 — `np.load(.npz)` NpzFile 不关闭 ✅确认
- line 402, 427, 563, 693, 712 — 全部无 `.close()`
- **后果**：5处 × 1000 shards → 极易耗尽 fd 1024 限制
- **修复**：用 `with np.load(cache_path) as data:` 或显式 `data.close()`

### H5. `jsonl_adapter.py:58,65` — `_detect_text_key` 返回值而非键名 ✅确认
- `_detect_text_key` (line 114-123) 返回文本值（`str(record[key])` 或 `v`）
- line 58 用返回值作 `text`（正确），但 line 65 把返回值存为 `text_key`（应为键名）
- **后果**：`detected_text_key` metadata 始终为空串 ""
- **修复**：拆分为 `_detect_text_key_name()` 和 `_extract_text_value()` 两个方法

### H6. `metadata_manager.py` — `from_shared(schema=None)` 崩溃 ✅确认
- line 624: `mgr._schema.domain_names` → AttributeError（当 schema=None 且 num_domains=None）
- `_validate` 的 `required_cols` 不包含 `text_col`
- **修复**：schema 改为必填参数，或在 required_cols 中加入 text_col

### H7. `types.py vs sampler.py` — scalar `sampling_value()` 无 exponent clipping ✅确认
- scalar 版 (types.py:76) 无 `np.clip(exponent, -100, 100)`
- vectorized 版 (sampler.py:75) 有 clipping
- **后果**：大 λ 值下两条路径结果不一致，scalar 版可能 NaN/溢出
- **修复**：在 scalar 版也加 `exponent = np.clip(exponent, -100, 100)`

### H8. `perf_timer.py` — `_stack/_timings` 类级别可变状态无锁 ✅确认
- `_stack.append/pop` 和 `_timings[key].append` 非原子
- **修复**：加 `threading.Lock`，或 `_stack` 用 `threading.local()`

### H9. 多文件 — 硬编码 `/home/ma-user/` 路径 ✅确认（3处）
- `constants.py:148`, `demo_run_stem.sh:43`, `demo_run_quick.sh:44`, `demo_run_full.sh:49`
- 其他机器必崩
- **修复**：用 `$QUADMIX_DIR/data/` 或 env var 替代

### H10. `demo_revalidate.sh/demo_reoptimize.sh` — 硬编码过期时间戳目录 ✅确认
- `RESULT_DIR="$QUADMIX_DIR/result/demo_full_20260630_170836"` → 目录不存在
- `demo_run_quick.sh` 缺少 `CUDA_VISIBLE_DEVICES` export
- **修复**：移除默认值，`--result-dir` 改为必填；补上 CUDA 变量

### H11. `reval_with_new_valset.py` — 缺 `core_bmk_v2` choices + SSL 不一致 ✅确认
- `--val-set` choices 缺少 `core_bmk_v2`（run_essential_web_v1.py 有）
- SSL 验证：主 runner 禁用 SSL，reval runner 不禁用 → HF mirror 环境失败
- **修复**：补充 `core_bmk_v2` choice；reval 也加 `ssl._create_unverified_context()`

---

## MEDIUM — 性能或边界问题（未修复）

### M1. `parallel_dispatch.py:397-418` — 3 处 `except: pass` 吞 KeyboardInterrupt ✅确认
- line 397, 411, 416 — worker crash handler 中
- **修复**：改为 `except Exception:`

### M2. `parallel_dispatch.py:130,237` — 全局修改 `os.environ` 不恢复
- `RAYON_NUM_THREADS/OMP_NUM_THREADS` 修改影响后续所有代码
- **修复**：函数结束后恢复原值

### M3. `quality_rank.py:72-83` — tied scores 得不同 rank ✅确认
- `np.argsort(-domain_scores)` 在 tied 组内不稳定排序 → tied 文档得不同 rank
- 论文定义 `r = |{x | q_x >= q}| / total`，tied 文档应得相同 rank
- **修复**：tied 组内取最大 rank

### M4. `types.py` — MergedQualityConfig `__eq__` 对 numpy array 失败
- dataclass 默认 `__eq__` 对 ndarray 返回 bool array → ValueError
- **修复**：override `__eq__` 用 `np.array_equal`

### M5. `types.py` — from_flattened/from_dict 无输入校验
- **修复**：加长度校验和一致性检查

### M6. `metadata_manager.py:91-92` — Python 循环做 domain remap → 极慢 ✅确认
- `np.array([remap[v] for v in domain_arr])` 对大数组极慢
- **修复**：用 `np.searchsorted` 向量化

### M7. `metadata_manager.py:708-715` — `read_texts` 加载整个 shard text 列 ✅部分已修复
- 并行化已做 (commit `a6b0522`)，但无 `row_in_shard_col` 时仍全列读取
- **修复**：只读需要的行范围

### M8. `metadata_manager.py:534-535` — `row_col_to_local` 每次创建 Python dict ✅确认
- 每次 `reverse_map = {int(v): i for i, v in enumerate(col_arr)}`
- **修复**：缓存 reverse_map dict

### M9. `run_essential_web_v1.py:140-730` — 8 个近相同 `ensure_*` 函数 ~400 行重复
- **修复**：抽象为通用函数

### M10. `parallel_dispatch.py` — worker crash 时剩余 SharedMemory 未 unlink
- **修复**：crash handler 中遍历剩余 task 清理 shm

### M11. `metadata_manager.py` — cache 写入非原子
- `np.savez` 和 `json.dump` 分两步写，中间崩溃 → cache 不一致
- **修复**：先写 temp 文件再 `os.replace` 原子替换

### M12. `quality_merger.py:61` — 负 domain labels 静默跳过无警告
- **修复**：加 warning 日志

### M13. `metadata_manager.py` — `_validate` 只检查第一个 shard
- **修复**：抽样检查首、中、末 shard

### M14. `proxy_model.py:208-209` — assert 用于运行时校验 ✅确认
- `assert T <= self.config.block_size` (line 208)，`-O` 可禁用
- **修复**：改为 `if T > self.config.block_size: raise ValueError(...)`

### M15. `concurrency.py` — `apply_env_vars` 在 numpy 已导入后无效
- **修复**：检测 `numpy in sys.modules` 后发 warning

### M16. `csv_adapter.py + parquet_adapter.py` — domain string 列 cast int64 失败
- **修复**：加 categorical encoding 支持

---

## LOW — 小优化或理论边界（未修复）

### L1. `parallel_dispatch.py + tokenize_worker.py` — 重复 tokenizer 缓存定义
### L2. `essential_proxy_runner.py:338-346` — `_memory_cache_add_rows` 不必要的数据拷贝
### L3. `parallel_dispatch.py:451-452` — 验证数据永久占 GPU 内存
### L4. `essential_proxy_runner.py:838-842` — legacy mode 硬编码 quality 列名
### L5. `param_sampler.py:52,64,119,123` — 除零风险无防护
### L6. `essential_proxy_runner.py:2031-2048` — 首批等待 900s 无 tokenize 线程存活检查
### L7. `normalization.py:68` — rank_normalize 不产生精确 1.0
### L8. `normalization.py:42,53` — 常量输入归一化静默返回零，无 warning
### L9. `perf_timer.py:34-36` — _timings 无界增长
### L10. `proxy_model.py:49-50,60-61` — rotary embedding 重复切片
### L11. `base.py:58-59` — MD5 截断 12 hex（48 bit）碰撞风险
### L12. `txt_adapter.py:41` — strip() 重复调用
### L13. `proxy_model.py:192-204` — tied lm_head/embed 双重初始化
### L14. `types.py:57-60` — SamplingConfig 参数无范围校验
### L15. `demo_run_quick.sh:206,242` — 显示 10 实验实际 8
### L16. `demo_reoptimize.sh:50,73` — help 文本默认值与实际不符
### L17. `demo_revalidate.sh:58,90` — 默认 npu 但 help 说 cpu
### L18. `demo_run_stem.sh:150` — search-mode 与其他 demo 不一致（r2_weighted vs r2_sigma_weighted）
### L19. `demo_revalidate.sh:108` — usage 中脚本名错误
### L20. `base.py:45-68` — `from_dataframe` 不保留 quality_scores
### L21. `metadata_manager.py:462` — 不必要的 `pd.read_parquet(columns=[])`
### L22. `metadata_manager.py` — `from_shared` 绕过 `__init__`，新增属性易遗漏

---

## 优先修复计划

### 第一梯队：必崩/数据错（当前 STEM 运行会触发）
1. **C1** — optimizer bootstrap NameError
2. **C3** — essential_proxy_runner M/N 未定义
3. **C4** — report.py QUALITY_SHORT NameError
4. **C5** — bincount 含 -1（STEM 有 unlabeled 文档时崩）
5. **C7** — 非连续 domain labels 索引越界

### 第二梯队：资源泄漏/并发安全（长时间运行触发）
6. **H1** — SharedMemory fd 泄漏
7. **H4** — np.load NpzFile 不关闭（5处 × 1000 shards → fd 耗尽）
8. **H3** — memory_cache 无锁（双线程竞争）

### 第三梯队：性能瓶颈
9. **M6** — Python 循环 remap（百万级数组分钟级）
10. **M8** — row_col_to_local 重复建 dict
11. **M7** — read_texts 无 row_in_shard_col 时全列读取
