# QuaDMix 项目审计问题清单

> 2026-07-23 第四轮更新。H13 SHA256+L1/L6/L13/L15-L19/L27 已修复并移至历史表。剩余LOW跳过。

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
| `a7e1744` | C1-C10 全部修复 | 多文件 | 9个 CRITICAL bug：optimizer NameError、M/N未定义、QUALITY_SHORT、bincount含-1、跨shard code不一致、非连续domain索引、$prev_arg、reval domain_names NameError、preprocess路径 |
| `9a84ee2` | H1+H4 修复 | shared_memory.py + 5处np.load | SharedMemory fd 泄漏 + NpzFile 不关闭 |
| `049f816` | H5-H11 修复 | 多文件 | jsonl text_key、from_shared崩溃、scalar exponent clip、perf_timer无锁、硬编码路径、demo过期目录、reval SSL+choices |
| `19c2371` | H2+H3+H8+H9 修复 | 多文件 | exp_shm_info无锁、memory_cache无锁、perf_timer类级状态、硬编码/home/ma-user路径 |
| `12822f0` | CJK 字体自动检测 | report.py | 动态检测matplotlib CJK字体支持 |
| `b6df3bb` | read_texts 性能优化 | metadata_manager.py | ProcessPoolExecutor(spawn) + pyarrow + 自适应策略 |
| `02fefaa` | tokenize 性能优化 | metadata_manager.py | pd.read_parquet → pyarrow + 自适应策略 |
| `94a5297` | shard_tasks unpack | parallel_dispatch.py | 5-tuple (sid, path, miss_rows, is_seq, total_rows) |
| `4629a34` | normalization + domain_indices skip | essential_proxy_runner.py | worker模式跳过归一化省100s |
| `b903f50` | NaN/Inf JSON + CJK字体 | 多文件 | sanitize JSON输出 + matplotlib CJK检测 |
| `568540e` | NPUGraph capture | parallel_dispatch.py | 消除Python dispatch间隙 |
| `c993ff8` | M1-M16 全部修复 + L22 | 多文件 | 16个MEDIUM bug：except:pass、env var不恢复、tied rank、__eq__ ndarray、from_flattened无校验、Python循环remap、read_texts全列、row_col_to_local重复建dict、ensure_*重复、crash shm清理、cache非原子写、负domain无warning、_validate只查首shard、assert运行时、concurrency numpy已导入后无效、csv/parquet cast、from_shared绕过__init__ |
| `0833478` | dead --seed + pyyaml | run_essential_web_v1.py + pyproject.toml + requirements.txt | 删除无效--seed参数 + 补缺失pyyaml依赖 |
| `eef4fff` | H12+H14+M18-M26+L4 | 多文件 | H12除零保护(max eps)、H14 README YAML修正、M18 search_weight_mode→r2_weighted、M20 num_criteria属性、M21 _negated_cols、M22 block-size→2048、M23 default_rng、M24 schema必填、M25 callback+分批写入+legacy移除、M26 删checkpoint-steps、L4 legacy路径移除 |
| `eef4fff` | H13 部分 | essential_proxy_runner + parallel_dispatch | weights_only=True 已用；SHA256 完整校验在 `60bee15` 补完 |
| `60bee15` | H13完整 + L1/L6/L13/L15-L19/L27 | 多文件 | H13 SHA256校验(_sha256_verify)、L1 tokenizer_utils共享模块、L6 tokenize线程存活检查、L13 tied weight tie后init、L15显示8、L16-L17 help默认值、L18 search-mode统一r2_weighted、L19脚本名、L27删除13垃圾文件 |

---

## 所有已知问题均已修复

以下 LOW 级问题经评估决定**跳过**（影响极小或改动风险大于收益）：

| ID | 理由 |
|----|------|
| L2 | 内存拷贝优化 — 0-length array concat 代价极低 |
| L3 | val占GPU — worker退出自动释放 |
| L7 | rank不精确1.0 — 标准percentile行为，下游无影响 |
| L8 | 常量无warning — 零化数学正确，warning增加噪音 |
| L9 | _timings无界 — PerfTimer默认关闭，pipeline运行有限步数 |
| L10 | rotary重复切片 — view不拷贝，零性能影响 |
| L11 | MD5截断12hex — doc_id仅用于cache，非安全场景 |
| L12 | strip重复 — O(1)操作，微秒级影响 |
| L14 | SamplingConfig无校验 — 参数由搜索算法生成，不会越界 |
| L20 | from_dataframe丢quality — quality走shard路径，不走DataFrame |
| L21 | pd.read_parquet空列 — 已不存在，用pyarrow替代 |
| L23 | 字段名无~标记 — 注释已说明映射，改名破坏API |
| L24 | fcntl Unix-only — NPU/CUDA环境必为Linux |
| L25 | 依赖无上限 — 研究代码惯例，上限可能引安装冲突 |
| L26 | perf_timer多进程 — 调试辅助，子进程时序已由print记录 |
| L28 | 中英混合 — 故意设计，CJK字体检测+fallback完备 |

---

## 非问题备注（二次分析确认）

- **noise_level `higher_better=True`**：STEM 数据的 noise_level 已在预处理时翻转（分越高=噪声越低=质量越好），不是 bug
- **--seed 无意义**：多次实验合并拟合需随机参数多样性，固定 seed 反而降低搜索覆盖。已从主入口移除
- **近似 vs 精确 rank**：proxy 实验用10K近似（速度），最终采样用精确rank（质量），是已知设计选择
- **论文 smaller=better vs 代码 higher=better**：全链路自洽，不影响结果

---

## 优先修复计划

> **当前状态：所有 CRITICAL/HIGH/MEDIUM 问题已修复。16个 LOW 级跳过。**
> 唯一剩余的未修 MEDIUM 级问题是 M17（文件拆分），属长期重构。

### 第一梯队（已完成 ✓）
- H13 SHA256 校验 + weights_only=True

### 第二梯队（已完成 ✓）
- L1 tokenizer共享模块、L6线程存活检查、L13 tied weight修复
- L15-L19 demo脚本文本修正、L27 repo清理

### 第三梯队：长期重构（可选）
- **M17** — essential_proxy_runner 拆分（2210行→4模块，大工程）

### 第四梯队：零测试覆盖（长期）
- 核心算法单元测试（Eq.1-3、Alg.1）
- Pipeline 集成测试
- pytest 框架搭建 + CI 配置
