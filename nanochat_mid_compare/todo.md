# nanochat_mid_compare 审查报告 — 修复状态

## 已修复 (12/21)

| # | 优先级 | 问题 | 修复文件 | 修复方式 |
|---|--------|------|----------|----------|
| 1 | P0 | STEM metric 不被 report 解析 | generate_report.py:81 | regex → `(?:CORE|STEM) metric:\s+([\d.-]+)` |
| 2 | P0 | run_experiment.sh 缺 data-ratio/num-scaling-params | run_experiment.sh:288-302 | 添加两参数到 PREP_ARGS |
| 3 | P0 | int(domain_data[i]) 崩溃 on string | prepare_data.py:263 | `int(v) if isinstance(v, (int, np.integer)) else v` |
| 4 | P0 | char_count_col 不存在 pq.read_table 崩溃 | prepare_data.py:215-227 | 先读 schema 验证列名，缺失列 fallback 到 text |
| 5 | P0 | Manual Ratio trim 破坏域比例 | prepare_data.py:461 | shuffle mr_docs before trim |
| 16 | P3→completed | docstring budget_cap 公式过期 | prepare_data.py:13 | `min(quadmix_total, target) × 1.1` → `target × 1.1` |
| 6 | P1 | ACTUAL_RATIO 1.1× 膨胀 | run_stem_experiment.sh:401, run_experiment.sh:420 | 用原始 TARGET_PARAM_DATA_RATIO 替代 |
| 7 | P1 | except ImportError 太窄 | get_model_info.py:95 | → `except Exception:` |
| 8 | P1 | docstring 1.3B vs 730M | get_model_info.py:7 | → "730M (d24 model)" |
| 10 | P1 | Skip logic 太粗糙 | run_stem_experiment.sh:259, run_experiment.sh:284 | 检查各 baseline 子目录 + stats.json |
| 11 | P2 | budget_cap default '0' 零步训练 | run_stem_experiment.sh:449, run_experiment.sh:468 | 加验证 exit 1 |
| 12 | P2 | NUM_ITERATIONS 整数截断 | run_stem_experiment.sh:410, run_experiment.sh:429 | 向上取整 ceiling div |
| 14 | P2→merged | CORE metric regex 不匹配负值 | generate_report.py:81 | 已合并入 P0 #1 修复 |

## 不修复/保留 (8/21)

| # | 原优先级 | 新优先级 | 问题 | 原因 |
|---|---------|---------|------|------|
| 9 | P1 | P3 | symlink cleanup on crash | symlink 残留无害（指向正确 base ckpt，retry 可复用） |
| 13 | P2 | merged | eval 缺 --eval-benchmarks | 默认 eval all 对 essential-web 合理，且已合并到 #1 |
| 15 | P2 | P3 | Quality datasets 常驻内存 | 当前实验未用 quality method，内存压力不大 |
| 17 | P3 | P3 | val_ratio=0 写 dummy shard | 不影响训练 |
| 18 | P3 | P3 | trim_docs_to_target undershoot | 所有基线一致 undershoot，反而保证公平 |
| 19 | P3 | P3 | 无 warning source data 不够 | 当前数据量充足，未来可加 |
| 20 | P3 | P3 | NCCL vars on HCCL | 需实测确认，盲改有风险 |

## 不是 bug (1/21)

| # | 原优先级 | 问题 | 原因 |
|---|---------|------|------|
| 21 | P3 | eval device-batch-size 不一致 | 不同机器 GPU/NPU 数量不同导致配置差异，不是代码 bug |

---

## 第二轮审查 (issues 22-31)

### 已修复

| # | 优先级 | 问题 | 修复文件 | 修复方式 |
|---|--------|------|----------|----------|
| 22 | P0 | `run_quadmix_only.sh` 硬编码 `NUM_SCALING_PARAMS=1300000000` (d24 实际 ~730M) | run_quadmix_only.sh | 用 `get_model_info.py` 自动检测；同时修复 `continue_experiment.sh` 同样问题 |
| 23 | P1 | 数据复用检查目录名不匹配 (`quadmix` vs `quadmix_data`)，永远跳过复用 | run_stem_experiment.sh, run_experiment.sh | 删除复用检查，每次重新准备数据 (保证正确性) |
| 24 | P1 | 数据复用缺少配置一致性校验 | (同 #23) | 删除复用检查后不存在此问题 |
| 25 | P1 | `run_experiment.sh` 缺少磁盘空间检查 | run_experiment.sh | 添加 pre-flight 检查 (同 run_stem_experiment.sh) |
| 26 | P2 | `run_experiment.sh` L70 注释 `d24 ≈ 1.3B` 错误 | run_experiment.sh | 改为 `≈ 730M` |
| 27 | P2 | `run_experiment.sh` L13 注释说 auto-detect 但实际硬编码 | run_experiment.sh | 修正注释为明确默认路径 |
| 28 | P2 | `run_experiment.sh` L39 `QUADMIX_SAMPLED_DATA="${QUADMIX_SAMPLED_DATA:-}"` 冗余 | run_experiment.sh | 删除 |
| 29 | P2 | `prepare_data.py` L830-867 temp 文件写/读无意义 (1.5T RAM) | prepare_data.py | 删除 temp write/read 块 |
| 30 | P2 | `run_mid_training()` 重复读 `total_batch_size` (已有 `CKPT_TOTAL_BATCH_SIZE`) | run_stem_experiment.sh, run_experiment.sh, run_quadmix_only.sh, continue_experiment.sh | 使用全局 `CKPT_TOTAL_BATCH_SIZE` |
| 31 | P2 | `--eval-benchmarks` 未传给 `mid_train.py` | run_stem_experiment.sh, run_experiment.sh | 加 `--eval-benchmarks` 参数 |
