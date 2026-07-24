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
