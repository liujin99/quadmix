# nanochat_mid_compare 审查报告 — 待修复清单

## P0 (必须修复)

### 1. STEM metric 不被 report 解析
- **文件**: `generate_report.py:81`
- **问题**: `core_pattern = re.compile(r"CORE metric:\s+([\d.]+)")` 只匹配 "CORE metric:"
- 当 `--eval-benchmarks=stem`，base_eval 输出 `"STEM metric: X.XXXX"`（不输出 "CORE metric:"），报告所有基线 CORE metric = N/A
- **修复**: 同时匹配 `STEM metric:`，或根据 eval_benchmarks 配置选择匹配

### 2. run_experiment.sh 缺 --data-ratio/--num-scaling-params
- **文件**: `run_experiment.sh:288-302`
- **问题**: PREP_ARGS 未传 `--data-ratio` 和 `--num-scaling-params` → prepare_data  else 分支，budget_cap = `quadmix_total × 1.1`（而非 target × 1.1）
- `run_stem_experiment.sh` 正确传了，`run_experiment.sh` 漏了
- **修复**: 添加 `--data-ratio "$TARGET_PARAM_DATA_RATIO"` 和 `--num-scaling-params "$NUM_SCALING_PARAMS"`

### 3. int(domain_data[i]) 在 string domain 值上崩溃
- **文件**: `prepare_data.py:263`
- **问题**: 当 `domain_names=None`（essential-web 默认）且 domain 列含字符串（如 "wikipedia.org"），line 236 存为 `np.array(domain_raw)` (dtype=object)，line 263 `int(domain_data[i])` 崩溃 ValueError
- STEM 场景不受影响（有 domain_names，走 Categorical 路径），但 essential-web 必崩
- **修复**: line 263 改为只在 Categorical 路径下转 int，string domain 直接保留原值

### 4. char_count_col 列不存在时 pq.read_table 崩溃
- **文件**: `prepare_data.py:219-224`
- **问题**: 如果 `char_count_col` 指定了但 shard 中没有该列，`pq.read_table(shard_path, columns=[...])` 直接 ArrowKeyError
- line 238 的防御性检查 `char_count_col in table.column_names` 是死代码（table 还没读到）
- **修复**: 先读 shard schema 验证列名，或去掉 columns= 参数读全部列后再筛选，或 fallback 到 text 计算

### 5. Manual Ratio 全局 trim 破坏域比例
- **文件**: `prepare_data.py:456-459`
- **问题**: `mr_selected` 按域顺序拼接（数学→物理→化学→生物学），`trim_docs_to_target` 从尾部裁剪，只裁最后域
- 如需裁 5%，几乎全裁生物学（12.5%比例被破坏）
- **修复**: trim 前 shuffle `mr_docs`，或逐域按比例 trim

## P1 (应当修复)

### 6. ACTUAL_RATIO 被 1.1× 膨胀（无害但误导）
- **文件**: `run_stem_experiment.sh:401`, `run_experiment.sh:418`
- **问题**: `ACTUAL_RATIO = TRAIN_TOKENS / NUM_SCALING_PARAMS` = `(target × 1.1) / params` ≈ 0.55（而非 0.5）
- **影响**: mid_train 的 weight_decay_scaled 公式 `D_REF/target_tokens` 中 ratio 同时出现在分子分母被消掉，**weight_decay 实际不受影响**；LR schedule 基于 num_iterations（正确），也不受影响
- 但 log 打印 "ratio=0.55" 误导，且未来代码修改公式时 1.1× 膨胀会变成真实 bug
- **修复**: ACTUAL_RATIO 改为用原始 `TARGET_PARAM_DATA_RATIO`（0.5），`--num-iterations` 单独控制训练步数

### 7. get_model_info.py except ImportError 太窄
- **文件**: `get_model_info.py:95`
- **问题**: `GPTConfig(**model_config)` 可抛 TypeError（extra keys），GPT() 可抛 RuntimeError，这些都未被 ImportError 捕获
- 脚本崩溃而非走 approximate 公式回退
- **修复**: 改为 `except Exception:` 或 `(ImportError, TypeError, RuntimeError, ValueError)`

### 8. get_model_info.py docstring 与 DEFAULT 值矛盾
- **文件**: `get_model_info.py:7 vs 20`
- **问题**: docstring 写 "Hardcoded default 1.3B (d24 model)"，但 `DEFAULT_NUM_SCALING_PARAMS = 730000000` (730M)
- 730M 是 d24 的正确值（已验证），docstring 应改为 730M

### 9. symlink 在 mid_train 失败时不清理
- **文件**: `run_stem_experiment.sh:419-436`, `run_experiment.sh:436-453`
- **问题**: `set -e` 使脚本在 mid_train crash 后立即退出，`rm "$LINK_DIR"` 永远不执行
- 留下孤立 symlink，后续 run 会误删不属于本次的 symlink
- **修复**: 用 trap 或手动 error handling，先清理 symlink 再判断是否继续

### 10. Skip logic 太粗糙
- **文件**: `run_stem_experiment.sh:259-262`, `run_experiment.sh:284-287`
- **问题**: `if [ -f "$DATA_DIR/dataset_stats.json" ]` → 整体跳过 prepare_data
- 新增 baseline（如加 --manual-ratio 或 --quality-method）时，已有 stats.json 但缺少对应数据目录
- **修复**: 检查每个 baseline 目录是否存在，只跳过已存在的；或删除 stats.json 强制重建

## P2 (建议修复)

### 11. budget_cap default 为字符串 '0' 导致零步训练
- **文件**: `run_stem_experiment.sh:443`, `run_experiment.sh:460`
- **问题**: `s['config'].get('budget_cap', '0')` → 旧版本 stats 中没有 budget_cap 时，BUDGET_CAP='0'，NUM_ITERATIONS=0
- **修复**: 加验证 `if [ "$BUDGET_CAP" -eq 0 ]; then echo ERROR; exit 1`

### 12. BUDGET_CAP / TOTAL_BATCH_SIZE 整数截断少训约 1 batch
- **文件**: `run_stem_experiment.sh:400`
- **问题**: Bash `$(( ))` truncates toward zero
- **修复**: 向上取整 `$(( (BUDGET_CAP + TOTAL_BATCH_SIZE - 1) / TOTAL_BATCH_SIZE ))`

### 13. run_experiment.sh eval 缺 --eval-benchmarks
- **文件**: `run_experiment.sh:527-533`
- **问题**: 只传 `--eval=core`，不传 `--eval-benchmarks`，使用 nanochat 默认（DCLM core）
- **修复**: 添加 `--eval-benchmarks="$EVAL_BENCHMARKS"`

### 14. CORE metric regex 不匹配负值
- **文件**: `generate_report.py:81`
- **问题**: `r"CORE metric:\s+([\d.]+)"` 不匹配负的 centered 值
- **修复**: 改为 `r"(?:CORE|STEM) metric:\s+([\d.-]+)"`

### 15. Quality datasets 不走 temp save/reload
- **文件**: `prepare_data.py:740-777`
- **问题**: 只存 quadmix/random/manual_ratio，quality datasets 常驻内存
- **修复**: 也存 quality docs 到 temp parquet，或对大数据集场景评估是否需要

## P3 (可接受或未来修复)

### 16. docstring budget_cap 公式过期
- **文件**: `prepare_data.py:13`
- 写的是 `min(quadmix_total, target) × 1.1`，实际已改为 `target × 1.1`

### 17. val_ratio=0 写 dummy shard
- **文件**: `prepare_data.py:828-833`
- 写 `[{"text": "dummy"}]`，nanochat 会 tokenize 它（可能导致 val loss 无意义）

### 18. trim_docs_to_target 总是略低于目标
- **文件**: `prepare_data.py:305-313`
- 不取超限的最后一个 doc，undershoot 可达 ~1 doc 大小（所有基线一致，公平但偏少）

### 19. 无 warning 当 source data 不够 budget_cap
- Random/ManualRatio/Quality 采样不够时静默返回不足数据

### 20. NCCL vars 在 HCCL 系统上可能无效或有副作用
- **文件**: `run_stem_experiment.sh:160-161`, `run_experiment.sh:180-181`

### 21. run_experiment.sh eval --device-batch-size=32 vs STEM 的 16
- 不同 batch size 可能导致微小 eval 差异
