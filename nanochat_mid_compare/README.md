# QuadMix vs Random vs Quality: Nanochat Mid-Training Comparison

对比实验：三种数据选取策略在 nanochat mid-training 阶段的效果对比。

1. **QuadMix** — quality+diversity 联合优化的采样策略
2. **Random** — 随机抽取等 token 数子集（baseline）
3. **Quality Top-K** — 按单一质量分数（DCLM/FineWeb-Edu）选 top-k 文档（可选）

## Quick Start

```bash
# 两组对比 (QuadMix vs Random)
QUADMIX_SAMPLED_DATA=/path/to/quadmix/result/sampled_dataset.parquet \
PREPROCESSED_DATA_DIR=/path/to/preprocessed \
NANOCHAT_MODEL_DIR=/path/to/.cache/nanochat \
NANOCHAT_REPO=/path/to/nanochat-npu \
bash nanochat_mid_compare/run_experiment.sh

# 多组对比 (QuadMix vs Random vs Quality-DCLM vs Quality-FineWeb-Edu)
QUADMIX_SAMPLED_DATA=/path/to/quadmix/result/sampled_dataset.parquet \
PREPROCESSED_DATA_DIR=/path/to/preprocessed \
QUALITY_METHODS=dclm,fineweb_edu \
NANOCHAT_MODEL_DIR=/path/to/.cache/nanochat \
NANOCHAT_REPO=/path/to/nanochat-npu \
bash nanochat_mid_compare/run_experiment.sh
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `QUADMIX_SAMPLED_DATA` | QuadMix 输出的 `sampled_dataset.parquet` 路径 | (required) |
| `PREPROCESSED_DATA_DIR` | 预处理 shards 目录 (含质量分数，用于 Random 和 Quality baseline) | (required) |
| `QUALITY_METHODS` | Quality baseline 的质量分数，逗号分隔 (dclm/fineweb_edu/english/math_general/math_openweb)，设为空则禁用 | `dclm,fineweb_edu` |
| `NANOCHAT_MODEL_DIR` | nanochat 模型目录 (含 tokenizer/, base_checkpoints/) | `/home/ma-user/work/nanochat_model_dir` |
| `NANOCHAT_REPO` | nanochat 代码仓库根目录 | `~/nanochat-npu` |
| `BASE_MODEL_TAG` | 预训练 base model tag (所有实验共享) | `d24_0320` |
| `MID_CHECKPOINTS_OUTPUT_DIR` | mid-training checkpoint 保存目录 (避免 EVS 空间不足) | `$HOME/.cache/nanochat_mid_compare/mid_checkpoints` |
| `TARGET_PARAM_DATA_RATIO` | 目标 tokens/params 比例 (自动 cap 防止过度训练) | `0.5` |
| `NUM_SCALING_PARAMS` | 模型 scaling params 数量 (d24 ≈ 1.3B) | `1300000000` |
| `DEVICE_BATCH_SIZE` | 每卡 batch size | `8` |
| `NUM_NPU` | NPU 卡数 | `8` |
| `SHARD_SIZE` | 输出 parquet 每 shard 文档数 | `10000` |
| `VAL_RATIO` | 验证集比例 (0=全量训练，写 dummy val shard 兼容 dataloader) | `0` |
| `EVAL_EVERY` | val BPB 评估间隔 (-1 = 禁用，因各组数据 val 不可比) | `-1` |
| `CORE_METRIC_EVERY` | 训练中 CORE metric 评估间隔 (-1 = 禁用，子集测试意义不大) | `-1` |
| `SEED` | 随机种子 | `42` |
| `MAX_RANDOM_SCAN` | 随机抽样扫描的最大 shard 数 | `500` |

## Token Budget 计算策略

训练 token 数取 `min(target_ratio * num_scaling_params, dataset_tokens)`：

```
target_tokens = TARGET_PARAM_DATA_RATIO * NUM_SCALING_PARAMS
actual_tokens = min(target_tokens, dataset_tokens)
num_iterations = actual_tokens / total_batch_size (524288)
```

**两种情况**：
- **数据稀缺** (data < target): 用全部数据（1 epoch），不重复训练
- **数据充足** (data > target): 按 ratio 截断，不过度训练

**d24 模型示例** (num_scaling_params ≈ 1.3B, ratio=0.5):

| 数据集大小 | target_tokens | actual_tokens | steps | 说明 |
|-----------|---------------|---------------|-------|------|
| 50M | 650M | 50M | ~95 | 数据稀缺，全量训练 |
| 300M | 650M | 300M | ~572 | 数据稀缺，全量训练 |
| 650M | 650M | 650M | ~1240 | 刚好匹配 |
| 1B | 650M | 650M | ~1240 | 数据充足，截断训练 |

## Prerequisites

无需修改 nanochat-npu 代码。脚本通过 symlink 实现 base model 复用：
- 在 `base_checkpoints/` 下创建 `MODEL_TAG -> BASE_MODEL_TAG` 的 symlink
- mid_train.py 从 symlink 加载 base model，保存到 `mid_checkpoints/<MODEL_TAG>/`
- 训练完成后自动清理 symlink

## Quality Top-K Baseline

Quality Top-K 是论文 Table 1 中的重要对比基线，用于验证 QuadMix 的 quality+diversity 联合优化是否优于纯质量筛选。

**实现方式**（支持多个质量分数同时对比）：
1. 扫描 `PREPROCESSED_DATA_DIR` 中的 preprocessed shards，读取指定质量分数列
2. 按质量分数降序排序所有文档
3. 贪心选择 top-k 文档直到累计 token 数达到 QuadMix 的 token 总量
4. 使用 nanochat tokenizer 精确计算 token 数并裁剪对齐
5. 每个质量分数独立生成数据集、独立训练和评估

**支持的质量分数**：

| Method | Column | Description |
|--------|--------|-------------|
| `dclm` | `qs_dclm` | DataComp-LM 学术质量分数（论文中贡献最大） |
| `fineweb_edu` | `qs_fineweb_edu_approx` | FineWeb-Edu 教育质量分数 |
| `english` | `qs_english` | 英语语言质量分数 |
| `math_general` | `qs_eai_general_math` | EAI 通用数学质量分数 |
| `math_openweb` | `qs_eai_open_web_math` | EAI 开放网络数学质量分数 |

**论文参考**：Table 1 中 QuaDMix-OH (39.0) vs DCLM (36.7) / Fineweb-edu (37.4)，提升 4-6%。

## Pipeline

```
1. prepare_data.py
   ├── 读取 QuadMix sampled_dataset.parquet
   ├── 使用 nanochat tokenizer 精确计算 token 数
   ├── 从 preprocessed shards 随机抽样等 token 数文档
   ├── (可选) 从 preprocessed shards 按多个质量分数分别选 top-k 文档
   ├── val_ratio=0 时全量训练 (写 dummy val shard 兼容 dataloader)
   └── 输出 2+N 组 sharded parquet (text column only)

2. 设置 checkpoint 输出目录 (可选 symlink 到大容量存储，通过 MID_CHECKPOINTS_OUTPUT_DIR)

3. 分别运行 nanochat mid-training
   ├── 创建 symlink: base_checkpoints/<tag> -> base_checkpoints/<BASE_MODEL_TAG>
   ├── QuadMix data -> mid_checkpoints/<tag>_quadmix_<timestamp>/
   ├── Random data  -> mid_checkpoints/<tag>_random_<timestamp>/
   └── (可选) Quality data -> mid_checkpoints/<tag>_quality_<method>_<timestamp>/

4. 分别运行评估 (base_eval --eval=core --model-type=mid)
   └── 仅运行 CORE metric 评估（不跑 BPB 和 sample，避免数据目录缺失错误）
```

## Output Structure

```
nanochat_mid_compare/results/<timestamp>/
├── data/
│   ├── quadmix_data/              # QuadMix 子集 (sharded parquet)
│   ├── random_data/               # 随机基线 (sharded parquet)
│   ├── quality_data_dclm/         # Quality DCLM 子集 (可选)
│   ├── quality_data_fineweb_edu/  # Quality FineWeb-Edu 子集 (可选)
│   └── dataset_stats.json         # 数据集统计
├── mid_train_quadmix.log
├── mid_train_random.log
├── mid_train_quality_dclm.log     # (可选)
├── mid_train_quality_fineweb_edu.log  # (可选)
├── eval_quadmix.log
├── eval_random.log
├── eval_quality_dclm.log          # (可选)
├── eval_quality_fineweb_edu.log   # (可选)
└── experiment_report.md           # 对比报告
```

## Base Model Evaluation

如需对比 base model 的 CORE metric（作为 baseline）：

```bash
bash nanochat_mid_compare/eval_base_model.sh
```

输出保存到 `results/base_eval/eval_base_<model_tag>.log`，可手动对比三个日志中的 CORE metric。

## Notes

- 所有基线按 **token 数精确对齐** QuadMix 子集 (使用 nanochat tokenizer)
- 各组实验共享同一验证集（CORE-22tasks），确保对比公平
- mid-training 从同一 base model checkpoint 出发（通过 symlink）
- 不复制 base checkpoint，节省磁盘空间（symlink 方式）
- `MID_CHECKPOINTS_OUTPUT_DIR` 可指向大容量存储，训练完成后 checkpoint 保存在此
- Quality baseline 可选：设置 `QUALITY_METHODS` 即可启用（默认 `dclm,fineweb_edu`），设为空则禁用。`QUALITY_METHODS` 支持逗号分隔多个方法（对应论文 Table 1 的 quality-only baseline）
