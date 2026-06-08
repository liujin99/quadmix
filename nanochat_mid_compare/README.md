# QuadMix vs Random: Nanochat Mid-Training Comparison

对比实验：QuadMix 选取的 essential-web 子集 vs 随机抽取的等 token 数子集，在 nanochat mid-training 阶段的效果对比。

## Quick Start

```bash
QUADMIX_DATASET=/path/to/quadmix/result/sampled_dataset.parquet \
ESSENTIAL_WEB_DIR=/path/to/essential-web-v1 \
NANOCHAT_BASE_DIR=/path/to/.cache/nanochat \
NANOCHAT_ROOT=/path/to/nanochat-npu \
bash nanochat_mid_compare/run_experiment.sh
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `QUADMIX_DATASET` | QuadMix 输出的 `sampled_dataset.parquet` 路径 | (required) |
| `ESSENTIAL_WEB_DIR` | essential-web 原始 parquet shards 目录 | (required) |
| `NANOCHAT_BASE_DIR` | nanochat 基础目录 (含 tokenizer/, base_checkpoints/) | `~/.cache/nanochat` |
| `NANOCHAT_ROOT` | nanochat-npu 仓库根目录 | `~/nanochat-npu` |
| `BASE_MODEL_TAG` | 预训练 base model tag (两组实验共享) | `d24_0320` |
| `MID_CHECKPOINTS_OUTPUT_DIR` | mid-training 训练完成后 checkpoint 保存目录 (避免 EVS 空间不足) | `$NANOCHAT_BASE_DIR/mid_checkpoints` |
| `TARGET_PARAM_DATA_RATIO` | mid-training 数据/参数比 (见下方说明) | `0.1` |
| `DEVICE_BATCH_SIZE` | 每卡 batch size | `8` |
| `NUM_NPU` | NPU 卡数 | `8` |
| `SHARD_SIZE` | 输出 parquet 每 shard 文档数 | `10000` |
| `VAL_RATIO` | 验证集比例 (0=全量训练，写 dummy val shard 兼容 dataloader) | `0` |
| `EVAL_EVERY` | val BPB 评估间隔 (-1 = 禁用，因两组数据 val 不可比) | `-1` |
| `SEED` | 随机种子 | `42` |
| `MAX_RANDOM_SCAN` | 随机抽样扫描的最大 shard 数 | `500` |

## target-param-data-ratio 说明

控制 mid-training 的 token budget：`tokens = ratio × num_scaling_params`

d24 模型约 500M scaling params：

| ratio | tokens | steps (batch=524K) | 适用场景 |
|-------|--------|-------------------|----------|
| 0.1 | ~50M | ~95 | 快速对比 |
| 1.0 | ~500M | ~950 | 较充分对比 |
| 10.0 | ~5B | ~9500 | 完整 mid-training |

建议先用 `0.1` 快速验证流程，再用 `1.0` 做正式对比。

## Prerequisites

无需修改 nanochat-npu 代码。脚本通过 symlink 实现 base model 复用：
- 在 `base_checkpoints/` 下创建 `MODEL_TAG -> BASE_MODEL_TAG` 的 symlink
- mid_train.py 从 symlink 加载 base model，保存到 `mid_checkpoints/<MODEL_TAG>/`
- 训练完成后自动清理 symlink

## Pipeline

```
1. prepare_data.py
   ├── 读取 QuadMix sampled_dataset.parquet
   ├── 使用 nanochat tokenizer 精确计算 token 数
   ├── 从 essential-web 随机抽样等 token 数文档
   ├── val_ratio=0 时全量训练 (写 dummy val shard 兼容 dataloader)
   └── 输出两组 sharded parquet (text column only)

2. 设置 checkpoint 输出目录 (可选 symlink 到大容量存储，通过 MID_CHECKPOINTS_OUTPUT_DIR)

3. 分别运行 nanochat mid-training
   ├── 创建 symlink: base_checkpoints/<tag> -> base_checkpoints/<BASE_MODEL_TAG>
   ├── QuadMix data -> mid_checkpoints/<tag>_quadmix_<timestamp>/
   └── Random data  -> mid_checkpoints/<tag>_random_<timestamp>/

4. 分别运行评估 (base_eval --model-type=mid)
```

## Output Structure

```
nanochat_mid_compare/results/<timestamp>/
├── data/
│   ├── quadmix_data/          # QuadMix 子集 (sharded parquet)
│   ├── random_data/           # 随机基线 (sharded parquet)
│   └── dataset_stats.json     # 数据集统计
├── mid_train_quadmix.log
├── mid_train_random.log
├── eval_quadmix.log
└── eval_random.log
```

## Notes

- 随机基线按 **token 数精确对齐** QuadMix 子集 (使用 nanochat tokenizer)
- 两组实验共享同一验证集，确保对比公平
- mid-training 从同一 base model checkpoint 出发（通过 symlink）
- 不复制 base checkpoint，节省磁盘空间（symlink 方式）
- `MID_CHECKPOINTS_OUTPUT_DIR` 可指向大容量存储，训练完成后 checkpoint 保存在此
