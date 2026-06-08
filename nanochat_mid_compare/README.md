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
| `BASE_MODEL_TAG` | 预训练 base model tag | `d24_0320` |
| `TARGET_PARAM_DATA_RATIO` | mid-training 数据/参数比 | `0.1` |
| `DEVICE_BATCH_SIZE` | 每卡 batch size | `8` |
| `NUM_NPU` | NPU 卡数 | `8` |
| `SHARD_SIZE` | 输出 parquet 每 shard 文档数 | `10000` |
| `VAL_RATIO` | 验证集比例 | `0.05` |
| `SEED` | 随机种子 | `42` |
| `MAX_RANDOM_SCAN` | 随机抽样扫描的最大 shard 数 | `500` |

## Pipeline

```
1. prepare_data.py
   ├── 读取 QuadMix sampled_dataset.parquet
   ├── 估算 token 数 (char_count // 4)
   ├── 从 essential-web 随机抽样等 token 数文档
   ├── 共享验证集 (从 QuadMix 子集中抽取)
   └── 输出两组 sharded parquet (text column only)

2. 复制 base checkpoint 为两个独立 tag

3. 分别运行 nanochat mid-training
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

- 随机基线按 **token 数对齐** QuadMix 子集 (char_count // 4 估算)
- 两组实验共享同一验证集，确保对比公平
- mid-training 从同一 base model checkpoint 出发
- 默认 `target-param-data-ratio=0.1`，对 d24 模型约 50M tokens
