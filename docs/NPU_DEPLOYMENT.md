# QuaDMix NPU 部署指南

在 NPU 服务器上运行 QuaDMix 完整流水线（3000 组代理实验，100K 搜索点）。

## 前置条件

### NPU 服务器环境

- Python ≥ 3.10
- PyTorch（匹配 CANN 版本）
- torch_npu + CANN toolkit（参见 https://www.hiascend.com/software/cann）
- transformers（`pip install transformers` — GPT-NeoX tokenizer）
- LightGBM（`pip install lightgbm`）
- matplotlib（`pip install matplotlib` — 报告图表）

### 需传输的数据

| 数据 | 大小 | 说明 |
|------|------|------|
| `quadmix/` 项目源码 | ~500KB | 仅源码，不含 venv/temp/result |
| `temp/preprocessed/` | ~360MB / ~700GB | 2 shard 测试 / 3291 shard 全量 |
| `openhermes_10k_tokenized.pt` | 177MB | 预分词验证集 |

> **注意**: `token_cache/` 不需要传输，NPU 上首次运行时自动生成。
> **注意**: 只需传 preprocessed shards（不含原始 text），原始数据可留在开发机。

## 部署步骤

### 1. 安装依赖

```bash
cd /path/to/quadmix
pip install -e .              # 安装 quadmix 包
pip install transformers      # GPT-NeoX tokenizer
pip install lightgbm          # 回归模型
pip install matplotlib        # 报告图表
```

**NPU 额外安装：**

```bash
# 确认 CANN 已安装
npu-smi info
cat /usr/local/Ascend/CANN_VERSION

# torch_npu 需与 CANN 版本匹配
pip install torch_npu         # 或从源码编译
```

### 2. 确认数据路径

```bash
ls temp/preprocessed/         # 应有 shard_index.json + preprocessed_*.parquet
ls /path/to/openhermes_10k_tokenized.pt   # 验证集
```

### 3. 运行流水线

#### 快速验证（2 实验，CPU，~15s）

```bash
cd /path/to/quadmix
bash scripts/demo_run_quick.sh
```

#### 完整论文配置（3000 实验）

```bash
# 自动检测 NPU/CUDA/CPU
bash scripts/demo_run_full.sh

# 或手动指定
python scripts/run_essential_web_v1.py \
  --preprocessed-dir temp/preprocessed \
  --full \
  --block-size 2048 \
  --tiny-steps 0 \
  --device-type npu \
  --output result/full_npu_run
```

#### 带目标 token 缩放（如 10B tokens）

```bash
python scripts/run_essential_web_v1.py \
  --preprocessed-dir temp/preprocessed \
  --full \
  --device-type npu \
  --target-tokens 10 \
  --output result/10B_target
```

### 4. 监控进度

代理实验每个实验会打印：

```
[Exp 0000] QuaDMix sampled 488K docs (from 275M, 0.002 avg prob)
[TokenLoad] 488K docs loaded, cache: 0/488K hits (0%) (125.7s)
[Exp 0000] Model on npu: 14,222,592 total, 1,312,000 non-emb
[Exp 0000] Training 25000 steps (grad_acc=128, micro_batch=4)...
  Step 1/25000, loss=43.21, lr=3.00e-07, 141 tok/s, ETA: 39h
  ...
  Step 25000/25000, loss=3.12, lr=1.00e-05, 15200 tok/s, ETA: 0s
[Exp 0000] Done. train_loss=3.12, val_loss=2.89 (ppl=18.0)
```

- 首次实验 cache miss（tokenize 全 shard）最慢
- NPU 上 25K steps（1B tokens）训练约需 **2-5 分钟/实验**
- 3000 实验 ≈ **100-250 小时**（连续运行）
- 建议用 `nohup` 或 `tmux` 后台运行

## 架构要点

```
原始 shard (3291 × 246MB)
  │
  ▼ 预处理（1次）
preprocessed shard (含 domain + quality + char_count)
  │
  ▼ ShardMetadataManager（元数据常驻内存 ~15GB）
domain_labels [275M] + quality_scores [275M×5] + doc_char_counts [275M]
  │
  ▼ 每实验（3000 次）
Eq.1+Eq.2+Eq.3 → 采样 → 加载 token（mmap cache）→ 训练 proxy model
  │
  ▼
LightGBM → 100K 搜索 → top-10 平均 → 最优参数 → 最终采样
```

### 内存估算

| 数据 | 大小 | 说明 |
|------|------|------|
| domain_labels | 2.2 GB | int64, 275M docs |
| quality_scores | 11 GB | float64, 275M×5 |
| doc_char_counts | 2.2 GB | int64, 275M docs |
| token cache（单 shard） | 340 MB | 活页式 mmap，不常驻 |
| Proxy 模型 | ~170 MB | tinyllama_1M 参数 |
| **总计常驻** | **~15 GB** | 仅 metadata，text 不载入 |

## 常见问题

### torch_npu 未安装

```
报错: [WARN] torch_npu not available. Falling back to CPU.
```

检查 `pip list | grep torch_npu`，安装匹配 CANN 版本的 torch_npu。

### CUDA 驱动过旧

```
报错: CUDA initialization: The NVIDIA driver on your system is too old
```

NPU 场景不用 CUDA，确保使用 `--device-type npu`。此警告无害，可忽略。

### 内存不足

```
报错: Out of Memory / Killed
```

检查 `npu-smi info` 确认可用内存。减小 `--val-limit` 或 `--rank-ref-size`。确保系统可用内存 ≥ 32GB。

### 训练时间过长

3000 实验在单 NPU 上需连续运行数天。

- 考虑分批次运行（如 3 组 × 1000 实验）并合并结果
- 或用 `nohup` + `tmux` 防止 SSH 断开终止

### Token 缓存重建

如果预处理数据有变（重新跑 preprocess），需删掉旧 cache：

```bash
rm -rf temp/token_cache/*
```

## 输出结构

```
result/<experiment_name>/
├── optimal_parameters.json        # 最优 θ
├── pipeline_summary.json          # 配置 + R² + 采样统计
├── sampled_dataset.parquet        # 最终采样数据集
├── quadmix_report.md              # 报告
├── fig1_domain_distribution.png   # 域分布对比图
├── fig2_quality_weights.png       # 质量信号权重图
└── proxy_experiments/             # 每个实验的详细结果
    ├── exp_0000/
    │   ├── meta.json               # 参数 + val_loss
    │   └── selected_indices.npy    # 采样文档索引
    └── exp_0001/ ...
```
