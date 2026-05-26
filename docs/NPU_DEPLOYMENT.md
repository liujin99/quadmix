# QuaDMix NPU 部署指南

在 Ascend NPU 服务器上完整运行 QuaDMix 流水线（下载原始数据 → 预处理 → 3000 组代理实验 → 最优参数搜索）。

> **本文档为参考指南**，涵盖常见步骤和注意事项。
> 实际部署时用户通常已有配置好的 NPU 环境（CANN、torch_npu 已就绪），
> 可直接跳到第 2 节或第 4 节。
>
> AI Agent 可将本文档作为步骤参考，结合 `npu-smi info` 等命令探测实际环境后自行适配。

---

## 1. 环境检查与准备

### 1.1 确认操作系统与 Python

```bash
# Python ≥ 3.10
python3 --version

# 确认有编译工具链（pip wheel 需要）
gcc --version
```

### 1.2 确认 CANN 已安装

```bash
npu-smi info
# 应显示 NPU 设备列表（如 910B2 x8）
# 若无输出，需联系管理员安装 CANN toolkit

cat /usr/local/Ascend/CANN_VERSION
# 记录版本号（如 8.0.RC2），后续 torch_npu 需匹配此版本
```

**常见 CANN 安装路径：**
| 文件 | 路径 |
|------|------|
| CANN 版本 | `/usr/local/Ascend/CANN_VERSION` |
| NPU 驱动 | `/usr/local/Ascend/driver/` |
| npu-smi | `/usr/local/bin/npu-smi` |

若 CANN 未安装，需先安装（参考 [昇腾文档](https://www.hiascend.com/software/cann)）：
```bash
# 从官网下载 Ascend-cann-toolkit_*.run
chmod +x Ascend-cann-toolkit_*.run
./Ascend-cann-toolkit_*.run --install --install-path=/usr/local/Ascend
```

### 1.3 克隆项目

```bash
git clone https://github.com/liujin99/quadmix.git
cd quadmix
export QUADMIX_DIR=$(pwd)
```

### 1.4 安装 Python 依赖

```bash
# 1) 核心包（自动装 numpy/pandas/lightgbm/scikit-learn）
pip install -e .

# 2) 必需辅助包
pip install transformers      # GPT-NeoX tokenizer
pip install matplotlib        # 报告图表

# 3) 可选：进度条（下载时显示进度）
pip install tqdm
```

### 1.5 确认 torch_npu 可用

NPU 环境通常已预装 CANN + torch_npu。确认方法：

```bash
python -c "
try:
    import torch
    import torch_npu
    print(f'NPU devices: {torch.npu.device_count()}')
    for i in range(torch.npu.device_count()):
        print(f'  NPU {i}: available')
except ImportError:
    print('torch_npu not installed — will fall back to CPU')
"
```

> 若 `torch_npu` 未安装，项目会自动降级到 CPU 运行（不影响功能，但速度极慢）。
> 如需安装，请参考昇腾官方文档选择匹配当前 CANN 版本的 torch_npu：
> - https://www.hiascend.com/developer/download/ascend-pytorch
> - 或 `pip install torch_npu -i https://pypi.huaweicloud.com/simple`（需匹配 CANN 版本）

### 1.6 设置 HuggingFace 认证（可选但推荐）

验证集自动下载不需要认证，但如果网络受限：
```bash
pip install huggingface_hub
huggingface-cli login
# 输入 token（不需要也可以，仓库是公开的）
```

---

## 2. 下载原始数据

QuaDMix 使用 [essential-web-v1.0](https://huggingface.co/datasets/nvidia/essential-web-v1.0) 数据集（3291 个 parquet shards，~791 GB）。

```bash
cd $QUADMIX_DIR

# 默认下载 2000 shards（~493 GB，适用于大多数场景）
# 若需要全量 3291，先设置环境变量：
#   export NUM_SHARDS=3291
bash scripts/demo_run_full.sh  # 自动触发下载
```

或手动下载指定数量：
```bash
python scripts/download_essential_web.py \
    --num-files 2000 \
    --output-dir data/essential-web-v1 \
    --workers 8  # 并行下载线程数
```

### 2.1 下载加速方法

**方法 1：并行下载（推荐）**

下载脚本支持多线程并行下载，通过 `--workers` 参数控制：
```bash
# 8 线程并行下载（推荐）
python scripts/download_essential_web.py --num-files 2000 --workers 8

# 或在 demo_run_full.sh 中设置环境变量
DOWNLOAD_WORKERS=8 bash scripts/demo_run_full.sh
```

**方法 2：使用 HF 镜像（国内用户推荐）**

设置 `HF_ENDPOINT` 环境变量使用国内镜像：
```bash
# 使用 hf-mirror.com（国内常用镜像）
HF_ENDPOINT=https://hf-mirror.com python scripts/download_essential_web.py --num-files 2000 --workers 8

# 或在 demo_run_full.sh 中
HF_ENDPOINT=https://hf-mirror.com bash scripts/demo_run_full.sh
```

**方法 3：使用 hf_transfer（最快，需额外安装）**

hf_transfer 是 HuggingFace 官方的多线程下载库，速度可达普通下载的 3-5 倍：
```bash
pip install hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

python scripts/download_essential_web.py --num-files 2000 --use-hf-transfer
```

**速度对比（以 2000 shards ~493 GB 为例）：**

| 方法 | 预计耗时 | 适用场景 |
|------|----------|----------|
| 单线程（默认） | 4-8 小时 | 网络不稳定 |
| 4 线程并行 | 1-2 小时 | 一般网络 |
| 8 线程并行 | 30-60 分钟 | 良好网络 |
| HF 镜像 + 8 线程 | 20-40 分钟 | 国内网络 |
| hf_transfer | 15-30 分钟 | 最佳条件 |

**速度预期：** 每个 shard ~246 MB，2000 shards 在网络良好的情况下约需 2-4 小时。

下载完成后应在 `data/essential-web-v1/` 下看到 `shard_00000.parquet` ~ `shard_01xxx.parquet`。

---

## 3. 数据预处理

从原始 parquet 提取 domain 标签和 quality 信号，输出为精简的预处理器 parquet（不含 text，大幅缩小体积）。

```bash
cd $QUADMIX_DIR

python scripts/preprocess_essential_web_v1_sharded.py \
    --input-dir data/essential-web-v1 \
    --output-dir temp/preprocessed
```

**耗时：** 每个 shard ~0.3-0.5s，2000 shards ≈ 10-15 分钟，3291 shards ≈ 20 分钟。

**验证：**
```bash
ls temp/preprocessed/preprocessed_*.parquet | wc -l
# 应等于输入 shard 数量

ls -lh temp/preprocessed/shard_index.json
# 约几千字节

python -c "
from quadmix.data.metadata_manager import ShardMetadataManager
mgr = ShardMetadataManager('temp/preprocessed')
print(f'Shards: {mgr.num_shards}')
print(f'Docs:   {mgr.num_docs:,}')
print(f'Domains:{len(mgr.unique_domains)}')
"
# 应输出类似：
#   Shards: 2000
#   Docs:   167,489,000
#   Domains: 10
```

---

## 4. 运行流水线

### 4.1 快速验证（2 实验，CPU，~15 秒）

验证环境配置正确、数据路径正常：

```bash
cd $QUADMIX_DIR
bash scripts/demo_run_quick.sh
```

成功标志：控制台打印 `All 2 experiments complete` 并在 `result/` 下生成结果目录。

### 4.2 单卡 NPU 完整运行（3000 实验）

```bash
cd $QUADMIX_DIR
bash scripts/demo_run_full.sh --device-type npu
```

或等价命令：
```bash
python scripts/run_essential_web_v1.py \
    --preprocessed-dir temp/preprocessed \
    --full \
    --block-size 2048 \
    --tiny-steps 0 \
    --micro-batch-size 4 \
    --global-batch-size 512 \
    --val-limit 1000 \
    --device-type npu \
    --output result/full_npu_run
```

**耗时预期（单卡 NPU）：**
| 阶段 | 耗时 |
|------|------|
| 预采样（Eq.1-3） | 5-10 分钟 |
| 首次实验（cache miss，tokenize） | ~2-3 分钟 |
| 后续实验（cache hit） | ~2-5 分钟/实验 |
| 3000 实验总计 | **100-250 小时**（连续运行） |
| LightGBM 回归 | < 1 分钟 |
| 最优参数搜索 | < 1 分钟 |

**建议用 tmux 后台运行：**
```bash
tmux new -s quadmix
bash scripts/demo_run_full.sh --device-type npu
# Ctrl+B, D 脱离
# tmux attach -t quadmix 重新连接
```

### 4.3 多卡 NPU 并行运行（推荐）

8 张 NPU 可以将 3000 实验从 ~100h 缩短到 **~15 小时**。

```bash
cd $QUADMIX_DIR
bash scripts/demo_run_full.sh --device-type npu --npu-devices 8
```

**动态任务队列架构：**
- Worker 完成任务后立即领取下一个，无批次边界
- 快的 NPU 自然做更多实验，慢的只影响自己
- CPU tokenize 线程独立运行，与 NPU 训练重叠
- 每张 NPU 绑定独立的 `worker_id` 作为 `npu_device_id`

**架构示意：**
```
Main Process
├─ Tokenize Thread (CPU) — 持续预 tokenize (lookahead=32)
├─ Dispatcher Thread — 推送就绪任务到 task_queue
├─ Collector Thread — 收集 result_queue 的完成结果
└─ Worker 0-7 (NPU) — 循环: task_queue.get() → run → result_queue.put()
```

**多卡验证：**
```bash
python -c "
import torch
import torch_npu
print(f'{torch.npu.device_count()} NPU devices available')
for i in range(torch.npu.device_count()):
    print(f'  NPU {i}: {torch.npu.get_device_properties(i).name}')
"
```

---

## 5. 监控与调试

### 5.1 运行时日志解读

```
[Setup] Loading ShardMetadataManager from: temp/preprocessed
[Setup] 167,489,000 docs across 2000 shards

[Stage 3] Sampling 3000 parameter configurations (Alg.1)...

[Stage 4] Running 3000 proxy experiments...
[Stage 4] Using 8 parallel workers (dynamic task queue)

[Dynamic Parallel] 3000 experiments, 8 workers, tokenize_lookahead=32
  Precomputing 3000 samples (Eq.1-3)... done in 381.2s

[Worker 0] Exp 0000: Training on NPU:0... cache: 0/488K hits
[Worker 1] Exp 0001: Training on NPU:1... cache: 0/492K hits
[Worker 2] Exp 0002: Training on NPU:2... cache: 0/475K hits
...
[Worker 0] Exp 0000: Done. train_loss=3.12, val_loss=2.89 (ppl=18.0)
[Worker 0] Exp 0008: Training on NPU:0... cache: 410K/495K hits

[Progress] 100/3000 done (avg: 45s/exp, ETA: 2175s)
...
```

**关键信号：**
- `cache: 0/488K hits` → 首次运行，正在生成 token cache
- `cache: 488K/488K hits` → 全部 cache hit，最快速度
- `loss=3.12` → 训练正常收敛
- `val_loss=2.89` → 验证集 loss，越低越好

### 5.2 常见问题

**torch_npu 未安装：**
```
[WARN] torch_npu not available. Falling back to CPU.
```
→ 安装匹配 CANN 版本的 torch_npu（见 1.5 节）

**NPU 内存不足：**
```
RuntimeError: NPU out of memory
```
→ 减小 `--micro-batch-size`（如 2）或检查 `npu-smi info` 可用内存

**下载超时：**
```
urllib.error.URLError: <urlopen error [Errno 110] Connection timed out>
```
→ 检查网络代理，或预下载验证集放到 `data/openhermes_10k_assistant_tokenized.pt`

**预处理数据路径错误：**
```
FileNotFoundError: No preprocessed_*.parquet files found in ...
```
→ 确认 `temp/preprocessed/` 下有 `preprocessed_*.parquet` 和 `shard_index.json`，或重新跑预处理

**多卡并行失败：**
```
AttributeError: Can't pickle local object ...
```
→ 检查 `mp.get_context("spawn")` 是否可用，某些环境需设置：
```bash
export PYTHONPATH=$QUADMIX_DIR:$PYTHONPATH
```

---

## 6. 输出结构

```
result/quadmix_20250525_143022/
├── optimal_parameters.json        # 最优 θ*（可直接用于下游采样）
├── pipeline_summary.json          # 运行配置 + R² + 采样统计
├── sampled_dataset.parquet        # 最终采样数据集（含 domain/文档索引）
├── quadmix_report.md              # 完整中文报告
├── fig1_domain_distribution.png   # 域分布对比图
├── fig2_quality_weights.png       # 质量信号权重图
└── proxy_experiments/             # 每个实验的详细结果
    ├── exp_0000/
    │   ├── meta.json              # (α_m, λ_m, ω_m, η_m, ε_m) + val_loss
    │   └── selected_indices.npy   # 该实验采样文档的全局索引
    └── exp_0001/ ...
```

---

## 7. 参考步骤速查

| # | 步骤 | 参考命令 |
|---|------|---------|
| 1 | Python ≥ 3.10 | `python3 --version` |
| 2 | CANN 已安装 | `npu-smi info` |
| 3 | 项目克隆 | `cd quadmix && pwd` |
| 4 | 核心依赖 | `pip list \| grep -E "numpy\|pandas\|lightgbm"` |
| 5 | torch_npu | `python -c "import torch_npu; print('ok')"` |
| 6 | 原始数据 | `ls data/essential-web-v1/shard_00000.parquet` |
| 7 | 预处理数据 | `ls temp/preprocessed/shard_index.json` |
| 8 | 快速测试 | `bash scripts/demo_run_quick.sh` |

---

## 附录：关键文件速查

| 文件 | 用途 |
|------|------|
| `scripts/run_essential_web_v1.py` | 主入口，所有 CLI 参数在此 |
| `scripts/essential_proxy_runner.py` | 代理模型训练、token cache、多卡调度 |
| `scripts/preprocess_essential_web_v1_sharded.py` | 数据预处理 |
| `scripts/download_essential_web.py` | 原始数据下载 |
| `scripts/demo_run_quick.sh` | 快速验证脚本（2 exp） |
| `scripts/demo_run_full.sh` | 完整论文配置脚本（3000 exp） |
| `src/quadmix/npu/device.py` | NPU 设备抽象层（try-import torch_npu） |
| `src/quadmix/data/metadata_manager.py` | Shard 元数据管理（~15GB 常驻内存） |
| `src/quadmix/core/proxy_model.py` | tinyllama_1M 代理模型 |
| `src/quadmix/pipeline/optimizer.py` | LightGBM 回归 + 最优搜索 |
| `src/quadmix/pipeline/real_pipeline.py` | 流水线编排（8 个 stage） |
| `docs/ARCHITECTURE.md` | 项目架构详细文档 |
