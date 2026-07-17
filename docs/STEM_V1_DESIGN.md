# QuaDMix-STEM: STEM Benchmark 验证集设计

> **版本**: v1.1  
> **日期**: 2026-07-17  
> **状态**: 设计完成，待实现脚本

---

## 1. 背景与动机

### 1.1 现有验证集对比

| 验证集 | 类型 | 效果 | 信号特点 |
|--------|------|------|---------|
| **openhermes** | SFT 对话训练数据 | ✓ 较好 | 语言流畅度、对话续写 |
| **core_bmk_v6** | 21 个 benchmark 测试数据 | ✗ 较差 | QA/选择题/短答案 |
| **cap_v1** | 外部训练数据 + benchmark train (70/30) | 待验证 | 推理密集 + 能力对齐 |

### 1.2 为什么需要 STEM 验证集

CAP v1 的 5 个 capability cluster 中，**world_knowledge** 和 **symbol_logic** 是搜索信号最弱的两个：
- world_knowledge: per-cluster R² = 0.26 (arc_easy 0.37, arc_challenge 0.35, qa_wikidata 0.05)
- symbol_logic: per-cluster R² = 0.37 (operators 0.72, 但 cs_algorithms 0.21, repeat_copy 0.32)

**STEM v1 的设计目标**：集中强化 STEM 类 benchmark 的验证信号，让 proxy 能感知到**数学推理、科学知识**能力的差异。

### 1.3 核心设计原则

**验证集任务必须与下游评估 benchmark 直接对应**，避免信号迁移不确定性。

不纳入无直接下游对应的任务（如 bigbench_operators、unit_conversion、periodic_elements 等），因为：
- 无法确认这些任务的搜索信号是否能迁移到下游 STEM benchmark
- 盲目加入可能引入噪声，干扰搜索方向

---

## 2. 下游评估 benchmark

中训练验证后测试以下 6 个 benchmark：

| 下游 benchmark | 能力维度 |
|---------------|---------|
| **arc_easy** | 基础科学知识 |
| **arc_challenge** | 进阶科学知识 |
| **mmlu_stem (0-shot, 22学科)** | 多学科 STEM 知识 |
| **gpqa_diamond** | 研究生级科学知识 |
| **gsm8k_cot** | 数学推理 |
| **math_cot** | 竞赛级数学推理 |

---

## 3. 验证集设计

### 3.1 4 个 task，与下游直接对应

| # | 验证集 task | 下游对应 | Train 样本 | 说明 |
|---|------------|---------|----------|------|
| 1 | **GSM8K** | gsm8k_cot, math_cot | 7,473 | 数学推理，含 step-by-step 解题 |
| 2 | **MMLU (22 STEM)** | mmlu_stem, gpqa_diamond | ~2,200 | 22 个 STEM 学科选择题 |
| 3 | **ARC-Easy** | arc_easy | 2,251 | 基础科学问答 |
| 4 | **ARC-Challenge** | arc_challenge | 1,119 | 进阶科学问答 |

**总样本量**：~13,043

每个验证集 task 和下游 benchmark 直接同名对应，信号迁移确定性高。

### 3.2 MMLU 22 STEM 学科

与下游 mmlu_stem 评估完全对齐：

abstract_algebra, anatomy, astronomy, college_biology, college_chemistry, college_computer_science, college_mathematics, college_physics, computer_security, conceptual_physics, electrical_engineering, elementary_mathematics, formal_logic, high_school_biology, high_school_chemistry, high_school_computer_science, high_school_mathematics, high_school_physics, high_school_statistics, machine_learning, medical_genetics, virology

---

## 4. 数据源

| Task | HF 数据源 | 官方性 | Split |
|------|----------|--------|-------|
| **GSM8K** | `openai/gsm8k` | ✅ OpenAI 官方 | train |
| **MMLU** | `cais/mmlu` | ⚠️ 社区公认镜像（与 Hendrycks 原始数据一致） | train (per subject) |
| **ARC-Easy** | `allenai/ai2_arc` (ARC-Easy) | ✅ AllenAI 官方 | train |
| **ARC-Challenge** | `allenai/ai2_arc` (ARC-Challenge) | ✅ AllenAI 官方 | train |

MMLU 数据源说明：
- **官方 repo**: `hendrycks/test` (GitHub)
- **HuggingFace**: `cais/mmlu` 是 CAIS 镜像，内容与 Hendrycks 原始数据一致，社区公认最稳定
- **Fallback**: `lukaemon/mmlu`

---

## 5. Loss 策略

### 5.1 全部 full-sequence loss

**所有 4 个 task 均使用 full-sequence loss**，不使用 answer-only。

### 5.2 理由

| Task | 答案类型 | answer-only 问题 |
|------|---------|----------------|
| **GSM8K** | step-by-step 推理 (~200-500 chars) | 无（答案本身很长） |
| **MMLU** | 选择题字母 (A/B/C/D) | ✗ 只有 1 token，噪声极高 |
| **ARC** | 选择题字母 (A/B/C/D) | ✗ 只有 1 token，噪声极高 |

full-sequence 让 context（问题+选项）参与 loss，信号充足。

---

## 6. 格式适配

### 6.1 选择题答案格式

**MMLU/ARC 统一使用字母答案 (A/B/C/D)**，与原始数据格式保持一致。

设计原则：**保持原始格式，不人为改变答案类型**。

- full-sequence 下答案只占 ~5% loss，格式选择对信号影响极小
- 格式自洽：问题展示 A/B/C/D 选项，答案也是字母
- 避免问题和答案脱节（问"选 A/B/C/D"，答案却是文本内容）

### 6.2 各 task 文本格式

| Task | 格式 |
|------|------|
| **GSM8K** | `Question: {question}\nSolution: {answer}` |
| **MMLU** | `Question: {question}\nChoices:\n  A. {c1}\n  B. {c2}\n  C. {c3}\n  D. {c4}\nAnswer: {letter}` |
| **ARC-Easy/Challenge** | `Question: {question}\nChoices:\n  {label}. {text}\nAnswer: {letter}` |

### 6.3 GSM8K 格式

GSM8K 原始 answer 包含 `#### 72` 格式标记，保留原始格式。`####` 是 GSM8K 标准格式。

---

## 7. 样本量

| Task | 可用 train | 目标采样 | 采样策略 |
|------|----------|---------|---------|
| **GSM8K** | 7,473 | 5,000 | 随机采样 |
| **MMLU** | ~2,200 (22 subjects) | 全部使用 | 跨学科全部使用 |
| **ARC-Easy** | 2,251 | 全部使用 | 不足 5K |
| **ARC-Challenge** | 1,119 | 全部使用 | 不足 5K |

**实际总量**：5,000 (GSM8K采样) + ~2,200 + 2,251 + 1,119 ≈ **~10,570**

### 7.1 MMLU 等比例采样

22 个 STEM subjects，每学科 ~100-150 train 样本。全部使用，不做截断或过采样。

---

## 8. 未纳入的候选 task 及理由

| 候选 task | Train 样本 | 未纳入理由 |
|-----------|----------|-----------|
| bigbench_operators | ~168 | 样本太少，且无直接下游对应 |
| bigbench_elementary_math_qa | 30,531 | 无直接下游对应，信号迁移不确定 |
| bigbench_arithmetic | 12,019 | 太简单，搜索区分度低 |
| bigbench_modified_arithmetic | 4,800 | 无直接下游对应 |
| bigbench_unit_conversion | 19,151 | 纯记忆性任务，不锻炼推理能力 |
| bigbench_periodic_elements | 524 | 样本少，和 MMLU/ARC 有重叠 |
| openbook_qa | 4,957 | 不纯 STEM（常识推理） |
| qa_wikidata | ~54 | 不纯 STEM（事实查询），样本极少 |
| MATH (竞赛数学) | ~7,500 | 1M proxy 学不动竞赛级内容 |

**设计原则**：只纳入与下游直接对应的 task，避免引入噪声。后续实验验证后，可考虑补充有效 task。

---

## 9. gpqa_diamond 和 math_cot 的覆盖 gap

### 9.1 gpqa_diamond

验证集中无 gpqa_diamond 数据。MMLU (22 STEM) 部分覆盖其科学知识维度，但难度有 gap（MMLU 是大学/高中级，gpqa_diamond 是研究生级）。

1M proxy 无法学会研究生级内容，所以无法直接用 gpqa_diamond 作为验证集 task。MMLU 的科学知识信号是当前最接近的替代。

### 9.2 math_cot

验证集中无 MATH dataset 数据。GSM8K 部分覆盖其数学推理维度，但难度有 gap（GSM8K 是小学级，math_cot 是竞赛级）。

同样，1M proxy 学不动竞赛级内容。GSM8K 的数学推理信号是当前最接近的替代。

**应对**：先跑实验验证 GSM8K 和 MMLU 的信号能否迁移到 math_cot 和 gpqa_diamond。如果迁移效果差，再考虑补充中间难度的 task。

---

## 10. 搜索策略

### 10.1 Per-task LightGBM

4 个 task_labels，每个 task 训练独立 LightGBM：

```
tasks = ["gsm8k", "mmlu", "arc_easy", "arc_challenge"]
```

### 10.2 R²-weighted 搜索

与 CAP v1 一致，使用 R²-weighted z-score 搜索：

```python
weighted_z_score = Σ R²_i * (loss_i - mean_loss_i) / std_loss_i
# R² ≤ 0 的 task 自动过滤
```

### 10.3 预期搜索方向

| 信号来源 | 预期 θ* 偏好 | 预期域分布变化 |
|---------|-------------|-------------|
| GSM8K | 选数学推理密集域 | ↑ Mathematics |
| MMLU | 选科学知识域 | ↑ Physics_and_Chemistry, ↑ Earth_and_Life_Sciences |
| ARC | 选科学理解域 | ↑ Medicine_and_Health, ↑ Physics_and_Chemistry |

---

## 11. 与现有验证集对比

| 维度 | openhermes | core_bmk_v6 | cap_v1 | **stem_v1** |
|------|-----------|-------------|--------|------------|
| Task 数 | 1 | 21 | 5 clusters | **4** |
| 数据来源 | SFT 对话 | Benchmark test | 外部+benchmark 70/30 | **Benchmark train (纯)** |
| Loss 策略 | full-seq | 混合 | full-seq | **full-seq** |
| 与下游对齐 | 间接 | 部分 | 部分 | **直接同名对应** |
| 样本量 | 10K | ~31K | 40K | **~10.5K** |

---

## 12. 潜在风险

### 12.1 样本量偏小

4 task 总样本 ~10.5K，比 CAP v1 (40K) 和 BMK v6 (~31K) 少。per-task LightGBM 的训练数据可能不足。

**应对**：R²-weighted 搜索会自动过滤信号弱的 task。如果 ARC-Challenge (1,119) 的 R² 太低，会被降权。

### 12.2 gpqa_diamond/math_cot 难度 gap

验证集无研究生级和竞赛级内容，信号可能无法迁移到 gpqa_diamond 和 math_cot。

**应对**：先实验验证，效果差再补充中间难度 task。

### 12.3 4 task 搜索方向可能偏窄

只有 4 个 task，搜索空间维度少，可能偏向单一方向（全部趋向 STEM 密集域），忽略数据多样性。

**应对**：对比 CAP v1 的搜索结果，检查域分布是否过于极端。

---

## 13. 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Task 数量 | 4 (不纳入5/6/7) | 只用与下游直接对应的 task，避免噪声 |
| MMLU 学科数 | 22 (与下游对齐) | 验证集和下游评估统一 |
| 选择题答案 | 字母 (A/B/C/D) | 保持原始格式一致性 |
| Loss 策略 | 全部 full-sequence | 选择题 answer-only 只有 1 token |
| 样本不足 | 全部使用可用 train | 不补充外部数据 |
| 搜索策略 | per-task LightGBM + R²-weighted z-score | 与 CAP v1 一致 |

---

## 14. 实施路线图

### Phase 1: 改脚本 + 生成验证集

1. 更新 `prepare_stem_v1.py`：4 task, MMLU 22 学科, 去掉 OpenBookQA/BigBench
2. 运行脚本生成 `data/stem_v1_tokenized.pt`
3. 检查样本量、格式

### Phase 2: 上传 HuggingFace

1. 上传到 `liujin99/quadmix-stem-v1`

### Phase 3: 对比实验

1. 用 stem_v1 跑搜索实验
2. 对比 CAP v1 vs STEM v1 的域分布和下游得分

---

## 15. 文件清单

| 文件 | 说明 |
|------|------|
| `scripts/validation_set/prepare_stem_v1.py` | 验证集生成脚本 |
| `src/quadmix/constants.py` | `HF_STEM_V1_DATASET`, `HF_STEM_V1_FILENAME` |
| `scripts/ensure_val_data.sh` | `stem_v1` download case |
| `scripts/runners/run_essential_web_v1.py` | `--val-set=stem_v1` |
| `scripts/runners/reval_with_new_valset.py` | `stem_v1` choice |
| `docs/STEM_V1_DESIGN.md` | 本设计文档 |
