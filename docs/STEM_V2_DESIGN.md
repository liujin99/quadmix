# QuaDMix-STEM v2: 补全 GPQA 和 MATH 代理覆盖

> **版本**: v2.0  
> **日期**: 2026-07-27  
> **状态**: 设计完成，实现中  
> **前置**: [STEM v1 设计](./STEM_V1_DESIGN.md)

---

## 1. 背景与动机

### 1.1 STEM v1 实验结果

STEM v1 实验完成后，下游评估结果：

| 方法 | CORE (6 task 均值) | 排名 |
|------|-------------------|------|
| Random | 0.1615 | 1 |
| QuadMix | 0.1530 | 2 |
| Manual | 0.1504 | 3 |

**QuadMix 未超过 Random**，根本原因分析：

### 1.2 三个结构性失配

| # | 失配 | 具体表现 |
|---|------|---------|
| 1 | Loss ≠ Accuracy | proxy CE loss (1M 模型) vs 下游 accuracy (730M 模型 + CoT) |
| 2 | R²-weighted → gsm8k 占 62% 权重 | 但 gsm8k_cot accuracy ≈ 0 at 0.5x scale，信号无效 |
| 3 | **gpqa_diamond 无直接 proxy 覆盖** | 从 mmlu 映射 (R²=0.105, 11% 权重)，信号极弱 |

### 1.3 gpqa_diamond 是决定性因素

QuadMix vs Random 的差距几乎全部来自 gpqa_diamond (normalized -0.0606)。排除 gpqa_diamond 后，QuadMix 略胜 Random。

**核心问题**：v1 验证集中没有 GPQA 数据，gpqa_diamond 下游得分完全依赖 MMLU 的间接迁移信号，而 MMLU 的 R² 只有 0.105。

### 1.4 v2 的设计目标

**补全 GPQA 和 MATH 的直接 proxy 覆盖**，消除 v1 的信号盲区。

---

## 2. v1 vs v2 对比

### 2.1 Task 列表

| # | v1 Task | v2 Task | 变化 |
|---|---------|---------|------|
| 1 | gsm8k | gsm8k | 不变 |
| 2 | mmlu (22 STEM) | mmlu (22 STEM) | 不变 |
| 3 | arc_easy | arc_easy | 不变 |
| 4 | arc_challenge | arc_challenge | 不变 |
| 5 | — | **gpqa (main, 448题)** | **新增** |
| 6 | — | **math (train, 5000题)** | **新增** |

### 2.2 下游覆盖对比

| 下游 benchmark | v1 proxy 覆盖 | v2 proxy 覆盖 |
|---------------|--------------|--------------|
| arc_easy | ✅ arc_easy | ✅ arc_easy |
| arc_challenge | ✅ arc_challenge | ✅ arc_challenge |
| mmlu_stem | ✅ mmlu | ✅ mmlu |
| gpqa_diamond | ⚠️ mmlu 间接 (R²=0.105) | ✅ **gpqa main 直接** |
| gsm8k_cot | ✅ gsm8k | ✅ gsm8k |
| math_cot | ⚠️ gsm8k 间接 | ✅ **math train 直接** |

### 2.3 Token 量对比

| 维度 | v1 | v2 |
|------|----|----|
| Task 数 | 4 | **6** |
| 总样本 | ~10.5K | **~13K** |
| 估计总 tokens | ~2.4M | **~4.5M** |
| gpqa tokens | 0 | **~110K** |
| math tokens | 0 | **~2000K** |

---

## 3. 新增 Task 设计

### 3.1 GPQA (gpqa_main)

| 属性 | 值 |
|------|---|
| HF 数据源 | `Idavidrein/gpqa`（官方，David Rein 等原作者上传） |
| Config | `gpqa_main` |
| Split | `train` |
| 题数 | 448 |
| 许可证 | CC BY 4.0 |
| 论文 | arxiv.org/abs/2311.12022 |
| 下游对应 | gpqa_diamond (198题，diamond ⊂ main) |

**为什么用 main (448) 而非 diamond (198)？**

- diamond ⊂ main：main 包含全部 198 diamond 题 + 250 额外题
- 198 题 ≈ 49K tokens，统计功效低
- 448 题 ≈ 110K tokens，2.3x 数据量，loss 估计更稳定
- 额外 250 题仍是 PhD 级科学 MC，同一领域
- 对齐不丢失：下游测的 diamond 是 proxy 用的 main 的子集

**GPQA 数据集结构**：

| Config | 题数 | 说明 |
|--------|------|------|
| gpqa_diamond | 198 | 最高质量子集（专家一致验证） |
| gpqa_main | 448 | 主集（diamond ⊂ main） |
| gpqa_extended | 546 | 扩展集（main + 额外题） |
| gpqa_experts | — | 专家信息 |

**CSV 关键列**（共 78 列，取 5 列）：

| 列名 | 用途 |
|------|------|
| `Question` | 题目文本 |
| `Correct Answer` | 正确答案文本 |
| `Incorrect Answer 1` | 错误答案 1 |
| `Incorrect Answer 2` | 错误答案 2 |
| `Incorrect Answer 3` | 错误答案 3 |

### 3.2 MATH (hendrycks/competition_math)

| 属性 | 值 |
|------|---|
| HF 数据源 | `EleutherAI/hendrycks_math`（社区镜像，与官方同数据） |
| Split | `train`（7 个 subject configs） |
| 题数 | 7,500（采样 5,000） |
| 许可证 | MIT |
| 下游对应 | math_cot (math500.jsonl，500题，train ≠ test 无重叠) |

**为什么用 EleutherAI/hendrycks_math？**

- 官方 `hendrycks/competition_math` 在 HF Hub 上不可达（连接重置）
- `EleutherAI/hendrycks_math` 包含同样的 Hendrycks MATH 数据（同题目、同解答）
- EleutherAI 是知名研究组织，镜像可信
- MIT 许可证（比官方 CC BY-NC-SA 更宽松）
- 7 个 subject configs：algebra, counting_and_probability, geometry, intermediate_algebra, number_theory, prealgebra, precalculus

**为什么用 train (7500) 而非 test (5000)？**

- 下游 eval (`math_cot`) 使用 MATH-500（从 test 采样的 500 题）
- 用 train split → 0% 重叠，无数据泄露风险
- train 有 7500 题，采样 5000 题足够
- train 和 test 同分布（同一竞赛数据集的不同 split）

**关键列**：

| 列名 | 用途 |
|------|------|
| `problem` | 题目文本（LaTeX） |
| `solution` | 完整解答（含 `\boxed{}`） |
| `level` | 难度等级 (Level 1-5) |
| `type` | 学科分类 |

---

## 4. 格式适配

### 4.1 新增 Task 格式

| Task | 格式 | 与 v1 对齐 |
|------|------|-----------|
| **GPQA** | `Question: {question}\nChoices:\n  A. {c0}\n  B. {c1}\n  C. {c2}\n  D. {c3}\nAnswer: {letter}` | 与 MMLU/ARC MC 格式一致 |
| **MATH** | `Question: {problem}\nSolution: {solution}` | 与 gsm8k 格式一致 |

### 4.2 GPQA choices shuffling

官方 CSV 中正确答案和错误答案分列存储。proxy 格式需要合并为 A/B/C/D 选项：

```python
choices = [correct, incorrect_1, incorrect_2, incorrect_3]
rng = random.Random(42)  # 固定种子，可复现
indices = list(range(4))
rng.shuffle(indices)
shuffled = [choices[i] for i in indices]
correct_idx = indices.index(0)
```

### 4.3 v1 保留 Task 格式（不变）

| Task | 格式 |
|------|------|
| GSM8K | `Question: {question}\nSolution: {answer}` |
| MMLU | `Question: {question}\nChoices:\n  A. {c1}\n  B. {c2}\n  C. {c3}\n  D. {c4}\nAnswer: {letter}` |
| ARC-Easy/Challenge | `Question: {question}\nChoices:\n  {label}. {text}\nAnswer: {key}` |

---

## 5. 样本量

| Task | 可用数据 | 目标采样 | 采样策略 |
|------|---------|---------|---------|
| GSM8K | 7,473 | 5,000 | 随机采样 |
| MMLU | ~2,200 | 全部使用 | 跨学科全部使用 |
| ARC-Easy | 2,251 | 全部使用 | 不足 5K |
| ARC-Challenge | 1,119 | 全部使用 | 不足 5K |
| **GPQA main** | 448 | **全部使用** | 不足 5K |
| **MATH train** | 7,500 | **5,000** | 随机采样 |

**实际总量**：5,000 + ~2,200 + 2,251 + 1,119 + 448 + 5,000 ≈ **~16,018**

---

## 6. Loss 策略

与 v1 一致：**全部 6 个 task 使用 full-sequence loss**。

| Task | 答案类型 | full-seq 理由 |
|------|---------|-------------|
| GSM8K | step-by-step 推理 | 答案本身很长，full-seq 信号充足 |
| MMLU | 选择题字母 | answer-only 只有 1 token，噪声高 |
| ARC | 选择题字母 | 同 MMLU |
| **GPQA** | 选择题字母 | 同 MMLU，但题目更长（PhD 级） |
| **MATH** | step-by-step 解答 | 同 GSM8K，解答含 LaTeX 推理 |

---

## 7. 搜索策略

### 7.1 Per-task LightGBM

6 个 task_labels，每个 task 训练独立 LightGBM：

```python
tasks = ["gsm8k", "mmlu", "arc_easy", "arc_challenge", "gpqa", "math"]
```

### 7.2 搜索模式选择

v1 使用 R²-weighted（默认），导致 gsm8k 占 62% 权重。

v2 待 R² 出来后决定：

| 模式 | 公式 | 预期效果 |
|------|------|---------|
| `r2_weighted` (v1 默认) | w_i = R²_i / Σ R² | gsm8k 可能仍占主导 |
| `equal_weight` | w_i = 1/N | 每 task 16.7%，gpqa/math 不被压制 |
| `r2_sigma_weighted` | w_i = R²_i × σ_i / Σ | 最差：放大高 R² 高 σ task |

**决策原则**：看 stem_v2 的 per-task R² 后再选搜索模式。如果 gsm8k 仍占 >40%，用 equal_weight。

### 7.3 预期搜索方向

| 信号来源 | 预期 θ* 偏好 | v1 新增效果 |
|---------|-------------|------------|
| GSM8K | 选数学推理密集域 | 不变 |
| MMLU | 选科学知识域 | 不变 |
| ARC | 选科学理解域 | 不变 |
| **GPQA** | **选研究生级科学域** | **新信号：Physics/Chemistry/Biology 高难度** |
| **MATH** | **选竞赛数学密集域** | **新信号：Mathematics 竞赛级推理** |

---

## 8. 数据源权威性总结

| Task | HF Repo | 上传者 | 官方性 | 许可证 |
|------|---------|--------|--------|--------|
| GSM8K | `openai/gsm8k` | OpenAI | ✅ 官方 | MIT |
| MMLU | `cais/mmlu` | CAIS | ⚠️ 社区公认镜像 | MIT |
| ARC-Easy | `allenai/ai2_arc` | AllenAI | ✅ 官方 | CC BY 4.0 |
| ARC-Challenge | `allenai/ai2_arc` | AllenAI | ✅ 官方 | CC BY 4.0 |
| **GPQA** | `Idavidrein/gpqa` | David Rein (原作者) | ✅ **官方** | CC BY 4.0 |
| **MATH** | `EleutherAI/hendrycks_math` | EleutherAI | ⚠️ 社区镜像（同数据） | MIT |

**MATH 数据源选择说明**：

| 候选 | 上传者 | 官方性 | 许可证 | 选择 |
|------|--------|--------|--------|------|
| `hendrycks/competition_math` | Dan Hendrycks | ✅ 官方 | CC BY-NC-SA | ✗ HF Hub 不可达 |
| `lighteval/MATH` | lighteval 团队 | 社区镜像 | CC BY-NC-SA | ✗ HF Hub 不可达 |
| `EleutherAI/hendrycks_math` | EleutherAI | 社区镜像 | MIT | **✅ 选择** |
| `HuggingFaceH4/MATH-500` | HF H4 团队 | OpenAI 子集 | MIT | ✗ 不选（下游用） |

选择理由：官方 `hendrycks/competition_math` 在 HF Hub 上连接重置不可达。`EleutherAI/hendrycks_math` 包含同样的 Hendrycks MATH 数据（同题目、同解答），由知名研究组织 EleutherAI 上传，MIT 许可证。7 个 subject configs 分别加载。

---

## 9. v1 预测 vs 实际结果回顾

### 9.1 v1 设计文档的预测

> **gpqa_diamond**: 验证集中无 gpqa_diamond 数据。MMLU (22 STEM) 部分覆盖其科学知识维度，但难度有 gap（MMLU 是大学/高中级，gpqa_diamond 是研究生级）。1M proxy 无法学会研究生级内容，所以无法直接用 gpqa_diamond 作为验证集 task。

> **math_cot**: 验证集中无 MATH dataset 数据。GSM8K 部分覆盖其数学推理维度，但难度有 gap。1M proxy 学不动竞赛级内容。GSM8K 的数学推理信号是当前最接近的替代。

### 9.2 v1 实验验证结果

v1 设计文档的预测部分正确：
- ✅ MMLU → gpqa_diamond 迁移效果差 (R²=0.105)
- ✅ GSM8K → math_cot 迁移效果差（gsm8k_cot accuracy ≈ 0 at 0.5x scale）
- ❌ "1M proxy 无法学会竞赛级内容" — 这个判断过于保守

### 9.3 v2 的修正

v2 放弃了"1M proxy 学不动"的假设，直接加入 GPQA 和 MATH 作为 proxy task：

- **即使 1M proxy 对 GPQA/MATH 的 loss 较高**（学不好），只要不同 θ 之间的 loss 差异与下游 accuracy 差异相关（R² > 0），就能提供搜索信号
- R² 衡量的是"loss 差异能否预测 accuracy 差异"，不是"loss 绝对值是否低"
- 即使 proxy model 对 MATH 的 loss 很高，如果 loss 的方差能反映 θ 的质量差异，就是有效信号

---

## 10. 潜在风险

### 10.1 gpqa 样本量仍偏小

448 题 ≈ 110K tokens，占总量的 ~2.4%。per-θ loss 估计噪声中等。

**应对**：比 v1 的 0 覆盖已大幅改善。如果 R² 仍偏低，后续可考虑 gpqa_extended (546)。

### 10.2 MATH 难度 gap

MATH 是竞赛级数学，1M proxy 的 loss 可能很高（学不好）。

**应对**：R² 衡量的是方差解释率，不是绝对 loss 值。即使 loss 高，只要方差与下游相关就有信号。

### 10.3 数据下载

`hendrycks/competition_math` 之前有被移除/重组的报告，可能不稳定。

**应对**：用 `HF_ENDPOINT=https://hf-mirror.com` 下载。`Idavidrein/gpqa` 已通过 gating（diamond 已缓存），main 应该可下载。

### 10.4 CC BY-NC-SA 许可证

MATH 的 CC BY-NC-SA 含 NC（非商业用途）限制。

**应对**：研究用途 OK。

---

## 11. 关键设计决策

| 决策点 | v1 | v2 | 理由 |
|--------|----|----|------|
| Task 数量 | 4 | 6 | 补全 GPQA + MATH 直接覆盖 |
| GPQA 来源 | 不纳入 | Idavidrein/gpqa main (448) | 官方源，diamond ⊂ main，统计功效 |
| MATH 来源 | 不纳入 | hendrycks/competition_math train (7500→5000) | 官方源，train ≠ test 无重叠 |
| MATH fallback | — | 无 fallback | 优先权威性，单一数据源 |
| 搜索模式 | R²-weighted (默认) | 待 R² 出来后决定 | 先看信号再选 |
| Loss 策略 | 全部 full-seq | 全部 full-seq | 不变 |

---

## 12. 实施路线图

### Phase 1: 生成验证集

1. 创建 `scripts/validation_set/prepare_stem_v2.py`（基于 v1）
2. 运行 `HF_ENDPOINT=https://hf-mirror.com python scripts/validation_set/prepare_stem_v2.py`
3. 验证输出 `data/stem_v2_tokenized.pt`（6 task，~16K 样本）

### Phase 2: 重评估 proxy 模型

1. 用 `reval_with_new_valset.py --val-path data/stem_v2_tokenized.pt` 重评估 144 个 proxy 模型
2. 获取 per-task R² 和 loss

### Phase 3: 搜索模式决策

1. 检查 per-task R² 和权重分布
2. 如果 gsm8k 仍占 >40%，切换到 equal_weight
3. 用 `resume_from_stage5.py --search-mode <mode>` 重跑搜索

---

## 13. 文件清单

| 文件 | 说明 |
|------|------|
| `scripts/validation_set/prepare_stem_v2.py` | v2 验证集生成脚本 |
| `data/stem_v2_tokenized.pt` | 生成的 tokenized 验证集 |
| `data/stem_v2.parquet` | 文本+元数据 parquet |
| `scripts/runners/reval_with_new_valset.py` | 重评估脚本 |
| `scripts/runners/resume_from_stage5.py` | 搜索重跑脚本 |
| `docs/STEM_V2_DESIGN.md` | 本设计文档 |
