# QuaDMix-CAP: Capability-Aligned Proxy 验证集设计

> **版本**: v1.4  
> **日期**: 2026-07-06  
> **状态**: 设计完成，验证集目标重新定义为"已证明有效的训练数据"，benchmark train 部分采用等比例采样，待 Phase 1 实施

---

## 1. 背景与问题

### 1.1 现有变体对比

| 变体 | 验证集 | 效果 |
|------|--------|------|
| **QuaDMix-OH** | OpenHermes SFT 对话训练数据 | ✓ 较好 |
| **QuaDMix-BMK** | 下游 benchmark 测试数据 | ✗ 较差 |

### 1.2 关键观察

**OH > BMK 的原因**：
- OH 使用 **SFT 对话训练数据**，格式丰富多样（长文本、多轮对话、详细解释、推理过程）
- BMK 使用 **benchmark 测试数据**，格式单一（QA、选择题、短答案）
- **验证集的数据格式决定了 proxy 学到的能力类型**
- 如果验证集都是短答案/选择题，proxy 学到的分布太窄，无法提升通用推理能力

### 1.3 核心问题

Proxy 优化目标（验证集上的 val_loss）与下游目标（21 个推理 benchmark 得分）不匹配：

```
Proxy 在 essential-web 上训练
  → 在验证集上测 val_loss
    → LightGBM 学: θ → val_loss
      → 搜索 θ* 最小化预测 val_loss
        → θ* 采样数据做中训练
          → 中训练模型在 21 benchmark 上评估
```

**问题链路**：
1. Proxy 在 essential-web 训练
2. 在 OpenHermes/BMK 上测 val_loss
3. LightGBM 学到"选语言流畅的文本 → loss 低"
4. θ* 倾向选"语言质量高"的文档
5. 和 DCLM quality-top-k 选出同样的东西
6. 对话续写能力 ≠ 推理能力 → QuaDMix 没赢

### 1.4 设计目标

用**能力对齐的训练数据**做验证集，让 proxy 学到的分布特征能迁移到下游 benchmark。

---

## 2. 核心设计原则

### 2.1 数据格式原则

1. **选择格式丰富的训练数据**：长文本、详细解释、推理过程，而不是简单 QA/选择题
2. **优先 full-sequence loss**：让 context 也参与 loss，信号更强
3. **避免短答案**：如果答案是 A/B/C/D，必须用 full-sequence

### 2.2 能力对齐原则

验证集应该覆盖下游 benchmark 的 5 类核心能力：
- 语言理解
- 常识推理
- 世界知识
- 阅读理解
- 符号/逻辑推理

### 2.3 样本量原则

- 每类 5K-10K 样本
- 总计 ~30-50K 样本
- 保证 loss 稳定，LightGBM 标签噪声小

---

## 3. 能力聚类方案

### 3.1 5 类聚类

| 聚类 | 下游 benchmark (21个) | 样本量 |
|------|---------------------|--------|
| **语言理解** | hellaswag, lambada, winogrande, winograd, language_id | 8,255 |
| **常识推理** | copa, piqa, openbook_qa, commonsense_qa | 6,500 |
| **世界知识** | arc_easy, arc_challenge, jeopardy, qa_wikidata | 8,000 |
| **阅读理解** | boolq, squad, coqa | 6,000 |
| **符号/逻辑** | lsat_ar, operators, dyck, cs_algorithms, repeat_copy | 2,472 |

### 3.2 为什么 5 类而非 21 类或 1 类？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **21 个 per-task** | 粒度最细 | ensemble R² 只有 0.32（误差叠加），训练成本高 |
| **5 个 per-cluster** | 平衡粒度和稳定性，预期 R² 0.5-0.7 | 同类任务可能有差异 |
| **1 个 aggregate** | 最简单，R²=0.79 | 掩盖能力差异，搜索方向不精准 |

**选择 5 类的理由**：
- 同类任务正相关，聚合后噪声抵消
- 训练成本适中，搜索时 5 个预测稳定
- 保留能力差异信息，比 aggregate 更精准

---

## 4. 验证数据源选择

### 4.1 数据源选择原则

1. **优先选有详细解释的训练数据**：
   - SocialIQA 有 "reason" 字段（解释为什么选这个答案）
   - ARC 有 explanation 版本
   - GSM8K 有 step-by-step 推理过程

2. **避免纯选择题格式**：如果数据源是选择题，确保有额外的解释文本

3. **优先 HuggingFace 可直接获取的数据集**

### 4.2 实际实现的数据源选择（v1）

| 能力类别 | 下游 benchmark | v1 验证数据源 | 可用样本 | 目标采样 | Loss 策略 |
|---------|---------------|-------------|---------|---------|----------|
| **language_understanding** | hellaswag, lambada, winogrande, winograd, language_id | HellaSwag train + WinoGrande train + LAMBADA test | 85,456 | 8,000 | full-sequence |
| **common_sense_reasoning** | copa, piqa, openbook_qa, commonsense_qa | CommonsenseQA train + PIQA train + COPA train + OpenBookQA train | 31,811 | 8,000 | full-sequence |
| **world_knowledge** | arc_easy, arc_challenge, jeopardy, qa_wikidata | ARC-Easy train + ARC-Challenge train | 3,370 | 3,370 | full-sequence |
| **reading_comprehension** | boolq, squad, coqa | SQuAD train + BoolQ train + CoQA train | 205,673 | 8,000 | full-sequence |
| **symbol_logic** | lsat_ar, operators, dyck, cs_algorithms, repeat_copy | GSM8K train + synthetic dyck + synthetic operators + synthetic repeat_copy | 27,396 | 8,000 | full-sequence |

**数据源变更说明**：
- SocialIQA → CommonsenseQA：SocialIQA 使用自定义加载脚本，新版 `datasets` 库不支持
- LogiQA 移除：同样使用自定义加载脚本
- WinoGrande 加入 language_understanding：覆盖代词消歧能力（下游 winogrande/winograd）
- LAMBADA 加入 language_understanding：覆盖长上下文语言建模能力（下游 lambada）
- COPA 加入 common_sense_reasoning：覆盖因果推理能力（下游 copa）
- OpenBookQA 移至 common_sense_reasoning：与下游 benchmark 分类对齐
- BoolQ 移至 reading_comprehension：与下游 benchmark 分类对齐
- CoQA 加入 reading_comprehension：覆盖多轮对话阅读理解（下游 coqa）
- Synthetic dyck/operators/repeat_copy 加入 symbol_logic：覆盖纯符号操作能力

### 4.3 数据源详细说明

#### 4.3.1 language_understanding

**下游 benchmark**：hellaswag, lambada, winogrande, winograd, language_id

**验证数据源**：
- **HellaSwag train** (39,905 样本): 句子补全任务
  - 格式: `{context} {correct_ending}`
  - 覆盖: hellaswag
- **WinoGrande train** (40,398 样本): 代词消歧
  - 格式: `Sentence: {sentence}\nAnswer: {correct}\nCompleted: {filled_sentence}`
  - 覆盖: winogrande, winograd
- **LAMBADA test** (5,153 样本): 长上下文语言建模
  - 格式: `{passage}`（完整段落，最后一个词是预测目标）
  - 覆盖: lambada

**Loss 策略**：full-sequence

#### 4.3.2 common_sense_reasoning

**下游 benchmark**：copa, piqa, openbook_qa, commonsense_qa

**验证数据源**：
- **CommonsenseQA train** (9,741 样本): 常识推理
  - 格式: `Question: {question}\nAnswer: {correct_answer}`
  - 覆盖: commonsense_qa
- **PIQA train** (16,113 样本): 物理直觉 QA
  - 格式: `Goal: {goal}\nSolution: {correct_solution}`
  - 覆盖: piqa
- **COPA train** (1,000 样本): 因果推理
  - 格式: `{premise}, {therefore|because} {correct_choice}`
  - 覆盖: copa
- **OpenBookQA train** (4,957 样本): 开卷科学问答
  - 格式: `Question: {question}\nAnswer: {correct_answer}`
  - 覆盖: openbook_qa

**Loss 策略**：full-sequence

#### 4.3.3 world_knowledge

**下游 benchmark**：arc_easy, arc_challenge, jeopardy, qa_wikidata

**验证数据源**：
- **ARC-Easy train** (2,251 样本): 简单科学问答
- **ARC-Challenge train** (1,119 样本): 挑战性科学问答
  - 格式: `Question: {question}\nAnswer: {correct_answer}`
  - 覆盖: arc_easy, arc_challenge

**Loss 策略**：full-sequence

**未覆盖 benchmark**：jeopardy（广泛事实知识）、qa_wikidata（Wikidata 属性查询）— 无对应训练数据源

#### 4.3.4 reading_comprehension

**下游 benchmark**：boolq, squad, coqa

**验证数据源**：
- **SQuAD train** (87,599 样本): 抽取式问答
  - 格式: `{context}\nQuestion: {question}\nAnswer: {answer}`
  - 覆盖: squad
- **BoolQ train** (9,427 样本): Yes/No 阅读理解
  - 格式: `Passage: {passage}\nQuestion: {question}\nAnswer: yes/no`
  - 覆盖: boolq
- **CoQA train** (108,647 样本): 多轮对话阅读理解
  - 格式: `Story: {story}\nQuestion: {question}\nAnswer: {answer}`
  - 覆盖: coqa

**Loss 策略**：full-sequence

#### 4.3.5 symbol_logic

**下游 benchmark**：agi_eval_lsat_ar, bigbench_operators, bigbench_dyck_languages, bigbench_cs_algorithms, bigbench_repeat_copy_logic

**验证数据源**：
- **GSM8K train** (7,473 样本): 数学应用题 + step-by-step 推理
  - 格式: `Question: {question}\nSolution: {step_by_step_answer}`
  - 覆盖: lsat_ar（逻辑推理部分）
- **Synthetic Dyck** (5,000 样本): 平衡括号序列
  - 格式: `Dyck language sequence: {bracket_seq}\nThis is a valid balanced bracket sequence.`
  - 覆盖: bigbench_dyck_languages
- **Synthetic Operators** (~10,000 样本): 自定义运算符求值
  - 格式: `Define: a @ b = {result}\nCompute: {expr} = {result}`
  - 覆盖: bigbench_operators
- **Synthetic Repeat Copy** (5,000 样本): 序列重复指令
  - 格式: `Q: say {word} {n} times [and ...]\nA: {repeated_output}`
  - 覆盖: bigbench_repeat_copy_logic

**Loss 策略**：full-sequence

**未覆盖 benchmark**：bigbench_cs_algorithms（算法题，无对应训练数据源）

---

## 5. Loss 计算策略

### 5.1 全部使用 full-sequence loss

**理由**：
1. 避免短答案噪声（A/B/C/D 只有 1-2 个 token）
2. Context 也参与 loss，信号更强
3. 与 OH 验证集策略一致（OH 也是 full-sequence）

### 5.2 Loss mask 机制

参考 v6 的 per-task loss mask 机制：
- 每个样本有 `loss_mask` 字段，标记哪些 token 参与 loss 计算
- 默认 full-sequence（所有 token 都参与）
- 特殊情况下可以 mask 掉某些 token（如系统提示、格式标记）

---

## 6. 加权策略

### 6.1 方案对比

| 方案 | 权重 | 优点 | 缺点 |
|------|------|------|------|
| **A. 等权** | 各 0.2 | 简单 | 忽略 R² 差异 |
| **B. 按 R² 加权** | R² 高的 cluster 权重高 | 预测更准的 cluster 权重高 | 需要交叉验证 |
| **C. 按 benchmark 数量加权** | 反映重要性 | 直观 | 忽略难度差异 |

### 6.2 选择方案 B：按 R² 加权

**实现方法**：
1. 用 5-fold CV 估计每个 cluster 的 R²
2. 计算权重：`weight_i = R²_i / sum(R²_j)`
3. 搜索时用加权 loss：`total_loss = sum(weight_i * cluster_loss_i)`

**理由**：
- R² 高的 cluster 预测更准，应该给更高权重
- 自动适应不同 cluster 的难度和信号强度

---

## 7. 模型训练策略

### 7.1 训练 5 个 per-cluster LightGBM

**输入特征**：
- θ 参数（N=5 个 merge 参数 + M=22 个 sampling 参数 = 27 维）
- 可选：域分布统计特征

**输出标签**：
- 每个 cluster 的 val_loss（5 个独立模型）

**训练配置**：
- 5-fold CV 估计 R²
- Early stopping（patience=50）
- 超参数调优（学习率、树深度、正则化）

### 7.2 与 per-task 的对比

| 方案 | 模型数量 | 训练成本 | 搜索成本 | 预期 R² |
|------|---------|---------|---------|---------|
| **per-task** | 21 | 高 | 高 | 0.32 (ensemble) |
| **per-cluster** | 5 | 中 | 中 | 0.5-0.7 (预期) |
| **aggregate** | 1 | 低 | 低 | 0.79 (单模型) |

---

## 8. 实现路线图

### Phase 1: 低成本验证（1-2 小时）

**目标**：验证设计合理性，避免盲目实现

**步骤**：
1. 写 `validate_cap_design.py`：
   - 加载 v6 per-task losses（874 组实验）
   - 按 5 类聚类计算 per-cluster loss
   - 训练 5 个 per-cluster LightGBM
   - 计算 per-cluster R²
   - 用 per-cluster 搜索 θ*
   - 对比域分布差异

2. 验证指标：
   - per-cluster ensemble R² > 0.5 ✓
   - 域分布与 aggregate 有显著差异（至少 3 个域变化 > 5%）✓

3. 根据验证结果调整设计：
   - 如果 per-cluster R² < 0.5 → 调整聚类方案或加权策略
   - 如果域分布相似 → 重新思考问题根因

### Phase 2: 实现 CAP 验证集（2-4 小时）

**步骤**：
1. 写 `prepare_cap_v1.py`：
   - 下载 5 类训练数据
   - 每类采样 5K-10K 样本
   - Tokenize 并保存
   - 生成 loss_mask

2. 修改 pipeline：
   - `_run_validation()` 返回 5 个 per-cluster loss
   - `_train_per_task_models()` 训练 5 个 per-cluster 模型
   - `search_optimal()` 用 per-cluster 加权预测

### Phase 3: 完整实验（1-2 天）

**步骤**：
1. 用 CAP 验证集跑 3000 proxy 实验
2. 用 per-cluster 搜索 θ*
3. 生成数据配比，跑中训练
4. 评估 21 个 benchmark，对比：
   - QuaDMix-OH
   - QuaDMix-BMK
   - QuaDMix-CAP
   - Quality-top-k (DCLM)
   - Quality-top-k (FineWeb-Edu)

---

## 9. 潜在风险与应对

### 9.1 数据源获取困难

**风险**：某些训练数据集可能不在 HuggingFace 上，或需要特殊处理

**应对**：
- 优先选 HuggingFace 上直接可用的数据集
- 准备 fallback 方案（如用 benchmark train split 代替）

### 9.2 per-cluster R² 仍然低

**风险**：即使聚合，某些 cluster 的 R² 可能仍低于 0.5

**应对**：
- 先用 Phase 1 快速验证
- 如果某类 R² < 0.3，考虑合并到其他类或降低权重

### 9.3 域分布和 aggregate 相似

**风险**：per-cluster 搜索出的域分布可能和 aggregate 几乎一样

**应对**：
- Phase 1 的关键验证点
- 如果域分布相似，说明问题不在验证集，而在搜索算法或数据源本身

---

## 10. 关键设计决策总结

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **聚类数量** | 5 类 | 平衡粒度和稳定性 |
| **数据格式** | 长文本、详细解释 | 避免短答案噪声 |
| **Loss 策略** | full-sequence | 信号强，与 OH 一致 |
| **加权策略** | 按 R² 加权 | 预测更准的 cluster 权重高 |
| **模型数量** | 5 个 per-cluster | 比 21 个 per-task 稳定，比 1 个 aggregate 精准 |
| **样本量** | 每类 5K-10K | 保证 loss 稳定 |

---

## 11. 与现有变体的关系

| 变体 | 验证集类型 | 数据格式 | 模型数量 | 预期效果 |
|------|-----------|---------|---------|---------|
| **QuaDMix-OH** | SFT 对话训练数据 | 长对话、多轮 | 1 (aggregate) | ✓ 较好 |
| **QuaDMix-BMK** | Benchmark 测试数据 | QA、选择题 | 21 (per-task) | ✗ 较差 |
| **QuaDMix-CAP** | 能力对齐训练数据 | 长文本、详细解释 | 5 (per-cluster) | ✓✓ 预期最好 |

---

## 12. 下一步行动

1. **Phase 1**: 写 `validate_cap_design.py`，验证设计合理性
2. **Phase 2**: 实现 `prepare_cap_v1.py`，生成 CAP 验证集
3. **Phase 3**: 跑完整实验，评估效果

---

## 附录 A: 下游 benchmark 详细列表

### A.1 语言理解（5 个）

| Benchmark | 样本量 | 任务描述 |
|-----------|--------|---------|
| hellaswag_zeroshot | 2,000 | 句子补全（选择最合理的结尾） |
| lambada_openai | 2,000 | 长文本最后一个词预测 |
| winogrande | 2,000 | 代词消歧（常识推理） |
| winograd | 255 | Winograd Schema Challenge |
| bigbench_language_identification | 2,000 | 语言识别 |

### A.2 常识推理（4 个）

| Benchmark | 样本量 | 任务描述 |
|-----------|--------|---------|
| copa | 500 | 因果推理（选择原因或结果） |
| piqa | 2,000 | 物理直觉 QA |
| openbook_qa | 2,000 | 开卷科学问答 |
| commonsense_qa | 2,000 | 常识推理 |

### A.3 世界知识（4 个）

| Benchmark | 样本量 | 任务描述 |
|-----------|--------|---------|
| arc_easy | 2,000 | 简单科学问答 |
| arc_challenge | 2,000 | 挑战性科学问答 |
| jeopardy | 2,000 | Jeopardy 风格问答 |
| bigbench_qa_wikidata | 2,000 | Wikidata 事实性问答 |

### A.4 阅读理解（3 个）

| Benchmark | 样本量 | 任务描述 |
|-----------|--------|---------|
| boolq | 2,000 | Yes/No 阅读理解 |
| squad | 2,000 | 抽取式问答 |
| coqa | 2,000 | 对话式问答 |

### A.5 符号/逻辑推理（5 个）

| Benchmark | 样本量 | 任务描述 |
|-----------|--------|---------|
| agi_eval_lsat_ar | 230 | LSAT 逻辑分析题 |
| bigbench_operators | 210 | 自定义运算符求值 |
| bigbench_dyck_languages | 1,000 | 括号序列补全 |
| bigbench_cs_algorithms | 1,320 | 算法追踪 |
| bigbench_repeat_copy_logic | 32 | 字符串重复 |

---

## 附录 B: v6 per-task R² 参考

| Task | R² | 聚类 |
|------|---:|------|
| lambada_openai | 0.8098 | 语言理解 |
| winograd | 0.8043 | 语言理解 |
| winogrande | 0.8039 | 语言理解 |
| copa | 0.7440 | 常识推理 |
| bigbench_operators | 0.7222 | 符号/逻辑 |
| bigbench_language_identification | 0.7068 | 语言理解 |
| piqa | 0.6687 | 常识推理 |
| hellaswag_zeroshot | 0.5807 | 语言理解 |
| boolq | 0.4916 | 阅读理解 |
| openbook_qa | 0.4819 | 常识推理 |
| coqa | 0.4714 | 阅读理解 |
| bigbench_dyck_languages | 0.4381 | 符号/逻辑 |
| squad | 0.4146 | 阅读理解 |
| arc_easy | 0.3675 | 世界知识 |
| arc_challenge | 0.3461 | 世界知识 |
| bigbench_repeat_copy_logic | 0.3151 | 符号/逻辑 |
| jeopardy | 0.2611 | 世界知识 |
| bigbench_cs_algorithms | 0.2143 | 符号/逻辑 |
| commonsense_qa | 0.1790 | 常识推理 |
| agi_eval_lsat_ar | 0.1382 | 符号/逻辑 |
| bigbench_qa_wikidata | 0.0466 | 世界知识 |

**聚类平均 R²**：
- 语言理解: 0.74
- 常识推理: 0.52
- 阅读理解: 0.46
- 符号/逻辑: 0.37
- 世界知识: 0.26

---

## 13. 覆盖缺口分析（2026-07-06 补充）

### 13.1 覆盖状态总结

CAP v1 覆盖缺口修复完成后，21 个下游 benchmark 的覆盖状态：

| 状态 | 数量 | Benchmark |
|------|------|-----------|
| **直接匹配** | 15 | hellaswag, lambada, winogrande, copa, piqa, commonsense_qa, openbook_qa, arc_easy, arc_challenge, boolq, squad, coqa, dyck_languages, operators, repeat_copy_logic |
| **近似/弱覆盖** | 3 | winograd（由 WinoGrande 近似覆盖）, lsat_ar（由 GSM8K 弱覆盖）, cs_algorithms（76% 由 synthetic_dyck 覆盖） |
| **未覆盖** | 3 | language_id, jeopardy, qa_wikidata |

### 13.2 未覆盖 benchmark 决策

| Benchmark | 不补充的理由 |
|-----------|-------------|
| **language_id** | HF 有 `papluca/language-identification`（70K 样本），但训练格式（直接识别）与 benchmark 格式（四选一 MC）差异大，1.3B 模型格式迁移不可靠 |
| **jeopardy** | 广泛事实知识，HF 无对应训练数据源 |
| **qa_wikidata** | Wikidata 属性查询，HF 无对应训练数据源 |
| **cs_algorithms** | 76%（括号匹配 1,000 题）已被 synthetic_dyck 覆盖，24%（LCS 320 题）合成格式与 benchmark 几乎相同，收益低 |

**决策**：接受 15/21 直接匹配现状，不补充任何未覆盖 benchmark。

---

## 14. 格式差异分析（2026-07-06 补充）

### 14.1 CAP 格式 vs Benchmark 格式对比

| Benchmark | CAP 格式 | Benchmark 格式 | Gap |
|-----------|---------|---------------|-----|
| lambada | `{full_text}` | `{context} {last_word}` | **NONE** |
| copa | `{premise}, {conn} {choice}` | `{premise}, {conn} {choice}` | **NONE** |
| arc_easy | `Question: {q}\nAnswer: {a}` | `Question: {q}\nAnswer: {a}` | **NONE** |
| arc_challenge | `Question: {q}\nAnswer: {a}` | `Question: {q}\nAnswer: {a}` | **NONE** |
| boolq | `Passage: {p}\nQuestion: {q}\nAnswer: {y/n}` | `{p}\nQuestion: {q}\nAnswer: {y/n}` | **SMALL** |
| squad | `{ctx}\nQuestion: {q}\nAnswer: {a}` | `{ctx}\nQuestion: {q}\nAnswer:  {a}` | **SMALL** |
| hellaswag | `{raw_ctx} {ending}` | `{cleaned_ctx} {ending}` | **MODERATE** |
| piqa | `Goal: {g}\nSolution: {s}` | `Question: {g}\n\nAnswer: {s}` | **MODERATE** |
| openbook_qa | `Question: {q}\nAnswer: {a}` | `{q} {a}` | **MODERATE** |
| operators | `Define: ...\nCompute: ... = {r}` | `Given the definition...\n{def}\n{expr} = {r}` | **MODERATE** |
| repeat_copy | `Q: {instr}\nA: {output}` | `repeat with logic:\n\nQ: {instr}\nA:{output}` | **MODERATE** |
| **commonsense_qa** | `Answer: {answer_text}` | `Choices: A...\nAnswer: {letter}` | **LARGE** |
| **winogrande** | `Sentence: ...\nAnswer: ...\nCompleted: ...` | `{prefix}{option} {suffix}` | **LARGE** |
| **coqa** | `Story: {s}\nQuestion: {q}\nAnswer: {a}` | `Story: {s}\nPreceding: ...\nFinal question:\nQuestion: {q}\nAnswer: {a}` | **LARGE** |
| **dyck** | `Dyck: {complete}\nThis is valid.` | `Complete...\nInput: {partial}\nOutput: {brackets}` | **LARGE** |

### 14.2 格式 Gap 统计

| Gap 级别 | 数量 | 能否迁移 |
|---------|------|---------|
| **NONE** | 4 | ✅ 完全可迁移 |
| **SMALL** | 2 | ✅ 几乎可迁移 |
| **MODERATE** | 5 | ⚠️ 大概率可迁移（答案类型一致） |
| **LARGE** | 4 | ❌ 迁移风险高 |

### 14.3 4 个 LARGE gap 的具体问题

1. **commonsense_qa**: CAP 教模型输出答案文本，benchmark 要求输出字母 A-E。1.3B 模型可能不会"选字母"
2. **winogrande**: CAP 用三行结构化格式，benchmark 是自然句子补全。模型学到的是"解析结构化格式"而非"理解代词"
3. **coqa**: CAP 每题独立，benchmark 包含多轮对话历史。模型没见过"Preceding questions"这种格式
4. **dyck**: CAP 是"这段括号序列是合法的"（验证），benchmark 是"补全剩余括号"（生成）。完全不同的任务

### 14.4 决策：不改格式

**理由**：

1. CAP 的核心优势是用 train data 做验证，避免数据泄露
2. 格式差异是"能力迁移"问题，不是"数据泄露"问题
3. 保持简单格式 → proxy 学到更通用的能力 → 泛化到 benchmark 的不同格式
4. 如果改成 benchmark 格式 → proxy 过拟合到特定格式 → 又回到 BMK 的"格式太窄"问题

**验证方法**：跑完实验看结果。如果 CAP 的 15 个"直接匹配"benchmark 里，LARGE gap 的 4 个（commonsense_qa/winogrande/coqa/dyck）不涨，说明格式差异确实是问题，再针对性修复。

---

## 15. Essential-web 数据泄露分析（2026-07-06 补充）

### 15.1 数据源

- **来源**: `EssentialAI/essential-web-v1.0`
- **Common Crawl dump**: `CC-MAIN-2024-38`
- **规模**: 275M 文档，~808 GB

### 15.2 Benchmark 数据在 Common Crawl 中的存在性

| Benchmark | 来源 | 在 Common Crawl 中？ | 泄露风险 |
|-----------|------|---------------------|---------|
| HellaSwag | wikiHow | **是** | 高 |
| SQuAD | Wikipedia | **是** | 高 |
| BoolQ | Wikipedia + Google | **是** | 高 |
| CoQA | web 文章 | **是** | 高 |
| COPA | 部分 web | 部分 | 中 |
| PIQA | 众包新写 | 否 | 低 |
| CommonsenseQA | 众包新写 | 否 | 低 |
| OpenBookQA | 众包新写 | 否 | 低 |
| GSM8K | 众包新写 | 否 | 低 |
| WinoGrande | 众包新写 | 否 | 低 |
| ARC | 标准化考试 | 否 | 低 |
| LAMBADA | BookCorpus | 否 | 低 |

### 15.3 去污染状态

**代码中没有任何去重/去污染步骤。**

- 无 n-gram 匹配
- 无 fuzzy deduplication (SimHash/MinHash)
- 无 exact-match filtering
- 无 URL-level exclusion

### 15.4 泄露对 BMK vs CAP 的影响

泄露对两者是**对称的**：

- BMK 用 benchmark test split → essential-web 可能包含同源 web 文本（wikiHow、Wikipedia）
- CAP 用 benchmark train split → essential-web 同样可能包含同源 web 文本

两者的泄露程度差不多，因为 wikiHow/Wikipedia 的 train 和 test 内容都在 Common Crawl 里。

**结论**: 泄露是个真实问题，但不影响 BMK vs CAP 的相对优劣。

---

## 16. Proxy 感知瓶颈分析（2026-07-06 补充）

### 16.1 核心问题

不管用 BMK 还是 CAP，**proxy 模型是同一个 1.3B 模型**，训练在同样的 essential-web 子集上。

| Proxy 能学到的 | Proxy 学不到的 |
|---------------|---------------|
| 语言流畅度（quality score 高的文本 → loss 低） | 推理深度（文本是否包含多步推理） |
| 主题分布（科学/数学文本 → loss 低） | 知识密度（文本是否包含有用事实） |
| 格式匹配（训练格式接近验证格式 → loss 低） | 能力迁移（训练数据是否培养推理能力） |

### 16.2 为什么 QuadMix = Quality(dclm)

Proxy 唯一能感知的维度就是"语言质量"。所以搜索收敛到 quality-top-k 的等效解。

### 16.3 CAP 能否突破瓶颈

CAP 的 val_loss 理论上编码了推理信号，但 proxy 能否感知到这些信号取决于：

1. 推理密集文本 vs 非推理文本在 next-token prediction loss 上是否有显著差异
2. Proxy 模型容量是否足够区分这些差异
3. 训练步数（5000 steps）是否足够学到这些差异

**预期**: CAP 的潜力更高，但能否兑现取决于 proxy 的感知能力是否足够。需要实验验证。

---

## 17. I3: 增加推理密度质量维度（2026-07-06 补充）

### 17.1 目标

当前 5 个质量分（DCLM、fineweb_edu 等）都衡量语言质量。I3 要加一个新维度：**推理密度**。

### 17.2 三种实现路径

| 方案 | 做法 | 成本 | 精度 |
|------|------|------|------|
| **A. 启发式关键词** | 统计推理标记词频次 | 1 天 | 低 |
| **B. 训练分类器** | 训练 FastText 二分类 | 2-3 天 | 中 |
| **C. LLM 蒸馏** | 大模型打分 → 训练小模型 | 5-7 天 | 高 |

### 17.3 方案 A: 启发式关键词计数

```python
REASONING_MARKERS = [
    "therefore", "because", "since", "thus", "hence",
    "if...then", "implies", "follows that", "conclude",
    "assume", "suppose", "given that", "it follows",
    "step 1", "step 2", "first...second...third",
]
score = count_markers(text) / len(text.split())
```

- 优点：零训练成本
- 缺点：噪声大，"because" 出现在日常对话里不代表有推理

### 17.4 方案 B: 训练 FastText 分类器

```
正样本: GSM8K 解答, MATH 解答, LogiQA 推理段落, 
        OpenHermes 中 CoT 对话, 代码+docstring
负样本: 新闻, 社交媒体, 日常对话, 产品描述
```

- 优点：比关键词准确，和现有 pipeline 兼容
- 缺点：需要标注数据，分类边界不好定

### 17.5 方案 C: LLM 蒸馏

```
1. 采样 50K 文档
2. 用 Qwen-72B / GPT-4 打分 (1-5 分: 推理密度)
3. 训练小模型 (BERT/Roberta) 拟合打分
4. 用训练好的小模型对 275M 文档批量打分
```

- 优点：最准确
- 缺点：API 成本高，周期长

### 17.6 I3 与 CAP 的关系

如果验证集是 CAP（推理导向），proxy 模型在推理题上的 val_loss **已经包含了推理信号**。LightGBM 不需要额外的"推理密度分"——它直接从 val_loss 学到"哪些 θ 让推理 loss 低"。

**所以 I3 可能不是独立于 CAP 的改进，而是 CAP 的补充**：

- CAP 解决"验证集测什么"
- I3 解决"LightGBM 有什么额外特征"
- 如果 CAP 的 val_loss 已经能区分推理好坏，I3 是多余的

### 17.7 决策

**先跑 CAP 实验**。如果 CAP 的 per-cluster val_loss 能给出足够的推理信号（R² > 0.5），I3 不需要做。如果 CAP 的 R² 仍然低（proxy 感知不到推理差异），再考虑 I3 方案 B（FastText 分类器）。

---

## 18. 推理密集数据集方案（2026-07-06 补充）

### 18.1 核心问题

当前 CAP v1 的数据源与 BMK 本质上是同一类数据（benchmark train split vs test split），只是换了 split。如果 BMK 失败的根本原因是 proxy 感知瓶颈（只能感知语言质量，感知不到推理能力），那 CAP v1 也会失败。

**关键洞察**：CAP 需要引入 BMK 没有的新信号类型——**推理密集的自然文本**，而不是 benchmark 题目。

### 18.2 为什么推理密集文本能给出不同信号

| 验证集类型 | proxy 学到的 θ 偏好 | 搜索出的数据 |
|-----------|-------------------|-------------|
| BMK（benchmark 题目） | 选语言流畅的文本 | ≈ quality-top-k |
| CAP v1（benchmark train） | 选语言流畅的文本 | ≈ quality-top-k（同 BMK） |
| **CAP v2（推理密集文本）** | **选推理密集的文本** | **不同于 quality-top-k** |

推理密集文本有更强的长程依赖（step 1 → step 2 → 结论），如果 θ 采样了更多科学/数学域，proxy 在这些文本上 loss 更低。这给 LightGBM 一个**域感知**信号，而不是纯语言质量信号。

### 18.3 按能力聚类的公认推理密集数据集

#### 18.3.1 symbol_logic（最成熟）

| 数据集 | 大小 | 格式 | 公认度 |
|--------|------|------|--------|
| **Orca-Math** | 200K | GPT-4 生成的 step-by-step 数学解答，1-5.5K chars | Shah et al. 2024，SLM 专用 |
| **MetaMathQA** | 395K | 增强版 GSM8K+MATH，8 种增强方式 | Yu et al. 2023，SOTA |
| **NuminaMath-CoT** | 860K | 竞赛数学（AMC/AIME）+ 完整证明 | AIMO 2024 冠军数据集 |
| **CAMEL-Math** | 50K | 25 个数学主题 × GPT-4 解答 | Li et al. 2023 |

#### 18.3.2 common_sense_reasoning

| 数据集 | 大小 | 格式 | 公认度 |
|--------|------|------|--------|
| **OpenOrca (CoT subset)** | ~300K | FLAN 任务 + GPT-4 CoT 解释 | Mukherjee et al. 2023 |
| **Magpie-Reasoning** | 150K | 指令 + 长回复 + 质量标注 | Xu et al. 2024 |

#### 18.3.3 language_understanding / reading_comprehension / world_knowledge

| 数据集 | 大小 | 格式 | 公认度 |
|--------|------|------|--------|
| **NaturalReasoning** (Meta) | 1.15M | 长文本问答，多步推导，覆盖物理/化学/数学/语言 | Meta 2025，MATH/GPQA 上 scaling 优于其他数据集 |
| **QASPER** | 5K Qs / 1.5K papers | 完整 NLP 论文 + 问答 + 证据段落 | Dasigi et al. 2021 |

### 18.4 推荐方案：混合验证集

每个 cluster 的验证集 = **推理密集文本（70%）+ benchmark 训练题（30%）**

| 聚类 | 推理密集文本（70%） | benchmark 训练题（30%） |
|------|-------------------|----------------------|
| **language_understanding** | NaturalReasoning（语言类子集） | HellaSwag + WinoGrande + LAMBADA |
| **common_sense_reasoning** | OpenOrca CoT subset | CommonsenseQA + PIQA + COPA + OpenBookQA |
| **world_knowledge** | NaturalReasoning（科学类子集） | ARC-Easy + ARC-Challenge |
| **reading_comprehension** | QASPER | SQuAD + BoolQ + CoQA |
| **symbol_logic** | Orca-Math 或 MetaMathQA | GSM8K + synthetic dyck/operators/repeat_copy |

**设计理由**：
- 推理密集文本（70%）：提供新信号（推理能力），让 proxy 学到域感知特征
- benchmark 训练题（30%）：保证下游覆盖，避免格式迁移问题

### 18.5 推理密集文本对 21 个下游 task 的覆盖分析

| 下游 task | 需要的核心能力 | 推理密集文本覆盖？ | 说明 |
|-----------|-------------|------------------|------|
| **hellaswag** | 句子补全、常识推断 | ⚠️ 弱 | NaturalReasoning 是长文本推理，不是句子补全 |
| **lambada** | 长上下文最后一个词预测 | ⚠️ 弱 | 需要语言建模能力，不是推理能力 |
| **winogrande** | 代词消歧 | ⚠️ 弱 | 需要指代理解，推理密集文本不专门训练这个 |
| **winograd** | 代词消歧 | ⚠️ 弱 | 同上 |
| **language_id** | 语言识别 | ❌ 无 | 纯分类任务，推理文本无关 |
| **copa** | 因果推理 | ✅ 强 | NaturalReasoning 含大量因果推导 |
| **piqa** | 物理直觉 | ✅ 强 | NaturalReasoning 含物理推理 |
| **openbook_qa** | 科学知识应用 | ✅ 强 | NaturalReasoning 含科学推理 |
| **commonsense_qa** | 常识推理 | ✅ 中 | OpenOrca CoT 含常识推理链 |
| **arc_easy** | 科学理解 | ✅ 强 | NaturalReasoning 含科学推导 |
| **arc_challenge** | 科学推理 | ✅ 强 | 同上 |
| **jeopardy** | 广泛事实知识 | ⚠️ 弱 | 推理文本侧重推理过程，不是事实记忆 |
| **qa_wikidata** | 属性查询 | ❌ 无 | 纯事实查询，推理文本无关 |
| **boolq** | 阅读理解 + 判断 | ✅ 强 | QASPER 是长文档 QA |
| **squad** | 抽取式问答 | ✅ 强 | QASPER 是抽取+推理 QA |
| **coqa** | 多轮对话理解 | ⚠️ 弱 | QASPER 是单轮，不是多轮 |
| **lsat_ar** | 逻辑分析 | ✅ 强 | NuminaMath/Orca-Math 含严格逻辑推导 |
| **operators** | 符号运算 | ✅ 强 | Orca-Math/MetaMathQA 含符号操作 |
| **dyck_languages** | 括号匹配 | ⚠️ 弱 | 数学推导含括号但目的不同 |
| **cs_algorithms** | 算法追踪 | ⚠️ 弱 | 代码数据集可覆盖，但推理文本不专门训练这个 |
| **repeat_copy** | 指令跟随 | ❌ 无 | 纯机械操作，推理文本无关 |

**覆盖统计**：

| 覆盖程度 | 数量 | Task |
|---------|------|------|
| ✅ 强 | 10 | copa, piqa, openbook_qa, arc_easy, arc_challenge, boolq, squad, lsat_ar, operators, commonsense_qa |
| ⚠️ 弱 | 8 | hellaswag, lambada, winogrande, winograd, jeopardy, coqa, dyck, cs_algorithms |
| ❌ 无 | 3 | language_id, qa_wikidata, repeat_copy |

**结论**：推理密集文本对**推理类 task（10/21）**覆盖很好，但对**语言理解类（hellaswag/lambada/winogrande/winograd）**和**机械操作类（repeat_copy/dyck）**覆盖弱。这就是为什么需要混合方案：推理密集文本（70%）+ benchmark 训练题（30%），后者补充覆盖缺口。

### 18.6 与 essential-web 的关系

**不从 essential-web 筛选推理密集子集**。理由：

1. essential-web 是训练池，从里面筛验证集等于自己验证自己，没有独立信号
2. 应该用**业界公认的、被证明能提升对应能力的训练数据集**
3. 这些数据集有明确的质量保证和推理密度标注

### 18.7 实施路线图

#### Phase 1: 数据获取（1-2 天）

1. 下载推理密集数据集：
   - `facebook/natural_reasoning` (1.15M)
   - `Open-Orca/OpenOrca` (2.94M，筛选 CoT subset)
   - `microsoft/orca-math-word-problems-200k` (200K)
   - `allenai/qasper` (5K)

2. 按能力聚类筛选和采样：
   - 每个 cluster 5,600 条推理密集文本（70% × 8,000）
   - 每个 cluster 2,400 条 benchmark 训练题（30% × 8,000）

#### Phase 2: 验证集生成（2-4 小时）

1. 修改 `prepare_cap_v1.py` → `prepare_cap_v2.py`
2. 加载推理密集数据集，按 cluster 分类
3. 混合推理密集文本 + benchmark 训练题
4. Tokenize 并保存

#### Phase 3: 完整实验（1-2 天）

1. 用 CAP v2 验证集跑 3000 proxy 实验
2. 用 per-cluster 搜索 θ*
3. 生成数据配比，跑中训练
4. 评估 21 个 benchmark，对比：
   - QuaDMix-OH
   - QuaDMix-BMK
   - QuaDMix-CAP-v1（benchmark train split）
   - **QuaDMix-CAP-v2（推理密集文本）**
   - Quality-top-k (DCLM)
   - Quality-top-k (FineWeb-Edu)

### 18.8 预期效果

| 变体 | 验证集信号 | 预期搜索方向 | 预期效果 |
|------|-----------|-------------|---------|
| OH | 对话续写 | 选语言流畅文本 | ≈ quality-top-k |
| BMK | benchmark 答题（过拟合） | 选语言流畅文本 | ≈ quality-top-k |
| CAP v1 | benchmark train 答题 | 选语言流畅文本 | ≈ quality-top-k |
| **CAP v2** | **推理密集文本理解** | **选推理密集文本** | **> quality-top-k** |

**关键假设**：推理密集文本的 val_loss 能区分不同 θ 的推理能力差异，而不仅仅是语言质量差异。

---

## 19. 验证集目标重新定义（2026-07-06 补充）

### 19.1 核心问题：C2 从未被验证

当前链路：
```
θ → 训练 proxy → val_loss(验证集) → LightGBM → θ* → 中训练 → benchmark
```

关键假设 C2：**val_loss 低 → benchmark 高**

这个假设从未被验证。OH 变体的 R²=0.79 只说明 LightGBM 能预测 val_loss，不说明 val_loss 低 → benchmark 高。事实上，QuadMix-OH 的 CORE = Quality-top-k 的 CORE，暗示 C2 可能不成立。

### 19.2 新思路：绕过 C2

**不改链路，改验证集的内容。**

把验证集从"benchmark 题目"或"推理密集文本"改成"**业界已证明能提升 benchmark 的训练数据**"。

新链路：
```
旧: θ → proxy → val_loss(验证集) → θ* → 中训练 → benchmark
     假设: val_loss 低 → benchmark 高 (C2，未验证)

新: θ → proxy → val_loss(已证明有效的训练数据) → θ* → 中训练 → benchmark
     假设: val_loss 低 → 数据分布接近已证明有效的数据 → benchmark 高
     这个假设依赖业界经验，而非 proxy 能力
```

**为什么这能绕过 C2**：
- 不需要 proxy 能预测 benchmark
- 只需要 proxy 能拟合已证明有效的训练数据
- 如果 θ* 让 proxy 在这些数据上 loss 低，说明 θ* 采样出的数据分布接近这些已证明有效的数据
- 中训练模型训练在类似分布的数据上，应该也能提升 benchmark

### 19.3 21 个 benchmark 对应的"已证明有效的训练数据"

#### 19.3.1 有外部训练数据的 benchmark（10 个）

| 下游 benchmark | 业界证明有效的**外部**训练数据 | 来源 | 数据量 |
|---------------|---------------------------|------|--------|
| **arc_easy** | ARC-EXPLAIN + 科学解释文本 | Clark et al. 2018 | ~2.5K |
| **arc_challenge** | ARC-EXPLAIN + NaturalReasoning 科学子集 | Clark et al. 2018, Meta 2025 | ~1.1K + 部分 |
| **openbook_qa** | OpenBookQA train + 科学教材文本 | Mihaylov et al. 2018 | ~5K |
| **commonsense_qa** | CommonsenseQA train + OpenOrca CoT | Talmor et al. 2019, Mukherjee 2023 | ~9.7K + ~300K |
| **copa** | COPA train + 因果推理文本 | Gordon et al. 2012 | ~1K |
| **piqa** | PIQA train + 物理常识文本 | Bisk et al. 2020 | ~16K |
| **boolq** | BoolQ train + QASPER | Clark et al. 2019, Dasigi 2021 | ~9.4K + 5K |
| **squad** | SQuAD train + NaturalInstructions RC | Rajpurkar et al. 2016 | ~87K |
| **lsat_ar** | Orca-Math + MetaMathQA + NuminaMath-CoT | Shah 2024, Yu 2023, AIMO 2024 | 200K + 395K + 860K |
| **operators** | MetaMathQA + CAMEL-Math | Yu 2023, Li 2023 | 395K + 50K |

#### 19.3.2 只有自身 train split 的 benchmark（8 个）

| 下游 benchmark | 可用的训练数据 | 问题 |
|---------------|-------------|------|
| **hellaswag** | HellaSwag train | 和 BMK 同源 |
| **lambada** | LAMBADA test / BookCorpus | 和 BMK 同源 |
| **winogrande** | WinoGrande train | 和 BMK 同源 |
| **winograd** | Winograd Schema | 和 BMK 同源 |
| **coqa** | CoQA train | 和 BMK 同源 |
| **dyck** | 合成括号序列 | 和 BMK 同源 |
| **cs_algorithms** | 合成算法追踪 | 和 BMK 同源 |
| **repeat_copy** | 合成指令跟随 | 和 BMK 同源 |

#### 19.3.3 无对应训练数据的 benchmark（3 个）

| 下游 benchmark | 问题 |
|---------------|------|
| **language_id** | 纯分类任务，无推理训练数据 |
| **jeopardy** | 广泛事实知识，无对应训练数据 |
| **qa_wikidata** | 属性查询，无对应训练数据 |

### 19.4 最终验证集设计

#### 19.4.1 混合策略

每个 cluster = **70% 外部训练数据 + 30% benchmark train split（等比例）**

**外部训练数据（70%）**：
- 业界已证明能提升对应 benchmark 的训练数据
- 每个 cluster 5,600 条

**Benchmark train split（30%，等比例）**：
- 每个 benchmark 在 30% 部分中占相同比例
- 确保每个下游 benchmark 在验证集中都有足够的信号
- 避免大数据集（如 SQuAD 87K）淹没小数据集（如 COPA 1K）
- 每个 cluster 2,400 条，按 benchmark 数量等分

#### 19.4.2 按 cluster 的具体设计（含等比例分配）

| Cluster | 外部训练数据（70% = 5.6K） | Benchmark train（30% = 2.4K，等比例） | 覆盖的 benchmark |
|---------|---------------------------|--------------------------------------|-----------------|
| **language_understanding** (8K) | NaturalReasoning 语言子集 (5.6K) | HellaSwag (800) + WinoGrande (800) + LAMBADA (800) | hellaswag, lambada, winogrande, winograd, language_id |
| **common_sense_reasoning** (8K) | OpenOrca CoT subset (5.6K) | CommonsenseQA (600) + PIQA (600) + COPA (600) + OpenBookQA (600) | copa, piqa, openbook_qa, commonsense_qa |
| **world_knowledge** (8K) | ARC-EXPLAIN + NaturalReasoning 科学子集 (5.6K) | ARC-Easy (1.2K) + ARC-Challenge (1.2K) | arc_easy, arc_challenge, jeopardy, qa_wikidata |
| **reading_comprehension** (8K) | QASPER + NaturalInstructions RC (5.6K) | SQuAD (800) + BoolQ (800) + CoQA (800) | boolq, squad, coqa |
| **symbol_logic** (8K) | Orca-Math + MetaMathQA + NuminaMath-CoT (5.6K) | GSM8K (600) + synthetic_dyck (600) + synthetic_operators (600) + synthetic_repeat_copy (600) | lsat_ar, operators, dyck, cs_algorithms, repeat_copy |

**等比例采样的理由**：
- 确保每个下游 benchmark 在验证集中都有足够的信号
- 避免大数据集（如 SQuAD 87K）淹没小数据集（如 COPA 1K）
- 让 per-cluster LightGBM 对每个 benchmark 的预测更均衡
- 如果某个 benchmark 的 train split 样本数不足目标数量，则使用全部可用样本

### 19.5 与之前方案的对比

| 方案 | 验证集内容 | 依赖的假设 | 风险 |
|------|-----------|-----------|------|
| **OH** | SFT 对话 | C2: val_loss 低 → benchmark 高 | C2 不成立 |
| **BMK** | benchmark test | C2 + 无数据泄露 | C2 不成立 + 泄露 |
| **CAP v1** | benchmark train | C2 | C2 不成立 |
| **CAP v2 (§18)** | 推理密集文本 | C2 + 推理信号可感知 | C2 不成立 |
| **CAP v3 (§19)** | **已证明有效的训练数据** | **数据分布匹配 → benchmark 高** | **业界经验可能不适用** |

### 19.6 关键优势

1. **绕过 C2**：不需要 proxy 能预测 benchmark，只需要 proxy 能拟合已证明有效的数据
2. **依赖业界经验**：Orca-Math、MetaMathQA、NaturalReasoning 等已被多篇论文证明能提升对应 benchmark
3. **信号更强**：这些数据集本身就是训练数据，格式丰富，proxy 更容易学到分布特征
4. **可解释性**：如果 θ* 让 proxy 在 Orca-Math 上 loss 低，说明 θ* 倾向选数学推理数据，这和数学 benchmark 提升直接对应

### 19.7 潜在风险

1. **业界经验可能不适用**：这些数据集在 7B+ 模型上证明有效，但在 1.3B proxy 上可能效果不同
2. **覆盖不均**：10 个 benchmark 有外部数据，8 个只有 train split，3 个无覆盖
3. **数据获取成本**：需要下载和处理多个大型数据集（Orca-Math 200K, MetaMathQA 395K, NuminaMath-CoT 860K）

### 19.8 实施路线图

#### Phase 1: 数据获取（2-3 天）

1. 下载外部训练数据集：
   - `microsoft/orca-math-word-problems-200k` (200K)
   - `meta-math/MetaMathQA` (395K)
   - `AI-MO/NuminaMath-CoT` (860K)
   - `facebook/natural_reasoning` (1.15M)
   - `Open-Orca/OpenOrca` (2.94M，筛选 CoT subset)
   - `allenai/qasper` (5K)

2. 按 cluster 分类和采样

#### Phase 2: 验证集生成（4-6 小时）

1. 修改 `prepare_cap_v1.py` → `prepare_cap_v3.py`
2. 加载外部训练数据 + benchmark train split
3. 按 70:30 混合
4. Tokenize 并保存

#### Phase 3: 完整实验（1-2 天）

1. 用 CAP v3 验证集跑 3000 proxy 实验
2. 用 per-cluster 搜索 θ*
3. 生成数据配比，跑中训练
4. 评估 21 个 benchmark，对比：
   - QuaDMix-OH
   - QuaDMix-BMK
   - QuaDMix-CAP-v1
   - QuaDMix-CAP-v2（如果实现）
   - **QuaDMix-CAP-v3**
   - Quality-top-k (DCLM)
   - Quality-top-k (FineWeb-Edu)

### 19.9 预期效果

| 变体 | 验证集信号 | 依赖假设 | 预期效果 |
|------|-----------|---------|---------|
| OH | 对话续写 | C2 | ≈ quality-top-k |
| BMK | benchmark 答题 | C2 + 无泄露 | ≈ quality-top-k |
| CAP v1 | benchmark train | C2 | ≈ quality-top-k |
| CAP v2 | 推理密集文本 | C2 + 可感知 | > quality-top-k（如果 C2 成立） |
| **CAP v3** | **已证明有效的训练数据** | **数据分布匹配** | **> quality-top-k（绕过 C2）** |

**关键假设**：如果 θ* 让 proxy 在已证明有效的训练数据上 loss 低，说明 θ* 采样出的数据分布接近这些已证明有效的数据，中训练模型应该也能提升 benchmark。

---

**文档结束**
