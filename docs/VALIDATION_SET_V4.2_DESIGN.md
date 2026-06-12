# CORE-BMK v4.2 验证集设计

> 2026-06-12 | 基于 per-task loss 新范式，对全部 21 个 CORE 基准的独立重新设计

## 1. Per-Task 范式如何改变验证集设计约束

### 1.1 v4 的核心约束：聚合稀释

v4 使用单一 aggregate `val_loss` 作为 LightGBM 的 target。每个任务对 aggregate 的贡献与其文档数成正比：

```
val_loss = (1/N_total) * Σ_i loss_i
```

**后果**：
- 高区分度任务的信号被低区分度任务稀释
- 小 N 任务（copa=100, winograd=273）天然被降权
- 必须排除所有"弱信号"任务，否则 aggregate 被污染
- 所有任务等权平均，无法区分"强信号"和"弱信号"

### 1.2 v4.2 的范式转变：独立评估 + 自适应加权

v4.2 为每个任务训练独立的 LightGBM 模型，搜索时使用区分度自适应加权：

```
predicted_loss = Σ_task w_task * loss_task
w_task = std_task / Σ std_all_tasks
```

**根本性变化**：

| 维度 | v4 (aggregate) | v4.2 (per-task) |
|------|----------------|-----------------|
| 任务间关系 | 相互竞争权重 | 独立评估，互不干扰 |
| 弱任务影响 | 稀释 aggregate 信号 | 自动获得低权重，不影响其他任务 |
| 小 N 任务 | 天然被降权 | 权重由区分度决定，非文档数 |
| 任务选择标准 | 必须全部高区分度 | **无需预先排除**，区分度由数据驱动 |
| 域覆盖 | 受限于高区分度任务 | 完整对齐下游评估 |

### 1.3 v4.2 的核心设计原则：不预先排除

**v3/v4 的排除逻辑**："1M 无法解题 → loss 接近随机 → 噪声 → 排除"

**per-task 范式下，这个逻辑不再成立**：
- 每个任务独立训练 LightGBM，弱信号任务自动获得低权重
- 零方差任务被自动排除（排除 + 警告），不需要人工预判
- 只要有非零方差，就有信号，就应该参与加权

**因此**：v4.2 纳入 nanochat CORE 评估的全部 21 个基准（去重后），不预先排除任何任务。

| 原则 | 说明 |
|------|------|
| **完整对齐下游** | nanochat CORE 评估包含的基准，验证集都应覆盖 |
| **数据驱动** | 区分度由实验数据的方差决定，不由人工预判 |
| **零方差自动处理** | per-task 机制自动排除零方差任务并打印警告 |

### 1.4 约束总结

**放松的约束**：
- ✅ 不再需要排除低区分度任务（自动降权）
- ✅ 不再需要排除"1M 无法解题"的任务（方差由数据决定）
- ✅ 不再需要排除符号/非自然语言任务（零方差自动排除）
- ✅ 小 N 任务不再被天然降权

**新增的约束**：
- ⚠️ 每个任务需要非零方差（零方差 = 自动排除）
- ⚠️ 更多任务 = 更多 per-task LightGBM 模型 = 更多训练数据需求

**不变的约束**：
- Loss 策略必须最大化每个任务的信号纯度
- 验证集大小需要在信号质量和评估效率间平衡

---

## 2. 全部 21 个 CORE 基准的独立分析

### 2.1 数据来源

core.yaml 定义了 22 个 ICL 任务，其中 hellaswag_zeroshot 和 hellaswag 共享同一 dataset_uri（仅 few-shot 数不同）。去重后得到 **21 个独立任务**。

### 2.2 完整评估表

对每个任务独立评估，不沿用 v3/v4 的排除结论。

| # | 任务 | 类别 | N | 数据结构 | Avg Ctx | Avg Ans | v4 状态 | v4.2 判断 |
|---|------|------|--:|---------|--------:|--------:|:-------:|:---------:|
| 1 | hellaswag_zeroshot | 语言理解 | 10,042 | MC | ~226c | ~84c | ✅ 保留 | ✅ 保留 |
| 2 | lambada_openai | 语言理解 | ~5K | LM | ~200c | ~6c | ✅ 保留 | ✅ 保留 |
| 3 | winogrande | 语言理解 | 1,267 | schema | ~60c | ~20c | ✅ 保留 | ✅ 保留 |
| 4 | winograd | 语言理解 | 273 | schema | ~60c | ~20c | ✅ 保留 | ✅ 保留 |
| 5 | piqa | 常识推理 | 1,838 | MC | ~47c | ~98c | ✅ 保留 | ✅ 保留 |
| 6 | arc_easy | 世界知识 | 2,376 | MC | ~124c | ~23c | ✅ 保留 | ✅ 保留 |
| 7 | arc_challenge | 世界知识 | 1,172 | MC | ~142c | ~30c | ✅ 保留 | ✅ 保留 |
| 8 | commonsense_qa | 常识推理 | 1,221 | MC | ~150c | ~10c | ✅ 保留 | ✅ 保留 |
| 9 | openbook_qa | 常识推理 | 500 | MC | ~55c | ~19c | ✅ 保留 | ✅ 保留 |
| 10 | copa | 常识推理 | 100 | MC | ~40c | ~30c | ✅ 保留 | ✅ 保留 |
| 11 | **jeopardy** | 世界知识 | 2,117 | LM | ~102c | ~11c | ❌ 排除 | ✅ **新纳入** |
| 12 | **bigbench_qa_wikidata** | 世界知识 | 20,321 | LM | ~38c | ~7c | ❌ 排除 | ✅ **新纳入** |
| 13 | **boolq** | 阅读理解 | 3,270 | MC | ~619c | ~3c | ❌ 排除 | ✅ **新纳入** |
| 14 | **squad** | 阅读理解 | 10,570 | LM | ~868c | ~19c | ❌ 排除 | ✅ **新纳入** |
| 15 | **coqa** | 阅读理解 | 7,983 | LM | ~2208c | ~14c | ❌ 排除 | ✅ **新纳入** |
| 16 | **agi_eval_lsat_ar** | 符号推理 | 230 | MC | ~847c | ~39c | ❌ 排除 | ✅ **新纳入** |
| 17 | **bigbench_dyck_languages** | 符号推理 | 1,000 | LM | ~201c | ~3c | ❌ 排除 | ✅ **新纳入** |
| 18 | **bigbench_cs_algorithms** | 符号推理 | 1,320 | LM | ~178c | ~1c | ❌ 排除 | ✅ **新纳入** |
| 19 | **bigbench_operators** | 符号推理 | 210 | LM | ~148c | ~2c | ❌ 排除 | ✅ **新纳入** |
| 20 | **bigbench_repeat_copy_logic** | 符号推理 | 32 | LM | ~100c | ~83c | ❌ 排除 | ✅ **新纳入** |
| 21 | **bigbench_language_identification** | 语言理解 | 10,000 | MC | ~302c | ~11c | ❌ 排除 | ✅ **新纳入** |

### 2.3 新纳入任务的数据特征

#### 世界知识（新纳入 2 个）

| 任务 | 数据特征 | 1M 可学性 |
|------|---------|----------|
| **jeopardy** | 自然语言线索 + 实体答案，覆盖历史/科学/体育/文学等广泛领域。答案 avg 11c，85.7% 在 5-20c 范围 | 线索文本词汇/主题分布高度可学；答案是有意义的实体名 |
| **bigbench_qa_wikidata** | 简短事实查询 + 实体答案。查询 avg 38c，答案 avg 7c | 查询文本覆盖人物/地点/概念等领域分布；答案虽短但是有意义的实体名 |

#### 阅读理解（新纳入 3 个）

| 任务 | 数据特征 | 1M 可学性 |a
|------|---------|----------|
| **boolq** | 新闻/维基段落 + yes/no 答案。段落 avg 619c，答案仅 2-3c | 段落文本分布可学（新闻/百科/科技等）；答案极短但 yes/no 分布本身可能对方差有贡献 |
| **squad** | 维基百科段落 + 抽取式答案。段落 avg 868c，答案 avg 19c | 维基文本分布高度可学；答案是有意义的文本片段 |
| **coqa** | 故事/对话 + 短答案。答案 avg 14c，49% 在 5-20c | 故事/对话文本分布可学；答案是有意义的短文本 |

#### 符号推理（新纳入 5 个）

| 任务 | 数据特征 | 1M 可学性 |
|------|---------|----------|
| **agi_eval_lsat_ar** | 逻辑推理段落 + MC 答案。答案 avg 39c，76.5% ≥ 20c | 法律/逻辑文本分布可学；答案较长，提供丰富信号 |
| **bigbench_dyck_languages** | 括号序列补全。指令 + 符号输入 → 符号输出 | 指令文本可学；符号部分可能零方差 → per-task 自动处理 |
| **bigbench_cs_algorithms** | 算法问题 + 数字答案。指令 + 随机字符串 → 单个数字 | 指令文本可学；答案极短（1c），可能低方差 |
| **bigbench_operators** | 自定义运算 + 数字答案。指令 + 表达式 → 数字 | 指令文本可学；答案极短（2c） |
| **bigbench_repeat_copy_logic** | 重复指令 + 重复文本。N=32，极小 | 指令+输出都是自然语言；N 极小但 per-task 不依赖 N 加权 |

#### 语言理解（新纳入 1 个）

| 任务 | 数据特征 | 1M 可学性 |
|------|---------|----------|
| **bigbench_language_identification** | 多语言句子 + 语言名称选择。句子 avg 302c | 多语言文本分布可学；语言名称是有意义的词 |

### 2.4 v3/v4 排除理由的重新审视

| v3/v4 排除理由 | per-task 范式下是否仍成立 | 说明 |
|---------------|:------------------------:|------|
| "1M 无法解题" | ❌ 不成立 | 解题能力 ≠ 文本分布可学性；即使 loss 高，方差仍可能非零 |
| "答案太短" | ❌ 不成立 | 短答案 → 低 loss tokens → 可能低方差 → per-task 自动低权重 |
| "符号任务" | ❌ 不成立 | 符号任务仍含自然语言指令；若完全零方差 → per-task 自动排除 |
| "阅读理解超出 1M" | ❌ 不成立 | 段落文本分布可学（新闻/维基/故事），答案有意义 |
| "知识回忆" | ❌ 不成立 | 知识问答的文本覆盖广泛领域（历史/科学/体育），分布可学 |

**结论**：v3/v4 的所有排除理由在 per-task 范式下都不再成立。任务选择应完全由数据驱动。

---

## 3. 每个任务的 Loss 策略设计

### 3.1 1M 代理模型的能力边界与提升路径

1M 代理模型有两个关键约束，决定了 loss 策略的设计：

**约束 1：1M 模型是分布学习器，不是推理器**

1M 模型未经过 SFT，无法像大模型一样理解问题并做出正确回答。它的能力是捕捉训练数据的**文本分布特征**。

**约束 2：预训练数据池中没有 QA 格式数据**

1M 模型的训练数据是自然文本（网页、书籍、代码等），不包含 "Question: ... Answer: ..." 这种 SFT 格式。

由此产生两条可能的下游提升路径：

| 路径 | 链条 | 对 1M 的适用性 |
|------|------|:-------------:|
| **路径 1：分布匹配** | 训练数据与基准所需文本分布相似 → 1M 学到该分布 → 下游基准提升 | ✅ **主导路径** |
| **路径 2：QA 格式学习** | 训练数据包含 QA 对 → 1M 学到 QA 关系 → QA 基准提升 | ❌ 数据池中几乎没有 QA 对 |

**结论**：路径 1（分布匹配）是主导路径。验证集的 loss 应该测量**训练数据与基准所需文本分布的相似度**。

### 3.2 Loss 策略分类原则

基于路径 1，每个任务的 loss 策略由以下判断决定：

> **Context 是否是预训练分布中的自然文本？**

| 判断 | Loss 策略 | 理由 |
|------|----------|------|
| ✅ Context 是自然文本 | **Full-sequence** | Context 的 loss 直接反映训练数据与基准所需分布的相似度。段落/句子本身就是信号 |
| ❌ Context 不是自然文本 | **Answer-only** | Context 是 Q+A 拼接/模板/符号，不在预训练分布中。Context loss 是噪声或恒定 offset |

**为什么自然文本 context 应该用 full-seq？**

对于阅读理解任务（如 boolq），context 是 Wikipedia 段落。不同实验的训练数据包含不同比例的 Wikipedia 类文本。Full-seq loss 直接测量训练数据与 Wikipedia 段落分布的匹配度 — 这正是路径 1 所要求的信号。

**为什么 Q+A 拼接 context 应该用 answer-only？**

对于纯 QA 任务（如 arc_easy），context 是 "Question: ... Answer: ..." 拼接。这种格式不在预训练分布中，1M 模型对其 loss 不反映任何有意义的分布匹配。只有答案文本（如 "photosynthesis"）是自然语言，其 loss 反映训练数据中科学词汇的分布。

**数学补充**（answer-only 对 Q+A 任务的方差优势）：

```
Full-seq:  L_full = (Σ_q loss_q + Σ_a loss_a) / (N_q + N_a)
Answer-only: L_ans = Σ_a loss_a / N_a

Var(L_full) = Var(Σ_a loss_a) / (N_q + N_a)²     [Q tokens 恒定，方差为 0]
            = N_a · σ²_a / (N_q + N_a)²

Var(L_ans)  = N_a · σ²_a / N_a² = σ²_a / N_a

Ratio: Var(L_full) / Var(L_ans) = N_a / (N_q + N_a) < 1
```

当 context 不是自然文本时，context loss 是恒定 offset（方差为 0），full-seq 稀释了答案的方差信号。Answer-only 保留全部信号方差 → 高方差 → 高权重。

### 3.3 逐任务 Loss 策略

#### A. Full-sequence loss — 12 个任务

Context + continuation 拼接后构成自然连贯文本，存在于预训练分布中。Full-seq loss 覆盖所有 tokens。

| # | 任务 | Context 示例 | Continuation 示例 | Full-seq 理由 |
|---|------|-------------|-------------------|--------------|
| 1 | hellaswag_zeroshot | "Roof shingle removal: A man is sitting on a roof. He starts..." | "...pulling up roofing on a roof." | 叙事文本，标准 LM 分布 |
| 2 | lambada_openai | "Ives hopped in deftly, after giving the boat a strong push. The pilot started..." | "...the outboard and" | 文学段落，预训练分布 |
| 3 | winogrande | "Sarah was a much better surgeon than Maria so..." | "...Maria always got the easier cases." | 自然句子 |
| 4 | winograd | "The city councilmen refused the demonstrators a permit because..." | "...the city councilmen feared violence." | 自然句子 |
| 5 | copa | "The man turned on the faucet, therefore..." | "...water flowed from the spout." | 自然因果句 |
| 6 | **jeopardy** | "WORLD HISTORY: This Navy commander flew from a base at Little America..." | "...Richard Byrd" | 知识线索 + 答案 = 自然语言陈述 |
| 7 | **boolq** | Wikipedia/新闻段落+问题 (avg 619c+~40c) | "yes" / "no" | 段落提供分布信号，question 使 answer 有意义 |
| 8 | **squad** | Wikipedia 段落 (avg 868c) | "Denver Broncos" | 段落是自然文本，分布信号强 |
| 9 | **coqa** | 故事/文章+问题 (avg 2208c+~30c) | "yes" / 短答案 | 故事提供分布信号，question 使 answer 有意义 |
| 10 | **bigbench_language_identification** | 指令+多语言句子+选项 (avg 302c+~50c) | "Spanish" | 句子提供分布信号，选项使 answer 有意义 |
| 11 | **bigbench_qa_wikidata** | "The native language of Daniel Schneidermann is" (38c) | "French" (7c) | **context + continuation = 自然陈述句** |
| 12 | **openbook_qa** | "...the best way to save money is to" (avg 55c) | "quit eating lunch out" (avg 19c) | **query stem + answer = 自然句子** |

**boolq/coqa/bigbench_language_identification 使用 full-seq 的关键理由**：

这些任务的 context 包含 Wikipedia 段落、故事文本、多语言句子 — 属于预训练数据的自然分布。Full-seq loss 覆盖所有 tokens：
- **段落/句子**（主体，数百 tokens）：提供分布匹配信号
- **Question/选项**（少量 tokens）：使 answer 有意义，不截断
- **Answer**（少量 tokens）：在 question 上下文中有意义

**bigbench_qa_wikidata / openbook_qa 使用 full-seq 的关键理由**：

- bigbench_qa_wikidata: context 是句子 stem，continuation 是实体名，拼接后是完整陈述句（如 "The native language of Daniel Schneidermann is French"），属于预训练分布
- openbook_qa: query 是句子 stem（如 "...the best way to save money is to"），answer 是自然语言补全，拼接后是完整自然句子

#### B. Answer-only mask — 9 个任务

Context 不是预训练分布中的自然文本（Q+A 拼接 / SFT 格式化文本 / 符号 / 人工构造）。只有答案文本提供分布信号。

| # | 任务 | Context 类型 | Answer 类型 | Answer-only 理由 |
|---|------|-------------|------------|-----------------|
| 13 | piqa | 物理场景问题 | 长答案 (~98c) | Q+A 拼接，非自然文本 |
| 14 | arc_easy | 科学问题 | 短答案 (~23c) | Q+A 拼接，非自然文本 |
| 15 | arc_challenge | 科学问题(难) | 短答案 (~30c) | Q+A 拼接，非自然文本 |
| 16 | commonsense_qa | SFT 格式化 MC（含嵌入选项） | **字母标签** ("A"/"B"/...) | query 含 "Question:...Choices:...Answer:" 格式，答案是字母 |
| 17 | **agi_eval_lsat_ar** | SFT 格式化 MC（含嵌入选项） | **字母标签** ("A"/"B"/...) | query 含 "Passage:...Q:...Choices:...Answer:" 格式，答案是字母 |
| 18 | **bigbench_dyck_languages** | 指令+括号序列 (201c) | 闭合括号 (~3c) | 符号输入，非自然文本 |
| 19 | **bigbench_cs_algorithms** | 指令+随机字符串 (178c) | 数字 (~1c) | 人工构造输入 |
| 20 | **bigbench_operators** | 指令+表达式 (148c) | 数字 (~2c) | 人工构造运算 |
| 21 | **bigbench_repeat_copy_logic** | 指令+模式 (100c) | 重复文本 (~83c) | 模板化指令 |

---

## 4. v4.2 验证集完整设计

### 4.1 任务列表

**21 个任务 = 10 个 v4 保留 + 11 个新纳入**

#### A. Full-sequence loss — 12 个（context+continuation 构成自然文本）

| # | 任务 | N (cap-2000) | 来源 | Context 分布 |
|---|------|:------------:|------|-------------|
| 1 | hellaswag_zeroshot | 2000 | v4 保留 | 叙事/动作描述 |
| 2 | lambada_openai | 2000 | v4 保留 | 文学文本 |
| 3 | winogrande | 1267 | v4 保留 | 代词消解句子 |
| 4 | winograd | 273 | v4 保留 | 代词消解句子 |
| 5 | copa | 100 | v4 保留 | 因果推理句子 |
| 6 | **jeopardy** | **2000** | **新纳入** | 知识线索文本 |
| 7 | **boolq** | **2000** | **新纳入** | Wikipedia/新闻段落+问题 |
| 8 | **squad** | **2000** | **新纳入** | Wikipedia 段落 |
| 9 | **coqa** | **2000** | **新纳入** | 故事/文章+问题 |
| 10 | **bigbench_language_identification** | **2000** | **新纳入** | 指令+多语言句子+选项 |
| 11 | **bigbench_qa_wikidata** | **2000** | **新纳入** | 句子 stem + 实体名 |
| 12 | **openbook_qa** | **500** | **v4 保留** | 句子 stem + 自然语言补全 |

#### B. Answer-only mask — 9 个（context 非自然文本）

| # | 任务 | N (cap-2000) | 来源 | Context 类型 |
|---|------|:------------:|------|-------------|
| 13 | piqa | 1838 | v4 保留 | Q+A 拼接 |
| 14 | arc_easy | 2000 | v4 保留 | Q+A 拼接 |
| 15 | arc_challenge | 1172 | v4 保留 | Q+A 拼接 |
| 16 | commonsense_qa | 1221 | v4 保留 | SFT 格式化 MC（答案=字母标签） |
| 17 | **agi_eval_lsat_ar** | **230** | **新纳入** | SFT 格式化 MC（答案=字母标签） |
| 18 | **bigbench_dyck_languages** | **1000** | **新纳入** | 符号序列 |
| 19 | **bigbench_cs_algorithms** | **1320** | **新纳入** | 随机字符串 |
| 20 | **bigbench_operators** | **210** | **新纳入** | 自定义运算 |
| 21 | **bigbench_repeat_copy_logic** | **32** | **新纳入** | 模板化指令 |

### 4.2 样本量分配

**策略：Uniform cap-2000，与 v4 一致**

| 任务 | 可用 | 采样 | 占比 |
|------|-----:|-----:|-----:|
| hellaswag_zeroshot | 10,042 | 2000 | 7.2% |
| lambada_openai | ~5,000 | 2000 | 7.2% |
| bigbench_qa_wikidata | 20,321 | 2000 | 7.2% |
| bigbench_language_identification | 10,000 | 2000 | 7.2% |
| squad | 10,570 | 2000 | 7.2% |
| coqa | 7,983 | 2000 | 7.2% |
| boolq | 3,270 | 2000 | 7.2% |
| arc_easy | 2,376 | 2000 | 7.2% |
| jeopardy | 2,117 | 2000 | 7.2% |
| piqa | 1,838 | 1838 | 6.6% |
| bigbench_cs_algorithms | 1,320 | 1320 | 4.7% |
| winogrande | 1,267 | 1267 | 4.5% |
| commonsense_qa | 1,221 | 1221 | 4.4% |
| arc_challenge | 1,172 | 1172 | 4.2% |
| bigbench_dyck_languages | 1,000 | 1000 | 3.6% |
| openbook_qa | 500 | 500 | 1.8% |
| winograd | 273 | 273 | 1.0% |
| agi_eval_lsat_ar | 230 | 230 | 0.8% |
| bigbench_operators | 210 | 210 | 0.8% |
| copa | 100 | 100 | 0.4% |
| bigbench_repeat_copy_logic | 32 | 32 | 0.1% |
| **总计** | | **~27,844** | **100%** |

### 4.3 为什么不调整样本量

在 per-task 范式下，样本量 N 的影响与 v4 不同：

| 维度 | v4 (aggregate) | v4.2 (per-task) |
|------|----------------|-----------------|
| N 对权重的影响 | N 大的任务权重高 | **无影响**（权重由区分度决定） |
| N 对 loss 估计的影响 | 影响 aggregate 稳定性 | 影响 per-task loss 估计精度 |
| N 对 LightGBM 的影响 | 所有实验共享一个模型 | 每个任务独立训练 |

**结论**：N 只影响 per-task loss 估计的精度（SE = σ/√N），不影响任务权重。

---

## 5. 数据处理通用规则

### 5.1 处理决策流程

对每个基准，按以下流程确定处理方案：

```
Step 1: 判断 context 是否包含预训练分布中的自然文本
  ├── 是 → Loss 策略 = full-seq
  │     └── Step 2a: 仅清理 SFT 模板前缀（如 "Passage:", "Context:", 指令前缀）
  │         保留 question、选项等全部内容（full-seq 覆盖所有 tokens，格式不影响）
  └── 否 → Loss 策略 = answer-only
        └── Step 2b: 保留原始 context（不计算 loss）
                     仅对 answer/continuation 计算 loss

Step 3: 确定答案提取方式
  ├── LM 格式（context + continuation）→ 直接使用 continuation
  ├── MC 格式（query + choices + gold）→ choices[gold] 或 parse_choice_from_query
  └── Schema 格式（context_options + gold）→ context_options[gold]
```

**关键原则**：对于 full-seq 任务，所有 tokens 都参与 loss 计算。Context 中是否包含 Q&A 格式、选项列表等不影响信号 — 段落文本提供分布信号，question 使 answer 有意义，全部 tokens 的 loss 共同构成 per-task 信号。因此只需清理 SFT 模板前缀，保留全部内容。

### 5.2 SFT Artifact 清理规则

eval bundle 中的数据经过 SFT 格式化处理，包含 SFT 模板前缀。对于 **full-seq 任务**，仅需清理这些模板前缀，**保留 question、选项等全部内容**：

| Artifact 类型 | 示例 | 清理方式 | 适用任务 |
|--------------|------|---------|---------|
| "Passage: " 前缀 | "Passage: All biomass goes through..." | 去除前缀 | boolq |
| "Context: " 前缀 | "Context: Super Bowl 50 was..." | 去除前缀 | squad |
| 指令前缀 | "Below is a story followed by..." | 去除前缀 | coqa |
| "Sentence: " 前缀 | "Given a sentence, select...\nSentence: ..." | 仅去除 "Sentence: "（保留指令和选项） | bigbench_language_identification |

**不清理的内容**（full-seq 下保留）：
- Question 部分（如 "\nQuestion: is biomass renewable?"）— 使 answer 有意义
- MC 选项部分（如 "\nA. Quechua\nB. Aymara"）— 全部 tokens 参与 loss
- 其他非模板前缀的内容

**对于 answer-only 任务**：context 不计算 loss，因此 SFT artifacts 不影响信号纯度，无需清理。

### 5.3 答案提取规则

| 数据格式 | 字段结构 | 答案提取方式 | 适用任务 |
|---------|---------|------------|---------|
| **LM** | context + continuation | 直接使用 continuation | lambada, jeopardy, bigbench_qa_wikidata, squad, coqa, 符号任务×4 |
| **MC-外部选项** | query + choices(完整文本) + gold | choices[gold] → 自然语言答案 | piqa, arc_easy/challenge, openbook_qa, boolq, copa |
| **MC-嵌入选项(文本)** | query(含选项文本) + choices(字母) + gold | parse_choice_from_query → 选项文本 | bigbench_language_identification |
| **MC-嵌入选项(字母)** | query(含选项文本) + choices(字母) + gold | choices[gold] → 字母标签 | commonsense_qa, agi_eval_lsat_ar |
| **Schema** | context_options + gold | context_options[gold] + continuation | winogrande, winograd |

---

## 6. 全部 21 个基准的处理总表

### 6.1 处理总表

| # | 任务 | Loss 策略 | Context 自然文本？ | Context 分布/类型 | SFT 清理 | 答案提取 | 理由 |
|---|------|:---------:|:-----------------:|-----------------|---------|---------|------|
| 1 | hellaswag_zeroshot | full-seq | ✅ | 叙事/动作描述 | 无 | LM: continuation | 叙事续写 = 标准 LM 分布 |
| 2 | lambada_openai | full-seq | ✅ | 文学段落 | 无 | LM: continuation | 文学段落 + 末词，完整文本 |
| 3 | winogrande | full-seq | ✅ | 代词消解句子 | 无 | Schema: context_options[gold] | 自然句子 |
| 4 | winograd | full-seq | ✅ | 代词消解句子 | 无 | Schema: context_options[gold] | 自然句子 |
| 5 | copa | full-seq | ✅ | 因果推理句子 | 无 | MC-外部: choices[gold] | 自然因果句 |
| 6 | jeopardy | full-seq | ✅ | 知识线索文本 | 无 | LM: continuation | 线索 + 答案 = 自然语言陈述 |
| 7 | boolq | full-seq | ✅ | Wikipedia/新闻段落+问题 | 去 "Passage:" 前缀，保留 question | MC-外部: choices[gold] | 段落提供分布信号，question 使 answer 有意义 |
| 8 | squad | full-seq | ✅ | Wikipedia 段落 | 去 "Context:" 前缀 | LM: continuation | 段落是自然文本，分布信号强 |
| 9 | coqa | full-seq | ✅ | 故事/文章+问题 | 去指令前缀，保留 question | LM: continuation | 故事提供分布信号，question 使 answer 有意义 |
| 10 | bigbench_language_identification | full-seq | ✅ | 指令+多语言句子+选项 | 去 "Sentence:" 前缀，保留指令和选项 | MC-嵌入(文本): parse_choice_from_query | 指令使逻辑完整，句子提供分布信号 |
| 11 | bigbench_qa_wikidata | full-seq | ✅ | 句子 stem + 实体名 | 无 | LM: continuation | stem + 答案 = 自然陈述句 |
| 12 | openbook_qa | full-seq | ✅ | 句子 stem + 自然语言补全 | 无 | MC-外部: choices[gold] | stem + 答案 = 自然句子 |
| 13 | piqa | ans-only | ❌ | Q+A 拼接 | 不需要 | MC-外部: choices[gold] | Q+A 拼接非自然文本 |
| 14 | arc_easy | ans-only | ❌ | Q+A 拼接 | 不需要 | MC-外部: choices[gold] | Q+A 拼接非自然文本 |
| 15 | arc_challenge | ans-only | ❌ | Q+A 拼接 | 不需要 | MC-外部: choices[gold] | Q+A 拼接非自然文本 |
| 16 | commonsense_qa | ans-only | ❌ | SFT 格式化 MC（含嵌入选项） | 不需要 | MC-嵌入: choices[gold] → 字母标签 | query 含选项，答案是字母 |
| 17 | agi_eval_lsat_ar | ans-only | ❌ | SFT 格式化 MC（含嵌入选项） | 不需要 | MC-嵌入: choices[gold] → 字母标签 | query 含选项，答案是字母 |
| 18 | bigbench_dyck_languages | ans-only | ❌ | 指令+括号序列 | 不需要 | LM: continuation | 符号输入，非自然文本 |
| 19 | bigbench_cs_algorithms | ans-only | ❌ | 指令+随机字符串 | 不需要 | LM: continuation | 人工构造输入 |
| 20 | bigbench_operators | ans-only | ❌ | 指令+表达式 | 不需要 | LM: continuation | 人工构造运算 |
| 21 | bigbench_repeat_copy_logic | ans-only | ❌ | 指令+模式 | 不需要 | LM: continuation | 模板化指令 |

### 6.2 逐任务处理详情

#### Full-seq 任务（10 个）

**1. hellaswag_zeroshot** — 叙事续写
```
原始: context="Roof shingle removal: A man is sitting on a roof. He starts..."
      continuation="pulling up roofing on a roof."
清理: 无（已是纯自然文本）
答案: 直接使用 continuation
Loss: full-sequence（context + continuation 全部计算）
```

**2. lambada_openai** — 文学末词预测
```
原始: context="Ives hopped in deftly, after giving the boat a strong push. The pilot started..."
      continuation="the outboard and"
清理: 无（已是纯自然文本）
答案: 直接使用 continuation
Loss: full-sequence
```

**3. winogrande** — 代词消解
```
原始: context_options=["Sarah was a much better surgeon than Maria so Sarah always got the harder cases.",
                       "Sarah was a much better surgeon than Maria so Maria always got the easier cases."]
      gold=1
清理: 无（已是纯自然句子）
答案: context_options[gold]
Loss: full-sequence
```

**4. winograd** — 代词消解
```
原始: context_options=["The city councilmen refused the demonstrators a permit because the city councilmen feared violence.",
                       "The city councilmen refused the demonstrators a permit because the demonstrators feared violence."]
      gold=0
清理: 无（已是纯自然句子）
答案: context_options[gold]
Loss: full-sequence
```

**5. copa** — 因果推理
```
原始: query="The man turned on the faucet, therefore"
      choices=["the toilet filled with water.", "water flowed from the spout."]  gold=1
清理: 无（已是纯自然句子 stem）
答案: choices[gold] → "water flowed from the spout."
Loss: full-sequence（stem + 答案全部计算）
```

**6. jeopardy** — 知识问答线索
```
原始: context="WORLD HISTORY: This Navy commander flew from a base at Little America to the South Pole & back Nov. 28-29, 1929"
      continuation="Richard Byrd"
清理: 无（线索文本已是自然语言）
答案: 直接使用 continuation
Loss: full-sequence
```

**7. boolq** — Wikipedia 段落阅读理解
```
原始: query="Passage: All biomass goes through at least some of these steps: it needs to be collected,
             processed, and then converted into a usable form of energy. Question: is biomass renewable?"
      choices=["yes", "no"]  gold=0
清理: 仅去除 "Passage: " 前缀，保留 question 部分
      → "All biomass goes through... Question: is biomass renewable?"
答案: choices[gold] → "yes"
Loss: full-sequence（段落 + question + 答案全部计算）
理由: 段落提供 Wikipedia 分布信号，question 使 answer "yes" 有意义
```

**8. squad** — Wikipedia 抽取式问答
```
原始: context="Context: Super Bowl 50 was an American football game to determine the champion of the
               National Football League (NFL) for the 2015 season."
      continuation="Denver Broncos"
清理: 去除 "Context: " 前缀 → 保留纯 Wikipedia 段落
答案: 直接使用 continuation
Loss: full-sequence（段落 + 答案全部计算）
理由: 段落是 Wikipedia 自然文本，属于预训练分布
```

**9. coqa** — 故事/对话阅读理解
```
原始: context="Below is a story followed by a series of related questions. [story text...] Question: Did he win?"
      continuation="yes"
清理: 仅去除指令前缀 "Below is a story followed by a series of related questions."
      保留 question 部分
      → "[story text...] Question: Did he win?"
答案: 直接使用 continuation
Loss: full-sequence（故事 + question + 答案全部计算）
理由: 故事提供分布信号，question 使 answer "yes" 有意义
```

**10. bigbench_language_identification** — 多语言识别
```
原始: query="Given a sentence, select the correct language among the choices\nSentence: Chaymanta Apolos
             yatiqirinakatakiwa.\nA. Quechua\nB. Aymara\nC. Guarani\nD. Spanish"
      choices=["A", "B", "C", "D"]  gold=3
清理: 仅去除 "Sentence: " 前缀，保留指令和选项
      → "Given a sentence, select the correct language among the choices\n
         Chaymanta Apolos yatiqirinakatakiwa.\nA. Quechua\nB. Aymara\nC. Guarani\nD. Spanish"
答案: parse_choice_from_query(query, "D") → "Spanish"
Loss: full-sequence（指令 + 句子 + 选项 + 答案全部计算）
理由: 指令使任务逻辑完整，句子提供分布信号，选项使 answer 有意义
注意: 需验证 parse_choice_from_query 对此格式的兼容性
```

**11. bigbench_qa_wikidata** — 事实查询（full-seq，自然陈述句）
```
原始: context="The native language of Daniel Schneidermann is"
      continuation="French"
清理: 无（已是自然语言 stem）
答案: 直接使用 continuation
Loss: full-sequence（stem + 答案全部计算）
理由: "The native language of Daniel Schneidermann is French" = 自然陈述句
```

**12. openbook_qa** — 教科书知识（full-seq，自然句子）
```
原始: query="A person wants to start saving money so that they can afford a nice vacation
             at the end of the year. After looking over their budget and expenses, they
             decide the best way to save money is to"
      choices=["make more phone calls", "quit eating lunch out",
               "buy less with monopoly money", "have lunch with friends"]  gold=1
清理: 无（query 已是自然语言 stem）
答案: choices[gold] → "quit eating lunch out"
Loss: full-sequence（stem + 答案全部计算）
理由: "...the best way to save money is to quit eating lunch out" = 自然句子
```

#### Answer-only 任务（9 个）

**13. piqa** — 物理直觉
```
原始: query="Which of the following is a way to prevent rust?"
      choices=["Paint the metal surface.", "Wash the metal surface with water."]  gold=0
清理: 不需要（context 不计算 loss）
答案: choices[gold] → "Paint the metal surface."
Loss: answer-only（仅答案 tokens 计算 loss）
```

**14-15. arc_easy / arc_challenge** — 科学问答
```
原始: query="What is the process by which plants convert light energy into chemical energy?"
      choices=["respiration", "photosynthesis", "fermentation", "digestion"]  gold=1
清理: 不需要
答案: choices[gold] → "photosynthesis"
Loss: answer-only
```

**16. commonsense_qa** — 常识推理（MC-嵌入选项）
```
原始: query="Question: A revolving door is convenient for two direction travel, but it also
             serves as a security measure at a what?\nChoices:\nA. bank\nB. department store
             \nC. mall\nD. new york\nAnswer:"
      choices=["A", "B", "C", "D"]  gold=0
清理: 不需要（context 不计算 loss）
答案: choices[gold] → "A"（字母标签，非选项文本 "bank"）
Loss: answer-only（仅字母标签 tokens 计算 loss）
理由: query 已嵌入选项文本，答案是字母标签，逻辑完整：问题 → 选项 → "A"
```

**17. agi_eval_lsat_ar** — 逻辑推理（MC-嵌入选项）
```
原始: query="Passage: Of the eight students—George, Helen, ... in a seminar...\nQ: Which one
             of the following could be the schedule?\nChoices:\nA.) Mon. morning: Irving;...
             \nB.) Mon. morning: Lenore;...\nC.) ...\nD.) ...\nAnswer:"
      choices=["A", "B", "C", "D"]  gold=1
清理: 不需要（context 不计算 loss）
答案: choices[gold] → "B"（字母标签，非选项文本）
Loss: answer-only（仅字母标签 tokens 计算 loss）
理由: query 已嵌入选项文本，答案是字母标签，逻辑完整：段落 → 问题 → 选项 → "B"
```

**18. bigbench_dyck_languages** — 括号序列补全
```
原始: context="Complete the rest of the sequence... Input: [ < < { } > ..."
      continuation="} ]"
清理: 不需要
答案: 直接使用 continuation
Loss: answer-only
预期: 可能零方差（符号任务），per-task 自动处理
```

**19. bigbench_cs_algorithms** — 算法问题
```
原始: context="Given two strings, determine the length... Strings: ZUIEJOBQXVLX..."
      continuation="8"
清理: 不需要
答案: 直接使用 continuation
Loss: answer-only
预期: 可能零方差（答案仅 1c），per-task 自动处理
```

**20. bigbench_operators** — 自定义运算
```
原始: context="Given the definition of the op operator... op (op 17) ="
      continuation="17"
清理: 不需要
答案: 直接使用 continuation
Loss: answer-only
预期: 可能零方差（答案仅 2c），per-task 自动处理
```

**21. bigbench_repeat_copy_logic** — 重复逻辑
```
原始: context="repeat with logic: Q: A watermelon has seven seeds. Repeat the sentence."
      continuation="A watermelon has seven seeds. A watermelon has seven seeds. ..."
清理: 不需要
答案: 直接使用 continuation
Loss: answer-only
预期: 可能零方差（N=32），per-task 自动处理
```

---

## 7. v4 vs v4.2 对比

| 维度 | v4 | v4.2 |
|------|-----|------|
| 任务数 | 10 | **21** |
| Full-seq（自然文本） | 5 | **12 (+jeopardy, boolq, squad, coqa, lang_id, qa_wikidata, openbook_qa)** |
| Answer-only（非自然文本） | 5 | **9 (+4)** |
| 总文档数 | 12,371 | **~27,844** |
| 域覆盖 | 9 个文本域 | **全部 CORE 域** |
| Loss 策略依据 | 续写=full-seq, QA=ans-mask | **context+continuation 是否构成自然文本** |
| Cap per task | 2000 | 2000（不变） |
| 预计文件大小 | 217 MB | **~490 MB** |
| 预计评估时间 | 1.0x | **~2.25x** |
| 下游对齐 | 部分 CORE 基准 | **完整 CORE 基准** |

### 7.1 域覆盖对比

| 文本域 | v4 | v4.2 | 新增任务 |
|--------|:--:|:----:|---------|
| 叙事续写 | ✅ | ✅ | — |
| 文学文本 | ✅ | ✅ | — |
| 代词消解 | ✅ | ✅ | — |
| 因果推理 | ✅ | ✅ | — |
| 物理直觉 | ✅ | ✅ | — |
| 科学(易/难/教科书) | ✅ | ✅ | — |
| 常识概念 | ✅ | ✅ | — |
| **知识问答** | ❌ | ✅ | **jeopardy** |
| **事实查询** | ❌ | ✅ | **bigbench_qa_wikidata** |
| **阅读理解** | ❌ | ✅ | **boolq, squad, coqa** |
| **逻辑推理** | ❌ | ✅ | **agi_eval_lsat_ar** |
| **符号推理** | ❌ | ✅ | **dyck, cs_algo, operators, repeat_copy** |
| **语言识别** | ❌ | ✅ | **bigbench_language_identification** |

---

## 8. 预期 Per-Task 行为

### 8.1 预期权重分布

基于理论分析（无实验数据），预估各任务的区分度：

| 任务 | Loss 策略 | 预期区分度 | 理由 |
|------|:---------:|:----------:|------|
| hellaswag_zeroshot | full-seq | 高 | 叙事文本对训练数据高度敏感 |
| lambada_openai | full-seq | 中高 | 文学风格敏感，长文本 |
| arc_easy | ans-only | 中高 | 简单科学文本可学 |
| **boolq** | **full-seq** | **中高** | **Wikipedia 段落分布信号强** |
| **squad** | **full-seq** | **中高** | **Wikipedia 段落分布信号强** |
| piqa | ans-only | 中 | 物理场景词汇分布 |
| arc_challenge | ans-only | 中 | 科学文本分布可学 |
| jeopardy | full-seq | 中 | 跨领域知识线索文本 |
| winogrande | full-seq | 中 | 代词模式有一定可学性 |
| **coqa** | **full-seq** | **中** | **故事文本分布可学** |
| **bigbench_qa_wikidata** | **full-seq** | **中** | **陈述句分布可学（full-seq 捕获完整句子方差）** |
| **openbook_qa** | **full-seq** | **中** | **句子 stem+答案分布可学** |
| commonsense_qa | ans-only | 低 | 答案仅字母标签（1c） |
| **bigbench_lang_id** | **full-seq** | **中** | **多语言句子分布可学** |
| agi_eval_lsat_ar | ans-only | 低 | N=230 + 答案仅字母标签 |
| winograd | full-seq | 低 | N=273 |
| bigbench_dyck_languages | ans-only | 低~零 | 符号任务，可能零方差 |
| bigbench_cs_algorithms | ans-only | 低~零 | 符号任务，答案 1c |
| bigbench_operators | ans-only | 低~零 | 符号任务，答案 2c |
| copa | full-seq | 低 | N=100 |
| bigbench_repeat_copy_logic | ans-only | 低~零 | N=32 |

### 8.2 可能的零方差任务

以下任务可能在实验中表现为零方差（被 per-task 机制自动排除）：

- bigbench_dyck_languages（括号序列，loss 可能恒定）
- bigbench_cs_algorithms（答案仅 1 个字符）
- bigbench_operators（答案仅 2 个字符）
- bigbench_repeat_copy_logic（N=32，统计不稳定）

**这正是 per-task 范式的价值**：不需要人工预判，数据会告诉我们哪些任务有信号。

---

## 9. 实现计划

### 9.1 准备脚本

基于 `prepare_core_bmk_v4.py` 创建 `prepare_core_bmk_v4.2.py`：

1. **从 core.yaml 加载全部 21 个任务**（去重 hellaswag）
2. **Full-seq 任务列表**（12 个）：hellaswag_zeroshot, lambada_openai, winogrande, winograd, copa, jeopardy, boolq, squad, coqa, bigbench_language_identification, bigbench_qa_wikidata, openbook_qa
3. **Answer-only 任务列表**（9 个）：piqa, arc_easy, arc_challenge, commonsense_qa, agi_eval_lsat_ar, bigbench_dyck_languages, bigbench_cs_algorithms, bigbench_operators, bigbench_repeat_copy_logic
4. **SFT artifact 清理**（full-seq 任务仅清理 SFT 模板前缀，保留 question/选项等全部内容）：
   - boolq: 去除 "Passage: " 前缀（保留 question）
   - squad: 去除 "Context: " 前缀
   - coqa: 去除指令前缀（保留 question）
   - bigbench_language_identification: 去除 "Sentence: " 前缀（保留指令和选项）
5. **答案解析**：
   - MC-外部选项（boolq, openbook_qa, copa, piqa, arc_easy/challenge）：choices[gold] → 自然语言答案
   - MC-嵌入选项(文本)（bigbench_language_identification）：parse_choice_from_query → 选项文本
   - MC-嵌入选项(字母)（commonsense_qa, agi_eval_lsat_ar）：choices[gold] → 字母标签
   - LM 任务（jeopardy, bigbench_qa_wikidata, squad, coqa, 符号任务）：直接使用 continuation
   - Schema 任务（winogrande, winograd）：context_options[gold] + continuation

### 9.2 输出文件

- `core_bmk_21tasks_v4.2_tokenized.pt` (~490 MB)
- `core_bmk_21tasks_v4.2.parquet` (~6 MB)

### 9.3 HuggingFace 上传

- Repo: `liujin99/quadmix-core-bmk-v4.2`
- 文件: `.pt`, `.parquet`, `README.md`

---

## 10. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 符号任务零方差 | 自动排除，per-task 模型数减少 | 预期行为，打印警告 |
| 评估时间增加 ~2.25x | 实验效率下降 | 可接受：完整域覆盖 + per-task 收益远大于时间成本 |
| bigbench_repeat_copy_logic N=32 | Loss 估计不稳定 | Per-task 自动处理（低方差 → 低权重或零方差 → 排除） |
| boolq full-seq 下答案信号被段落稀释 | 答案仅 2-3c，方差主要来自段落 | **预期行为**：段落分布是主要信号（路径 1），question 保留使 answer 有意义 |
| coqa context 过长 (avg 2208c) | Tokenize 后序列很长 | 可接受：full-seq 需要完整段落 |
| bigbench_language_identification 答案解析 | parse_choice_from_query 可能不兼容 | 需要测试并添加 fallback |
| 文件大小 ~490 MB | 下载/存储成本增加 | 可接受 |

---

## 11. 总结

v4.2 验证集的核心设计决策：

1. **纳入全部 21 个 CORE 基准** — 不预先排除任何任务，完整对齐 nanochat 下游评估
2. **数据驱动的任务权重** — per-task 机制自动处理区分度：高方差 → 高权重，低方差 → 低权重，零方差 → 自动排除
3. **基于分布匹配的 Loss 策略** — context+continuation 构成自然文本 → full-seq (12 个)，非自然文本 → answer-only (9 个)
4. **样本量 cap-2000** — per-task 范式下 N 不影响权重，只影响 loss 估计精度
5. **21 个任务，~27,844 文档** — 完整覆盖 CORE 评估的所有文本域

### 11.1 关键设计理念

**路径 1（分布匹配）是主导路径**：1M 代理模型是分布学习器，验证集 loss 应测量训练数据与基准所需文本分布的相似度。

**Loss 策略判断标准**：context + continuation 拼接后是否构成预训练分布中的自然文本？
- ✅ 是 → full-seq：所有 tokens 参与 loss，段落/句子提供分布信号，question/选项使 answer 有意义
- ❌ 否 → answer-only：context 是 Q+A 拼接/SFT 格式化文本/符号，只有答案提供分布信号

**答案逻辑一致性**：对于 MC-嵌入选项任务（commonsense_qa, agi_eval_lsat_ar），query 已包含选项文本，答案是字母标签（"A"/"B"），而非提取的选项内容。
