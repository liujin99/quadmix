# CORE-BMK v5 验证集设计

> 2026-06-17 | 基于 v4.3 实验结果的验证集数据补充设计

## 1. 背景：v4.3 的问题

### 1.1 v4.3 实验结果

674 experiments, 90 维参数空间, 5-fold CV, v4.3 验证集：

| 指标 | 数值 |
|------|------|
| Overall R² | 0.4586 |
| Overall MAE | 0.2756 (z-score 空间) |
| Aggregate loss mean | 6.7068 |
| Aggregate loss std | 0.1636 |

### 1.2 过拟合加剧

| 现象 | 证据 |
|------|------|
| Train R² 暴涨 | 几乎所有 task → 0.9+ |
| CV R² 几乎不变 | 部分 task 甚至下降 |
| Gap 显著增大 | jeopardy 0.27→0.52, operators 0.22→0.53, lsat_ar 0.30→0.69 |

### 1.3 低 R² task 分析

| Task | CV R² | 样本数 | 失败模式 |
|------|-------|--------|---------|
| repeat_copy_logic | -0.05 | 32 | 符号任务，无信号 |
| piqa | 0.05 | 1,838 | 参数-loss 映射弱 |
| copa | 0.10 | 100 | 样本太少 |
| operators | 0.12 | 211 | 无 split 划分 |
| commonsense_qa | 0.13 | 1,221 | 参数-loss 映射弱 |
| lsat_ar | 0.15 | 230 | 纯测试集 |
| winograd | 0.18 | 273 | 纯测试集 |
| jeopardy | 0.20 | 2,000 | 纯测试集 |

### 1.4 v5 改进方向

**核心假设**：增加验证集样本量可以提升低 R² task 的信号质量，尤其是样本数过少的 task（copa=100, winograd=273）。

**策略**：合并 train+test/validation splits，在不增加 cap 的前提下扩充可用样本池。

---

## 2. 数据补充可行性分析

### 2.1 可合并 train+test/validation 的 11 个 task

| # | Task | v4.3 样本数 | 可用 train | 可用 test | 可用 val | 合并后总量 | v5 cap | 补充潜力 |
|---|------|------------|-----------|----------|---------|-----------|--------|---------|
| 1 | hellaswag_zeroshot | 2,000 | 39,905 | 10,003 | 10,042 | 60,000+ | 2,000 | ✓ 充足 |
| 2 | winogrande | 1,267 | 40,398 | 1,767 | 1,267 | 43,000+ | 2,000 | ✓ 充足 |
| 3 | copa | 100 | 400 | 500 | 100 | 1,000 | 900 | ✓ 充足 |
| 4 | boolq | 2,000 | 9,427 | - | 3,270 | 12,000+ | 2,000 | ✓ 充足 |
| 5 | squad | 2,000 | 87,599 | - | 10,570 | 98,000+ | 2,000 | ✓ 充足 |
| 6 | coqa | 2,000 | 108,647 | - | 7,983 | 116,000+ | 2,000 | ✓ 充足 |
| 7 | openbook_qa | 500 | 4,957 | 500 | 500 | 5,957 | 2,000 | ✓ 充足 |
| 8 | arc_easy | 2,000 | 2,251 | 2,376 | 570 | 5,197 | 2,000 | ✓ 充足 |
| 9 | arc_challenge | 1,172 | 1,119 | 1,172 | 299 | 2,590 | 2,000 | ✓ 充足 |
| 10 | commonsense_qa | 1,221 | 9,741 | 1,140 | 1,221 | 12,102 | 2,000 | ✓ 充足 |
| 11 | piqa | 1,838 | 16,113 | - | 1,838 | 17,951 | 2,000 | ✓ 充足 |

**小计**: 11 个 task 可补充，v5 cap 2,000 → 实际可用 21,900 samples

### 2.2 有额外数据但暂不需要的 1 个 task

| # | Task | v4.3 样本数 | 额外数据 | 说明 |
|---|------|------------|---------|------|
| 12 | lambada_openai | 2,000 | dev=4,869 | Zenodo 原始数据，cap 2000 下暂不需要 |

**备注**：lambada_openai 的 development set (4,869 samples) 可从 Zenodo 下载，但当前 cap=2000 已足够，暂不纳入 v5。

### 2.3 无法补充的 9 个 task

| # | Task | v4.3 样本数 | 原因 | 备注 |
|---|------|------------|------|------|
| 13 | winograd | 273 | 纯测试集 (273) | XML 有 285 schemas，但 eval bundle 用 273 子集 |
| 14 | jeopardy | 2,000 | 纯测试集 (2,117) | j-archive 无批量下载，Kaggle 需登录 |
| 15 | agi_eval_lsat_ar | 230 | 纯测试集 (230) | AGIEval 官方只有 test split |
| 16 | bigbench_language_identification | 2,000 | 无 split 划分 (10,000) | 单一数据集，无 train/test 区分 |
| 17 | bigbench_qa_wikidata | 2,000 | 无 split 划分 (20,442) | 单一数据集，无 train/test 区分 |
| 18 | bigbench_dyck_languages | 1,000 | 无 split 划分 (1,000) | 单一数据集，无 train/test 区分 |
| 19 | bigbench_cs_algorithms | 1,320 | 无数据 (MosaicML 生成) | GitHub task.json 无 examples |
| 20 | bigbench_operators | 211 | 无 split 划分 (211) | 单一数据集，无 train/test 区分 |
| 21 | bigbench_repeat_copy_logic | 32 | 无 split 划分 (32) | 单一数据集，无 train/test 区分 |

**小计**: 9 个 task 无法补充，保持现状

---

## 3. 低 R² task 补充分析

### 3.1 低 R² task 可补充性

| Task | v4.3 CV R² | v4.3 样本数 | v5 可补充 | 可用数据 | 预期影响 |
|------|-----------|------------|----------|---------|---------|
| repeat_copy_logic | -0.05 | 32 | ✗ 无法补充 | 无 split 划分 | 无变化 |
| piqa | 0.05 | 1,838 | ✓ 可补充 | 16,113 train | 可能提升 |
| copa | 0.10 | 100 | ✓ 可补充 | 400 train + 500 test | 可能提升 |
| operators | 0.12 | 211 | ✗ 无法补充 | 无 split 划分 | 无变化 |
| commonsense_qa | 0.13 | 1,221 | ✓ 可补充 | 9,741 train + 1,140 test | 可能提升 |
| lsat_ar | 0.15 | 230 | ✗ 无法补充 | 纯测试集 | 无变化 |
| winograd | 0.18 | 273 | ✗ 无法补充 | 纯测试集 | 无变化 |
| jeopardy | 0.20 | 2,000 | ✗ 无法补充 | 纯测试集 | 无变化 |

### 3.2 关键发现

- **可补充**: 3/8 低 R² task (piqa, copa, commonsense_qa)
- **无法补充**: 5/8 低 R² task (repeat_copy_logic, operators, lsat_ar, winograd, jeopardy)
- **受限原因**: 纯测试集 (3 个) 或无 split 划分 (2 个)

### 3.3 补充预期

| Task | 当前样本数 | v5 可用样本数 | 增幅 | 预期 R² 变化 |
|------|-----------|--------------|------|-------------|
| copa | 100 | 900 | +800% | 显著提升（样本量 9×） |
| piqa | 1,838 | 2,000 | +8% | 小幅提升（样本池扩大 8×，但 cap 限制） |
| commonsense_qa | 1,221 | 2,000 | +64% | 中等提升（样本池扩大 10×，但 cap 限制） |

**备注**：piqa 和 commonsense_qa 虽然可用数据池大幅增加，但受 cap=2000 限制，实际增量有限。如果 cap 提升到 5000，效果可能更显著。

---

## 4. v5 vs v4.3 对比

### 4.1 汇总统计

| 指标 | v4.3 (当前) | v5 (可补充) | 增量 |
|------|------------|------------|------|
| 总样本数 | 27,163 | 21,900 + 9,166 = 31,066 | +3,903 (+14%) |
| 可补充 task 数 | 0 | 11 | +11 |
| 无法补充 task 数 | 21 | 9 | -12 |
| 平均每个 task | 1,293 | 1,479 | +186 |

### 4.2 低 R² task 覆盖

| 指标 | v4.3 | v5 | 变化 |
|------|------|-----|------|
| 低 R² task 总数 | 8 | 8 | 不变 |
| 可补充的低 R² task | 0 | 3 | +3 |
| 无法补充的低 R² task | 8 | 5 | -3 |

### 4.3 数据源分类

#### 可合并 train+test/validation 的 11 个 task

| Task | 数据来源 | 格式 | 合并策略 |
|------|---------|------|---------|
| hellaswag_zeroshot | HuggingFace (Rowan/hellaswag) | parquet | train + val |
| winogrande | HuggingFace (allenai/winogrande) | parquet | train + test |
| copa | HuggingFace (aps/super_glue, copa config) | parquet | train + test |
| boolq | HuggingFace (google/boolq) | parquet | train + val |
| squad | HuggingFace (rajpurkar/squad) | parquet | train + val |
| coqa | HuggingFace (stanfordnlp/coqa) | parquet | train + val |
| openbook_qa | HuggingFace (allenai/openbookqa) | parquet | train + test + val |
| arc_easy | HuggingFace (allenai/ai2_arc, ARC-Easy) | parquet | train + test + val |
| arc_challenge | HuggingFace (allenai/ai2_arc, ARC-Challenge) | parquet | train + test + val |
| commonsense_qa | HuggingFace (tau/commonsense_qa) | parquet | train + test |
| piqa | 原始 JSONL (yonatanbisk.com/piqa) | JSONL | train + val |

#### 保持现状的 10 个 task

| Task | 原因 | 样本数 |
|------|------|--------|
| lambada_openai | dev=4,869 可用但 cap 2000 下暂不需要 | 2,000 |
| winograd | 纯测试集 (273) | 273 |
| jeopardy | 纯测试集，无批量下载 (2,117) | 2,000 |
| agi_eval_lsat_ar | 纯测试集 (230) | 230 |
| bigbench_language_identification | 无 train/test 划分 (10,000) | 2,000 |
| bigbench_qa_wikidata | 无 train/test 划分 (20,442) | 2,000 |
| bigbench_dyck_languages | 无 train/test 划分 (1,000) | 1,000 |
| bigbench_cs_algorithms | GitHub 无数据，MosaicML 生成 (1,320) | 1,320 |
| bigbench_operators | 无 train/test 划分 (211) | 211 |
| bigbench_repeat_copy_logic | 无 train/test 划分 (32) | 32 |

---

## 5. 数据格式转换分析

### 5.1 v4.3 数据流管线

v4.3 的数据处理管线为：

```
eval_bundle JSONL → extract_pairs() → (context, continuation, is_full_seq) → tokenize → (token_ids, loss_mask)
```

eval_bundle JSONL 有三种 schema：

| Schema | 字段 | 使用 task |
|--------|------|----------|
| `multiple_choice` | `{query, choices[], gold}` | hellaswag, copa, boolq, openbook_qa, arc_easy, arc_challenge, commonsense_qa, piqa |
| `language_modeling` | `{context, continuation}` | squad, coqa |
| `schema` | `{context_options[], continuation, gold}` | winogrande |

`extract_pairs()` 根据 task_type 和 task_label 将 JSONL 转换为 `(context, continuation)` 对：

- **multiple_choice**: context = query, continuation = choices[gold]，部分 task 有 SFT prefix 清理和 `\nAnswer: ` 分隔符
- **language_modeling**: context = context（清理 SFT prefix 后），continuation = continuation
- **schema**: context = context_options[gold], continuation = continuation

### 5.2 逐 task 格式转换分析

#### 5.2.1 hellaswag_zeroshot (multiple_choice, full-seq)

| 维度 | eval_bundle | HuggingFace |
|------|-------------|-------------|
| 来源 | eval_bundle JSONL | `Rowan/hellaswag` |
| 字段 | `query`, `choices[]`, `gold` | `ctx`, `endings[]`, `label` (str) |
| 可用 splits | validation (10,042) | train (39,905) + test (10,003) + validation (10,042) |

**转换逻辑**：
```python
item = {
    "query": hf_sample["ctx"],
    "choices": hf_sample["endings"],
    "gold": int(hf_sample["label"])
}
```

**注意事项**：
- `label` 是 string 类型，需要 `int()` 转换
- `ctx` 已包含 activity_label 前缀，与 eval_bundle 的 `query` 格式一致
- **无需额外格式化**，直接字段映射

#### 5.2.2 winogrande (schema, full-seq)

| 维度 | eval_bundle | HuggingFace |
|------|-------------|-------------|
| 来源 | eval_bundle JSONL | `allenai/winogrande` / `winogrande_xl` |
| 字段 | `context_options[]`, `continuation`, `gold` | `sentence`, `option1`, `option2`, `answer` (str) |
| 可用 splits | validation (1,267) | train (40,398) + test (1,767) + validation (1,267) |

**转换逻辑**：
```python
sentence = hf_sample["sentence"]
# sentence 格式: "Sarah was a much better surgeon than Maria so _ always got the easier cases."
# 需要在 "_" 处分割为 context_options 和 continuation
parts = sentence.split("_", 1)
prefix = parts[0].rstrip()
continuation = parts[1].lstrip() if len(parts) > 1 else ""
item = {
    "context_options": [
        prefix + " " + hf_sample["option1"],
        prefix + " " + hf_sample["option2"]
    ],
    "continuation": continuation,
    "gold": int(hf_sample["answer"]) - 1  # "1"/"2" → 0/1
}
```

**注意事项**：
- `answer` 是 `"1"` 或 `"2"`，需要 `-1` 转为 0-indexed
- 分割点在 `_` 字符处，前缀 + option 构成 context_options，后缀构成 continuation
- **需要验证**：分割后的 context_options 与 eval_bundle 的格式是否完全一致

#### 5.2.3 copa (multiple_choice, full-seq)

| 维度 | eval_bundle | HuggingFace |
|------|-------------|-------------|
| 来源 | eval_bundle JSONL | `aps/super_glue` / `copa` config |
| 字段 | `query`, `choices[]`, `gold` | `premise`, `choice1`, `choice2`, `question`, `label` |
| 可用 splits | validation (100) | train (400) + test (500) + validation (100) |

**转换逻辑**：
```python
connective = "therefore" if hf_sample["question"] == "effect" else "because"
query = hf_sample["premise"].rstrip(". ") + ", " + connective
item = {
    "query": query,
    "choices": [hf_sample["choice1"], hf_sample["choice2"]],
    "gold": hf_sample["label"]
}
```

**注意事项**：
- eval_bundle 的 `query` 格式：`"The man turned on the faucet, therefore"`
- HF 的 `premise` + `question` 需要重构为相同格式
- `question` 值为 `"cause"` 或 `"effect"`，映射为 `"because"` 或 `"therefore"`
- **需要验证**：连接词和标点格式是否与 eval_bundle 完全一致

#### 5.2.4 boolq (multiple_choice, full-seq)

| 维度 | eval_bundle | HuggingFace |
|------|-------------|-------------|
| 来源 | eval_bundle JSONL | `google/boolq` |
| 字段 | `query`, `choices[]`, `gold` | `passage`, `question`, `answer` (bool) |
| 可用 splits | validation (3,270) | train (9,427) + validation (3,270) |

**转换逻辑**：
```python
item = {
    "query": "Passage: " + hf_sample["passage"] + "\nQuestion: " + hf_sample["question"],
    "choices": ["no", "yes"],
    "gold": 1 if hf_sample["answer"] else 0
}
```

**注意事项**：
- eval_bundle 的 `query` 格式：`"Passage: ...\nQuestion: ..."`
- `clean_sft_prefix()` 会移除 `"Passage: "` 前缀（full-seq 模式下）
- `choices` 固定为 `["no", "yes"]`
- `answer` 是 bool 类型，True→1 (yes)，False→0 (no)

#### 5.2.5 squad (language_modeling, full-seq)

| 维度 | eval_bundle | HuggingFace |
|------|-------------|-------------|
| 来源 | eval_bundle JSONL | `rajpurkar/squad` |
| 字段 | `context`, `continuation` | `context`, `question`, `answers` (dict) |
| 可用 splits | validation (10,570) | train (87,599) + validation (10,570) |

**转换逻辑**：
```python
answer = hf_sample["answers"]["text"][0]
item = {
    "context": "Context: " + hf_sample["context"] + "\nQuestion: " + hf_sample["question"] + "\nAnswer: ",
    "continuation": answer
}
```

**注意事项**：
- eval_bundle 的 `context` 格式：`"Context: ...\nQuestion: ...\nAnswer: "`
- `clean_sft_prefix()` 会移除 `"Context: "` 前缀
- `answers` 是 dict，`text` 字段是 list，取第一个元素
- 一个 question 可能有多个 answer，只取第一个

#### 5.2.6 coqa (language_modeling, full-seq)

| 维度 | eval_bundle | HuggingFace |
|------|-------------|-------------|
| 来源 | eval_bundle JSONL | `stanfordnlp.github.io/coqa` (JSON) 或 HF `coqa` |
| 字段 | `context`, `continuation` | `source`, `story`, `questions[]`, `answers` (dict) |
| 可用 splits | validation (7,983 QA) | train (7,199 stories → 108,647 QA) + dev (500 stories → 7,983 QA) |

**转换逻辑**：
```python
# 每个 story 有多个 QA pairs，需要展开
for i, (q, a) in enumerate(zip(hf_sample["questions"], hf_sample["answers"]["input_text"])):
    # 构建 context: instruction + story + 前面的 QA history + 当前 question
    history = ""
    for j in range(i):
        history += f"\nQuestion: {hf_sample['questions'][j]}\nAnswer: {hf_sample['answers']['input_text'][j]}"
    context = f"Below is a story followed by a series of related questions.\n\nStory: {hf_sample['story']}{history}\n\nFinal question:\nQuestion: {q}\nAnswer: "
    item = {
        "context": context,
        "continuation": a
    }
```

**注意事项**：
- **多轮对话展开**：每个 story 的 N 个 QA pairs 展开为 N 个独立样本
- eval_bundle 的 context 包含 instruction header `"Below is a story..."`
- `clean_sft_prefix()` 会查找 `"Story:"` 并从那里开始
- **样本数**：train 7,199 stories → ~108,647 QA pairs，dev 500 stories → ~7,983 QA pairs
- **计算量大**：需要处理多轮对话的展开逻辑

#### 5.2.7 openbook_qa (multiple_choice, full-seq)

| 维度 | eval_bundle | HuggingFace |
|------|-------------|-------------|
| 来源 | eval_bundle JSONL | `allenai/openbookqa` / `main` |
| 字段 | `query`, `choices[]`, `gold` | `question_stem`, `choices` (dict), `answerKey` (str) |
| 可用 splits | test (500) | train (4,957) + test (500) + validation (500) |

**转换逻辑**：
```python
choices_text = hf_sample["choices"]["text"]
choices_label = hf_sample["choices"]["label"]  # ["A", "B", "C", "D"]
gold_idx = choices_label.index(hf_sample["answerKey"])
item = {
    "query": hf_sample["question_stem"],
    "choices": choices_text,
    "gold": gold_idx
}
```

**注意事项**：
- `answerKey` 是 string ("A"/"B"/"C"/"D")，需要转为 index
- `choices` 是 dict `{text: [...], label: [...]}`，需要提取
- eval_bundle 的 `query` 无 `"Question: "` 前缀，直接使用 `question_stem`

#### 5.2.8 arc_easy / arc_challenge (multiple_choice, full-seq, `\nAnswer: ` separator)

| 维度 | eval_bundle | HuggingFace |
|------|-------------|-------------|
| 来源 | eval_bundle JSONL | `allenai/ai2_arc` / `ARC-Easy` 或 `ARC-Challenge` |
| 字段 | `query`, `choices[]`, `gold` | `question`, `choices` (dict), `answerKey` (str) |
| 可用 splits | test (2,376 / 1,172) | train + test + validation |

**转换逻辑**：
```python
choices_text = hf_sample["choices"]["text"]
choices_label = hf_sample["choices"]["label"]
gold_idx = choices_label.index(hf_sample["answerKey"])
item = {
    "query": "Question: " + hf_sample["question"],
    "choices": choices_text,
    "gold": gold_idx
}
```

**注意事项**：
- eval_bundle 的 `query` 有 `"Question: "` 前缀
- 使用 `"\nAnswer: "` 分隔符（`MC_FULLSEQ_ANSWER_SEP_TASKS`）
- 格式与 openbook_qa 几乎相同，仅多了 `"Question: "` 前缀

#### 5.2.9 commonsense_qa (multiple_choice, answer-only, MC-embedded)

| 维度 | eval_bundle | HuggingFace |
|------|-------------|-------------|
| 来源 | eval_bundle JSONL | `tau/commonsense_qa` |
| 字段 | `query`, `choices[]`, `gold` | `question`, `choices` (dict), `answerKey` (str) |
| 可用 splits | validation (1,221) | train (9,741) + test (1,140) + validation (1,221) |

**转换逻辑**：
```python
choices_text = hf_sample["choices"]["text"]
choices_label = hf_sample["choices"]["label"]  # ["A", "B", "C", "D", "E"]

# 构建 embedded query: 选项嵌入到文本中
query = "Question: " + hf_sample["question"] + "\nChoices:\n"
for label, text in zip(choices_label, choices_text):
    query += f"{label}. {text}\n"
query += "Answer:"

# answer 是 letter label（MC-embedded 模式）
gold_idx = choices_label.index(hf_sample["answerKey"]) if hf_sample["answerKey"] else -1
item = {
    "query": query,
    "choices": choices_label,  # letter labels: ["A", "B", ...]
    "gold": gold_idx
}
```

**注意事项**：
- **MC-embedded 模式**：选项文本嵌入到 query 中，choices 只保留 letter labels
- eval_bundle 的 `query` 格式：`"Question: ...\nChoices:\nA. ...\nB. ...\nAnswer:"`
- `extract_answer()` 返回 `choices[gold]` 即 letter label（如 `"A"`）
- **test split 的 answerKey 为空**：需要跳过无 answer 的样本
- answer-only loss：只对 letter label 的 token 计算 loss

#### 5.2.10 piqa (multiple_choice, full-seq, `\nAnswer: ` separator)

| 维度 | eval_bundle | HuggingFace |
|------|-------------|-------------|
| 来源 | eval_bundle JSONL | `baber/piqa` (或 `ybisk/piqa` 原始 JSONL) |
| 字段 | `query`, `choices[]`, `gold` | `goal`, `sol1`, `sol2`, `label` (int) |
| 可用 splits | validation (1,838) | train (16,113) + validation (1,838) + test (3,084, 无 label) |

**转换逻辑**：
```python
item = {
    "query": "Question: " + hf_sample["goal"] + "\n",
    "choices": [hf_sample["sol1"], hf_sample["sol2"]],
    "gold": hf_sample["label"]
}
```

**注意事项**：
- eval_bundle 的 `query` 格式：`"Question: ...\n"`（注意末尾换行）
- 使用 `"\nAnswer: "` 分隔符
- **test split 的 label 为 -1**：需要跳过无 label 的样本
- `baber/piqa` 可直接通过 HF datasets 加载；`ybisk/piqa` 需要 script，不兼容新版 datasets

### 5.3 转换复杂度汇总

| Task | 转换难度 | 关键挑战 | 风险点 |
|------|---------|---------|--------|
| hellaswag_zeroshot | **低** | 直接字段映射 | 无 |
| winogrande | **中** | `_` 分割 sentence | 分割逻辑需验证 |
| copa | **中** | premise + connective 重构 | 标点/连接词格式需验证 |
| boolq | **低** | 拼接 passage + question | 无 |
| squad | **低** | 拼接 context + question + answer | 无 |
| coqa | **高** | 多轮对话展开 | 展开逻辑复杂，需验证 |
| openbook_qa | **低** | answerKey → index | 无 |
| arc_easy | **低** | 同 openbook_qa + `"Question: "` 前缀 | 无 |
| arc_challenge | **低** | 同 arc_easy | 无 |
| commonsense_qa | **中** | MC-embedded query 构建 | 格式需精确匹配 |
| piqa | **低** | 字段重命名 + `"Question: "` 前缀 | test split 无 label |

### 5.4 需要验证的格式细节

以下 task 的转换逻辑需要与 eval_bundle 原始数据逐样本对比验证：

| Task | 验证方法 | 验证内容 |
|------|---------|---------|
| winogrande | 对比 HF train 样本 vs eval_bundle val 样本 | `_` 分割后 context_options 和 continuation 是否一致 |
| copa | 对比 HF val 样本 vs eval_bundle val 样本 | premise + connective 格式（逗号、空格）是否一致 |
| commonsense_qa | 对比 HF val 样本 vs eval_bundle val 样本 | embedded query 格式（换行、标点、Answer:）是否一致 |
| coqa | 对比 HF dev 样本 vs eval_bundle val 样本 | 多轮对话展开格式（instruction header、history）是否一致 |

**建议**：先对 eval_bundle 和 HF 的 val split 做 diff 验证，确认转换逻辑正确后再合并 train。

### 5.5 统一数据加载架构

v5 脚本采用 **双源加载 + 统一中间格式** 架构：

```
┌─────────────────────────────────────────────────────────┐
│                    v5 数据加载管线                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  11 个可补充 task:                                       │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │ HuggingFace  │───→│ convert_to_  │───→ (context,     │
│  │ datasets     │    │ eval_bundle  │     continuation,  │
│  │ (train+test  │    │ _format()    │     is_full_seq)   │
│  │  +val)       │    └──────────────┘                    │
│  └──────────────┘                                       │
│                                                         │
│  10 个保持现状 task:                                     │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │ eval_bundle  │───→│ extract_     │───→ (context,     │
│  │ JSONL        │    │ pairs()      │     continuation,  │
│  │              │    │ (v4.3 原逻辑) │     is_full_seq)   │
│  └──────────────┘    └──────────────┘                    │
│                                                         │
│  ─────────────────────────────────────→ merge + cap      │
│                                        → tokenize        │
│                                        → output          │
└─────────────────────────────────────────────────────────┘
```

**核心原则**：
1. HF 数据通过 `convert_to_eval_bundle_format()` 转为与 eval_bundle JSONL 相同的中间格式
2. 然后复用 v4.3 的 `extract_pairs()` 逻辑生成 `(context, continuation)` 对
3. 10 个保持现状的 task 继续从 eval_bundle 加载，逻辑不变
4. 最终 merge + cap + tokenize 逻辑不变

---

## 6. v5 实施策略

### 6.1 核心原则

1. **合并 train+test/validation**：利用完整数据集，不局限于 eval bundle 的单一 split
2. **保持 cap=2000**：与 v4.3 一致，控制验证集规模
3. **随机采样**：从合并后的数据池中随机采样，确保多样性
4. **保持 loss 策略不变**：沿用 v4.3 的 full-seq/answer-only 分类

### 6.2 数据加载策略

| Task | 当前 v4.3 来源 | v5 来源 | 变化 |
|------|---------------|---------|------|
| hellaswag_zeroshot | eval_bundle val | HF train + val | 合并 |
| winogrande | eval_bundle val | HF train + test | 合并 |
| copa | eval_bundle val | HF train + test | 合并 |
| boolq | eval_bundle val | HF train + val | 合并 |
| squad | eval_bundle val | HF train + val | 合并 |
| coqa | eval_bundle val | HF train + val | 合并 |
| openbook_qa | eval_bundle test | HF train + test + val | 合并 |
| arc_easy | eval_bundle test | HF train + test + val | 合并 |
| arc_challenge | eval_bundle test | HF train + test + val | 合并 |
| commonsense_qa | eval_bundle val | HF train + test | 合并 |
| piqa | eval_bundle val | 原始 JSONL train + val | 合并 |
| 其余 10 个 task | eval_bundle | eval_bundle | 不变 |

### 6.3 预期效果

| 维度 | v4.3 | v5 预期 |
|------|------|---------|
| Overall R² | 0.4586 | 提升（低 R² task 样本增加） |
| 低 R² task 数量 | 8 | 减少（copa/piqa/commonsense_qa 可能提升） |
| 过拟合程度 | 严重 (gap 0.3-0.7) | 可能缓解（验证集更大，信号更稳定） |

---

## 7. 结论

### 7.1 v5 可行性

- **可补充**: 11 个 task (hellaswag, winogrande, copa, boolq, squad, coqa, openbook_qa, arc_easy, arc_challenge, commonsense_qa, piqa)
- **无法补充**: 9 个 task (lambada_openai 暂不需要, winograd, jeopardy, lsat_ar, 5 个 BIG-bench)
- **低 R² task**: 仅 3/8 可补充，其余 5 个受限于纯测试集或无 split 划分

### 7.2 预期收益

- **样本量增加**: +14% (27,163 → 31,066)
- **低 R² task 覆盖**: 3/8 可补充 (copa, piqa, commonsense_qa)
- **信号质量提升**: 样本量增加可能提升低 R² task 的预测准确性

### 7.3 局限性

- **cap=2000 限制**: piqa/commonsense_qa 虽然数据池扩大 8-10×，但实际增量有限
- **5/8 低 R² task 无法补充**: repeat_copy_logic, operators, lsat_ar, winograd, jeopardy 受限于数据源
- **过拟合问题**: 增加验证集样本可能缓解，但 90D 空间的样本密度问题仍存在

### 7.4 下一步

1. 创建 `prepare_core_bmk_v5.py` 脚本
2. 生成 v5 验证集 parquet 文件
3. 上传到 HuggingFace，更新 `constants.py` 中的 `VAL_FILE`
4. 重新运行实验，评估 Overall R² 是否提升
5. 对比 v4.3 vs v5 结果，分析 train+test 合并对低 R² task 的影响

---

**文档版本**: v1.0  
**最后更新**: 2026-06-17  
**基于**: v4.3 实验结果 (674 experiments, Overall R²=0.4586)
