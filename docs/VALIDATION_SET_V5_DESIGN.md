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
| operators | 0.12 | 210 | 无 split 划分 |
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
| 3 | copa | 100 | 400 | 500 (label=-1) | 100 | 500 可用 | 2,000 | ✓ 充足 |
| 4 | boolq | 2,000 | 9,427 | - | 3,270 | 12,000+ | 2,000 | ✓ 充足 |
| 5 | squad | 2,000 | 87,599 | - | 10,570 | 98,000+ | 2,000 | ✓ 充足 |
| 6 | coqa | 2,000 | 108,647 | - | 7,983 | 116,000+ | 2,000 | ✓ 充足 |
| 7 | openbook_qa | 500 | 4,957 | 500 | 500 | 5,957 | 2,000 | ✓ 充足 |
| 8 | arc_easy | 2,000 | 2,251 | 2,376 | 570 | 5,197 | 2,000 | ✓ 充足 |
| 9 | arc_challenge | 1,172 | 1,119 | 1,172 | 299 | 2,590 | 2,000 | ✓ 充足 |
| 10 | commonsense_qa | 1,221 | 9,741 | 1,140 | 1,221 | 12,102 | 2,000 | ✓ 充足 |
| 11 | piqa | 1,838 | 16,113 | - | 1,838 | 17,951 | 2,000 | ✓ 充足 |

**小计**: 11 个 task 可补充，v5 cap 2,000 → 实际可用 21,500 samples（copa test 500 条因 label=-1 被过滤，仅 500 可用）

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
| 20 | bigbench_operators | 210 | 无 split 划分 (210) | 单一数据集，无 train/test 区分 |
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
| operators | 0.12 | 210 | ✗ 无法补充 | 无 split 划分 | 无变化 |
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

**实际结果修正**（详见 §9.2）：
- **copa 实际为 500**（非 900）：test split 500 条因 label=-1 被过滤，仅 train(400) + val(100) = 500 可用
- **winogrande 实际为 2,000**（非 ~1,947）：数据池 41,665 远超 cap，过滤 53 条后仍够 2,000
- **bigbench_operators 实际为 210**（非 211）：过滤 1 条异常样本

---

## 4. v5 vs v4.3 对比

### 4.1 汇总统计

| 指标 | v4.3 (当前) | v5 (可补充) | 增量 |
|------|------------|------------|------|
| 总样本数 | 27,163 | 31,547 | +4,384 (+16.1%) |
| 可补充 task 数 | 0 | 11 | +11 |
| 无法补充 task 数 | 21 | 9 | -12 |
| 平均每个 task | 1,293 | 1,502 | +209 |

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
| coqa | HuggingFace (`coqa`) | parquet | train + val |
| openbook_qa | HuggingFace (allenai/openbookqa) | parquet | train + test + val |
| arc_easy | HuggingFace (allenai/ai2_arc, ARC-Easy) | parquet | train + test + val |
| arc_challenge | HuggingFace (allenai/ai2_arc, ARC-Challenge) | parquet | train + test + val |
| commonsense_qa | HuggingFace (tau/commonsense_qa) | parquet | train + test |
| piqa | HuggingFace (`baber/piqa`) | parquet | train + val |

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
| bigbench_operators | 无 train/test 划分 (210) | 210 |
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
- **句号-only 过滤**：`extract_pairs()` 的 `schema` handler 会过滤 continuation 仅为 `.`/`,`/`!`/`?` 的样本（winogrande 约 53 条，winograd 约 18 条），这些样本的 `_` 占位符在句末，loss 信号几乎为零
- **需要验证**：分割后的 context_options 与 eval_bundle 的格式是否完全一致

#### 5.2.3 copa (multiple_choice, full-seq)

| 维度 | eval_bundle | HuggingFace |
|------|-------------|-------------|
| 来源 | eval_bundle JSONL | `aps/super_glue` / `copa` config |
| 字段 | `query`, `choices[]`, `gold` | `premise`, `choice1`, `choice2`, `question`, `label` |
| 可用 splits | validation (100) | train (400) + test (500) + validation (100) |

**转换逻辑**：
```python
gold = hf_sample["label"]
if gold < 0:
    return None  # test split 无真实标签，直接过滤
premise = hf_sample["premise"]
if premise.endswith("."):
    premise = premise[:-1]
connective = "therefore" if hf_sample["question"] == "effect" else "because"
c1 = hf_sample["choice1"]
c2 = hf_sample["choice2"]
item = {
    "query": f"{premise}, {connective}",
    "choices": [
        c1[0].lower() + c1[1:] if c1 else c1,
        c2[0].lower() + c2[1:] if c2 else c2,
    ],
    "gold": gold,
}
```

**注意事项**：
- eval_bundle 的 `query` 格式：`"The man turned on the faucet, therefore"`
- HF 的 `premise` + `question` 需要重构为相同格式
- `question` 值为 `"cause"` 或 `"effect"`，映射为 `"because"` 或 `"therefore"`
- **test split (500 样本) label=-1**：直接过滤（`return None`），不使用默认 choice1，避免错误答案污染数据
- choices 首字母小写以匹配 eval_bundle 格式
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

#### 5.2.7 openbook_qa (multiple_choice, full-seq, per-sample 格式)

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

**Per-sample 格式逻辑**（`extract_pairs()` 中）：

openbook_qa 的题目有两种风格，v5 按样本动态选择格式以最大化自然文本匹配度：

| 风格 | 判断条件 | 格式 | 数量 | 示例 |
|------|---------|------|------|------|
| Q&A | `endswith("?")` 或首词为疑问词 | `Question: ...?\nAnswer: ...` | 198 (40%) | `Question: Which requires energy to move?\nAnswer: weasel` |
| 续写 | 其他 | `... answer` | 302 (60%) | `Predators eat bunnies` |

**疑问词集合**：`{how, what, why, which, where, when, who, can, do, does, is, are, should}`

**注意事项**：
- `answerKey` 是 string ("A"/"B"/"C"/"D")，需要转为 index
- `choices` 是 dict `{text: [...], label: [...]}`，需要提取
- eval_bundle 的 `query` 无 `"Question: "` 前缀，直接使用 `question_stem`
- Q&A 模式下自动添加 `"Question: "` 前缀和 `"\nAnswer: "` 分隔符
- 续写模式下保持原始文本，空格拼接 answer，更匹配预训练文本分布

#### 5.2.8 arc_easy / arc_challenge (multiple_choice, full-seq, per-sample 格式)

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

**Per-sample 格式逻辑**（`extract_pairs()` 中）：

arc_easy / arc_challenge 的题目有两种风格，v5 按样本动态选择格式：

| 风格 | 判断条件 | 格式 | arc_easy | arc_challenge |
|------|---------|------|----------|---------------|
| Q&A | `endswith("?")` 或首词为疑问词 | `Question: ...?\nAnswer: ...` | 1,933 (81%) | 996 (84%) |
| 续写 | 其他 | `... answer`（去掉 `Question:` 前缀） | 443 (18%) | 176 (15%) |

**疑问词集合**：`{how, what, why, which, where, when, who, can, do, does, is, are, should}`

**注意事项**：
- eval_bundle 的 `query` 有 `"Question: "` 前缀
- Q&A 模式保留 `"Question: "` 前缀和 `"\nAnswer: "` 分隔符
- 续写模式去掉 `"Question: "` 前缀，空格拼接 answer，更匹配预训练文本分布
- **选项未嵌入 query**：所有 ARC 题目的选项仅在 `choices[]` 字段中，query 不包含 A/B/C/D 选项文本。answer 为全文本（如 `"Sunlight is the source of energy..."`），非字母标签

#### 5.2.9 commonsense_qa (multiple_choice, full-seq, MC-embedded, `\nAnswer: ` separator)

| 维度 | eval_bundle | HuggingFace |
|------|-------------|-------------|
| 来源 | eval_bundle JSONL | `tau/commonsense_qa` |
| 字段 | `query`, `choices[]`, `gold` | `question`, `choices` (dict), `answerKey` (str) |
| 可用 splits | validation (1,221) | train (9,741) + test (1,140) + validation (1,221) |

**转换逻辑**：
```python
choices_text = hf_sample["choices"]["text"]
choices_label = hf_sample["choices"]["label"]  # ["A", "B", "C", "D", "E"]

# 构建 embedded query: 选项嵌入到文本中（不含 "Answer:"，由 separator 添加）
query = "Question: " + hf_sample["question"] + "\nChoices:\n"
for label, text in zip(choices_label, choices_text):
    query += f"{label}. {text}\n"

# answer 是 letter label（MC-embedded 模式）
gold_idx = choices_label.index(hf_sample["answerKey"]) if hf_sample["answerKey"] else -1
item = {
    "query": query,
    "choices": choices_label,  # letter labels: ["A", "B", ...]
    "gold": gold_idx
}
```

**注意事项**：
- **MC-embedded 模式**：选项文本嵌入到 query 中，choices 只保留 letter labels（`["A", "B", "C", "D", "E"]`）
- eval_bundle 的 `query` 格式：`"Question: ...\nChoices:\nA. ...\nB. ...\nAnswer:"`（v5 移除末尾 `Answer:`）
- `extract_answer()` 统一返回 `choices[gold]`，对 MC-embedded task 即 letter label（如 `"A"`）
- **test split 的 answerKey 为空**：需要跳过无 answer 的样本
- **full-seq loss**：对 query + answer 全部 token 计算 loss（v5 从 answer-only 改为 full-seq，提升 R²）
- 使用 `"\nAnswer: "` 分隔符（`MC_FULLSEQ_ANSWER_SEP_TASKS`）

#### 5.2.10 piqa (multiple_choice, full-seq, per-sample 格式)

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

**Per-sample 格式逻辑**（`extract_pairs()` 中）：

piqa 的题目有两种风格，v5 按样本动态选择格式以最大化自然文本匹配度：

| 风格 | 判断条件 | 格式 | 数量 | 示例 |
|------|---------|------|------|------|
| Q&A | `endswith("?")` 或首词为疑问词 | `Question: ...?\nAnswer: ...` | 867 (47%) | `Question: How do you attach toilet paper to a glass jar?\nAnswer: Press a piece of double-sided tape...` |
| 续写 | 其他 | `... answer`（去掉 `Question:` 前缀） | 971 (53%) | `To fight Ivan Drago in Rocky for sega master system. You have to defeat Apollo Creed and Clubber Lang first.` |

**疑问词集合**：`{how, what, why, which, where, when, who, can, do, does, is, are, should}`

**注意事项**：
- eval_bundle 的 `query` 格式：`"Question: ...\n"`（注意末尾换行）
- Q&A 模式保留 `"Question: "` 前缀和 `"\nAnswer: "` 分隔符
- 续写模式去掉 `"Question: "` 前缀，空格拼接 answer，更匹配预训练文本分布
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
| openbook_qa | **中** | answerKey → index + per-sample 格式判断 | 问答/续写分类需验证 |
| arc_easy | **中** | 同 openbook_qa + per-sample 格式判断 | 问答/续写分类需验证 |
| arc_challenge | **中** | 同 arc_easy | 问答/续写分类需验证 |
| commonsense_qa | **中** | MC-embedded query 构建 | 格式需精确匹配 |
| piqa | **中** | 字段重命名 + per-sample 格式判断 | 问答/续写分类需验证，test split 无 label |

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

### 5.6 continuation_delimiter 机制

v5 从 `core.yaml` 读取每个 task 的 `continuation_delimiter` 字段，作为 `extract_pairs()` 的参数传入。这解决了 v4.3 中硬编码分隔符导致 Q&A 格式 task 丢失 `\nAnswer: ` 的问题。

**delimiter 优先级**（`multiple_choice` handler，full-seq 模式）：

| 优先级 | 条件 | 使用的分隔符 | 示例 task |
|--------|------|-------------|----------|
| 1 | task 在 `MC_PER_SAMPLE_TASKS` | **per-sample**：Q&A 用 `\nAnswer: `，续写用空格 | openbook_qa, piqa, arc_easy, arc_challenge |
| 2 | core.yaml 有显式 delimiter（非空格） | 使用 core.yaml 的 delimiter | boolq, jeopardy (`\nAnswer: `) |
| 3 | delimiter 为默认空格 + task 在 `MC_FULLSEQ_ANSWER_SEP_TASKS` | `\nAnswer: ` | commonsense_qa, agi_eval_lsat_ar |
| 4 | 其他 | 根据 question 末尾是否有空格决定 | hellaswag, copa |

**Per-sample 格式（openbook_qa / piqa / arc_easy / arc_challenge）**：

这四个 task 的题目混合了问答式和续写式，v5 按样本动态选择格式：

```python
has_prefix = question.startswith("Question: ")
core = question[len("Question: "):] if has_prefix else question
first_word = core.split()[0].lower() if core.split() else ""

if core.endswith("?") or first_word in QUESTION_WORDS:
    question = "Question: " + core
    sep = "\nAnswer: "
else:
    question = core
    sep = " "
```

**QUESTION_WORDS** = `{how, what, why, which, where, when, who, can, do, does, is, are, should}`

| Task | 有 `Question:` 前缀 | Q&A 比例 | 续写比例 | 实际 Q&A/Cont (2000 样本) |
|------|---------------------|----------|----------|--------------------------|
| openbook_qa | 无（自动添加） | 40% | 60% | 806 / 1194 |
| piqa | 有（续写时移除） | 49% | 51% | 971 / 1029 |
| arc_easy | 有（续写时移除） | 81% | 19% | 1629 / 371 |
| arc_challenge | 有（续写时移除） | 85% | 15% | 1698 / 302 |

**`language_modeling` handler**（full-seq 模式）：直接使用 core.yaml 的 delimiter 拼接 context 和 continuation。

| Task | core.yaml delimiter | 效果 |
|------|---------------------|------|
| jeopardy | `\nAnswer: ` | `context\nAnswer: continuation` |
| squad | `None` (默认空格) | `context continuation` |
| coqa | `None` (默认空格) | `context continuation` |
| 其他 language_modeling | `None` (默认空格) | `context continuation` |

**v4.3 的 bug**：`extract_pairs()` 中 `language_modeling` handler 硬编码了空格分隔符，导致 jeopardy 的 context 和 answer 直接拼接（`"...1929 Admiral Richard Byrd"`），丢失了 `\nAnswer: ` 分隔。v5 通过读取 core.yaml 修复此问题。

---

## 6. v5 实施策略

### 6.1 核心原则

1. **合并 train+test/validation**：利用完整数据集，不局限于 eval bundle 的单一 split
2. **保持 cap=2000**：与 v4.3 一致，控制验证集规模
3. **随机采样**：从合并后的数据池中随机采样，确保多样性
4. **优化 loss 策略**：将 3 个低 R² task 从 answer-only 改为 full-seq（详见 §7）

### 6.2 数据加载策略

| Task | 当前 v4.3 来源 | v5 来源 | 变化 |
|------|---------------|---------|------|
| hellaswag_zeroshot | eval_bundle val | HF train + val | 合并 |
| winogrande | eval_bundle val | HF train + val (test answer="" 跳过) | 合并 |
| copa | eval_bundle val | HF train + val (test label=-1 过滤) | 合并 |
| boolq | eval_bundle val | HF train + val | 合并 |
| squad | eval_bundle val | HF train + val | 合并 |
| coqa | eval_bundle val | HF train + val | 合并 |
| openbook_qa | eval_bundle test | HF train + test + val | 合并 |
| arc_easy | eval_bundle test | HF train + test + val | 合并 |
| arc_challenge | eval_bundle test | HF train + test + val | 合并 |
| commonsense_qa | eval_bundle val | HF train + test | 合并 |
| piqa | eval_bundle val | HF `baber/piqa` train + val | 合并 |
| 其余 10 个 task | eval_bundle | eval_bundle | 不变 |

### 6.3 预期效果

| 维度 | v4.3 | v5 预期 |
|------|------|---------|
| Overall R² | 0.4586 | 提升（低 R² task 样本增加 + loss 策略优化） |
| 低 R² task 数量 | 8 | 减少（copa/piqa/commonsense_qa 样本增加；commonsense_qa/lsat_ar/operators 改 full-seq） |
| 过拟合程度 | 严重 (gap 0.3-0.7) | 可能缓解（验证集更大，信号更稳定） |

---

## 7. Loss 策略优化：answer-only → full-seq

### 7.1 问题分析

v4.3 中 6 个 answer-only task 的 R² 表现：

| Task | Loss tokens | Unique answers | v4.3 R² | 1M 模型能力 |
|------|------------|----------------|---------|------------|
| commonsense_qa | **1** (A-E) | 5 | 0.13 | 不能做常识推理 |
| agi_eval_lsat_ar | **1** (A-E) | 4 | 0.15 | 不能做逻辑推理 |
| bigbench_operators | 1-3 | 43 | 0.12 | 不能做自定义运算 |
| bigbench_cs_algorithms | 1 | 11 | **0.38** | 能做（Valid/Invalid 简单分类） |
| bigbench_dyck_languages | 1-3 | 8 | **0.42** | 能做（括号补全，语法匹配） |
| bigbench_repeat_copy_logic | 4-35 | 166 | -0.05 | 不能做（样本太少） |

**规律**：1M 模型做不了的任务 + 只有 1 个 loss token → loss 方差极小 → meta-model 无信号可学 → R² 极低。

### 7.2 改进策略

将 3 个低 R² task 从 answer-only 改为 full-seq：

| Task | v4.3 loss | v5 loss | 改动理由 |
|------|----------|---------|---------|
| commonsense_qa | answer-only (1 token) | **full-seq** | 模型答不对，但可学 question/choices pattern |
| agi_eval_lsat_ar | answer-only (1 token) | **full-seq** | 同上，长 passage 提供丰富信号 |
| bigbench_operators | answer-only (1-3 tokens, 210 samples) | **full-seq** | 同上，问题文本有 pattern 可学 |

保持 3 个 task 不变：

| Task | loss 策略 | 理由 |
|------|----------|------|
| bigbench_cs_algorithms | answer-only | R²=0.38 已可用，1M 能做简单分类 |
| bigbench_dyck_languages | answer-only | R²=0.42 已可用，1M 能做括号补全 |
| bigbench_repeat_copy_logic | answer-only | 改了也没用（只有 32 样本） |

### 7.3 实现细节

**commonsense_qa / agi_eval_lsat_ar**（MC-embedded tasks）：
- 加入 `MC_FULLSEQ_ANSWER_SEP_TASKS`，使用 `"\nAnswer: "` 分隔符
- `clean_sft_prefix()` 移除 query 末尾的 `"Answer:"`（eval_bundle 原始格式自带）
- commonsense_qa 使用全部 5 choices (A-E)，不再降为 4

**bigbench_operators**（language_modeling task）：
- context + continuation 格式，直接改为 full-seq，无需额外处理

### 7.4 预期效果

| 维度 | v4.3 | v5 |
|------|------|-----|
| Full-seq tasks | 15 | **18** (+3) |
| Answer-only tasks | 6 | **3** (-3) |
| Full-seq samples | 27,273 | **29,195** |
| Answer-only samples | 4,792 | **2,352** |

---

## 8. 结论

### 8.1 v5 可行性

- **可补充**: 11 个 task (hellaswag, winogrande, copa, boolq, squad, coqa, openbook_qa, arc_easy, arc_challenge, commonsense_qa, piqa)
- **无法补充**: 9 个 task (lambada_openai 暂不需要, winograd, jeopardy, lsat_ar, 5 个 BIG-bench)
- **低 R² task**: 仅 3/8 可补充，其余 5 个受限于纯测试集或无 split 划分

### 8.2 预期收益

- **样本量增加**: +16.1% (27,163 → 31,547)
- **低 R² task 覆盖**: 3/8 可补充样本量 + 3/8 改进 loss 策略
- **信号质量提升**: 样本量增加 + full-seq loss 提供更多学习信号
- **数据质量提升**: 过滤句号-only 样本（winograd -18, winogrande -53）+ 过滤 COPA test 错误标签（-500）+ 修复 continuation_delimiter（jeopardy/boolq/arc/piqa）+ bigbench_language_identification answer 格式统一 + MC per-sample 格式优化（openbook_qa/piqa/arc_easy/arc_challenge）

### 8.3 局限性

- **cap=2000 限制**: piqa/commonsense_qa 虽然数据池扩大 8-10×，但实际增量有限
- **5/8 低 R² task 无法补充样本**: repeat_copy_logic, winograd, jeopardy 受限于数据源（但 operators/lsat_ar 已改 full-seq）
- **过拟合问题**: 增加验证集样本可能缓解，但 90D 空间的样本密度问题仍存在

### 8.4 下一步

1. ~~创建 `prepare_core_bmk_v5.py` 脚本~~ ✓
2. ~~生成 v5 验证集 parquet 文件~~ ✓
3. ~~上传到 HuggingFace，更新 `constants.py` 中的 `VAL_FILE`~~ ✓
4. ~~重新生成 v5 验证集~~（COPA 过滤 + 句号过滤 + continuation_delimiter 修复 + answer 格式统一 + MC per-sample 格式优化）✓
5. ~~重新上传 v5 到 HuggingFace~~ ✓ (2026-06-18)
6. ~~Python runner 添加 `core_bmk_v5` 支持~~ ✓
7. 重新运行实验，评估 Overall R² 是否提升
8. 对比 v4.3 vs v5 结果，分析 train+test 合并对低 R² task 的影响

---

## 9. 生成结果

### 9.1 格式验证结果

对 4 个中/高风险 task 做了 HF val → eval_bundle 格式的逐样本 diff 验证：

| Task | 转换难度 | 验证结果 | 备注 |
|------|---------|---------|------|
| winogrande | 中 | **100% PASS** | `_` 分割逻辑完全匹配 |
| copa | 中 | **100% PASS** | premise + connective 重构完全匹配 |
| commonsense_qa | 中 | **38.8% → 改用 5 choices + full-seq** | eval_bundle 随机丢弃 1/5 choices，v5 改用全部 5 choices；loss 从 answer-only 改为 full-seq |
| coqa | 高 | **100% PASS** | 多轮对话展开格式完全匹配 |

### 9.2 实际生成统计

| Task | v4.3 | v5 | 变化 | v5 数据池 | 来源 | 备注 |
|------|------|-----|------|----------|------|------|
| hellaswag_zeroshot | 2,000 | 2,000 | — | 49,947 | HF train+val | |
| lambada_openai | 2,000 | 2,000 | — | 5,153 | eval_bundle | |
| winogrande | 1,267 | **2,000** | **+733** | 41,665 | HF train+val | 过滤 ~53 条句号-only（数据池仍远超 cap） |
| winograd | 273 | **255** | **-18** | 255 | eval_bundle | 过滤 18 条句号-only |
| copa | 100 | **500** | **+400** | 500 | HF train+val | test 500 条 label=-1 已过滤 |
| jeopardy | 2,000 | 2,000 | — | 2,117 | eval_bundle | 修复 `\nAnswer: ` 分隔符 |
| boolq | 2,000 | 2,000 | — | 12,697 | HF train+val | 修复 `\nAnswer: ` 分隔符 |
| squad | 2,000 | 2,000 | — | 98,169 | HF train+val | |
| coqa | 2,000 | 2,000 | — | 116,630 | HF train+val | |
| bigbench_language_identification | 2,000 | 2,000 | — | 10,000 | eval_bundle | answer 格式统一为字母 |
| bigbench_qa_wikidata | 2,000 | 2,000 | — | 20,321 | eval_bundle | |
| openbook_qa | 500 | **2,000** | **+1,500** | 5,957 | HF train+val | per-sample 格式（Q&A 40%, 续写 60%） |
| piqa | 1,838 | **2,000** | **+162** | 17,951 | HF train+val | per-sample 格式（Q&A 47%, 续写 53%） |
| arc_easy | 2,000 | **2,000** | — | 5,197 | HF train+val | per-sample 格式（Q&A 81%, 续写 18%） |
| arc_challenge | 1,172 | **2,000** | **+828** | 2,590 | HF train+val | per-sample 格式（Q&A 84%, 续写 15%） |
| commonsense_qa | 1,221 | **2,000** | **+779** | 10,962 | HF train+val | |
| agi_eval_lsat_ar | 230 | 230 | — | 230 | eval_bundle | |
| bigbench_dyck_languages | 1,000 | 1,000 | — | 1,000 | eval_bundle | |
| bigbench_cs_algorithms | 1,320 | 1,320 | — | 1,320 | eval_bundle | |
| bigbench_operators | 211 | **210** | **-1** | 210 | eval_bundle | 过滤 1 条异常样本 |
| bigbench_repeat_copy_logic | 32 | 32 | — | 32 | eval_bundle | |
| **Total** | **27,163** | **31,547** | **+4,384 (+16.1%)** | | |

### 9.3 关键发现

1. **Test splits 标签缺失处理**：
   - **copa test (500)**: label = -1 → **直接过滤**（`return None`），不使用默认 choice1，避免错误答案污染数据
   - winogrande test (1,767): answer = "" → 跳过
   - commonsense_qa test (1,140): answerKey = "" → 跳过
   - piqa test (3,084): label = -1 → 跳过

2. **copa 从 100 → 500**：5× 提升（train 400 + val 100，test 500 条因 label=-1 被过滤）

3. **winogrande 从 1,267 → 2,000**：train+val 提供 41,665 样本（过滤 ~53 条句号-only），数据池远超 cap，采样到 2,000

4. **winograd 从 273 → 255**：过滤 18 条 continuation 仅为 `.` 的样本（`_` 占位符在句末，loss 信号几乎为零）

5. **openbook_qa 从 500 → 2,000**：train+val 提供 5,957 样本，cap 到 2,000

6. **arc_challenge 从 1,172 → 2,000**：train+val 提供 2,590 样本，cap 到 2,000

7. **commonsense_qa 从 1,221 → 2,000**：使用全部 5 choices (A-E)，loss 从 answer-only 改为 full-seq

8. **continuation_delimiter 修复**：从 core.yaml 读取 delimiter，修复 jeopardy/boolq/arc/piqa 丢失 `\nAnswer: ` 分隔符的问题（详见 §5.6）

9. **bigbench_language_identification answer 格式统一**：v4.3 中 `extract_answer()` 对该 task 做了特殊处理，将字母 answer（如 `"D"`）映射回选项全文（如 `"Desano"`），导致与其他 MC 任务不一致。v5 移除该特殊处理，统一返回 `choices[gold]`（字母），删除不再使用的 `MC_EMBEDDED_TASKS` 常量和 `parse_choice_from_query()` 函数

10. **MC per-sample 格式优化**：openbook_qa / piqa / arc_easy / arc_challenge 四个 task 的题目混合了问答式和续写式。v5 按样本动态选择格式：问答式用 `Question: ...?\nAnswer: ...`，续写式用自然文本拼接 `... answer`。这使续写式样本更匹配预训练文本分布，提升 loss 信号质量（详见 §5.2.7, §5.2.8, §5.2.10）

11. **10 个 eval_bundle tasks 完全不变**：输出与 v4.3 一致

### 9.4 输出文件

- `data/core_bmk_21tasks_v5_tokenized.pt` (555 MB，已生成 ✓)
- `data/core_bmk_21tasks_v5.parquet` (13.1 MB，已生成 ✓)
- `data/README_v5.md` (HF dataset card，已生成 ✓)
- HF repo: `liujin99/quadmix-core-bmk-v5` (已上传 ✓, 2026-06-18)

### 9.5 下一步

1. ~~上传 v5 parquet 到 HuggingFace~~ ✓
2. ~~更新 `constants.py` 中的 `VAL_FILE`~~ ✓
3. ~~重新生成 v5 验证集~~（应用 COPA 过滤、句号过滤、continuation_delimiter 修复、bigbench_language_identification answer 格式统一、MC per-sample 格式优化）✓
4. ~~重新上传 v5 到 HuggingFace~~ ✓ (2026-06-18, `liujin99/quadmix-core-bmk-v5`)
5. ~~Python runner 添加 `core_bmk_v5` 支持~~ ✓ (`run_essential_web_v1.py`, `reval_with_new_valset.py`)
6. 重新运行实验，评估 Overall R² 是否提升
7. 对比 v4.3 vs v5 结果，分析 train+val 合并对低 R² task 的影响

---

## 10. Python Runner 集成 (2026-06-18)

### 10.1 Shell 脚本（已默认 v5）

所有 demo 脚本在 v5 设计阶段已更新为默认使用 v5 验证集：

| 脚本 | VAL_FILE | HF repo | --val-set |
|------|----------|---------|-----------|
| `demo_run_full.sh` | `v5_tokenized.pt` | `quadmix-core-bmk-v5` | `core_bmk_v5` |
| `demo_run_quick.sh` | `v5_tokenized.pt` | `quadmix-core-bmk-v5` | `core_bmk_v5` |
| `demo_run_cpu.sh` | `v5_tokenized.pt` | `quadmix-core-bmk-v5` | `core_bmk_v5` |
| `demo_revalidate.sh` | — | — | `core_bmk_v5` (默认) |

### 10.2 Python Runner（2026-06-18 补充）

Shell 脚本传入 `--val-set core_bmk_v5`，但 Python runner 原先缺少对应的 argparse choice 和 ensure 函数，直接执行会报 `invalid choice` 错误。已补充：

| 文件 | 修改内容 |
|------|---------|
| `scripts/runners/run_essential_web_v1.py` | +import `HF_CORE_BMK_V5_*` 常量, +`ensure_core_bmk_v5_data()` 函数, +argparse `core_bmk_v5` choice, +elif 分支路由 |
| `scripts/runners/reval_with_new_valset.py` | +import `HF_CORE_BMK_V5_*` 常量, +`resolve_val_path()` 中 `core_bmk_v5` 分支, +argparse choice |

`ensure_core_bmk_v5_data()` 逻辑与 v4.3 一致：先检查本地文件 → 对比 HF 远端大小 → 下载或回退本地生成。

---

**文档版本**: v6.2  
**最后更新**: 2026-06-18  
**基于**: v4.3 实验结果 (674 experiments, Overall R²=0.4586)  
**脚本**: `scripts/validation_set/prepare_core_bmk_v5.py`  
**HF repo**: `liujin99/quadmix-core-bmk-v5` (已上传, 2026-06-18)  
**主要改动**: 
1. 11 个 task 合并 train+test/validation splits
2. 3 个低 R² task 从 answer-only 改为 full-seq (commonsense_qa, lsat_ar, operators)
3. commonsense_qa 使用全部 5 choices (A-E)
4. COPA test split (label=-1) 直接过滤，不再使用默认 choice1
5. schema 类型过滤句号-only continuation (winograd -18, winogrande -53)
6. 从 core.yaml 读取 continuation_delimiter，修复 jeopardy/boolq 分隔符丢失
7. bigbench_language_identification answer 格式统一为字母（移除 `MC_EMBEDDED_TASKS` 特殊处理）
8. MC per-sample 格式优化：openbook_qa / piqa / arc_easy / arc_challenge 按样本动态选择问答式或续写式格式（优先级高于 core.yaml delimiter）
9. Python runner 添加 `core_bmk_v5` 支持（`run_essential_web_v1.py`, `reval_with_new_valset.py`）
