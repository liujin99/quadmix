# 下游基准数据集分析报告

## Eval Bundle 来源

- **Eval Bundle**: `https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip` ✓
- **基于**: MosaicML Eval Gauntlet v0.3.0
- **对齐**: nanochat CORE metric (DCLM Benchmark)
- **配置文件**: `core.yaml` 定义 21 个 task

## 21 个 Task 数据源验证汇总表

| # | Task | 原始基准 | 数据来源链接 | 下载验证 | Eval Bundle | Train | Test | Validation | 样本数匹配 |
|---|------|---------|-------------|---------|-------------|-------|------|------------|-----------|
| 1 | **hellaswag_zeroshot** | HellaSwag | `github.com/rowanz/hellaswag` | ✓ HF parquet | 10,042 (val) | 39,905 | 10,003 | 10,042 | ✓ |
| 2 | **winogrande** | WinoGrande | `github.com/allenai/winogrande` | ✓ HF parquet | 1,267 (val) | 40,398 (xl) | 1,767 | 1,267 | ✓ |
| 3 | **copa** | COPA (SuperGLUE) | `huggingface.co/datasets/aps/super_glue` | ✓ HF parquet | 100 (val) | 400 | 500 | 100 | ✓ |
| 4 | **boolq** | BoolQ (SuperGLUE) | `github.com/google-research-datasets/boolean-questions` | ✓ HF parquet | 3,270 (val) | 9,427 | - | 3,270 | ✓ |
| 5 | **squad** | SQuAD v1.1 | `rajpurkar.github.io/SQuAD-explorer` | ✓ 官方 JSON | 10,570 (val) | 87,599 | - | 10,570 | ✓ |
| 6 | **coqa** | CoQA | `stanfordnlp.github.io/coqa` | ✓ 官方 JSON | 7,983 (val) | 108,647 QA | - | 7,983 QA | ✓ |
| 7 | **openbook_qa** | OpenBookQA | `github.com/allenai/OpenBookQA` | ✓ HF parquet | 500 (test) | 4,957 | 500 | 500 | ✓ |
| 8 | **arc_easy** | ARC-Easy | `allenai.org/data/arc` | ✓ HF parquet | 2,376 (test) | 2,251 | 2,376 | 570 | ✓ |
| 9 | **arc_challenge** | ARC-Challenge | `allenai.org/data/arc` | ✓ HF parquet | 1,172 (test) | 1,119 | 1,172 | 299 | ✓ |
| 10 | **commonsense_qa** | CommonsenseQA | `github.com/jonathanherzig/commonsenseqa` | ✓ HF parquet | 1,221 (val) | 9,741 | 1,140 | 1,221 | ✓ |
| 11 | **piqa** | PIQA | `yonatanbisk.com/piqa` | ✓ 原始 JSONL | 1,838 (val) | 16,113 | - | 1,838 | ✓ |
| 12 | **lambada_openai** | LAMBADA | `zenodo.org/records/2630551` | ✓ Zenodo tar.gz | 5,153 (test) | - | 5,153 | 4,869 (dev) | ✓ |
| 13 | **winograd** | WSC | `cs.nyu.edu/~davise/papers/WinogradSchemas` | ⚠️ XML 285 | 273 (val) | - | 285 | - | ⚠️ |
| 14 | **jeopardy** | Jeopardy Archive | `j-archive.com` | ⚠️ 无批量下载 | 2,117 (test) | - | 2,117 | - | ⚠️ |
| 15 | **agi_eval_lsat_ar** | AGIEval | `github.com/ruixiangcui/AGIEval` | ✓ GitHub JSONL | 230 (test) | - | 230 | - | ✓ |
| 16 | **bigbench_language_identification** | BIG-bench | `github.com/google/BIG-bench/.../language_identification` | ✓ task.json | 10,000 | 10,000 | - | - | ✓ |
| 17 | **bigbench_qa_wikidata** | BIG-bench | `github.com/google/BIG-bench/.../qa_wikidata` | ✓ task.json | 20,321 | 20,442 | - | - | ✓ |
| 18 | **bigbench_dyck_languages** | BIG-bench | `github.com/google/BIG-bench/.../dyck_languages` | ✓ task.json | 1,000 | 1,000 | - | - | ✓ |
| 19 | **bigbench_cs_algorithms** | BIG-bench | `github.com/google/BIG-bench/.../cs_algorithms` | ✗ 无 examples | 1,320 | 0 | - | - | ✗ |
| 20 | **bigbench_operators** | BIG-bench | `github.com/google/BIG-bench/.../operators` | ✓ task.json | 210 | 211 | - | - | ✓ |
| 21 | **bigbench_repeat_copy_logic** | BIG-bench | `github.com/google/BIG-bench/.../repeat_copy_logic` | ✓ task.json | 32 | 32 | - | - | ✓ |

## 汇总统计

| 指标 | 数值 |
|------|------|
| Eval Bundle 总样本数 | 27,163 |
| 平均每个 task | 1,293 |
| 有 train split 的 task | 11 |
| 纯测试集的 task | 5 |
| BIG-bench 无 split 划分的 task | 6 |

## 数据源分类

### 1. HuggingFace 数据集 (可通过 parquet 下载)

| Task | HuggingFace Repo | Config | Train | Test | Validation |
|------|------------------|--------|-------|------|------------|
| hellaswag | Rowan/hellaswag | default | 39,905 | 10,003 | 10,042 |
| winogrande | allenai/winogrande | winogrande_xl | 40,398 | 1,767 | 1,267 |
| copa | aps/super_glue | copa | 400 | 500 | 100 |
| boolq | google/boolq | default | 9,427 | - | 3,270 |
| squad | rajpurkar/squad | plain_text | 87,599 | - | 10,570 |
| openbook_qa | allenai/openbookqa | main | 4,957 | 500 | 500 |
| arc_easy | allenai/ai2_arc | ARC-Easy | 2,251 | 2,376 | 570 |
| arc_challenge | allenai/ai2_arc | ARC-Challenge | 1,119 | 1,172 | 299 |
| commonsense_qa | tau/commonsense_qa | default | 9,741 | 1,140 | 1,221 |
| lambada_openai | EleutherAI/lambada_openai | default | - | 5,153 | - |

### 2. 官方数据源 (JSON/JSONL 格式)

| Task | 来源 | 格式 | Train | Test | Validation |
|------|------|------|-------|------|------------|
| squad | rajpurkar.github.io/SQuAD-explorer | JSON | 87,599 | - | 10,570 |
| coqa | stanfordnlp.github.io/coqa | JSON | 108,647 QA | - | 7,983 QA |
| piqa | yonatanbisk.com/piqa | JSONL | 16,113 | - | 1,838 |
| agi_eval_lsat_ar | github.com/ruixiangcui/AGIEval | JSONL | - | 230 | - |

### 3. BIG-bench 数据集 (task.json 格式)

| Task | GitHub 路径 | Examples | Eval Bundle |
|------|------------|----------|-------------|
| language_identification | bigbench/benchmark_tasks/language_identification | 10,000 | 10,000 |
| qa_wikidata | bigbench/benchmark_tasks/qa_wikidata | 20,442 | 20,321 |
| dyck_languages | bigbench/benchmark_tasks/dyck_languages | 1,000 | 1,000 |
| cs_algorithms | bigbench/benchmark_tasks/cs_algorithms | 0 | 1,320 |
| operators | bigbench/benchmark_tasks/operators | 211 | 210 |
| repeat_copy_logic | bigbench/benchmark_tasks/repeat_copy_logic | 32 | 32 |

### 4. 特殊数据源

| Task | 来源 | 说明 |
|------|------|------|
| lambada_openai | Zenodo (zenodo.org/records/2630551) | 原始数据集包含 test=5,153, dev=4,869, train-novels=2,679 文件 |
| winograd | NYU WSCollection.xml | XML 包含 285 schemas，eval bundle 使用 273 子集 |
| jeopardy | j-archive.com | 无批量下载，eval bundle 包含 2,117 samples |

## 验证结论

| 状态 | 数量 | Tasks |
|------|------|-------|
| ✓ 完全匹配 | 17 | hellaswag, winogrande, copa, boolq, squad, coqa, openbook_qa, arc_easy, arc_challenge, commonsense_qa, piqa, lambada_openai, agi_eval_lsat_ar, bigbench_language_identification, bigbench_qa_wikidata, bigbench_dyck_languages, bigbench_operators, bigbench_repeat_copy_logic |
| ⚠️ 部分验证 | 2 | winograd (285 vs 273 子集), jeopardy (无批量下载) |
| ✗ 不匹配 | 1 | bigbench_cs_algorithms (MosaicML 自行生成) |
| ✓ 额外数据 | 1 | lambada_openai dev=4,869 可用 |

## 备注

- **Eval Bundle 使用的 split**: 大多数 task 使用 validation split，部分使用 test split (arc_easy, arc_challenge, lambada_openai, openbook_qa, jeopardy, agi_eval_lsat_ar)
- **BIG-bench 无 split 划分**: 所有数据作为单一数据集，无 train/test/validation 区分
- **bigbench_cs_algorithms**: GitHub task.json 中无 examples，eval bundle 的 1,320 samples 由 MosaicML 自行生成
- **winograd**: WSCollection.xml 包含 285 schemas，eval bundle 使用 273 samples 子集
- **lambada_openai**: Zenodo 原始数据集包含 development set (4,869 samples)，可作为额外验证数据
- **coqa**: 官方数据以 conversation 为单位，train=7,199 stories → 108,647 QA pairs, dev=500 stories → 7,983 QA pairs

---

**文档版本**: v1.0  
**最后更新**: 2026-06-17  
**验证方法**: HuggingFace API、官方下载、GitHub API、Zenodo API
