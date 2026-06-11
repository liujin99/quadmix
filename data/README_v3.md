# CORE-BMK v3 Validation Set

Benchmark-aligned validation set for QuaDMix proxy model, designed based on **1M proxy model learnability** rather than answer ratio or sample count.

## Motivation

Analysis of BMK-v2 revealed critical issues:
- 54% of data came from `bigbench_qa_wikidata` (weak signal: 7-char entity answers)
- Selection based on Ans% > 10% included symbolic tasks with zero natural language signal
- Tasks requiring deep reasoning, reading comprehension, or knowledge recall cannot be learned by 1M proxy

**Key insight**: Even if 1M proxy cannot "solve" a task, loss has discriminative power if the task's text vocabulary/topic/style distribution is sensitive to training data quality.

## Design Principle: 1M Proxy Learnability

A 1M-parameter proxy model (2-layer transformer, 256 dim) can learn:
- Word frequency distributions
- Simple syntactic patterns
- Topic/domain distributions
- Surface text style and formatting

A 1M proxy **cannot** learn:
- Multi-step logical reasoning
- Deep reading comprehension
- Long-range dependencies (>512 tokens)
- Factual knowledge recall

**Selection criterion**: Include only tasks whose text distribution a 1M proxy can learn, regardless of whether it can "solve" the task.

## Task Selection

### 10 Tasks Selected (by 1M learnability)

| Task | Type | N | Ans% | 1M Learnability | Rationale |
|------|------|---|------|-----------------|-----------|
| hellaswag_zeroshot | MC | 2000 | 37.4% | **Strong** | Narrative continuation = standard LM task |
| arc_easy | MC | 2000 | 14.7% | **Strong** | Simple science QA, learnable vocabulary |
| piqa | MC | 1838 | 64.1% | **Medium-Strong** | Intuitive physics, concrete scenarios |
| lambada | LM | 0 | 1.8% | **Medium** | Literary style sensitivity (excluded: not in eval bundle) |
| arc_challenge | MC | 1172 | 16.7% | **Medium** | Science text distribution learnable |
| winogrande | schema | 1267 | 22.5% | **Medium** | Simple sentence structure |
| winograd | schema | 273 | 21.8% | **Medium** | Same as winogrande, small N |
| copa | MC | 100 | 39.1% | **Medium** | Simple causal scenarios, small N |
| openbook_qa | MC | 500 | 25.0% | **Medium** | Science scenarios, small N |
| commonsense_qa | MC | 1221 | 6.0% | **Medium-Weak** | Concept knowledge, borderline |

**Note**: `lambada` was planned but not found in the eval bundle, resulting in 9 tasks instead of 10.

### 11 Tasks Excluded

**Reading comprehension / knowledge recall** (beyond 1M capacity):
- `boolq` (0.5%): Long passages, yes/no answers
- `squad` (2.2%): Extractive QA, long context
- `jeopardy` (9.9%): Knowledge recall
- `coqa` (0.6%): Conversational QA

**Weak signal**:
- `bigbench_qa_wikidata` (14.5%): 7-char entity answers, simple fact lookup

**Logical reasoning** (beyond 1M):
- `agi_eval_lsat_ar` (4.2%): Analytical reasoning

**Symbolic / non-natural language** (zero NL signal):
- `bigbench_dyck_languages` (1.8%): Bracket sequences
- `bigbench_repeat_copy_logic` (44.5%): Pattern repetition
- `bigbench_operators` (1.5%): Mathematical operators
- `bigbench_cs_algorithms` (3.5%): Algorithmic strings
- `bigbench_language_id` (3.3%): Language identification tokens

## Key Differences from v2

| Aspect | v2 (CORE-BMK) | v3 (CORE-BMK) |
|--------|---------------|---------------|
| Selection principle | Ans% > 10% | 1M proxy learnability |
| Tasks | 10 | 9 (lambada missing) |
| Cap per task | 20,000 | 2,000 |
| Total docs | 37,600 | 10,371 |
| bigbench_qa_wikidata | Included (54% of data) | **Excluded** (weak signal) |
| Symbolic tasks | Included | **Excluded** (zero NL signal) |
| lambada | Excluded (Ans% = 1.8%) | Planned (literary style) |

## Why Ans% is Not the Selection Criterion

- **lambada** (Ans% = 1.8%): Included in v3 plan because paragraph distribution is learnable
- **boolq** (Ans% = 0.5%): Excluded because reading comprehension is beyond 1M capacity
- **bigbench_repeat_copy_logic** (Ans% = 44.5%): Excluded because it's symbolic, not natural language

The deciding factor is whether 1M can learn the **text distribution**, not whether it can solve the task.

## Why N (Sample Count) is Not a Core Issue

Small N tasks naturally get lower weight in val_loss calculation:
- `copa` (N=100): Only 1% weight in val_loss
- `hellaswag` (N=2000): 19% weight in val_loss

Task instability has minimal impact when N is small. The focus should be on **quality of signal**, not quantity.

## Loss Strategy

**Full-sequence loss**: All non-padding tokens contribute to the loss (loss_mask = True for all tokens).

This allows the proxy model to learn the overall distribution of benchmark text, similar to the QuaDMix paper's BMK approach.

## Statistics

| Metric | OpenHermes-10k | CORE-22tasks v1 | CORE-BMK v2 | **CORE-BMK v3** |
|--------|----------------|-----------------|-------------|-----------------|
| Documents | 10,000 | 46,926 | 37,600 | **10,371** |
| Non-padding tokens | 2,235,498 | 6,166,003 | 1,237,907 | **435,065** |
| Loss tokens | 2,235,498 | 317,561 | 1,237,907 | **435,065** |
| Loss tokens/doc | 223.5 | 6.8 | 32.9 | **41.9** |
| Loss% of non-padding | 100% | 5.2% | 100% | **100%** |
| File size (.pt) | 176 MB | 825 MB | 661 MB | **182 MB** |

v3 provides higher loss tokens/doc (41.9 vs 32.9) with much smaller footprint, focusing on quality over quantity.

## Files

- `core_bmk_10tasks_v3_tokenized.pt` (182 MB): PyTorch tensor format for proxy model validation
  - `token_ids`: LongTensor [10371, 2048] (padded)
  - `loss_mask`: BoolTensor [10371, 2048] (True for all non-padding tokens)
  - `task_labels`: list[str] (per-doc task label)
  - `metadata`: dict (generation config and task stats)

- `core_bmk_10tasks_v3.parquet` (13.8 MB): Pandas-readable format for inspection
  - Columns: `text`, `task`, `num_tokens`, `num_loss_tokens`

## Usage

```python
import torch

data = torch.load("core_bmk_10tasks_v3_tokenized.pt", weights_only=True)
token_ids = data["token_ids"]      # [10371, 2048]
loss_mask = data["loss_mask"]      # [10371, 2048]
task_labels = data["task_labels"]  # list of 10371 strings
```

Or with pandas:

```python
import pandas as pd

df = pd.read_parquet("core_bmk_10tasks_v3.parquet")
print(df["task"].value_counts())
```

## Generation

```bash
python scripts/validation_set/prepare_core_bmk_v3.py \
    --eval-bundle /path/to/eval_bundle \
    --output-dir data \
    --num-samples-per-task 2000
```

## Comparison with v1 and v2

| Aspect | v1 (CORE-22tasks) | v2 (CORE-BMK) | v3 (CORE-BMK) |
|--------|-------------------|---------------|---------------|
| Tasks | 21 | 10 | 9 |
| Selection principle | All CORE tasks | Ans% > 10% | 1M learnability |
| Loss strategy | continuation-only | full-sequence | full-sequence |
| Avg answer ratio | 5.7% | 30.5% | 27.5% |
| Loss tokens | 317,561 | 1,237,907 | 435,065 |
| Loss tokens/doc | 6.8 | 32.9 | 41.9 |
| Weak signal tasks | Many | bigbench_qa_wikidata (54%) | **None** |
| Symbolic tasks | 5 | 1 | **0** |

v3 eliminates weak-signal and non-NL tasks, focusing purely on what 1M proxy can learn from text distribution.

## License

Derived from public benchmark datasets. Individual task licenses vary.
