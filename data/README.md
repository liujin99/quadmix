# CORE-BMK v2 Validation Set

Benchmark-aligned validation set for QuaDMix proxy model, designed to improve signal-to-noise ratio compared to the original CORE-22tasks (v1).

## Motivation

Analysis of the 21 CORE benchmark tasks revealed that most tasks have very low answer-to-context ratios (average 5.7%), causing the 1M-parameter proxy model to produce near-random predictions (perplexity ~4800). This resulted in poor LightGBM generalization (val_r2 = 0.179 vs 0.825 for OpenHermes-10k).

The QuaDMix paper's BMK baseline uses only 5 tasks (HellaSwag, ARC-E, ARC-C, MMLU, TriviaQA), all with answer ratios > 10%. Following this principle, CORE-BMK v2 selects 10 "BMK-like" tasks with answer ratios > 10% and uses **full-sequence loss** instead of continuation-only loss.

## Design

### Task Selection

Only tasks with **answer character ratio > 10%** are included:

| Task | Type | N | Ans% | Description |
|------|------|---|------|-------------|
| hellaswag_zeroshot | MC | 5000 | 37.4% | Commonsense reasoning (sentence completion) |
| piqa | MC | 1838 | 64.1% | Physical intuition QA |
| bigbench_repeat_copy_logic | LM | 32 | 44.5% | Symbolic pattern repetition |
| copa | MC | 100 | 39.1% | Cause/effect reasoning |
| openbook_qa | MC | 500 | 25.0% | Open-book science QA |
| winogrande | schema | 1267 | 22.5% | Pronoun disambiguation |
| winograd | schema | 273 | 21.8% | Winograd Schema Challenge |
| arc_challenge | MC | 1172 | 16.7% | Grade-school science (hard) |
| arc_easy | MC | 2376 | 14.6% | Grade-school science (easy) |
| bigbench_qa_wikidata | LM | 5000 | 14.5% | Wikidata-based QA |

### Excluded Tasks (11 tasks, avg Ans% = 3.2%)

boolq (0.5%), coqa (0.6%), lambada (1.8%), bigbench_dyck (1.8%), bigbench_operators (1.5%), squad (2.2%), bigbench_language_id (3.3%), bigbench_cs_algorithms (3.5%), agi_eval_lsat_ar (4.2%), commonsense_qa (6.0%), jeopardy (9.9%)

These tasks have long context passages or very short answers, causing context to dominate the loss signal.

### Loss Strategy

**Full-sequence loss**: All non-padding tokens contribute to the loss (loss_mask = True for all tokens).

This differs from CORE-22tasks v1 which used continuation-only loss (only answer tokens). Full-sequence loss allows the proxy model to learn the overall distribution of benchmark text, similar to the QuaDMix paper's BMK approach.

## Statistics

| Metric | OpenHermes-10k | CORE-22tasks v1 | **CORE-BMK v2** |
|--------|----------------|-----------------|-----------------|
| Documents | 10,000 | 46,926 | **37,600** |
| Non-padding tokens | 2,235,498 | 6,166,003 | **1,237,907** |
| Loss tokens | 2,235,498 | 317,561 | **1,237,907** |
| Loss tokens/doc | 223.5 | 6.8 | **32.9** |
| Loss% of non-padding | 100% | 5.2% | **100%** |
| File size | 176 MB | 825 MB | **661 MB** |

## Files

- `core_bmk_10tasks_v2_tokenized.pt` (661 MB): PyTorch tensor format for proxy model validation
  - `token_ids`: LongTensor [37600, 2048] (padded)
  - `loss_mask`: BoolTensor [37600, 2048] (True for all non-padding tokens)
  - `task_labels`: list[str] (per-doc task label)
  - `metadata`: dict (generation config and task stats)

- `core_bmk_10tasks_v2.parquet` (3.1 MB): Pandas-readable format for inspection
  - Columns: `text`, `task`, `num_tokens`, `num_loss_tokens`

## Usage

```python
import torch

data = torch.load("core_bmk_10tasks_v2_tokenized.pt", weights_only=True)
token_ids = data["token_ids"]      # [37600, 2048]
loss_mask = data["loss_mask"]      # [37600, 2048]
task_labels = data["task_labels"]  # list of 37600 strings
```

Or with pandas:

```python
import pandas as pd

df = pd.read_parquet("core_bmk_10tasks_v2.parquet")
print(df["task"].value_counts())
```

## Generation

```bash
python scripts/validation_set/prepare_core_bmk_v2.py \
    --eval-bundle /path/to/eval_bundle \
    --output-dir data \
    --num-samples-per-task 20000
```

## Comparison with v1

| Aspect | v1 (CORE-22tasks) | v2 (CORE-BMK) |
|--------|-------------------|---------------|
| Tasks | 21 | 10 |
| Loss strategy | continuation-only | full-sequence |
| Avg answer ratio | 5.7% | 30.5% |
| Loss tokens | 317,561 | 1,237,907 |
| Loss tokens/doc | 6.8 | 32.9 |

v2 provides 3.9x more loss tokens than v1, with 100% token utilization (vs 5.2% in v1).

## License

Derived from public benchmark datasets. Individual task licenses vary.


