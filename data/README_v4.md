# CORE-BMK v4 Validation Set

Benchmark-aligned validation set for QuaDMix proxy model, with **task-type-aware loss masking**.

## Design Principles

v4 is built on three insights derived from v3 (val R²=0.596) analysis:

1. **Continuation tasks are naturally coherent** — context + continuation forms fluent text that exists in pretraining distribution → use **full-sequence loss**
2. **QA tasks have Q+A concatenation that is NOT natural text** — "Question: X? Y" is an SFT artifact, not web text → use **answer-only loss mask** (Q provides topic context, loss focuses on answer generation)
3. **SFT format artifacts pollute loss** — "Question:", "Answer:", "Choices:" prefixes are not common in pretraining data → **remove all SFT artifacts**

This mirrors OpenHermes-10k's approach: user message provides context, loss only on assistant response.

## Task Classification

### A. Continuation Tasks (full-sequence loss) — 5 tasks

| Task | N | Type | Example |
|------|--:|------|---------|
| hellaswag_zeroshot | 2000 | MC→continuation | `Roof shingle removal: A man is sitting on a roof. He starts pulling up roofing on a roof.` |
| lambada_openai | 2000 | LM | `Ives hopped in deftly, after giving the boat a strong, hard push. The pilot started the outboard and` |
| winogrande | 1267 | schema | `Sarah was a much better surgeon than Maria so Maria always got the easier cases.` |
| winograd | 273 | schema | `The city councilmen refused the demonstrators a permit because the city councilmen feared violence.` |
| copa | 100 | MC→continuation | `The man turned on the faucet, therefore water flowed from the spout.` |

### B. QA Tasks (answer-only mask) — 5 tasks

| Task | N | Avg Ans | Example (context **\|** answer) |
|------|--:|--------:|-------------------------------|
| piqa | 1838 | 98c | `How do I ready a guinea pig cage?` **\|** `Provide the guinea pig with a cage full of bedding...` |
| arc_challenge | 1172 | 30c | `An astronomer observes that a planet rotates faster...?` **\|** `Planetary days will become shorter.` |
| arc_easy | 2000 | 23c | `Which statement best explains why photosynthesis...?` **\|** `Sunlight is the source of energy...` |
| openbook_qa | 500 | 19c | `A person wants to start saving money...` **\|** `quit eating lunch out` |
| commonsense_qa | 1221 | 10c | `A revolving door... at a what?` **\|** `bank` |

**`|` marks the Q/A boundary** — model sees both, but loss_mask is True only for answer tokens.

## Key Changes from v3

| Change | v3 | v4 | Rationale |
|--------|-----|-----|-----------|
| QA loss strategy | full-sequence | **answer-only mask** | Q+A concatenation is not natural text; loss should focus on answer generation |
| "Question:" prefix | kept | **removed** | SFT artifact, not in pretraining distribution |
| "Answer:" prefix | kept (commonsense_qa) | **removed** | Same |
| "Choices:" section | kept (commonsense_qa) | **removed** | 70c overhead vs 10c answer signal (11% efficiency) |
| commonsense_qa answer | "A"/"B"/"C"/"D" | **real option text** ("bank", "complete job") | Meaningful text instead of labels |
| QA delimiter | from core.yaml (`\nAnswer: `) | **space** (` `) | Remove SFT artifact from delimiter |

## Why Answer-Only Mask for QA Tasks

1M proxy (2-layer transformer, 256 dim) **cannot** understand question semantics or Q-A correspondence. It learns:
- Word frequency / topic / domain distributions
- Surface text style and formatting

Question text provides **topic context** (e.g., "photosynthesis" → biology domain), helping the model build internal representations. But the Q+A concatenation is not natural pretraining text, so computing loss on Q tokens adds noise.

Answer-only mask:
- Model sees Q (topic context) + A (answer text)
- Loss computed only on A tokens
- Similar to OpenHermes: user message provides context, loss on assistant response

## Statistics

| Metric | OpenHermes-10k | v3 | **v4** |
|--------|---------------|-----|--------|
| Documents | 10,000 | 10,371 | **12,371** |
| Non-padding tokens | 2,235,498 | 435,065 | **553,974** |
| Loss tokens | 2,235,498 | 435,065 | **430,115** |
| Loss% of non-padding | 100% | 100% | **77.6%** |
| Continuation loss tokens | — | — | 366,327 |
| QA answer-only loss tokens | — | — | 63,788 |
| File size (.pt) | 176 MB | 182 MB | **217 MB** |

## Files

- `core_bmk_10tasks_v4_tokenized.pt` (217 MB): PyTorch tensor format
  - `token_ids`: LongTensor [12371, 2048] (padded)
  - `loss_mask`: BoolTensor [12371, 2048] (True for continuation=all, QA=answer only)
  - `task_labels`: list[str] (per-doc task label)
  - `metadata`: dict (generation config and task stats)

- `core_bmk_10tasks_v4.parquet` (2.9 MB): Pandas-readable format
  - Columns: `task`, `category`, `loss_strategy`, `context`, `answer`, `text`

## Usage

```python
import torch

data = torch.load("core_bmk_10tasks_v4_tokenized.pt", weights_only=True)
token_ids = data["token_ids"]      # [12371, 2048]
loss_mask = data["loss_mask"]      # [12371, 2048]
task_labels = data["task_labels"]  # list of 12371 strings
```

## Generation

```bash
python scripts/validation_set/prepare_core_bmk_v4.py \
    --eval-bundle /path/to/eval_bundle \
    --output-dir data \
    --num-samples-per-task 2000
```

## License

Derived from public benchmark datasets. Individual task licenses vary.
