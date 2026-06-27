# QuadMix CORE Benchmark Validation Set v6

**Version:** v6.0  
**Tasks:** 21 (all nanochat CORE benchmarks, deduplicated)  
**Documents:** ~31,547 (same as v5)  
**Loss Strategy:** Per-task (full-seq + answer-only hybrid)

## Overview

This is the **v6** validation set for the [QuadMix](https://github.com/liujin99/quadmix) proxy model pipeline. It covers all 21 unique benchmarks from the nanochat CORE evaluation suite, using a per-task loss strategy that independently selects the optimal loss mask for each task.

### Previous Versions

| Version | Repo | Tasks | Docs | Loss Strategy |
|---------|------|-------|------|---------------|
| v3 | `liujin99/quadmix-core-22tasks` | 22 (with dup) | 46,926 | Continuation-only |
| v4 | `liujin99/quadmix-core-bmk-v4` | 10 | 12,371 | Continuation-only |
| v4.2 | `liujin99/quadmix-core-bmk-v4.2` | 21 | 27,163 | Per-task hybrid (12 full-seq, 9 answer-only) |
| v4.3 | `liujin99/quadmix-core-bmk-v4.3` | 21 | 27,163 | Per-task hybrid (15 full-seq, 6 answer-only) |
| v5 | `liujin99/quadmix-core-bmk-v5` | 21 | 31,547 | Per-task hybrid (18 full-seq, 3 answer-only) |
| **v6** | **`liujin99/quadmix-core-bmk-v6`** | **21** | **~31,547** | **Per-task hybrid (18 full-seq, 3 answer-only)** |

## Key Changes in v6

v6 fixes two preprocessing issues in v5 where HF-loaded tasks did not match eval_bundle format.

### Fix 1: HellaSwag Tag Cleaning

**Problem in v5:** HellaSwag data loaded directly from HuggingFace (`Rowan/hellaswag`) contained raw structural tags from wikiHow and ActivityNet sources:

```
[header] How to become a fashion consultant [title] Obtain your high school diploma... [step] This job requires...
```

These tags appeared in ~67% of HellaSwag validation samples, causing inconsistency with eval_bundle format and potentially affecting proxy model training.

**Fix in v6:** Added `clean_hellaswag_ctx()` function that matches eval_bundle preprocessing:

| Tag | Transformation | Example |
|-----|----------------|---------|
| `[header] X [title]` | `X. ` (period added) | `[header] How to cook [title] Prepare` → `How to cook. Prepare` |
| `[title]` | `. ` (sentence separator) | `...done. [title] Next step` → `...done. Next step` |
| `[step]`, `[substeps]` | ` ` (removed) | `[step] Mix ingredients` → `Mix ingredients` |
| Other brackets `[...]` | ` ` (removed) | `[www.example.com]` → (removed) |

Additional processing:
- `activity_label` field prepended as prefix: `{activity}: {cleaned_text}`
- First letter after `. ` capitalized (sentence boundaries)
- First letter of text capitalized

**Result:**

```
HF raw:     [header] How to become a fashion consultant [title] Obtain your high school diploma...
v6 cleaned: Personal Care and Style: How to become a fashion consultant. Obtain your high school diploma...
```

**Validation:** 99.7% match with eval_bundle format (10,015/10,042 examples). Remaining 0.3% are edge cases with abbreviations (`u.s.`, `dr.`, `e.g.`, `i.e.`) where capitalization differs.

### Fix 2: MC Task Format Consistency (arc/piqa/openbook_qa)

**Problem in v5:** The `MC_PER_SAMPLE_TASKS` logic in `extract_pairs()` applied per-sample branching based on whether the question text ends with `?` or starts with a question word. This created format inconsistency with eval_bundle:

| Task | eval_bundle format (100% consistent) | v5 fill-in-blank samples | v5 question samples |
|------|--------------------------------------|--------------------------|---------------------|
| arc_challenge | `"Question: " + "\nAnswer: "` | `question + " " + answer` ❌ (~16%) | `"Question: " + "\nAnswer: "` ✅ (~84%) |
| arc_easy | `"Question: " + "\nAnswer: "` | `question + " " + answer` ❌ (~20%) | `"Question: " + "\nAnswer: "` ✅ (~80%) |
| piqa | `"Question: " + "\nAnswer: "` | `question + " " + answer` ❌ (~51%) | `"Question: " + "\nAnswer: "` ✅ (~49%) |
| openbook_qa | `question + " " + answer` | `question + " " + answer` ✅ (~60%) | `"Question: " + "\nAnswer: "` ❌ (~40%) |

**Root cause:** eval_bundle uses a fixed format per task (defined by `continuation_delimiter` in core.yaml), but v5 branched per-sample based on text content.

**Fix in v6:** Removed `MC_PER_SAMPLE_TASKS` branch entirely. All 4 tasks now use the standard `continuation_delimiter` path:

- arc_challenge/arc_easy/piqa: `continuation_delimiter="\nAnswer: "` from core.yaml → `"Question: " + question + "\nAnswer: " + answer`
- openbook_qa: default delimiter (space) → `question + " " + answer`

**Examples:**

```
arc_challenge fill-in-blank:
  v5: "Biological evolution can occur through all of these except none of these"
  v6: "Question: Biological evolution can occur through all of these except\nAnswer: none of these"

piqa non-question:
  v5: "lid can be put on Tupperware"
  v6: "Question: lid\n\nAnswer: can be put on Tupperware"

openbook_qa question:
  v5: "Question: When a needle points north on a compass and you are thirsty?\nAnswer: head towards water"
  v6: "When a needle points north on a compass and you are thirsty? head towards water"
```

**Rationale:** The validation set format should match eval_bundle to maximize proxy signal accuracy. The proxy model's loss on validation data predicts the 1.3B model's benchmark performance — format consistency ensures this prediction is reliable.

## Files

| File | Size | Description |
|------|------|-------------|
| `core_bmk_21tasks_v6_tokenized.pt` | ~555 MB | Pre-tokenized PyTorch tensor (ready for proxy eval) |
| `core_bmk_21tasks_v6.parquet` | ~13 MB | Human-readable parquet (context, answer, task label, loss mask info) |

## Task List

### Full-Sequence Tasks (18 tasks, ~29,195 docs)

Context + Continuation forms natural text; all tokens contribute to loss.

| # | Task | Docs | Source | Description |
|---|------|------|--------|-------------|
| 1 | hellaswag_zeroshot | 2,000 | HF | Sentence completion (v6: tags cleaned) |
| 2 | lambada_openai | 2,000 | eval_bundle | Last-word prediction |
| 3 | winogrande | 2,000 | HF | Pronoun resolution |
| 4 | winograd | 255 | eval_bundle | Winograd Schema Challenge |
| 5 | copa | 500 | HF | Causal reasoning |
| 6 | jeopardy | 2,000 | eval_bundle | Trivia clues |
| 7 | boolq | 2,000 | HF | Yes/no reading comprehension |
| 8 | squad | 2,000 | HF | Extractive QA |
| 9 | coqa | 2,000 | HF | Conversational QA |
| 10 | bigbench_language_identification | 2,000 | eval_bundle | Language identification |
| 11 | bigbench_qa_wikidata | 2,000 | eval_bundle | Wikidata factual QA |
| 12 | openbook_qa | 2,000 | HF | Open-book science QA |
| 13 | piqa | 2,000 | HF | Physical intuition QA |
| 14 | arc_easy | 2,000 | HF | Easy science questions |
| 15 | arc_challenge | 2,000 | HF | Hard science questions |
| 16 | commonsense_qa | 2,000 | HF | Commonsense reasoning |
| 17 | agi_eval_lsat_ar | 230 | eval_bundle | LSAT analytical reasoning |
| 18 | bigbench_operators | 210 | eval_bundle | Custom operator evaluation |

### Answer-Only Tasks (3 tasks, ~2,352 docs)

Context is question+options/SFT template/symbolic; only answer tokens contribute to loss.

| # | Task | Docs | Source | Description |
|---|------|------|--------|-------------|
| 19 | bigbench_dyck_languages | 1,000 | eval_bundle | Bracket completion |
| 20 | bigbench_cs_algorithms | 1,320 | eval_bundle | CS algorithm tracing |
| 21 | bigbench_repeat_copy_logic | 32 | eval_bundle | String repetition |

## File Format

### Tokenized (.pt)

```python
{
    "token_ids":    torch.LongTensor,   # [~31547, 2048], padded with pad_token_id (0)
    "loss_mask":    torch.BoolTensor,   # [~31547, 2048], True = include in loss
    "task_labels":  list[str],          # per-doc task label
    "metadata":     dict,               # source info, tokenizer, strategy description
}
```

### Parquet

| Column | Type | Description |
|--------|------|-------------|
| `context` | string | Input text (question/passage/context) |
| `answer` | string | Correct answer/continuation |
| `task` | string | Task label (e.g., "hellaswag_zeroshot") |
| `category` | string | "full_sequence" or "answer_only" |
| `loss_strategy` | string | "full_sequence" or "answer_only" |
| `text` | string | context + answer (full text) |

## HellaSwag Data Sources

HellaSwag contains data from two sources with different formats:

| Source | % of Data | Has Tags | Example |
|--------|-----------|----------|---------|
| ActivityNet Captions | ~33% | No | `A man is sitting on a roof. He...` |
| wikiHow | ~67% | Yes | `[header] How to cook [title] Prepare ingredients...` |

v6 preprocessing unifies both formats to match eval_bundle.

## Technical Details

- **Tokenizer:** GPT-NeoX-20B (vocab 50,432)
- **Block size:** 2048 tokens
- **Cap:** 2,000 docs/task (no up-sampling for data-poor tasks)
- **Deduplication:** hellaswag_zeroshot and hellaswag (10-shot) share the same dataset; only zeroshot is kept
- **Separator:** piqa/arc_easy/arc_challenge use `\nAnswer: ` from core.yaml; openbook_qa uses default space; boolq uses `\nAnswer: ` from core.yaml

## Usage

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/validation_set/prepare_core_bmk_v6.py
```

Output files will be saved to `data/`:
- `core_bmk_21tasks_v6_tokenized.pt`
- `core_bmk_21tasks_v6.parquet`

## References

- [DCLM Benchmark](https://www.datacomp.ai/dclm/)
- [nanochat CORE metric](https://github.com/karpathy/nanochat)
- [QuadMix pipeline](https://github.com/liujin99/quadmix)
- [HellaSwag paper](https://arxiv.org/abs/1905.07830)
- [HellaSwag HuggingFace dataset](https://huggingface.co/datasets/Rowan/hellaswag)
