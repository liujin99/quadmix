# QuadMix CORE Benchmark Validation Set v4.2

**Version:** v4.2  
**Tasks:** 21 (all nanochat CORE benchmarks, deduplicated)  
**Documents:** 27,163  
**Loss Strategy:** Per-task (full-seq + answer-only hybrid)

## Overview

This is the **v4.2** validation set for the [QuadMix](https://github.com/liujin99/quadmix) proxy model pipeline. It covers all 21 unique benchmarks from the nanochat CORE evaluation suite, using a per-task loss strategy that independently selects the optimal loss mask for each task.

### Previous Versions

| Version | Repo | Tasks | Docs | Loss Strategy |
|---------|------|-------|------|---------------|
| v3 | `liujin99/quadmix-core-22tasks` | 22 (with dup) | 46,926 | Continuation-only |
| v4 | `liujin99/quadmix-core-bmk-v4` | 10 | 12,371 | Continuation-only |
| **v4.2** | **`liujin99/quadmix-core-bmk-v4.2`** | **21** | **27,163** | **Per-task hybrid** |

## Key Changes in v4.2

1. **All 21 CORE benchmarks included** — no pre-filtering; per-task zero-variance detection automatically excludes non-discriminative tasks
2. **Per-task loss strategy** — each task independently classified as full-seq or answer-only based on whether context+continuation forms natural text
3. **Per-task weighted prediction** — optimizer uses `weight_i = std_i / sum(std_j)` for per-task loss aggregation
4. **SFT template cleaning** — removes instruction prefixes (e.g., "Passage:", "Context:") while preserving question/option content

## Files

| File | Size | Description |
|------|------|-------------|
| `core_bmk_21tasks_v4.2_tokenized.pt` | 478 MB | Pre-tokenized PyTorch tensor (ready for proxy eval) |
| `core_bmk_21tasks_v4.2.parquet` | 12.3 MB | Human-readable parquet (context, answer, task label, loss mask info) |

## Task List

### Full-Sequence Tasks (12 tasks, 18,140 docs)

Context + continuation forms natural text; all tokens contribute to loss.

| # | Task | Docs | Description |
|---|------|------|-------------|
| 1 | hellaswag_zeroshot | 2,000 | Sentence completion |
| 2 | lambada_openai | 2,000 | Last-word prediction |
| 3 | winogrande | 1,267 | Pronoun resolution |
| 4 | winograd | 273 | Winograd Schema Challenge |
| 5 | copa | 100 | Causal reasoning |
| 6 | jeopardy | 2,000 | Trivia clues |
| 7 | boolq | 2,000 | Yes/no reading comprehension |
| 8 | squad | 2,000 | Extractive QA |
| 9 | coqa | 2,000 | Conversational QA |
| 10 | bigbench_language_identification | 2,000 | Language identification |
| 11 | bigbench_qa_wikidata | 2,000 | Wikidata factual QA |
| 12 | openbook_qa | 500 | Open-book science QA |

### Answer-Only Tasks (9 tasks, 9,023 docs)

Context is question+options/SFT template/symbolic; only answer tokens contribute to loss.

| # | Task | Docs | Description |
|---|------|------|-------------|
| 13 | piqa | 1,838 | Physical intuition QA |
| 14 | arc_easy | 2,000 | Easy science questions |
| 15 | arc_challenge | 1,172 | Hard science questions |
| 16 | commonsense_qa | 1,221 | Commonsense reasoning |
| 17 | agi_eval_lsat_ar | 230 | LSAT analytical reasoning |
| 18 | bigbench_dyck_languages | 1,000 | Bracket completion |
| 19 | bigbench_cs_algorithms | 1,320 | CS algorithm tracing |
| 20 | bigbench_operators | 210 | Custom operator evaluation |
| 21 | bigbench_repeat_copy_logic | 32 | String repetition |

## File Format

### Tokenized (.pt)

```python
{
    "token_ids":    torch.LongTensor,   # [27163, 2048], padded with pad_token_id (0)
    "loss_mask":    torch.BoolTensor,   # [27163, 2048], True = include in loss
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
| `loss_type` | string | "full_sequence" or "answer_only" |
| `n_context_tokens` | int32 | Token count of context |
| `n_answer_tokens` | int32 | Token count of answer |

## Usage

```bash
python scripts/runners/run_essential_web_v1.py --quick --val-set=core-v4.2
```

## Technical Details

- **Tokenizer:** GPT-NeoX-20B (vocab 50,432)
- **Block size:** 2048 tokens
- **Cap:** 2,000 docs/task (no up-sampling for data-poor tasks)
- **Deduplication:** hellaswag_zeroshot and hellaswag (10-shot) share the same dataset; only zeroshot is kept

## References

- [DCLM Benchmark](https://www.datacomp.ai/dclm/)
- [nanochat CORE metric](https://github.com/karpathy/nanochat)
- [QuadMix pipeline](https://github.com/liujin99/quadmix)
