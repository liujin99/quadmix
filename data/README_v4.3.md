# QuadMix CORE Benchmark Validation Set v4.3

**Version:** v4.3  
**Tasks:** 21 (all nanochat CORE benchmarks, deduplicated)  
**Documents:** 27,163  
**Loss Strategy:** Per-task (full-seq + answer-only hybrid)

## Overview

This is the **v4.3** validation set for the [QuadMix](https://github.com/liujin99/quadmix) proxy model pipeline. It covers all 21 unique benchmarks from the nanochat CORE evaluation suite, using a per-task loss strategy that independently selects the optimal loss mask for each task.

### Previous Versions

| Version | Repo | Tasks | Docs | Loss Strategy |
|---------|------|-------|------|---------------|
| v3 | `liujin99/quadmix-core-22tasks` | 22 (with dup) | 46,926 | Continuation-only |
| v4 | `liujin99/quadmix-core-bmk-v4` | 10 | 12,371 | Continuation-only |
| v4.2 | `liujin99/quadmix-core-bmk-v4.2` | 21 | 27,163 | Per-task hybrid (12 full-seq, 9 answer-only) |
| **v4.3** | **`liujin99/quadmix-core-bmk-v4.3`** | **21** | **27,163** | **Per-task hybrid (15 full-seq, 6 answer-only)** |

## Key Changes in v4.3

1. **piqa, arc_easy, arc_challenge moved to full-sequence** — these MC tasks have full-text answers that form natural Q&A pairs with the question
2. **Natural Q&A separator** — piqa/arc_easy/arc_challenge now use `\nAnswer: ` separator instead of direct concatenation
3. **Improved signal quality** — full-sequence loss on these 3 tasks increases loss tokens from ~5% to ~92% of non-padding tokens
4. **Better proxy model learning** — longer loss sequences provide stronger training signal for the 1M proxy model

### Comparison with v4.2

| Metric | v4.2 | v4.3 | Change |
|--------|------|------|--------|
| Full-seq tasks | 12 | 15 | +3 |
| Answer-only tasks | 9 | 6 | -3 |
| Full-seq docs | 18,140 | 23,150 | +5,010 |
| Answer-only docs | 9,023 | 4,013 | -5,010 |
| Loss tokens (% of non-padding) | ~30% | 92.3% | +62% |

## Files

| File | Size | Description |
|------|------|-------------|
| `core_bmk_21tasks_v4.3_tokenized.pt` | 478 MB | Pre-tokenized PyTorch tensor (ready for proxy eval) |
| `core_bmk_21tasks_v4.3.parquet` | 12.3 MB | Human-readable parquet (context, answer, task label, loss mask info) |

## Task List

### Full-Sequence Tasks (15 tasks, 23,150 docs)

Context + Continuation forms natural text; all tokens contribute to loss.

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
| 13 | piqa | 1,838 | Physical intuition QA |
| 14 | arc_easy | 2,000 | Easy science questions |
| 15 | arc_challenge | 1,172 | Hard science questions |

### Answer-Only Tasks (6 tasks, 4,013 docs)

Context is question+options/SFT template/symbolic; only answer tokens contribute to loss.

| # | Task | Docs | Description |
|---|------|------|-------------|
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
| `category` | string | "full_sequence" or "answer_only" |
| `loss_strategy` | string | "full_sequence" or "answer_only" |
| `text` | string | context + answer (full text) |

## Example: piqa (v4.3 full-sequence)

```
Question: What is the best way to feel at least a bit happier every day?
Answer: The best way to feel a little bit better every day is to do something nice for someone else.
```

All tokens (question + answer) contribute to loss, providing stronger training signal.

## Example: commonsense_qa (answer-only, unchanged)

```
Question: Where would you find magazines along side many other printed works?
Choices:
A. doctor
B. bookstore
C. train station
D. mortuary
Answer:
```

Only the answer token ("B") contributes to loss.

## Usage

```bash
python scripts/runners/run_essential_web_v1.py --quick --val-set=core_bmk_v4.2
```

Note: The pipeline currently uses `core_bmk_v4.2` as the val-set identifier. Update the pipeline code to support `core_bmk_v4.3` if needed.

## Technical Details

- **Tokenizer:** GPT-NeoX-20B (vocab 50,432)
- **Block size:** 2048 tokens
- **Cap:** 2,000 docs/task (no up-sampling for data-poor tasks)
- **Deduplication:** hellaswag_zeroshot and hellaswag (10-shot) share the same dataset; only zeroshot is kept
- **Separator:** piqa/arc_easy/arc_challenge use `\nAnswer: ` for natural Q&A format

## References

- [DCLM Benchmark](https://www.datacomp.ai/dclm/)
- [nanochat CORE metric](https://github.com/karpathy/nanochat)
- [QuadMix pipeline](https://github.com/liujin99/quadmix)
