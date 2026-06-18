# QuadMix CORE Benchmark Validation Set v5

**Version:** v5.0  
**Tasks:** 21 (all nanochat CORE benchmarks, deduplicated)  
**Documents:** 31,547  
**Loss Strategy:** Per-task (full-seq + answer-only hybrid)

## Overview

This is the **v5** validation set for the [QuadMix](https://github.com/liujin99/quadmix) proxy model pipeline. It covers all 21 unique benchmarks from the nanochat CORE evaluation suite, using a per-task loss strategy that independently selects the optimal loss mask for each task.

### Previous Versions

| Version | Repo | Tasks | Docs | Loss Strategy |
|---------|------|-------|------|---------------|
| v3 | `liujin99/quadmix-core-22tasks` | 22 (with dup) | 46,926 | Continuation-only |
| v4 | `liujin99/quadmix-core-bmk-v4` | 10 | 12,371 | Continuation-only |
| v4.2 | `liujin99/quadmix-core-bmk-v4.2` | 21 | 27,163 | Per-task hybrid (12 full-seq, 9 answer-only) |
| v4.3 | `liujin99/quadmix-core-bmk-v4.3` | 21 | 27,163 | Per-task hybrid (15 full-seq, 6 answer-only) |
| **v5** | **`liujin99/quadmix-core-bmk-v5`** | **21** | **31,547** | **Per-task hybrid (18 full-seq, 3 answer-only)** |

## Key Changes in v5

1. **11 tasks now load from HuggingFace** (train+test+val merged) instead of eval_bundle single split — larger data pool, up to 2,000 samples per task
2. **3 tasks moved from answer-only to full-seq** — commonsense_qa, agi_eval_lsat_ar, bigbench_operators (1M model can't do these tasks, single answer token has no signal)
3. **commonsense_qa uses all 5 choices (A-E)** — eval_bundle randomly dropped 1/5 for 25% random baseline alignment; v5 keeps all 5
4. **COPA test split filtered** — 500 test samples with label=-1 are excluded (previously defaulted to choice1)
5. **Schema period-only filtering** — winograd (-18) and winogrande (-53) samples with punctuation-only continuations removed
6. **continuation_delimiter from core.yaml** — fixes jeopardy/boolq missing `\nAnswer: ` separator
7. **bigbench_language_identification answer format unified** — letter labels (A/B/C/D) instead of full text
8. **MC per-sample format optimization** — openbook_qa/piqa/arc_easy/arc_challenge dynamically select Q&A vs continuation format per sample

### Comparison with v4.3

| Metric | v4.3 | v5 | Change |
|--------|------|-----|--------|
| Full-seq tasks | 15 | 18 | +3 |
| Answer-only tasks | 6 | 3 | -3 |
| Total docs | 27,163 | 31,547 | +4,384 (+16%) |
| Non-padding tokens | 2.6M | 2.9M | +11% |
| Loss tokens (% of non-padding) | 92.3% | 96.5% | +4.2% |
| HF-loaded tasks | 0 | 11 | +11 |
| Eval-bundle tasks | 21 | 10 | -11 |

## Files

| File | Size | Description |
|------|------|-------------|
| `core_bmk_21tasks_v5_tokenized.pt` | 555 MB | Pre-tokenized PyTorch tensor (ready for proxy eval) |
| `core_bmk_21tasks_v5.parquet` | 13.1 MB | Human-readable parquet (context, answer, task label, loss mask info) |

## Task List

### Full-Sequence Tasks (18 tasks, 29,195 docs)

Context + Continuation forms natural text; all tokens contribute to loss.

| # | Task | Docs | Source | Description |
|---|------|------|--------|-------------|
| 1 | hellaswag_zeroshot | 2,000 | HF | Sentence completion |
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

### Answer-Only Tasks (3 tasks, 2,352 docs)

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
    "token_ids":    torch.LongTensor,   # [31547, 2048], padded with pad_token_id (0)
    "loss_mask":    torch.BoolTensor,   # [31547, 2048], True = include in loss
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

## Data Quality Improvements

### Per-Sample Format (MC_PER_SAMPLE_TASKS)

openbook_qa, piqa, arc_easy, arc_challenge have mixed question styles. v5 dynamically selects format per sample:

| Format | Condition | Example |
|--------|-----------|---------|
| Q&A | ends with `?` or starts with question word | `Question: ...?\nAnswer: ...` |
| Continuation | otherwise | `... answer` (natural text) |

**Question words:** `{how, what, why, which, where, when, who, can, do, does, is, are, should}`

| Task | Q&A % | Cont % |
|------|-------|--------|
| openbook_qa | 40% | 60% |
| piqa | 49% | 51% |
| arc_easy | 81% | 19% |
| arc_challenge | 85% | 15% |

### Loss Strategy Change (answer-only -> full-seq)

3 tasks moved from answer-only to full-seq because:
- 1M proxy model can't solve these tasks (commonsense reasoning, LSAT logic, custom operators)
- Single answer token has near-zero loss variance -> no signal for meta-model
- Full-seq loss on question+answer text provides context signal

| Task | v4.3 R² | v4.3 Strategy | v5 Strategy | Rationale |
|------|---------|---------------|-------------|-----------|
| commonsense_qa | 0.13 | answer-only | full-seq | 1M can't do commonsense reasoning |
| agi_eval_lsat_ar | 0.15 | answer-only | full-seq | Long passage provides signal |
| bigbench_operators | 0.12 | answer-only | full-seq | Question text has pattern |

## Technical Details

- **Tokenizer:** GPT-NeoX-20B (vocab 50,432)
- **Block size:** 2048 tokens
- **Cap:** 2,000 docs/task (no up-sampling for data-poor tasks)
- **Deduplication:** hellaswag_zeroshot and hellaswag (10-shot) share the same dataset; only zeroshot is kept
- **Separator:** piqa/arc_easy/arc_challenge/openbook_qa use per-sample format; jeopardy/boolq use `\nAnswer: ` from core.yaml

## References

- [DCLM Benchmark](https://www.datacomp.ai/dclm/)
- [nanochat CORE metric](https://github.com/karpathy/nanochat)
- [QuadMix pipeline](https://github.com/liujin99/quadmix)
