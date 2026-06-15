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
python scripts/runners/run_essential_web_v1.py --quick --val-set=core_bmk_v4.3
```

The pipeline automatically downloads v4.3 from HuggingFace if not present locally.

## Design Decisions

### Search Weight: Pure R²

**Formula:**
```python
weight_i = max(R²_i, 0) / Σ max(R²_j, 0)
score = Σ weight_i × (pred_i - mean_i) / std_i
```

**Why not R² × std?**

Previously used `R² × std` as search weight, but this caused problems:
- `std` and z-score denominator cancel out: `(R² × std) × (pred - mean) / std = R² × (pred - mean)`
- Returns to raw loss space where scales are incomparable
- High-std tasks (e.g., bigbench_cs_algorithms, std=0.63) dominate low-std tasks (e.g., squad, std=0.05)

**Why pure R²?**

- Downstream evaluation is equal-weight (21 benchmarks average)
- Search should only weight by "prediction accuracy" (R²), not "variance magnitude" (std)
- Z-score normalization already handles scale differences
- High R² tasks (clear signal) get more trust; low R² tasks (noisy) get less trust

### Overall R² and MAE: Z-Score Normalized, R²-Weighted

**Formula:**
```python
# For each active task i:
z_pred_i = (pred_i - mean_i) / std_i
z_actual_i = (actual_i - mean_i) / std_i

# R²-weighted ensemble:
z_ensemble = Σ(R²_i × z_i) / Σ(R²_i)

# Overall metrics:
overall_r2 = 1 - Σ(z_ensemble_actual - z_ensemble_pred)² / Σ(z_ensemble_actual - mean(z_ensemble_actual))²
overall_mae = mean(|z_ensemble_pred - z_ensemble_actual|)
```

**Why z-score normalization?**

- Different tasks have different loss scales (e.g., bigbench_cs_algorithms ~3.0, squad ~0.5)
- Raw loss averaging gives high-loss tasks more influence
- Z-score puts all tasks on comparable scale

**Why R²-weighted (not equal-weight)?**

- Equal-weight: noisy tasks (low R²) degrade overall metric quality
- R²-weighted: high R² tasks (clear signal) contribute more, low R² tasks contribute less
- Consistent with search weight logic (pure R²)
- Prevents tasks like squad (R²=0.18) from dragging down overall R²

**Why not std-weighted?**

Previously used `Σ(std_i × R²_i) / Σ(std_i)`, but:
- High std doesn't mean strong signal (could be noise)
- bigbench_cs_algorithms (std=0.63) would have 12× more weight than squad (std=0.05)
- Contradicts search weight logic (pure R²)

### K-Fold Cross-Validation for R² Estimation

**Approach:**
```python
# 5-fold CV for each task (parallel across tasks):
for task in tasks:  # parallel
    for fold in range(5):
        cv_train = 4 folds
        cv_val = 1 fold
        cv_model.fit(cv_train)
        fold_r2 = cv_model.score(cv_val)

    cv_r2 = mean(fold_r2s)  # More stable R² estimate

# Final model trained on train_idx for prediction
final_model.fit(train_idx)
```

**Why K-fold CV?**

- Single 80/20 split: Val R² estimate has high variance (only ~40 samples)
- 5-fold CV: R² estimated from 5 × 40 = 200 samples (each sample validated once)
- R² estimate variance reduced ~5×
- More reliable search weights, especially for borderline tasks (R² ≈ 0)

**Parallel Training:**

- 21 tasks trained in parallel using joblib (one process per task)
- Each task does 5-fold CV + final model training independently
- CPU utilization: ~100% (vs ~10% with sequential training)
- Speedup: ~10-15× on 16-core machines

**Trade-off:**
- Computational cost: 5× more LightGBM training per task
- But LightGBM is fast, and parallelization offsets this
- Final prediction model still trained on train_idx (consistent with search)

**Config:**
```python
QuaDMixConfig.regression_cv_folds = 5  # Set to 0 or 1 for single split
```

### Unified Logic

All three components use R² as trust indicator:

| Component | Formula | Rationale |
|-----------|---------|-----------|
| Search weight | `weight_i = R²_i / Σ R²_j` | Trust accurate predictions |
| Search score | `Σ weight_i × z_score_i` | Z-score for scale invariance |
| Overall R²/MAE | R²-weighted z-score ensemble | Consistent with search logic |

### Per-Task R² Distribution (v4.2 baseline)

| Task | Val R² | Weight | Analysis |
|------|--------|--------|----------|
| bigbench_cs_algorithms | 0.4972 | 0.2153 | Symbolic task, surprisingly strong signal |
| bigbench_language_identification | 0.6481 | 0.1002 | Multilingual, high R² |
| bigbench_operators | 0.3628 | 0.0954 | Symbolic, moderate signal |
| arc_easy | 0.5207 | 0.0615 | Science QA, good signal |
| winogrande | 0.7501 | 0.0593 | Pronoun resolution, highest R² |
| squad | 0.1779 | 0.0065 | Reading comprehension, weak signal |
| boolq | 0.2205 | 0.0075 | Reading comprehension, weak signal |
| commonsense_qa | -0.0221 | 0.0000 | Filtered (negative R²) |

**Observation:** Symbolic tasks (bigbench_*) have stronger signal than expected; reading comprehension tasks (squad, boolq) have weaker signal than expected. This is because:
- Symbolic tasks: answer-only loss focuses on high-variance tokens
- Reading comprehension: full-seq loss diluted by context (Wikipedia text is "universal" across mixtures)

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
