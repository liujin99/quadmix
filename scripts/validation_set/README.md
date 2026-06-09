# QuaDMix Validation Sets

This directory contains scripts for preparing validation sets used by the QuaDMix proxy model pipeline. These validation sets measure how well a given data mixture trains a small proxy model — lower validation loss indicates a better mixture.

## Overview

QuaDMix uses a 1M-parameter proxy model to rapidly evaluate different data sampling strategies. The validation set serves as the **quality signal**: we train the proxy model on sampled data, then measure its perplexity on the validation set. A lower loss means the sampled data distribution better matches the capabilities we want the model to learn.

We provide two validation sets:

| Validation Set | Purpose | Source | Docs |
|----------------|---------|--------|------|
| **OpenHermes-10k** | General instruction-following quality | OpenHermes-2.5-1M | [Below](#openhermes-10k) |
| **CORE-22tasks** | Benchmark-aligned capability | DCLM CORE benchmark | [Below](#core-22tasks-recommended) |

Both sets share the same file format and are automatically downloaded from HuggingFace when first used.

---

## File Format

All validation sets are saved as PyTorch `.pt` files containing:

```python
{
    "token_ids":    torch.LongTensor,   # [num_docs, block_size], padded with pad_token_id (0)
    "loss_mask":    torch.BoolTensor,   # [num_docs, block_size], True = include in loss
    "task_labels":  list[str],          # per-doc task label (CORE-22tasks only)
    "metadata":     dict,               # source info, tokenizer, strategy description
}
```

The proxy runner computes validation loss as:

```
val_loss = mean over docs of (sum(loss * mask) / count(mask))
```

where `mask` is shifted by 1 position to align with next-token prediction targets.

---

## OpenHermes-10k

**Script:** `prepare_openhermes_assistant_10k.py`  
**HuggingFace:** [`liujin99/quadmix-openhermes-10k`](https://huggingface.co/datasets/liujin99/quadmix-openhermes-10k)  
**File:** `openhermes_10k_assistant_tokenized.pt`

### Design

OpenHermes-10k measures **general instruction-following quality**. It samples 10,000 assistant responses from the OpenHermes-2.5-1M dataset and tokenizes them with GPT-NeoX-20B.

Since only assistant text is extracted (no system/user prompts), every non-padding token is meaningful assistant output. The loss mask marks all non-padding tokens as `True`.

### Characteristics

- **Size:** 10,000 documents × 2048 tokens
- **Loss strategy:** `assistant_only_all_tokens` — all tokens contribute to loss
- **Tokenizer:** GPT-NeoX-20B (vocab 50,432)
- **Use case:** Default validation set; good for general-purpose data mixture evaluation

### Usage

```bash
python scripts/run_essential_web_v1.py --quick --val-set=openhermes
```

---

## CORE-22tasks (Recommended)

**Script:** `prepare_core_val_set.py`  
**HuggingFace:** [`liujin99/quadmix-core-22tasks`](https://huggingface.co/datasets/liujin99/quadmix-core-22tasks)  
**File:** `core_22tasks_tokenized.pt`

### Design Rationale

The CORE-22tasks validation set is designed to align with the **CORE metric** used by [nanochat](https://github.com/karpathy/nanochat) and the [DCLM benchmark](https://www.datacomp.ai/dclm/). The CORE metric evaluates models on 22 diverse tasks spanning world knowledge, language understanding, commonsense reasoning, symbolic problem solving, and reading comprehension.

**Key insight:** The proxy model cannot actually *solve* benchmark tasks (it's only 1M parameters). Instead, we measure whether the sampled data distribution produces a model with lower perplexity on benchmark-style text. This is a **distribution matching** problem, not a task-solving problem.

**Alignment with QuaDMix paper:** The QuaDMix paper (Liu et al., 2025) defines two proxy experiment variants:
- **QuaDMix-OH**: Uses OpenHermes-10k as the validation target
- **QuaDMix-BMK**: Uses benchmark training data (HellaSwag, ARC-E, ARC-C, MMLU, TriviaQA) as the validation target

Their results show QuaDMix-BMK outperforms QuaDMix-OH on 4/5 downstream tasks (Table 3), proving that benchmark-aligned validation sets produce better data mixtures for specific capabilities. Our CORE-22tasks follows this principle, covering all 21 CORE metric tasks rather than just 5.

### Text Extraction Strategy

For each CORE task, we extract `(context, continuation)` pairs based on the task type:

| Task Type | Context | Continuation | Example Tasks |
|-----------|---------|--------------|---------------|
| `multiple_choice` | `query + delimiter` | `choices[gold]` | ARC, HellaSwag, PIQA |
| `language_modeling` | `context.strip() + delimiter` | `continuation` | LAMBADA, SQuAD, Jeopardy |
| `schema` | `context_options[gold] + delimiter` | `continuation` | Winograd, Winogrande |

The `continuation` is the correct answer or completion. For `multiple_choice` tasks where the `choices` field only contains labels (A/B/C/D), we parse the full answer text from the query string.

### Continuation Delimiter

Each task specifies a `continuation_delimiter` in `core.yaml` (typically `" "` or `"\nAnswer: "`). This delimiter is inserted between context and continuation, affecting tokenization at the boundary. We read this from the YAML config to ensure correct token boundaries.

### Loss Strategy: Full Sequence Loss

**Design choice:** The loss mask marks **all non-padding tokens** as `True`.

```
[context tokens] [continuation tokens] [padding]
      True              True            False
```

#### Evolution: Continuation-Only → Full Sequence

We initially implemented **continuation-only loss** — masking only the answer/completion tokens, matching the scoring scope of the actual CORE metric evaluation. The reasoning was that context tokens (the question) are easy to predict and would dilute the signal.

However, this created a severe token count problem:

| Strategy | Docs | Tokens | vs OpenHermes Gap | SE Ratio |
|----------|------|--------|-------------------|----------|
| Continuation-only (cap-1000) | 16,345 | 100,018 | 22.4x | 4.7x |
| Full sequence (cap-1000) | 16,345 | 1,540,450 | 1.5x | 1.2x |
| Full sequence (cap-2000) | 27,163 | 2,837,174 | 0.79x | 0.89x |

The continuation-only approach produced only ~100k tokens because benchmark answers are inherently short:

| Continuation Length | Tasks | Avg tokens/doc |
|---------------------|-------|----------------|
| Very short (<5 chars) | BoolQ (yes/no), BigBench Operators (numbers), BigBench Dyck (brackets) | ~1-2 |
| Short (5-15 chars) | LAMBADA (last word), SQuAD (extractive QA), CommonsenseQA (option text) | ~3-6 |
| Medium (15-50 chars) | ARC, PIQA, HellaSwag (full sentence answers) | ~10-30 |

Even with all available data (80,995 docs), continuation-only loss yields only ~530k tokens — still a 4.2x gap vs OpenHermes. This is a fundamental property of benchmark task design, not an extraction bug.

**Why full sequence loss is valid for our use case:**

We are comparing data mixtures (QuadMix vs Random), not measuring absolute model capability. The comparison signal is:

```
Δloss = loss_A - loss_B
      = (L_context_A + L_answer_A) - (L_context_B + L_answer_B)
      = (L_context_A - L_context_B) + (L_answer_A - L_answer_B)
```

- `L_answer_A - L_answer_B`: the signal we care about (which mixture teaches better answers)
- `L_context_A - L_context_B`: adds noise from context prediction differences, but since both groups see the same validation set, this term is small and symmetric

The additional context tokens (~15x more) reduce variance far more than the added noise increases it. This matches the QuaDMix paper's approach, which computes loss on the full benchmark sequence without distinguishing context from continuation.

### Sampling Strategy: Cap-2000

Each task samples `min(2000, available_data)` documents — no up-sampling:

- **Data-rich tasks** (e.g., HellaSwag with 10,042 docs): sample 2,000
- **Data-poor tasks** (e.g., COPA with 100 docs, bigbench_repeat_copy_logic with 32): use all available

#### Cap Selection Process

We evaluated multiple cap values to find the best tradeoff between token coverage and evaluation time:

| Cap | Docs | Tokens | vs OH Gap | SE Ratio | Eval Time | File Size |
|-----|------|--------|-----------|----------|-----------|-----------|
| 500 | 8,845 | 0.52M | 42.9x | 6.5x | 0.9x | 69 MB |
| 1000 | 16,345 | 1.54M | 1.5x | 1.2x | 1.6x | 287 MB |
| 1500 | 22,325 | 2.10M | 1.1x | 1.0x | 2.2x | 392 MB |
| **2000** | **27,163** | **2.84M** | **0.79x** | **0.89x** | **2.7x** | **478 MB** |
| 3000 | 34,656 | 3.27M | 0.68x | 0.83x | 3.5x | 547 MB |
| 5000 | 46,926 | 4.42M | 0.51x | 0.71x | 4.7x | 741 MB |
| All | 80,995 | 7.63M | 0.29x | 0.54x | 8.1x | 1.3 GB |

- **SE Ratio** = standard error of CORE loss estimate / standard error of OpenHermes loss estimate. Lower is better; 1.0x means equal precision.
- **Eval Time** = relative to OpenHermes-10k (10,000 docs). Proxy model evaluation time scales linearly with doc count.
- **vs OH Gap** = OpenHermes tokens / CORE tokens. <1.0x means CORE has more tokens.

We chose **cap-2000** because:
1. It exceeds OpenHermes in token count (2.84M vs 2.24M), giving SE ratio 0.89x — slightly better precision than OpenHermes
2. Empirically, OpenHermes-10k produces ~8% val loss difference between QuadMix and Random strategies. Matching its token count ensures CORE has comparable statistical power to detect similar effect sizes
3. The 2.7x evaluation time cost is acceptable given the improved signal-to-noise ratio

#### Why No Up-Sampling for Data-Poor Tasks?

6 tasks have fewer than 2,000 documents:

| Task | Available | Reason |
|------|-----------|--------|
| bigbench_repeat_copy_logic | 32 | Tiny benchmark by design |
| copa | 100 | Small curated dataset |
| bigbench_operators | 210 | Symbolic reasoning subset |
| agi_eval_lsat_ar | 230 | LSAT analytical reasoning |
| openbook_qa | 500 | Science questions |
| winograd | 273 | Winograd Schema Challenge |

Up-sampling (sampling with replacement) would artificially inflate their token count but not their information content — the proxy model would see the same documents multiple times, reducing effective diversity. We use all available data without replacement.

### Deduplication

The original CORE benchmark includes `hellaswag_zeroshot` (0-shot) and `hellaswag` (10-shot), which reference the same underlying dataset. Since the proxy model has no few-shot capability, these would produce identical text. We deduplicate by `dataset_uri`, keeping only the first occurrence.

**Result:** 21 unique tasks, capped at 2,000 docs/task = **27,163 documents** (exact count depends on per-task data availability)

### Task Coverage

The 21 deduplicated tasks span 5 categories:

| Category | Tasks | Count |
|----------|-------|-------|
| **World Knowledge** | Jeopardy, ARC Easy, ARC Challenge, BigBench QA Wikidata | 4 |
| **Language Understanding** | HellaSwag, LAMBADA, Winograd, Winogrande, BigBench Language ID | 5 |
| **Commonsense Reasoning** | COPA, CommonsenseQA, PIQA, OpenBookQA | 4 |
| **Symbolic Problem Solving** | BigBench Dyck, Operators, CS Algorithms, Repeat Copy Logic, AGI Eval LSAT-AR | 5 |
| **Reading Comprehension** | SQuAD, CoQA, BoolQ | 3 |

### Usage

```bash
python scripts/run_essential_web_v1.py --quick --val-set=core
```

The validation set will be automatically downloaded from `liujin99/quadmix-core-22tasks` on first use. If the download fails, the script falls back to local generation from the CORE eval bundle (requires `eval_bundle/` directory).

### Regenerating Locally

If you need to regenerate the validation set (e.g., to customize parameters):

```bash
python scripts/validation_set/prepare_core_val_set.py \
  --eval-bundle /path/to/eval_bundle \
  --num-samples-per-task 2000 \
  --block-size 2048 \
  --output-dir data
```

**Requirements:**
- CORE eval bundle (downloaded from `https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip`)
- `transformers` library (for GPT-NeoX-20B tokenizer)
- `pyyaml` library

### Comparison with OpenHermes-10k

| Aspect | OpenHermes-10k | CORE-22tasks |
|--------|----------------|--------------|
| **Focus** | General instruction quality | Benchmark-aligned capabilities |
| **Source** | OpenHermes-2.5-1M (chat data) | DCLM CORE benchmark (21 tasks) |
| **Loss mask** | All non-padding tokens | All non-padding tokens |
| **Tokens** | 2.24M | 2.84M |
| **SE ratio** | 1.0x (baseline) | 0.89x |
| **Sampling** | 10,000 docs from single source | Cap-2000 per task (no up-sampling) |
| **Size** | 10,000 docs | ~27,000 docs (21 tasks, capped at 2000) |
| **Eval time** | 1.0x | ~2.7x |
| **Best for** | General-purpose mix | Capability-targeted mixes |

---

## Technical Details

### Tokenizer

Both validation sets use the **GPT-NeoX-20B tokenizer** (vocab size 50,432), matching the proxy model's vocabulary. This is different from nanochat's internal tokenizer (32K vocab), but intentional — the proxy model is a separate, smaller model with its own vocabulary.

### Block Size

Default block size is **2048 tokens**, matching the proxy model's sequence length. Documents longer than 2048 tokens are truncated; shorter documents are padded with `pad_token_id` (0).

### Loss Computation

The proxy runner computes validation loss as follows:

```python
# For each document:
ids_in  = token_ids[:-1]          # input: positions 0..T-2
ids_tgt = token_ids[1:]           # target: positions 1..T-1
mask_tgt = loss_mask[1:]          # mask: shifted to align with targets

# Per-token cross-entropy loss
loss = cross_entropy(model(ids_in), ids_tgt)

# Masked mean per document
doc_loss = sum(loss * mask_tgt) / count(mask_tgt)

# Global mean across all documents
val_loss = mean(doc_loss)
```

The 1-position shift aligns the mask with next-token prediction targets: `mask[t]` indicates whether predicting `token_ids[t+1]` should contribute to the loss.

### Per-Task Loss Analysis

The CORE-22tasks validation set includes `task_labels` (a list of task names per document), enabling per-task loss analysis:

```python
val_data = torch.load("core_22tasks_tokenized.pt")
task_labels = val_data["task_labels"]

# Group losses by task
from collections import defaultdict
task_losses = defaultdict(list)
for i, label in enumerate(task_labels):
    task_losses[label].append(doc_losses[i])

# Per-task mean loss
for task, losses in task_losses.items():
    print(f"{task}: {mean(losses):.4f}")
```

This can help diagnose whether a data mixture improves specific capability areas.

---

## References

- **DCLM Benchmark:** [Li et al., "DataComp-LM: In search of the next generation of multimodal datasets"](https://arxiv.org/abs/2406.11580)
- **nanochat CORE metric:** [Karpathy's nanochat](https://github.com/karpathy/nanochat) — `scripts/base_eval.py` and `nanochat/core_eval.py`
- **OpenHermes-2.5:** [Teknium's OpenHermes-2.5-1M](https://huggingface.co/datasets/teknium/openhermes-2.5-1m)
- **QuaDMix:** See the main [README](../../README.md) for the full pipeline documentation
