# QuaDMix Validation Sets

This directory contains scripts for preparing validation sets used by the QuaDMix proxy model pipeline. These validation sets measure how well a given data mixture trains a small proxy model — lower validation loss indicates a better mixture.

## Overview

QuaDMix uses a 1M-parameter proxy model to rapidly evaluate different data sampling strategies. The validation set serves as the **quality signal**: we train the proxy model on sampled data, then measure its perplexity on the validation set. A lower loss means the sampled data distribution better matches the capabilities we want the model to learn.

We provide two validation sets:

| Validation Set | Purpose | Source | Docs | Loss Strategy |
|----------------|---------|--------|------|---------------|
| **OpenHermes-10k** | General instruction-following quality | OpenHermes-2.5-1M | [Below](#openhermes-10k) | Full sequence |
| **CORE-22tasks** | Benchmark-aligned capability | DCLM CORE benchmark | [Below](#core-22tasks-recommended) | Continuation-only |

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
python scripts/runners/run_essential_web_v1.py --quick --val-set=openhermes
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

### Loss Strategy: Continuation-Only Loss

**Design choice:** The loss mask marks **only continuation tokens** as `True`.

```
[context tokens] [continuation tokens] [padding]
      False             True            False
```

#### Why Continuation-Only Loss

We initially implemented **full sequence loss** — masking all non-padding tokens (context + continuation), following the QuaDMix paper's approach. However, empirical testing revealed a critical problem: **16 proxy model experiments showed only std=0.0626 loss variance**, indicating the validation set could not discriminate between different data mixtures.

**Root cause analysis:**

| Component | Tokens | % of Total | Signal Quality |
|-----------|--------|------------|----------------|
| Context (questions) | 5,848,442 | 94.8% | Near-zero (identical across experiments) |
| Continuation (answers) | 317,561 | 5.2% | High (varies by data mixture quality) |

The problem: 94.8% of the loss came from predicting context tokens (benchmark questions), which are identical regardless of training data. The actual signal — how well the model predicts answers — was drowned out by this noise.

**Why continuation-only loss works:**

```
Δloss = loss_A - loss_B
      = L_answer_A - L_answer_B  (context masked out)
```

By masking context tokens, we eliminate the noise term entirely. The tradeoff: fewer total tokens (317k vs 2.84M), but **100% signal purity**.

**Compensation strategy:**

To offset the reduced token count, we increased the cap from 2000 to **5000 docs/task**, yielding 46,926 documents. While continuation tokens per doc are few (avg 6.8), the large doc count provides sufficient statistical power through the standard error formula:

```
SE = σ / √n
```

With n=46,926 independent documents, the SE is small enough to detect meaningful differences between data mixtures.

### Sampling Strategy: Cap-5000

Each task samples `min(5000, available_data)` documents — no up-sampling:

- **Data-rich tasks** (e.g., HellaSwag with 10,042 docs): sample 5,000
- **Data-poor tasks** (e.g., COPA with 100 docs, bigbench_repeat_copy_logic with 32): use all available

#### Cap Selection Process

With continuation-only loss, each document contributes few signal tokens (avg 6.8 continuation tokens). We need more documents to compensate:

| Cap | Docs | Cont Tokens | Eval Time | File Size |
|-----|------|-------------|-----------|-----------|
| 1000 | 16,345 | ~110k | 1.6x | 287 MB |
| 2000 | 27,163 | ~185k | 2.7x | 478 MB |
| **5000** | **46,926** | **318k** | **4.7x** | **825 MB** |
| All | 80,995 | ~530k | 8.1x | 1.3 GB |

- **Cont Tokens** = total continuation tokens (loss_mask=True). This is the actual signal.
- **Eval Time** = relative to OpenHermes-10k (10,000 docs).
- **File Size** = size of the tokenized .pt file.

We chose **cap-5000** because:
1. Continuation-only loss requires more documents to achieve statistical power (fewer tokens per doc)
2. 46,926 docs provides ~4.7x evaluation time vs OpenHermes, acceptable for the improved signal purity
3. All 21 tasks are retained (no filtering), preserving capability diversity

#### Why No Up-Sampling for Data-Poor Tasks?

Several tasks have fewer than 5,000 documents:

| Task | Available | Reason |
|------|-----------|--------|
| bigbench_repeat_copy_logic | 32 | Tiny benchmark by design |
| copa | 100 | Small curated dataset |
| bigbench_operators | 210 | Symbolic reasoning subset |
| agi_eval_lsat_ar | 230 | LSAT analytical reasoning |
| openbook_qa | 500 | Science questions |
| winograd | 273 | Winograd Schema Challenge |
| bigbench_dyck_languages | 1,000 | Symbolic bracket completion |
| arc_challenge | 1,172 | Hard science questions |
| commonsense_qa | 1,221 | Commonsense reasoning |
| winogrande | 1,267 | Pronoun resolution |
| piqa | 1,838 | Physical intuition QA |
| jeopardy | 2,117 | Trivia questions |
| arc_easy | 2,376 | Easy science questions |
| boolq | 3,270 | Yes/no reading comprehension |

Up-sampling (sampling with replacement) would artificially inflate their token count but not their information content — the proxy model would see the same documents multiple times, reducing effective diversity. We use all available data without replacement.

### Deduplication

The original CORE benchmark includes `hellaswag_zeroshot` (0-shot) and `hellaswag` (10-shot), which reference the same underlying dataset. Since the proxy model has no few-shot capability, these would produce identical text. We deduplicate by `dataset_uri`, keeping only the first occurrence.

**Result:** 21 unique tasks, capped at 5,000 docs/task = **46,926 documents** (exact count depends on per-task data availability)

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
python scripts/runners/run_essential_web_v1.py --quick --val-set=core
```

The validation set will be automatically downloaded from `liujin99/quadmix-core-22tasks` on first use. If the download fails, the script falls back to local generation from the CORE eval bundle (requires `eval_bundle/` directory).

### Regenerating Locally

If you need to regenerate the validation set (e.g., to customize parameters):

```bash
python scripts/validation_set/prepare_core_val_set.py \
  --eval-bundle /path/to/eval_bundle \
  --num-samples-per-task 5000 \
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
| **Loss mask** | All non-padding tokens | Continuation tokens only |
| **Signal tokens** | 2.24M (100% of non-padding) | 318k (5.2% of non-padding) |
| **Docs** | 10,000 | 46,926 |
| **Eval time** | 1.0x | ~4.7x |
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
