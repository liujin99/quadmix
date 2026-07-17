---
license: other
task_categories:
  - question-answering
  - text-generation
language:
  - en
tags:
  - stem
  - mathematics
  - science
  - mmlu
  - gsm8k
  - arc
  - proxy-validation
  - data-mixing
size_categories:
  - 10K-100K
---

# QuaDMix-STEM v1: STEM-Focused Proxy Validation Set

**Script:** `scripts/validation_set/prepare_stem_v1.py`
**HuggingFace:** [`liujin99/quadmix-stem-v1`](https://huggingface.co/datasets/liujin99/quadmix-stem-v1)
**Files:** `stem_v1_tokenized.pt`, `stem_v1.parquet`

## Overview

STEM v1 is a validation set designed to focus the proxy model's optimization signal on **STEM capabilities** — mathematics, science knowledge, and logical reasoning. Unlike CAP v1 (broad capability coverage) or core_bmk (benchmark test format), STEM v1 uses only tasks that **directly correspond** to downstream evaluation benchmarks, ensuring maximum signal transfer certainty.

### Key Design Principle: Direct Downstream Correspondence

Every task in STEM v1 maps directly to a downstream benchmark:

| Validation Task | Downstream Benchmark(s) | Signal Path |
|----------------|------------------------|-------------|
| GSM8K | gsm8k_cot, math_cot | Math reasoning → Math reasoning |
| MMLU (22 STEM) | mmlu_stem, gpqa_diamond | Science knowledge → Science knowledge |
| ARC-Easy | arc_easy | Basic science → Basic science |
| ARC-Challenge | arc_challenge | Advanced science → Advanced science |

No task is included without a direct downstream counterpart. Tasks like `bigbench_operators`, `openbook_qa`, or `unit_conversion` were deliberately excluded because their signal transfer to downstream STEM benchmarks is uncertain.

## Tasks

### 1. GSM8K — Grade School Math (5,000 samples)

Math word problems with step-by-step solutions. Sampled from 7,473 training examples.

**Format:** `Question: {question}\nSolution: {answer}`

The solution includes the full reasoning chain (not just the final number), providing rich math reasoning signal.

### 2. MMLU — 22 STEM Subjects (2,762 samples)

Multiple-choice questions across 22 STEM disciplines, using the **test split** (standard evaluation data). Subjects aligned with downstream mmlu_stem 0-shot evaluation:

| Category | Subjects |
|----------|---------|
| Mathematics | abstract_algebra, college_mathematics, elementary_mathematics, high_school_mathematics, high_school_statistics, formal_logic |
| Physics | college_physics, conceptual_physics, high_school_physics, electrical_engineering |
| Computer Science | college_computer_science, computer_security, high_school_computer_science, machine_learning |
| Biology | anatomy, college_biology, high_school_biology, medical_genetics, virology |
| Chemistry | college_chemistry, high_school_chemistry, astronomy |

**Format:** `Question: {question}\nChoices:\n  A. {c1}\n  B. {c2}\n  C. {c3}\n  D. {c4}\nAnswer: {letter}`

Answers use **letter format** (A/B/C/D), matching the original data format. Under full-sequence loss, the answer token is ~5% of total loss, so format choice has minimal signal impact.

### 3. ARC-Easy — Basic Science (2,251 samples)

Grade-school science questions from AllenAI's ARC corpus. All available training data used.

**Format:** `Question: {question}\nChoices:\n  {label}. {text}\nAnswer: {key}`

### 4. ARC-Challenge — Advanced Science (1,119 samples)

Harder science questions requiring deeper reasoning. All available training data used.

**Format:** Same as ARC-Easy.

## Data Sources

| Task | HuggingFace Source | Split | Official |
|------|-------------------|-------|----------|
| GSM8K | `openai/gsm8k` | train | OpenAI official |
| MMLU | `cais/mmlu` | test (per subject) | CAIS mirror (content = Hendrycks original) |
| ARC-Easy | `allenai/ai2_arc` (ARC-Easy) | train | AllenAI official |
| ARC-Challenge | `allenai/ai2_arc` (ARC-Challenge) | train | AllenAI official |

## Loss Strategy: Full-Sequence

All tasks use **full-sequence loss** — every non-padding token contributes to the loss.

### Why Not Answer-Only?

| Task | Answer Tokens | Answer-Only Problem |
|------|-------------|-------------------|
| GSM8K | ~200-500 chars | Not applicable (long reasoning chains) |
| MMLU | 1 token (A/B/C/D) | Only 1 token → extreme noise |
| ARC | 1 token (A/B/C/D) | Only 1 token → extreme noise |

Full-sequence loss captures context signal (question + choices), which is essential for multiple-choice tasks where the answer alone provides insufficient gradient.

### Token Statistics

```
Total tokens:            22,798,336 (11,132 × 2,048)
Non-padding tokens:      1,249,585 (5.5%)
Loss tokens (full-seq):  1,249,585 (100% of non-padding)
Truncated (>2048):       0/11,132 (0.0%)
```

## File Format

```python
{
    "token_ids":    torch.LongTensor,   # [11132, 2048], padded with pad_token_id (0)
    "loss_mask":    torch.BoolTensor,   # [11132, 2048], True = include in loss
    "task_labels":  list[str],          # per-doc task label ("gsm8k"/"mmlu"/"arc_easy"/"arc_challenge")
    "metadata":     dict,               # source info, tokenizer, task details, subject list
}
```

### Metadata Structure

```python
{
    "num_docs": 11132,
    "block_size": 2048,
    "tokenizer": "gpt-neox-20b",
    "tokenizer_vocab": 50254,
    "model_vocab": 50432,
    "loss_strategy": "full_sequence (all tasks)",
    "seed": 42,
    "num_samples_per_task": 5000,
    "downstream_benchmarks": [
        "arc_easy", "arc_challenge",
        "mmlu_stem (0-shot, 22 subjects)",
        "gpqa_diamond", "gsm8k_cot", "math_cot",
    ],
    "tasks": {
        "gsm8k": {"loaded": 7473, "sampled": 5000, ...},
        "mmlu": {"loaded": 2762, "sampled": 2762, ...},
        "arc_easy": {"loaded": 2251, "sampled": 2251, ...},
        "arc_challenge": {"loaded": 1119, "sampled": 1119, ...},
    },
    "mmlu_subjects": ["abstract_algebra", "anatomy", ...],  # 22 subjects
}
```

## Sampling Strategy

- **GSM8K**: 5,000 sampled from 7,473 (random, seed=42)
- **MMLU**: All 2,762 test samples used (22 subjects, ~100-270 per subject)
- **ARC-Easy**: All 2,251 training samples used
- **ARC-Challenge**: All 1,119 training samples used

Tasks with fewer than 5,000 available samples use all data without oversampling.

## Usage

```bash
python scripts/runners/run_essential_web_v1.py --quick --val-set=stem_v1
```

The validation set will be automatically downloaded from `liujin99/quadmix-stem-v1` on first use.

## Regenerating Locally

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/validation_set/prepare_stem_v1.py
```

**Requirements:**
- `datasets` library (for HuggingFace dataset loading)
- `transformers` library (for GPT-NeoX-20B tokenizer)
- `torch` library
- Internet access (downloads ~200MB of datasets on first run)

## Comparison with Other Validation Sets

| Aspect | OpenHermes-10k | CORE-22tasks | CAP v1 | **STEM v1** |
|--------|----------------|--------------|--------|------------|
| Focus | General chat quality | Broad benchmark capabilities | Proven training data | **STEM-specific, direct correspondence** |
| Source | OpenHermes-2.5 (chat) | DCLM CORE benchmark test | External training + benchmark train | **Benchmark data only** |
| Loss | Full-seq | Mixed (continuation-only for MC) | Full-seq | **Full-seq** |
| Signal tokens | 2.24M | 318K (5.2%) | 15.96M | **1.25M** |
| Docs | 10,000 | 46,926 | 40,000 | **11,132** |
| Tasks | 1 | 21 | 5 clusters | **4 tasks** |
| Downstream alignment | Indirect | Partial | Partial | **Direct (name-matched)** |

## Coverage Gap Analysis

### Direct Coverage (4/6 downstream benchmarks)

- **arc_easy** ← ARC-Easy
- **arc_challenge** ← ARC-Challenge
- **mmlu_stem** ← MMLU (22 STEM subjects)
- **gsm8k_cot** ← GSM8K

### Indirect Coverage (2/6 downstream benchmarks)

- **gpqa_diamond** ← MMLU (science knowledge signal, but difficulty gap: MMLU is college/HS level, gpqa_diamond is graduate level)
- **math_cot** ← GSM8K (math reasoning signal, but difficulty gap: GSM8K is grade school, math_cot is competition level)

The difficulty gap for gpqa_diamond and math_cot is inherent — a 1M-token proxy cannot learn graduate-level or competition-level content. MMLU and GSM8K provide the closest available signals. Experimental validation is needed to confirm transfer effectiveness.

## Excluded Tasks

| Task | Available Samples | Reason for Exclusion |
|------|-----------------|---------------------|
| bigbench_operators | 168 | Too few samples, no direct downstream benchmark |
| bigbench_elementary_math_qa | 30,531 | No direct downstream benchmark |
| bigbench_arithmetic | 12,019 | Too simple, low search discrimination |
| bigbench_unit_conversion | 19,151 | Pure memorization, not reasoning |
| bigbench_periodic_elements | 524 | Few samples, overlaps with MMLU/ARC |
| openbook_qa | 4,957 | Not pure STEM (commonsense reasoning) |
| qa_wikidata | ~54 | Not STEM (fact lookup), too few samples |
| MATH (competition math) | ~7,500 | 1M proxy cannot learn competition-level content |

## Search Strategy

Per-task LightGBM with R²-weighted z-score search:

```python
tasks = ["gsm8k", "mmlu", "arc_easy", "arc_challenge"]
weighted_z_score = Σ R²_i * (loss_i - mean_loss_i) / std_loss_i
# Tasks with R² ≤ 0 are automatically filtered
```

Expected search direction:
- GSM8K signal → prefer math reasoning-intensive domains
- MMLU signal → prefer science knowledge domains
- ARC signal → prefer science understanding domains

## Technical Details

### Tokenizer

GPT-NeoX-20B tokenizer (vocab size 50,254), matching the proxy model's vocabulary.

### Block Size

2048 tokens, matching the proxy model's sequence length.

### Loss Computation

```python
ids_in  = token_ids[:-1]
ids_tgt = token_ids[1:]
mask_tgt = loss_mask[1:]

loss = cross_entropy(model(ids_in), ids_tgt)
doc_loss = sum(loss * mask_tgt) / count(mask_tgt)
val_loss = mean(doc_loss)
```

## References

- **GSM8K**: [Cobbe et al., "Training Verifiers to Solve Math Word Problems"](https://arxiv.org/abs/2110.14168)
- **MMLU**: [Hendrycks et al., "Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300)
- **ARC**: [Clark et al., "ARC: A New Challenge Dataset for AI"](https://arxiv.org/abs/1803.04449)
- **DCLM Benchmark**: [Li et al., "DataComp-LM: In search of the next generation of multimodal datasets"](https://arxiv.org/abs/2406.11580)

## License

This dataset is released under the same license as the source datasets. Please check individual dataset licenses before commercial use.
