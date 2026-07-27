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
  - gpqa
  - math
  - proxy-validation
  - data-mixing
size_categories:
  - 10K-100K
---

# QuaDMix-STEM v2: STEM-Focused Proxy Validation Set with GPQA & MATH

**Script:** `scripts/validation_set/prepare_stem_v2.py`
**HuggingFace:** [`liujin99/quadmix-stem-v2`](https://huggingface.co/datasets/liujin99/quadmix-stem-v2)
**Files:** `stem_v2_tokenized.pt`, `stem_v2.parquet`

## Overview

STEM v2 is an upgraded validation set that fixes the two critical coverage gaps in STEM v1. In the v1 experiment, QuaDMix lost to Random downstream (CORE 0.1530 vs 0.1615), and root-cause analysis revealed:

1. **gpqa_diamond had no direct proxy** — mapped from MMLU (R²=0.105, only 11% search weight), causing a -0.0606 normalized gap that decided the entire experiment
2. **gsm8k dominated 62% of search weight** — but gsm8k_cot accuracy ≈ 0 at 0.5x scale, meaning the proxy signal was poorly aligned with downstream

STEM v2 adds **GPQA** (graduate-level science) and **Hendrycks MATH** (competition-level math) as direct proxy tasks, ensuring all 6 downstream benchmarks have direct signal coverage.

### Key Change: v1 → v2

| Aspect | STEM v1 | STEM v2 |
|--------|---------|---------|
| Tasks | 4 (gsm8k, mmlu, arc_easy, arc_challenge) | **6** (+ gpqa, math) |
| gpqa_diamond coverage | Indirect (via MMLU, R²=0.105) | **Direct** (gpqa_main, 448 questions) |
| math_cot coverage | Indirect (via gsm8k) | **Direct** (hendrycks_math, 5000 questions) |
| Total docs | 11,132 | **16,455** |
| Non-padding tokens | 1,249,585 | **2,788,917** (2.23x) |
| GPQA token share | 0% (not included) | ~110K (~4.0%) |
| MATH token share | 0% (excluded) | ~480K (~17.2%) |

## Tasks

### 1. GSM8K — Grade School Math (5,000 samples)

Math word problems with step-by-step solutions. Sampled from 7,473 training examples.

**Format:** `Question: {question}\nSolution: {answer}`

The solution includes the full reasoning chain (not just the final number), providing rich math reasoning signal.

### 2. MMLU — 22 STEM Subjects (2,637 samples)

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

> Note: MMLU astronomy subject failed to download (connection reset), resulting in 2,637 instead of the expected 2,720 samples (~83 questions lost, 3% of MMLU).

### 3. ARC-Easy — Basic Science (2,251 samples)

Grade-school science questions from AllenAI's ARC corpus. All available training data used.

**Format:** `Question: {question}\nChoices:\n  {label}. {text}\nAnswer: {key}`

### 4. ARC-Challenge — Advanced Science (1,119 samples)

Harder science questions requiring deeper reasoning. All available training data used.

**Format:** Same as ARC-Easy.

### 5. GPQA — Graduate-Level Science (448 samples, NEW in v2)

GPQA (Google-Proof Q&A) is a benchmark of biology, physics, chemistry, and math questions at expert (PhD) level. We use the **gpqa_main** config (448 questions), which includes all 198 diamond questions plus 250 additional questions.

**Why GPQA Main instead of Diamond?**
- Diamond (198 questions) generates only ~49K tokens (1.1% of total) — too small for reliable loss estimation
- Main (448 questions) generates ~110K tokens (4.0% of total) — 2.3x improvement
- Diamond ⊂ Main: every diamond question is included in main
- Main questions have the same format and quality as diamond

**Format:** Multiple-choice (A/B/C/D), choices shuffled with seed=42 to avoid answer position bias:
```
Question: {question}
Choices:
  A. {choice_1}
  B. {choice_2}
  C. {choice_3}
  D. {choice_4}
Answer: {correct_letter}
```

Choices are shuffled from the original format (which stores 1 correct + 3 incorrect answers separately).

### 6. MATH — Competition Mathematics (5,000 samples, NEW in v2)

Hendrycks MATH dataset containing competition-level mathematics problems from AMC, AIME, and other competitions. Covers 7 subject areas:

| Subject | Description |
|---------|-------------|
| algebra | Algebraic manipulations, equations, inequalities |
| counting_and_probability | Combinatorics, probability theory |
| geometry | Euclidean geometry, coordinate geometry |
| intermediate_algebra | Advanced algebra (polynomials, sequences) |
| number_theory | Divisibility, primes, modular arithmetic |
| prealgebra | Basic algebra, fractions, percentages |
| precalculus | Functions, trigonometry, complex numbers |

**Source:** `EleutherAI/hendrycks_math` (MIT license). The official `hendrycks/competition_math` was unreachable on HuggingFace Hub (connection reset). EleutherAI's mirror contains identical data.

**Sampling:** 5,000 sampled from 7,500 training examples (seed=42). The train split has **0% overlap** with downstream `math_cot` evaluation (which uses the MATH-500 test subset).

**Format:** `Question: {problem}\nSolution: {solution}`

The solution includes the full LaTeX-formatted proof/reasoning chain, providing rich competition math signal.

## Data Sources

| Task | HuggingFace Source | Split | License | Official |
|------|-------------------|-------|---------|----------|
| GSM8K | `openai/gsm8k` | train | MIT | OpenAI official |
| MMLU | `cais/mmlu` | test (per subject) | MIT | CAIS mirror (content = Hendrycks original) |
| ARC-Easy | `allenai/ai2_arc` (ARC-Easy) | train | CC-BY 4.0 | AllenAI official |
| ARC-Challenge | `allenai/ai2_arc` (ARC-Challenge) | train | CC-BY 4.0 | AllenAI official |
| GPQA | `Idavidrein/gpqa` | gpqa_main | CC-BY 4.0 | David Rein (official) |
| MATH | `EleutherAI/hendrycks_math` | train (7 subjects) | MIT | EleutherAI mirror (= Hendrycks original) |

## Loss Strategy: Full-Sequence

All tasks use **full-sequence loss** — every non-padding token contributes to the loss.

### Why Not Answer-Only?

| Task | Answer Tokens | Answer-Only Problem |
|------|-------------|-------------------|
| GSM8K | ~200-500 chars | Not applicable (long reasoning chains) |
| MMLU | 1 token (A/B/C/D) | Only 1 token → extreme noise |
| ARC | 1 token (A/B/C/D) | Only 1 token → extreme noise |
| GPQA | 1 token (A/B/C/D) | Only 1 token → extreme noise |
| MATH | ~500-2000 chars | Not applicable (long proof chains) |

Full-sequence loss captures context signal (question + choices), which is essential for multiple-choice tasks where the answer alone provides insufficient gradient.

### Token Statistics

```
Total tokens:            33,693,440 (16,455 × 2,048)
Non-padding tokens:       2,788,917 (8.3%)
Loss tokens (full-seq):   2,788,917 (100% of non-padding)
Truncated (>2048):       7/16,455 (0.0%)

Per-task breakdown:
  gsm8k:          5,000 docs
  mmlu:           2,637 docs
  arc_easy:       2,251 docs
  arc_challenge:  1,119 docs
  gpqa:             448 docs
  math:           5,000 docs
```

## File Format

```python
{
    "token_ids":    torch.LongTensor,   # [16455, 2048], padded with pad_token_id (0)
    "loss_mask":    torch.BoolTensor,   # [16455, 2048], True = include in loss
    "task_labels":  list[str],          # per-doc task label ("gsm8k"/"mmlu"/"arc_easy"/"arc_challenge"/"gpqa"/"math")
    "metadata":     dict,               # source info, tokenizer, task details, subject list
}
```

### Metadata Structure

```python
{
    "num_docs": 16455,
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
        "gsm8k":         {"loaded": 7473, "sampled": 5000, ...},
        "mmlu":          {"loaded": 2637, "sampled": 2637, ...},
        "arc_easy":      {"loaded": 2251, "sampled": 2251, ...},
        "arc_challenge": {"loaded": 1119, "sampled": 1119, ...},
        "gpqa":          {"loaded": 448,  "sampled": 448,  ...},
        "math":          {"loaded": 7500, "sampled": 5000, ...},
    },
    "mmlu_subjects": ["abstract_algebra", "anatomy", ...],  # 22 subjects
    "changes_from_v1": "v2 adds gpqa (gpqa_main 448) and math (hendrycks_math train 5000) ...",
}
```

## Sampling Strategy

- **GSM8K**: 5,000 sampled from 7,473 (random, seed=42)
- **MMLU**: All 2,637 test samples used (22 subjects, ~100-270 per subject)
- **ARC-Easy**: All 2,251 training samples used
- **ARC-Challenge**: All 1,119 training samples used
- **GPQA**: All 448 main questions used (no sampling)
- **MATH**: 5,000 sampled from 7,500 train examples (random, seed=42, 7 subjects)

Tasks with fewer than 5,000 available samples use all data without oversampling.

## Usage

```bash
# In demo_revalidate.sh (default val-set is stem_v2):
bash scripts/demo_revalidate.sh --result-dir result/demo_stem_xxx

# In demo_run_stem.sh (default val-set is stem_v2):
bash scripts/demo_run_stem.sh

# Or specify explicitly:
bash scripts/demo_run_stem.sh --val-set stem_v2
```

The validation set will be automatically downloaded from `liujin99/quadmix-stem-v2` on first use.

## Regenerating Locally

```bash
HF_ENDPOINT=https://hf-mirror.com python3 scripts/validation_set/prepare_stem_v2.py
```

**Requirements:**
- `datasets` library (for HuggingFace dataset loading)
- `transformers` library (for GPT-NeoX-20B tokenizer)
- `torch` library
- Internet access (downloads ~400MB of datasets on first run)
- `HF_ENDPOINT=https://hf-mirror.com` recommended for China users

## Coverage Analysis

### Direct Coverage (6/6 downstream benchmarks) — NEW in v2

- **arc_easy** ← ARC-Easy (direct name match)
- **arc_challenge** ← ARC-Challenge (direct name match)
- **mmlu_stem** ← MMLU (22 STEM subjects, direct)
- **gsm8k_cot** ← GSM8K (direct, same task type)
- **gpqa_diamond** ← GPQA Main (198 diamond ⊂ 448 main, direct) **[NEW]**
- **math_cot** ← MATH (5000 from 7500 train, direct, 0% overlap with MATH-500 test) **[NEW]**

### v1 Coverage Gaps (FIXED in v2)

| Downstream | v1 Proxy | v1 R² | v1 Weight | Issue | v2 Fix |
|------------|----------|-------|-----------|-------|--------|
| gpqa_diamond | MMLU | 0.105 | 11.3% | Graduate-level ≠ college/HS level | GPQA Main (direct, same questions) |
| math_cot | GSM8K | 0.576 | 62.2% | Grade school ≠ competition level | MATH (direct, same data source) |

## Comparison with Other Validation Sets

| Aspect | OpenHermes-10k | CORE-22tasks | CAP v1 | STEM v1 | **STEM v2** |
|--------|----------------|--------------|--------|---------|------------|
| Focus | General chat | Broad benchmark | Proven training data | STEM direct | **STEM direct + gap fix** |
| Source | OpenHermes-2.5 | DCLM CORE test | External training | Benchmark train | **Benchmark train + GPQA + MATH** |
| Loss | Full-seq | Mixed | Full-seq | Full-seq | **Full-seq** |
| Signal tokens | 2.24M | 318K | 15.96M | 1.25M | **2.79M** |
| Docs | 10,000 | 46,926 | 40,000 | 11,132 | **16,455** |
| Tasks | 1 | 21 | 5 clusters | 4 | **6** |
| gpqa_diamond coverage | None | None | None | Indirect (MMLU) | **Direct** |
| math_cot coverage | None | None | None | Indirect (GSM8K) | **Direct** |

## Search Strategy

Per-task LightGBM with R²-weighted z-score search:

```python
tasks = ["gsm8k", "mmlu", "arc_easy", "arc_challenge", "gpqa", "math"]
weighted_z_score = Σ R²_i * (loss_i - mean_loss_i) / std_loss_i
# Tasks with R² ≤ 0 are automatically filtered
```

### Search Mode Considerations

With 6 tasks instead of 4, the weight distribution changes:
- **r2_weighted** (default): weights proportional to R². If gsm8k still dominates (R²≈0.58), consider equal_weight.
- **equal_weight**: all tasks weighted equally, prevents any single task from dominating.
- **r2_sigma_weighted**: R² × (1/std), can amplify noisy tasks.

Recommended workflow:
1. Run with `r2_weighted` first to see per-task R² values
2. If gsm8k > 40% weight, re-run with `equal_weight` using `demo_reoptimize.sh`

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

### GPQA Choice Shuffling

GPQA original data stores answers as `Question`, `Correct Answer`, `Incorrect Answer 1/2/3`. We shuffle the 4 choices using seed=42 to randomize the correct answer position, preventing position bias in the proxy model.

### MATH LaTeX Format

MATH solutions contain LaTeX markup (e.g., `$\frac{a}{b}$`, `\begin{align*}...`). The GPT-NeoX-20B tokenizer handles LaTeX tokens reasonably well, but some complex expressions may tokenize into many subwords. This is acceptable for proxy loss estimation.

## References

- **GSM8K**: [Cobbe et al., "Training Verifiers to Solve Math Word Problems"](https://arxiv.org/abs/2110.14168)
- **MMLU**: [Hendrycks et al., "Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300)
- **ARC**: [Clark et al., "ARC: A New Challenge Dataset for AI"](https://arxiv.org/abs/1803.04449)
- **GPQA**: [Rein et al., "GPQA: A Graduate-Level Google-Proof Q&A Benchmark"](https://arxiv.org/abs/2311.12022)
- **MATH**: [Hendrycks et al., "Measuring Mathematical Problem Solving With the MATH Dataset"](https://arxiv.org/abs/2103.03874)
- **DCLM Benchmark**: [Li et al., "DataComp-LM: In search of the next generation of multimodal datasets"](https://arxiv.org/abs/2406.11580)

## License

This dataset is released under the same license as the source datasets:

- GSM8K: MIT
- MMLU: MIT
- ARC: CC-BY 4.0
- GPQA: CC-BY 4.0
- MATH: MIT

Please check individual dataset licenses before commercial use.
