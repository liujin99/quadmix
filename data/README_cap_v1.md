# QuaDMix-CAP v1: Capability-Aligned Proxy Validation Set

**Script:** `scripts/validation_set/prepare_cap_v1.py`  
**HuggingFace:** [`liujin99/quadmix-cap-v1`](https://huggingface.co/datasets/liujin99/quadmix-cap-v1)  
**Files:** `cap_v1_tokenized.pt`, `cap_v1.parquet`

## Overview

CAP v1 (Capability-Aligned Proxy) is a validation set designed to align the proxy model's optimization signal with downstream benchmark capabilities. Unlike traditional validation sets that measure general language quality, CAP v1 uses **proven training data** from the literature to guide the proxy toward capability-specific improvements.

### Key Innovation: Bypassing the C2 Hypothesis

Traditional proxy validation assumes: *low val_loss → high benchmark score* (C2 hypothesis). This assumption is **unverified** and may fail when the validation set format differs from benchmark format.

CAP v1 bypasses this by using **externally validated training data** — datasets proven in published research to improve specific capabilities. The proxy learns to select data mixtures that produce low loss on these proven sources, implicitly targeting the same capabilities.

## Design Principles

1. **External training data (70%)**: Use datasets proven effective in literature (Orca-Math, MetaMathQA, NaturalReasoning, etc.)
2. **Benchmark train split (30%)**: Include benchmark training data for format alignment
3. **Equal-ratio sampling**: Each benchmark contributes equally within its cluster, preventing large datasets from dominating
4. **Full-sequence loss**: All non-padding tokens contribute to loss (unlike continuation-only in CORE-22tasks)

## Capability Clusters

CAP v1 organizes data into 5 capability clusters, each targeting specific downstream benchmarks:

| Cluster | Target Capabilities | External Sources (70%) | Benchmark Sources (30%) |
|---------|-------------------|----------------------|------------------------|
| **language_understanding** | HellaSwag, WinoGrande, LAMBADA | NaturalReasoning-lang (5,600) | HellaSwag (800) + WinoGrande (800) + LAMBADA (800) |
| **common_sense_reasoning** | CommonsenseQA, PIQA, COPA, OpenBookQA | OpenOrca-CoT (5,600) | CommonsenseQA (600) + PIQA (600) + COPA (600) + OpenBookQA (600) |
| **world_knowledge** | ARC-Easy, ARC-Challenge | NaturalReasoning-sci (5,600) | ARC-Easy (1,281) + ARC-Challenge (1,119) |
| **reading_comprehension** | SQuAD, BoolQ, CoQA | HotpotQA (4,990) + QASPER (610) | SQuAD (800) + BoolQ (800) + CoQA (800) |
| **symbol_logic** | GSM8K, Dyck, Operators, Repeat-Copy | Orca-Math (1,868) + MetaMathQA (1,866) + NuminaMath-CoT (1,866) | GSM8K (600) + synthetic_dyck (600) + synthetic_operators (600) + synthetic_repeat_copy (600) |

**Total:** 40,000 samples (8,000 per cluster)

## Data Sources

### External Training Data (Proven Effective)

| Source | Size | Capability | Reference |
|--------|------|-----------|-----------|
| **NaturalReasoning** | 1.15M | Language understanding, scientific reasoning | [Li et al., 2024](https://arxiv.org/abs/2406.08492) |
| **OpenOrca-CoT** | 74K | Commonsense reasoning with chain-of-thought | [Mukherjee et al., 2023](https://arxiv.org/abs/2308.12067) |
| **HotpotQA** | 90K | Multi-hop reading comprehension | [Yang et al., 2018](https://arxiv.org/abs/1809.09600) |
| **QASPER** | 610 | Scientific reading comprehension | [Dasigi et al., 2021](https://arxiv.org/abs/2105.03011) |
| **Orca-Math** | 200K | Mathematical reasoning via agent interactions | [Mitra et al., 2024](https://arxiv.org/abs/2402.14370) |
| **MetaMathQA** | 395K | Mathematical reasoning with augmented questions | [Yu et al., 2023](https://arxiv.org/abs/2309.12284) |
| **NuminaMath-CoT** | 860K | Competition math with chain-of-thought | [Numina AI](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |

### Benchmark Train Splits

Standard benchmark training data for format alignment:
- HellaSwag, WinoGrande, LAMBADA, CommonsenseQA, PIQA, COPA, OpenBookQA, ARC-Easy, ARC-Challenge, SQuAD, BoolQ, CoQA, GSM8K

### Synthetic Data

For symbol_logic tasks with limited real data:
- **synthetic_dyck**: 5,000 Dyck language samples (balanced parentheses)
- **synthetic_operators**: 9,923 operator precedence samples
- **synthetic_repeat_copy**: 5,000 repeat-copy logic samples

## File Format

```python
{
    "token_ids":    torch.LongTensor,   # [40000, 2048], padded with pad_token_id (0)
    "loss_mask":    torch.BoolTensor,   # [40000, 2048], True = include in loss
    "task_labels":  list[str],          # per-doc cluster label
    "metadata":     dict,               # source info, tokenizer, mix_ratio, cluster details
}
```

### Metadata Structure

```python
{
    "num_docs": 40000,
    "block_size": 2048,
    "tokenizer": "EleutherAI/gpt-neox-20b",
    "tokenizer_vocab": 50254,
    "loss_strategy": "full_sequence",
    "mix_ratio": "external=70%, benchmark=30%",
    "clusters": {
        "language_understanding": {
            "total": 8000,
            "external_total": 5600,
            "benchmark_total": 2400,
            "external_counts": {"natural_reasoning_lang": 5600},
            "benchmark_counts": {"hellaswag": 800, "winogrande": 800, "lambada": 800}
        },
        # ... other clusters
    }
}
```

## Loss Strategy: Full-Sequence Loss

Unlike CORE-22tasks (continuation-only loss), CAP v1 uses **full-sequence loss** — all non-padding tokens contribute to the loss.

### Rationale

1. **External training data is diverse**: NaturalReasoning, Orca-Math, etc. contain rich reasoning chains where context tokens carry signal
2. **No format mismatch**: External data is already in training format (Q&A, CoT), not benchmark format (multiple-choice)
3. **More signal tokens**: 15.96M tokens (19.5% of total) vs 318K in CORE-22tasks

### Token Statistics

```
Total tokens:           81,920,000 (40,000 × 2,048)
Non-padding tokens:     15,963,192 (19.5%)
Loss tokens (full-seq): 15,963,192 (100% of non-padding)
Truncated (>2048):      36/40000 (0.1%)
```

## Usage

```bash
python scripts/runners/run_essential_web_v1.py --quick --val-set=cap_v1
```

The validation set will be automatically downloaded from `liujin99/quadmix-cap-v1` on first use.

## Regenerating Locally

```bash
python scripts/validation_set/prepare_cap_v1.py \
  --output-dir data \
  --block-size 2048 \
  --seed 42
```

**Requirements:**
- `datasets` library (for HuggingFace dataset loading)
- `transformers` library (for GPT-NeoX-20B tokenizer)
- `torch` library
- Internet access (downloads ~2GB of external datasets on first run)

## Comparison with Other Validation Sets

| Aspect | OpenHermes-10k | CORE-22tasks | CAP v1 |
|--------|----------------|--------------|--------|
| **Focus** | General instruction quality | Benchmark-aligned capabilities | Proven training data |
| **Source** | OpenHermes-2.5-1M (chat) | DCLM CORE benchmark (21 tasks) | External training + benchmark train |
| **Loss mask** | All non-padding tokens | Continuation tokens only | All non-padding tokens |
| **Signal tokens** | 2.24M (100%) | 318K (5.2%) | 15.96M (100%) |
| **Docs** | 10,000 | 46,926 | 40,000 |
| **Clusters** | None (single task) | 21 tasks | 5 capability clusters |
| **C2 assumption** | Required | Required | **Bypassed** |
| **Best for** | General-purpose mix | Capability-targeted mixes | Capability-targeted with proven data |

## Coverage Analysis

### Direct Coverage (15/21 benchmarks)

These benchmarks have corresponding training data in CAP v1:
- **language_understanding**: HellaSwag, WinoGrande, LAMBADA
- **common_sense_reasoning**: CommonsenseQA, PIQA, COPA, OpenBookQA
- **world_knowledge**: ARC-Easy, ARC-Challenge
- **reading_comprehension**: SQuAD, BoolQ, CoQA
- **symbol_logic**: GSM8K, Dyck, Operators, Repeat-Copy

### Indirect Coverage (3/21 benchmarks)

These benchmarks are covered by similar capabilities:
- **Winograd**: Covered by WinoGrande (same pronoun resolution task)
- **CS Algorithms**: 76% covered by synthetic_dyck (bracket matching)
- **AGI Eval LSAT-AR**: Covered by symbol_logic cluster (logical reasoning)

### Not Covered (3/21 benchmarks)

These benchmarks lack corresponding training data:
- **Language ID**: No reliable training data for 1.3B model format transfer
- **Jeopardy**: No open-source training data available
- **QA Wikidata**: No open-source training data available

## Technical Details

### Tokenizer

GPT-NeoX-20B tokenizer (vocab size 50,254), matching the proxy model's vocabulary.

### Block Size

2048 tokens, matching the proxy model's sequence length.

### Sampling Strategy

**Equal-ratio sampling** within each cluster's benchmark portion:
- Each benchmark contributes equally (e.g., 800 samples each for 3-benchmark clusters)
- If a benchmark has fewer samples than its equal share, the deficit is redistributed to benchmarks with surplus capacity
- This prevents large datasets (e.g., SQuAD 87K) from dominating small datasets (e.g., COPA 1K)

### Loss Computation

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

## References

- **NaturalReasoning**: [Li et al., "NaturalReasoning: A Challenging Benchmark for Logical Reasoning"](https://arxiv.org/abs/2406.08492)
- **OpenOrca**: [Mukherjee et al., "Orca: Progressive Learning from Complex Explanation Traces of GPT-4"](https://arxiv.org/abs/2308.12067)
- **HotpotQA**: [Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering"](https://arxiv.org/abs/1809.09600)
- **QASPER**: [Dasigi et al., "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers"](https://arxiv.org/abs/2105.03011)
- **Orca-Math**: [Mitra et al., "Orca-Math: Unlocking the potential of SLMs in Grade School Math"](https://arxiv.org/abs/2402.14370)
- **MetaMathQA**: [Yu et al., "MetaMath: Bootstrap Your Own Mathematical Questions with Large Language Models"](https://arxiv.org/abs/2309.12284)
- **NuminaMath-CoT**: [Numina AI](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- **DCLM Benchmark**: [Li et al., "DataComp-LM: In search of the next generation of multimodal datasets"](https://arxiv.org/abs/2406.11580)

## License

This dataset is released under the same license as the source datasets. Please check individual dataset licenses before commercial use.
