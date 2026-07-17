#!/usr/bin/env python3
"""
Prepare STEM v1 validation set for QuaDMix proxy model.

STEM-focused validation set with 4 benchmarks (aligned with downstream evaluation):
  1. GSM8K: grade school math word problems → gsm8k_cot, math_cot
  2. MMLU: 22 STEM subjects → mmlu_stem, gpqa_diamond
  3. arc_easy: ARC-Easy science questions → arc_easy
  4. arc_challenge: ARC-Challenge science questions → arc_challenge

All tasks use full-sequence loss. Data loaded entirely from HuggingFace.

Usage:
  HF_ENDPOINT=https://hf-mirror.com python scripts/validation_set/prepare_stem_v1.py

Output:
  data/stem_v1_tokenized.pt
  data/stem_v1.parquet
"""

import os
import random
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer


OUTPUT_DIR = "data"
SEED = 42
BLOCK_SIZE = 2048
NUM_SAMPLES_PER_TASK = 5000

STEM_TASKS = [
    "gsm8k",
    "mmlu",
    "arc_easy",
    "arc_challenge",
]

MMLU_STEM_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
    "medical_genetics",
    "virology",
]


def _load_gsm8k():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    texts = []
    for s in ds:
        question = s.get("question", "")
        answer = s.get("answer", "")
        if not question or not answer:
            continue
        text = f"Question: {question}\nSolution: {answer}"
        if len(text) < 20:
            continue
        texts.append(text)
    return texts


def _load_mmlu():
    from datasets import load_dataset
    texts = []
    for subject in MMLU_STEM_SUBJECTS:
        # Official: hendrycks/test (GitHub), data at people.eecs.berkeley.edu/~hendrycks/data.tar
        # HF: cais/mmlu is community-recognized mirror (same content as Hendrycks original)
        # lukaemon/mmlu is personal mirror (fallback)
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
        except Exception as e:
            print(f"    WARNING: failed to load MMLU {subject}: {e}")
            continue
        for s in ds:
            question = s.get("question", "")
            choices = s.get("choices", [])
            answer = s.get("answer", "")
            if not question or not choices or not answer:
                continue
            answer_idx = -1
            if isinstance(answer, int):
                answer_idx = answer
            elif isinstance(answer, str):
                answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                answer_idx = answer_map.get(answer.upper(), -1)
            if answer_idx < 0 or answer_idx >= len(choices):
                continue
            answer_letter = chr(65 + answer_idx)
            choice_str = "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(choices))
            text = f"Question: {question}\nChoices:\n{choice_str}\nAnswer: {answer_letter}"
            if len(text) < 20:
                continue
            texts.append(text)
        print(f"    MMLU {subject}: {len(texts)} cumulative")
    return texts


def _load_arc(config_name):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", config_name, split="train")
    texts = []
    for s in ds:
        question = s.get("question", "")
        choices = s.get("choices", {})
        answer_key = s.get("answerKey", "")
        if not question or not choices or not answer_key:
            continue
        choice_texts = choices.get("text", [])
        choice_labels = choices.get("label", [])
        if not choice_texts or not choice_labels:
            continue
        choice_str = "\n".join(f"  {lbl}. {t}" for lbl, t in zip(choice_labels, choice_texts))
        text = f"Question: {question}\nChoices:\n{choice_str}\nAnswer: {answer_key}"
        if len(text) < 20:
            continue
        texts.append(text)
    return texts


_SOURCE_LOADERS = {
    "gsm8k": _load_gsm8k,
    "mmlu": _load_mmlu,
    "arc_easy": lambda: _load_arc("ARC-Easy"),
    "arc_challenge": lambda: _load_arc("ARC-Challenge"),
}


def _load_source(src_name):
    loader = _SOURCE_LOADERS.get(src_name)
    if loader is None:
        print(f"  WARNING: no loader for source '{src_name}'")
        return []
    print(f"  Loading {src_name}...")
    try:
        texts = loader()
    except Exception as e:
        print(f"  ERROR loading {src_name}: {e}")
        texts = []
    print(f"  {src_name}: {len(texts)} samples loaded")
    return texts


def _sample_equal_ratio(all_texts, target_count):
    if len(all_texts) <= target_count:
        return all_texts
    return random.sample(all_texts, target_count)


def main():
    parser = argparse.ArgumentParser(
        description="Create STEM v1 validation set for QuaDMix"
    )
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help=f"Block size for tokenization (default: {BLOCK_SIZE})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--num-samples-per-task", type=int, default=NUM_SAMPLES_PER_TASK,
                        help=f"Target samples per task (default: {NUM_SAMPLES_PER_TASK})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    args = parser.parse_args()

    block_size = args.block_size
    output_dir = args.output_dir
    seed = args.seed
    n_per_task = args.num_samples_per_task
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer: GPT-NeoX-20B, vocab={tokenizer.vocab_size}")
    print(f"Tasks: {STEM_TASKS}")
    print(f"Target per task: {n_per_task}")

    all_texts = []
    all_labels = []
    task_meta = {}

    for task_name in STEM_TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")

        texts = _load_source(task_name)
        if not texts:
            print(f"  SKIPPED: no samples loaded")
            task_meta[task_name] = {"loaded": 0, "sampled": 0}
            continue

        sampled = _sample_equal_ratio(texts, n_per_task)

        avg_chars = sum(len(t) for t in sampled) / len(sampled)
        min_chars = min(len(t) for t in sampled)
        max_chars = max(len(t) for t in sampled)

        print(f"  Loaded: {len(texts)}, Sampled: {len(sampled)}")
        print(f"  Chars: avg={avg_chars:.0f}, min={min_chars}, max={max_chars}")
        if sampled:
            print(f"  Example: \"{sampled[0][:150]}\"")

        all_texts.extend(sampled)
        all_labels.extend([task_name] * len(sampled))

        task_meta[task_name] = {
            "loaded": len(texts),
            "sampled": len(sampled),
            "avg_chars": round(avg_chars, 1),
            "min_chars": min_chars,
            "max_chars": max_chars,
        }

    print(f"\n{'='*60}")
    print(f"Total samples: {len(all_texts)}")
    for tn in STEM_TASKS:
        count = sum(1 for l in all_labels if l == tn)
        print(f"  {tn}: {count}")

    if not all_texts:
        print("ERROR: no samples collected, aborting")
        return

    print(f"\nTokenizing (block_size={block_size})...")
    print(f"  Batch encoding {len(all_texts)} texts...")
    encoded = tokenizer(all_texts, add_special_tokens=False, verbose=False)
    ids_list = encoded["input_ids"]

    print(f"  Building tensors...")
    n_samples = len(all_texts)
    token_ids = torch.zeros(n_samples, block_size, dtype=torch.long)
    loss_mask = torch.zeros(n_samples, block_size, dtype=torch.bool)

    for idx in range(n_samples):
        doc_ids = ids_list[idx]
        seq_len = min(len(doc_ids), block_size)
        if seq_len > 0:
            token_ids[idx, :seq_len] = torch.tensor(doc_ids[:seq_len], dtype=torch.long)
            loss_mask[idx, :seq_len] = True
        if (idx + 1) % 10000 == 0:
            print(f"  {idx+1}/{n_samples}")

    total_tokens = token_ids.numel()
    non_padding = (token_ids != tokenizer.pad_token_id).sum().item()
    masked_tokens = loss_mask.sum().item()
    print(f"\n  Tokenized: {token_ids.shape}")
    print(f"  Non-padding tokens: {non_padding:,} / {total_tokens:,} ({non_padding/total_tokens*100:.1f}%)")
    print(f"  Loss tokens (full-seq): {masked_tokens:,} ({masked_tokens/max(non_padding,1)*100:.1f}% of non-padding)")

    truncated = sum(1 for ids in ids_list if len(ids) > block_size)
    print(f"  Truncated (>block_size): {truncated}/{n_samples} ({truncated/n_samples*100:.1f}%)")

    output = {
        "token_ids": token_ids,
        "loss_mask": loss_mask,
        "task_labels": all_labels,
        "metadata": {
            "num_docs": len(token_ids),
            "block_size": block_size,
            "tokenizer": "gpt-neox-20b",
            "tokenizer_vocab": tokenizer.vocab_size,
            "model_vocab": 50432,
            "loss_strategy": "full_sequence (all tasks)",
            "source": "STEM-v1 (Science, Technology, Engineering, Mathematics benchmarks)",
            "seed": seed,
            "num_samples_per_task": n_per_task,
            "downstream_benchmarks": [
                "arc_easy", "arc_challenge",
                "mmlu_stem (0-shot, 22 subjects)",
                "gpqa_diamond", "gsm8k_cot", "math_cot",
            ],
            "description": (
                f"STEM v1: {len(STEM_TASKS)} STEM-focused benchmarks, "
                f"aligned with downstream evaluation. "
                f"All tasks use full-sequence loss. "
                f"Tasks: {STEM_TASKS}. "
                f"MMLU includes {len(MMLU_STEM_SUBJECTS)} STEM subjects. "
                f"Total samples: {len(token_ids)}."
            ),
            "tasks": task_meta,
            "mmlu_subjects": MMLU_STEM_SUBJECTS,
        },
    }

    save_path = os.path.join(output_dir, "stem_v1_tokenized.pt")
    torch.save(output, save_path)
    file_size = os.path.getsize(save_path) / 1024**2
    print(f"\n  Saved: {save_path} ({file_size:.0f} MB)")

    print(f"\nExporting Parquet...")
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_rows = {
        "task": [],
        "text": [],
        "char_len": [],
        "token_len": [],
    }
    for i in range(n_samples):
        parquet_rows["task"].append(all_labels[i])
        parquet_rows["text"].append(all_texts[i])
        parquet_rows["char_len"].append(len(all_texts[i]))
        parquet_rows["token_len"].append(len(ids_list[i]))

    table = pa.table(parquet_rows)
    parquet_path = os.path.join(output_dir, "stem_v1.parquet")
    pq.write_table(table, parquet_path)
    pq_size = os.path.getsize(parquet_path) / 1024**2
    print(f"  Saved: {parquet_path} ({pq_size:.1f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
