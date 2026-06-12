#!/usr/bin/env python3
"""
Prepare CORE-BMK v4.2 validation set for QuaDMix proxy model.

Design: docs/VALIDATION_SET_V4.2_DESIGN.md

21 tasks (all CORE benchmarks, deduped hellaswag).
Per-task loss paradigm: no tasks pre-excluded, zero-variance handled at runtime.

Task classification:
  A. Full-sequence loss (12 tasks) — context+continuation = natural text:
     hellaswag_zeroshot, lambada_openai, winogrande, winograd, copa,
     jeopardy, boolq, squad, coqa, bigbench_language_identification,
     bigbench_qa_wikidata, openbook_qa
  B. Answer-only loss (9 tasks) — context is Q+A/SFT/symbolic:
     piqa, arc_easy, arc_challenge, commonsense_qa, agi_eval_lsat_ar,
     bigbench_dyck_languages, bigbench_cs_algorithms, bigbench_operators,
     bigbench_repeat_copy_logic

Usage:
  python scripts/validation_set/prepare_core_bmk_v4.2.py

Output:
  data/core_bmk_21tasks_v4.2_tokenized.pt
  data/core_bmk_21tasks_v4.2.parquet
"""

import os
import re
import random
import argparse
import json
import yaml

import numpy as np
import torch
from transformers import AutoTokenizer


EVAL_BUNDLE_DIR = "/tmp/opencode/eval_bundle"
OUTPUT_DIR = "data"
SEED = 42
BLOCK_SIZE = 2048
NUM_SAMPLES_PER_TASK = 2000

FULL_SEQ_TASKS = [
    "hellaswag_zeroshot",
    "lambada_openai",
    "winogrande",
    "winograd",
    "copa",
    "jeopardy",
    "boolq",
    "squad",
    "coqa",
    "bigbench_language_identification",
    "bigbench_qa_wikidata",
    "openbook_qa",
]

ANSWER_ONLY_TASKS = [
    "piqa",
    "arc_easy",
    "arc_challenge",
    "commonsense_qa",
    "agi_eval_lsat_ar",
    "bigbench_dyck_languages",
    "bigbench_cs_algorithms",
    "bigbench_operators",
    "bigbench_repeat_copy_logic",
]

ALL_V42_TASKS = FULL_SEQ_TASKS + ANSWER_ONLY_TASKS

MC_EMBEDDED_TASKS = {"commonsense_qa", "agi_eval_lsat_ar", "bigbench_language_identification"}


def parse_choice_from_query(query, choice_label):
    pattern = rf'{re.escape(choice_label)}[.)]\s*(.+?)(?=\n[A-Z][.)]|\nAnswer:|$)'
    match = re.search(pattern, query, re.DOTALL)
    if match:
        return match.group(1).strip()
    return choice_label


def clean_sft_prefix(text, task_label):
    if task_label == "boolq":
        if text.startswith("Passage: "):
            text = text[len("Passage: "):]
        elif text.startswith("Passage:"):
            text = text[len("Passage:"):]
    elif task_label == "squad":
        if text.startswith("Context: "):
            text = text[len("Context: "):]
        elif text.startswith("Context:"):
            text = text[len("Context:"):]
    elif task_label == "coqa":
        story_idx = text.find("Story:")
        if story_idx > 0:
            text = text[story_idx:]
    elif task_label == "bigbench_language_identification":
        text = re.sub(r'\nSentence: ', '\n', text, count=1)
        if text.startswith("Sentence: "):
            text = text[len("Sentence: "):]
    return text


def extract_answer(item, task_label):
    choices = item.get("choices", [])
    gold = item.get("gold", 0)
    if not choices or gold < 0 or gold >= len(choices):
        return ""
    if task_label in MC_EMBEDDED_TASKS:
        letter = choices[gold]
        if task_label == "bigbench_language_identification":
            return parse_choice_from_query(item.get("query", ""), letter)
        return letter
    return choices[gold]


def load_task_items(eval_bundle_dir, dataset_uri):
    filepath = os.path.join(eval_bundle_dir, "eval_data", dataset_uri)
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_pairs(task_type, task_label, items):
    is_full_seq = task_label in FULL_SEQ_TASKS
    pairs = []

    if task_type == "schema":
        for item in items:
            context_options = item.get("context_options", [])
            gold = item.get("gold", 0)
            continuation = item.get("continuation", "")
            if context_options and 0 <= gold < len(context_options):
                ctx = context_options[gold]
                sep = "" if ctx and ctx[-1].isspace() else " "
                pairs.append((ctx, sep + continuation, is_full_seq))
        return pairs

    if task_type == "language_modeling":
        for item in items:
            context = item.get("context", "").strip()
            continuation = item.get("continuation", "")
            if not continuation.strip():
                continue
            if is_full_seq:
                context = clean_sft_prefix(context, task_label)
            if context:
                if is_full_seq and not context[-1].isspace():
                    continuation = " " + continuation
                pairs.append((context, continuation, is_full_seq))
            else:
                pairs.append(("", continuation, is_full_seq))
        return pairs

    if task_type == "multiple_choice":
        for item in items:
            raw_query = item.get("query", "")
            if not raw_query:
                continue
            answer = extract_answer(item, task_label)
            if not answer.strip():
                continue
            question = raw_query.rstrip()
            if is_full_seq:
                question = clean_sft_prefix(question, task_label)
            if not question.strip():
                continue
            if is_full_seq:
                sep = "" if question and question[-1].isspace() else " "
                pairs.append((question, sep + answer, is_full_seq))
            else:
                pairs.append((question, answer, is_full_seq))
        return pairs

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Create CORE-BMK v4.2 validation set for QuaDMix"
    )
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help=f"Block size for tokenization (default: {BLOCK_SIZE})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--eval-bundle", type=str, default=EVAL_BUNDLE_DIR,
                        help=f"Path to CORE eval bundle directory (default: {EVAL_BUNDLE_DIR})")
    parser.add_argument("--num-samples-per-task", type=int, default=NUM_SAMPLES_PER_TASK,
                        help=f"Max samples per task, 0=all available (default: {NUM_SAMPLES_PER_TASK})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    args = parser.parse_args()

    block_size = args.block_size
    output_dir = args.output_dir
    seed = args.seed
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer: GPT-NeoX-20B, vocab={tokenizer.vocab_size}")

    core_yaml_path = os.path.join(args.eval_bundle, "core.yaml")
    with open(core_yaml_path, "r") as f:
        core_config = yaml.safe_load(f)

    all_tasks = core_config["icl_tasks"]
    print(f"Loaded {len(all_tasks)} tasks from core.yaml")

    task_map = {}
    for task in all_tasks:
        label = task["label"]
        if label not in task_map:
            task_map[label] = task

    print(f"\nFiltering to {len(ALL_V42_TASKS)} tasks (deduped hellaswag)...")
    print(f"  Full-seq tasks ({len(FULL_SEQ_TASKS)}): {FULL_SEQ_TASKS}")
    print(f"  Answer-only tasks ({len(ANSWER_ONLY_TASKS)}): {ANSWER_ONLY_TASKS}")
    for t in ALL_V42_TASKS:
        if t not in task_map:
            print(f"  WARNING: {t} not found in core.yaml")

    all_samples = []
    task_meta = []

    for label in ALL_V42_TASKS:
        if label not in task_map:
            continue

        task = task_map[label]
        task_type = task["icl_task_type"]
        dataset_uri = task["dataset_uri"]
        is_full_seq = label in FULL_SEQ_TASKS
        loss_strategy = "full_sequence" if is_full_seq else "answer_only"

        print(f"\nLoading task: {label} (type={task_type}, loss={loss_strategy})")
        items = load_task_items(args.eval_bundle, dataset_uri)
        print(f"  Total items: {len(items)}")

        pairs = extract_pairs(task_type, label, items)
        print(f"  Valid pairs: {len(pairs)}")

        if len(pairs) == 0:
            print(f"  SKIPPED: no valid pairs")
            continue

        if args.num_samples_per_task > 0:
            target = min(args.num_samples_per_task, len(pairs))
            sampled = random.sample(pairs, target)
            capped = len(pairs) > target
        else:
            sampled = pairs
            capped = False

        avg_ctx_chars = sum(len(ctx) for ctx, _, _ in sampled) / len(sampled)
        avg_ans_chars = sum(len(ans) for _, ans, _ in sampled) / len(sampled)
        avg_full_chars = avg_ctx_chars + avg_ans_chars
        ans_ratio = avg_ans_chars / avg_full_chars * 100 if avg_full_chars > 0 else 0

        print(f"  Sampled: {len(sampled)}" + (f" (capped from {len(pairs)})" if capped else f" (all used)"))
        print(f"  Avg chars: ctx={avg_ctx_chars:.0f}, ans={avg_ans_chars:.0f}, full={avg_full_chars:.0f}, ans%={ans_ratio:.1f}%")

        if sampled:
            ctx0, ans0, _ = sampled[0]
            full0 = ctx0 + ans0
            print(f"  Example: \"{full0[:120]}\"")

        all_samples.extend(sampled)
        task_meta.append({
            "label": label,
            "task_type": task_type,
            "category": "full_sequence" if is_full_seq else "answer_only",
            "loss_strategy": loss_strategy,
            "dataset_uri": dataset_uri,
            "items_loaded": len(items),
            "valid_pairs": len(pairs),
            "sampled": len(sampled),
            "capped": capped,
            "avg_context_chars": round(avg_ctx_chars, 1),
            "avg_answer_chars": round(avg_ans_chars, 1),
            "avg_answer_ratio": round(ans_ratio, 1),
        })

    print(f"\n{'='*60}")
    print(f"Total samples across {len(task_meta)} tasks: {len(all_samples)}")

    n_full = sum(1 for _, _, is_f in all_samples if is_f)
    n_ans = len(all_samples) - n_full
    print(f"  Full-seq: {n_full}")
    print(f"  Answer-only: {n_ans}")

    print(f"\nTokenizing (block_size={block_size})...")
    all_ids = []
    all_masks = []

    for idx, (context, continuation, is_full_seq) in enumerate(all_samples):
        context_ids = tokenizer.encode(context, add_special_tokens=False) if context else []
        cont_ids = tokenizer.encode(continuation, add_special_tokens=False) if continuation else []

        full_ids = context_ids + cont_ids
        seq_len = min(len(full_ids), block_size)

        ids_tensor = torch.zeros(block_size, dtype=torch.long)
        for i in range(seq_len):
            ids_tensor[i] = full_ids[i]

        loss_mask = torch.zeros(block_size, dtype=torch.bool)
        if is_full_seq:
            loss_mask[:seq_len] = True
        else:
            ans_start = min(len(context_ids), seq_len)
            loss_mask[ans_start:seq_len] = True

        all_ids.append(ids_tensor)
        all_masks.append(loss_mask)

        if (idx + 1) % 5000 == 0:
            print(f"  {idx+1}/{len(all_samples)}")

    task_labels = []
    for tm in task_meta:
        for _ in range(tm["sampled"]):
            task_labels.append(tm["label"])

    token_ids = torch.stack(all_ids)
    loss_mask = torch.stack(all_masks)

    total_tokens = token_ids.numel()
    non_padding = (token_ids != tokenizer.pad_token_id).sum().item()
    masked_tokens = loss_mask.sum().item()
    print(f"\n  Tokenized: {token_ids.shape}")
    print(f"  Non-padding tokens: {non_padding:,} / {total_tokens:,} ({non_padding/total_tokens*100:.1f}%)")
    print(f"  Loss tokens (loss_mask=True): {masked_tokens:,} ({masked_tokens/non_padding*100:.1f}% of non-padding)")

    full_mask_tokens = 0
    ans_mask_tokens = 0
    _offset = 0
    for tm in task_meta:
        count = tm["sampled"]
        for i in range(count):
            masked = all_masks[_offset + i].sum().item()
            if tm["category"] == "full_sequence":
                full_mask_tokens += masked
            else:
                ans_mask_tokens += masked
        _offset += count

    print(f"  Full-seq loss tokens: {full_mask_tokens:,}")
    print(f"  Answer-only loss tokens: {ans_mask_tokens:,}")

    output = {
        "token_ids": token_ids,
        "loss_mask": loss_mask,
        "task_labels": task_labels,
        "metadata": {
            "num_docs": len(token_ids),
            "block_size": block_size,
            "tokenizer": "gpt-neox-20b",
            "tokenizer_vocab": tokenizer.vocab_size,
            "model_vocab": 50432,
            "loss_strategy": "mixed (full_sequence=12 tasks, answer_only=9 tasks)",
            "source": f"CORE-BMK-{len(task_meta)}tasks-v4.2",
            "eval_bundle": args.eval_bundle,
            "num_samples_per_task": args.num_samples_per_task,
            "seed": seed,
            "description": (
                f"CORE-BMK v4.2: {len(task_meta)} tasks (all CORE benchmarks, deduped). "
                "Full-seq tasks: context+continuation = natural text, all tokens in loss. "
                "Answer-only tasks: context is Q+A/SFT/symbolic, only answer tokens in loss. "
                "SFT artifacts cleaned for full-seq tasks (Passage:/Context:/instruction prefixes). "
                "MC-embedded tasks (commonsense_qa, agi_eval_lsat_ar) use letter labels as answers."
            ),
            "tasks": task_meta,
        }
    }

    save_path = os.path.join(output_dir, "core_bmk_21tasks_v4.2_tokenized.pt")
    torch.save(output, save_path)
    file_size = os.path.getsize(save_path) / 1024**2
    print(f"\n  Saved: {save_path} ({file_size:.0f} MB)")

    print(f"\nExporting Parquet...")
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_rows = []
    offset = 0
    for tm in task_meta:
        count = tm["sampled"]
        for i in range(count):
            ctx, ans, _ = all_samples[offset + i]
            parquet_rows.append({
                "task": tm["label"],
                "category": tm["category"],
                "loss_strategy": tm["loss_strategy"],
                "context": ctx,
                "answer": ans,
                "text": ctx + ans,
            })
        offset += count

    columns = {}
    for key in parquet_rows[0]:
        columns[key] = [row[key] for row in parquet_rows]
    table = pa.table(columns)
    parquet_path = os.path.join(output_dir, "core_bmk_21tasks_v4.2.parquet")
    pq.write_table(table, parquet_path)
    pq_size = os.path.getsize(parquet_path) / 1024**2
    print(f"  Saved: {parquet_path} ({pq_size:.1f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
