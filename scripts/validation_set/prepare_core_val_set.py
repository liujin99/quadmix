#!/usr/bin/env python3
"""
Prepare CORE benchmark-based validation set for QuaDMix proxy model.

Loads 22 tasks from the CORE eval bundle, extracts context + continuation
per task type, tokenizes with GPT-NeoX-20B. The loss_mask marks ALL
non-padding tokens (full sequence loss), following the QuaDMix paper's
approach of using benchmark training data as the validation target.

Text extraction per icl_task_type (context | continuation):
  - multiple_choice:    query + delimiter  |  choices[gold]
  - language_modeling:  context + delimiter |  continuation
  - schema:             context_options[gold] + delimiter |  continuation

Usage:
  python scripts/validation_set/prepare_core_val_set.py

Output: core_22tasks_tokenized.pt
  - token_ids:    LongTensor [num_docs, block_size]   padded
  - loss_mask:    BoolTensor  [num_docs, block_size]   True for all non-padding tokens
  - task_labels:  list[str]   per-doc task label (e.g., "hellaswag_zeroshot")
  - metadata:     dict        source info
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


EVAL_BUNDLE_DIR = "/home/ma-user/work/nanochat-master-multi/eval_bundle"
OUTPUT_DIR = "data"
SEED = 42
BLOCK_SIZE = 2048
NUM_SAMPLES_PER_TASK = 2000


def parse_choice_from_query(query, choice_label):
    """Parse the full choice text from query when choices field only contains labels.
    
    Handles formats like:
        "Choices:\nA. option text\nB. another option\nAnswer:"
        "Choices:\nA) option text\nB) another option"
    """
    pattern = rf'{re.escape(choice_label)}[.)]\s*(.+?)(?=\n[A-Z][.)]|\nAnswer:|$)'
    match = re.search(pattern, query, re.DOTALL)
    if match:
        return match.group(1).strip()
    return choice_label


def load_task_items(eval_bundle_dir, dataset_uri):
    """Load JSONL items for a given dataset_uri relative to eval_bundle/eval_data/."""
    filepath = os.path.join(eval_bundle_dir, "eval_data", dataset_uri)
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_context_continuation(task_type, item, delimiter=" "):
    """Extract (context, continuation) from a CORE item."""
    if task_type == "multiple_choice":
        query = item.get("query", "").rstrip()
        choices = item.get("choices", [])
        gold = item.get("gold", 0)
        if not query or not choices:
            return "", ""
        if 0 <= gold < len(choices):
            choice = choices[gold]
            if len(choice) <= 2 and choice.isalpha():
                choice = parse_choice_from_query(query, choice)
            return query + delimiter, choice
        return "", ""
    elif task_type == "language_modeling":
        context = item.get("context", "").strip()
        continuation = item.get("continuation", "")
        if not continuation.strip():
            return "", ""
        if context:
            return context + delimiter, continuation
        return "", continuation
    elif task_type == "schema":
        context_options = item.get("context_options", [])
        gold = item.get("gold", 0)
        continuation = item.get("continuation", "")
        if context_options and 0 <= gold < len(context_options):
            return context_options[gold] + delimiter, continuation
        return "", ""
    else:
        return "", ""


def main():
    parser = argparse.ArgumentParser(
        description="Create CORE benchmark validation set for QuaDMix"
    )
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help=f"Block size for tokenization (default: {BLOCK_SIZE})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--eval-bundle", type=str, default=EVAL_BUNDLE_DIR,
                        help=f"Path to CORE eval bundle directory (default: {EVAL_BUNDLE_DIR})")
    parser.add_argument("--num-samples-per-task", type=int, default=NUM_SAMPLES_PER_TASK,
                        help=f"Number of samples per task (default: {NUM_SAMPLES_PER_TASK})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    args = parser.parse_args()

    block_size = args.block_size
    output_dir = args.output_dir
    seed = args.seed
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    # Tokenizer (same as openhermes ref script)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer: GPT-NeoX-20B, vocab={tokenizer.vocab_size}")

    # Load CORE task config
    core_yaml_path = os.path.join(args.eval_bundle, "core.yaml")
    with open(core_yaml_path, "r") as f:
        core_config = yaml.safe_load(f)

    tasks = core_config["icl_tasks"]
    print(f"Loaded {len(tasks)} tasks from core.yaml")

    # Per-task collection: list of (context, continuation, task_label) tuples
    all_samples = []
    task_meta = []
    seen_uris = set()

    for task in tasks:
        label = task["label"]
        task_type = task["icl_task_type"]
        dataset_uri = task["dataset_uri"]
        delimiter = task.get("continuation_delimiter", " ")

        # Deduplicate: skip if same dataset_uri already processed (e.g., hellaswag 0-shot vs 10-shot)
        if dataset_uri in seen_uris:
            print(f"\nSkipping task: {label} (duplicate of {dataset_uri})")
            continue
        seen_uris.add(dataset_uri)

        print(f"\nLoading task: {label} (type={task_type}, uri={dataset_uri})")
        items = load_task_items(args.eval_bundle, dataset_uri)
        print(f"  Total items: {len(items)}")

        # Extract (context, continuation) for each item
        pairs = []
        for item in items:
            ctx, cont = extract_context_continuation(task_type, item, delimiter)
            if cont.strip():
                pairs.append((ctx, cont))

        print(f"  Valid pairs (non-empty continuation): {len(pairs)}")

        # Cap at available data (no up-sampling)
        target = min(args.num_samples_per_task, len(pairs))
        sampled = random.sample(pairs, target)
        capped = len(pairs) > target

        print(f"  Sampled: {len(sampled)}" + (f" (capped from {len(pairs)})" if capped else f" (all {len(pairs)} used)"))

        all_samples.extend([(ctx, cont, label) for ctx, cont in sampled])
        task_meta.append({
            "label": label,
            "task_type": task_type,
            "dataset_uri": dataset_uri,
            "continuation_delimiter": delimiter,
            "items_loaded": len(items),
            "valid_pairs": len(pairs),
            "sampled": len(sampled),
            "capped": capped,
        })

    print(f"\n{'='*60}")
    print(f"Total samples across all tasks: {len(all_samples)}")

    # Tokenize with full sequence loss mask (all non-padding tokens)
    print(f"\nTokenizing (block_size={block_size})...")
    all_ids = []
    all_masks = []
    task_labels = []

    for idx, (context, continuation, task_label) in enumerate(all_samples):
        full_text = context + continuation
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        ids_tensor = torch.LongTensor(full_ids)
        seq_len = min(len(ids_tensor), block_size)

        # Truncate or pad to block_size
        if len(ids_tensor) > block_size:
            ids_tensor = ids_tensor[:block_size]
        else:
            pad_len = block_size - len(ids_tensor)
            ids_tensor = torch.cat([ids_tensor, torch.zeros(pad_len, dtype=torch.long)])

        # Loss mask: True for all non-padding tokens (full sequence loss)
        loss_mask = torch.zeros(block_size, dtype=torch.bool)
        loss_mask[:seq_len] = True

        all_ids.append(ids_tensor)
        all_masks.append(loss_mask)
        task_labels.append(task_label)

        if (idx + 1) % 2000 == 0:
            print(f"  {idx+1}/{len(all_samples)}")

    token_ids = torch.stack(all_ids)
    loss_mask = torch.stack(all_masks)

    # Stats
    total_tokens = token_ids.numel()
    non_padding = (token_ids != tokenizer.pad_token_id).sum().item()
    masked_tokens = loss_mask.sum().item()
    print(f"\n  Tokenized: {token_ids.shape}")
    print(f"  Non-padding tokens: {non_padding:,} / {total_tokens:,} ({non_padding/total_tokens*100:.1f}%)")
    print(f"  Loss tokens (loss_mask=True): {masked_tokens:,} ({masked_tokens/non_padding*100:.1f}% of non-padding)")

    # Save
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
            "loss_strategy": "full_sequence",
            "source": f"CORE-benchmark-{len(task_meta)}tasks",
            "eval_bundle": args.eval_bundle,
            "num_samples_per_task": args.num_samples_per_task,
            "seed": seed,
            "description": (
                f"CORE benchmark {len(task_meta)}-task validation set (deduplicated); "
                "loss_mask=True on all non-padding tokens (full sequence loss), "
                "following QuaDMix paper's approach of using benchmark data as validation target"
            ),
            "tasks": task_meta,
        }
    }

    save_path = os.path.join(output_dir, "core_22tasks_tokenized.pt")
    torch.save(output, save_path)
    file_size = os.path.getsize(save_path) / 1024**2
    print(f"\n  Saved: {save_path} ({file_size:.0f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
