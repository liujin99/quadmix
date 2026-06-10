#!/usr/bin/env python3
"""Export CORE-22tasks validation set as Parquet for easy viewing on HuggingFace."""

import os
import re
import json
import random
import argparse

import yaml
import pyarrow as pa
import pyarrow.parquet as pq


EVAL_BUNDLE_DIR = "/tmp/opencode/eval_bundle"
OUTPUT_DIR = "data"
SEED = 42
NUM_SAMPLES_PER_TASK = 5000
MIN_CONTINUATION_TOKENS = 1


def load_task_items(eval_bundle_dir, dataset_uri):
    filepath = os.path.join(eval_bundle_dir, "eval_data", dataset_uri)
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def parse_choice_from_query(query, choice_label):
    """Parse the full choice text from query when choices field only contains labels."""
    pattern = rf'{re.escape(choice_label)}[.)]\s*(.+?)(?=\n[A-Z][.)]|\nAnswer:|$)'
    match = re.search(pattern, query, re.DOTALL)
    if match:
        return match.group(1).strip()
    return choice_label


def extract_context_continuation(task_type, item, delimiter=" "):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-bundle", type=str, default=EVAL_BUNDLE_DIR)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--num-samples-per-task", type=int, default=NUM_SAMPLES_PER_TASK)
    parser.add_argument("--min-continuation-tokens", type=int, default=MIN_CONTINUATION_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    min_cont_tok = args.min_continuation_tokens

    core_yaml_path = os.path.join(args.eval_bundle, "core.yaml")
    with open(core_yaml_path, "r") as f:
        core_config = yaml.safe_load(f)

    tasks = core_config["icl_tasks"]
    print(f"Loaded {len(tasks)} tasks from core.yaml")

    rows = []
    seen_uris = set()

    for task in tasks:
        label = task["label"]
        task_type = task["icl_task_type"]
        dataset_uri = task["dataset_uri"]
        delimiter = task.get("continuation_delimiter", " ")

        if dataset_uri in seen_uris:
            print(f"Skipping: {label} (duplicate)")
            continue
        seen_uris.add(dataset_uri)

        items = load_task_items(args.eval_bundle, dataset_uri)
        pairs = []
        for item in items:
            ctx, cont = extract_context_continuation(task_type, item, delimiter)
            if cont.strip():
                cont_tok = tokenizer.encode(cont, add_special_tokens=False)
                if len(cont_tok) >= min_cont_tok:
                    pairs.append((ctx, cont))

        if len(pairs) == 0:
            print(f"  {label}: FILTERED OUT (no samples >= {min_cont_tok} cont tokens)")
            continue

        if args.num_samples_per_task > 0:
            target = min(args.num_samples_per_task, len(pairs))
            sampled = random.sample(pairs, target)
        else:
            sampled = pairs

        for ctx, cont in sampled:
            rows.append({
                "task_label": label,
                "task_type": task_type,
                "dataset_uri": dataset_uri,
                "context": ctx,
                "continuation": cont,
            })

        print(f"  {label}: {len(sampled)} samples")

    print(f"\nTotal rows: {len(rows)}")

    columns = {}
    for key in rows[0]:
        columns[key] = [row[key] for row in rows]
    table = pa.table(columns)
    output_path = os.path.join(args.output_dir, "core_22tasks.parquet")
    pq.write_table(table, output_path)
    file_size = os.path.getsize(output_path) / 1024**2
    print(f"Saved: {output_path} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()
