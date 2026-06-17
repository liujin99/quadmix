#!/usr/bin/env python3
"""
Show sample comparison between train and test splits for each task.
"""

import os
import json
import yaml
import random

EVAL_BUNDLE_DIR = "/tmp/opencode/eval_bundle"
random.seed(42)


def load_jsonl(filepath):
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def item_to_text(item, task_type, label):
    if task_type == "schema":
        ctx_opts = item.get("context_options", [])
        gold = item.get("gold", 0)
        cont = item.get("continuation", "")
        if ctx_opts and 0 <= gold < len(ctx_opts):
            return ctx_opts[gold] + " " + cont
        return ""
    elif task_type == "language_modeling":
        ctx = item.get("context", "").strip()
        cont = item.get("continuation", "")
        return ctx + " " + cont
    elif task_type == "multiple_choice":
        query = item.get("query", "")
        choices = item.get("choices", [])
        gold = item.get("gold", 0)
        ans = choices[gold] if choices and 0 <= gold < len(choices) else ""
        return query + " → " + ans
    return str(item)


def main():
    core_yaml_path = os.path.join(EVAL_BUNDLE_DIR, "core.yaml")
    with open(core_yaml_path, "r") as f:
        core_config = yaml.safe_load(f)

    task_map = {}
    for task in core_config["icl_tasks"]:
        label = task["label"]
        if label not in task_map:
            task_map[label] = task

    all_labels = [
        "hellaswag_zeroshot", "lambada_openai", "winogrande", "winograd",
        "copa", "jeopardy", "boolq", "squad", "coqa",
        "bigbench_language_identification", "bigbench_qa_wikidata",
        "openbook_qa", "piqa", "arc_easy", "arc_challenge",
        "commonsense_qa", "agi_eval_lsat_ar",
        "bigbench_dyck_languages", "bigbench_cs_algorithms",
        "bigbench_operators", "bigbench_repeat_copy_logic",
    ]

    for label in all_labels:
        if label not in task_map:
            continue

        task = task_map[label]
        task_type = task["icl_task_type"]
        dataset_uri = task["dataset_uri"]
        filepath = os.path.join(EVAL_BUNDLE_DIR, "eval_data", dataset_uri)

        print(f"\n{'='*80}")
        print(f"  {label}  (type={task_type})")
        print(f"{'='*80}")

        items = load_jsonl(filepath)
        print(f"  Total items (test/eval): {len(items)}")

        samples = random.sample(items, min(3, len(items)))
        for i, item in enumerate(samples):
            text = item_to_text(item, task_type, label)
            print(f"\n  [TEST #{i+1}]")
            print(f"  {text[:300]}")


if __name__ == "__main__":
    main()
