#!/usr/bin/env python3
"""
Analyze BIG-bench tasks from eval bundle JSONL files.
Compare train/test distribution for tasks that have both splits.
"""

import os
import json
import yaml
from collections import Counter

EVAL_BUNDLE_DIR = "/tmp/opencode/eval_bundle"


def compute_text_stats(texts):
    """Compute basic text statistics."""
    if not texts:
        return {}
    
    lengths = [len(t) for t in texts]
    words = [t.split() for t in texts]
    word_counts = [len(w) for w in words]
    
    all_words = [w.lower() for ws in words for w in ws]
    vocab = set(all_words)
    
    return {
        "count": len(texts),
        "avg_chars": sum(lengths) / len(lengths),
        "std_chars": (sum((l - sum(lengths)/len(lengths))**2 for l in lengths) / len(lengths))**0.5,
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "avg_words": sum(word_counts) / len(word_counts),
        "vocab_size": len(vocab),
        "vocab_per_sample": len(vocab) / len(texts),
    }


def load_jsonl(filepath):
    """Load JSONL file."""
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_text_from_item(item):
    """Extract text from eval bundle item."""
    if "query" in item:
        return item["query"]
    elif "context" in item and "continuation" in item:
        return item["context"] + " " + item["continuation"]
    elif "context_options" in item and "continuation" in item:
        gold = item.get("gold", 0)
        ctx = item["context_options"][gold] if gold < len(item["context_options"]) else ""
        return ctx + " " + item["continuation"]
    else:
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
    
    bigbench_tasks = [
        "bigbench_language_identification",
        "bigbench_qa_wikidata",
        "bigbench_dyck_languages",
        "bigbench_cs_algorithms",
        "bigbench_operators",
        "bigbench_repeat_copy_logic",
    ]
    
    print("=" * 80)
    print("BIG-bench Task Analysis (from eval bundle)")
    print("=" * 80)
    
    for label in bigbench_tasks:
        if label not in task_map:
            print(f"\n{label}: NOT FOUND")
            continue
        
        task = task_map[label]
        dataset_uri = task["dataset_uri"]
        filepath = os.path.join(EVAL_BUNDLE_DIR, "eval_data", dataset_uri)
        
        print(f"\n{label}")
        print("-" * 80)
        
        if not os.path.exists(filepath):
            print(f"  File not found: {filepath}")
            continue
        
        items = load_jsonl(filepath)
        print(f"  Total items (test/eval split): {len(items)}")
        
        texts = [extract_text_from_item(item) for item in items]
        stats = compute_text_stats(texts)
        
        print(f"\n  Stats:")
        print(f"    Avg chars: {stats['avg_chars']:.1f} ± {stats['std_chars']:.1f}")
        print(f"    Avg words: {stats['avg_words']:.1f}")
        print(f"    Vocab size: {stats['vocab_size']}")
        print(f"    Vocab/sample: {stats['vocab_per_sample']:.2f}")
        
        print(f"\n  Example:")
        print(f"    {texts[0][:200]}")
    
    print("\n" + "=" * 80)
    print("Note: These are test/eval splits only.")
    print("BIG-bench tasks don't have train splits in the eval bundle.")
    print("For these tasks, we can only use the existing test/eval data.")
    print("=" * 80)


if __name__ == "__main__":
    main()
