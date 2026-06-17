#!/usr/bin/env python3
"""
Analyze train/test distribution differences for each benchmark task.
This helps decide which tasks can safely merge train+test for validation set.

Usage:
  python scripts/validation_set/analyze_train_test_distribution.py

Output:
  Prints distribution comparison for each task
"""

import os
import json
import yaml
from collections import Counter
from datasets import load_dataset

EVAL_BUNDLE_DIR = "/tmp/opencode/eval_bundle"
OUTPUT_DIR = "data"

HF_DATASET_MAP = {
    "hellaswag_zeroshot": ("hellaswag", None, "validation"),
    "lambada_openai": ("EleutherAI/lambada_openai", None, "test"),
    "winogrande": ("winogrande", "winogrande_xl", "validation"),
    "winograd": ("winograd_wsc", "wsc273", "test"),
    "copa": ("super_glue", "copa", "validation"),
    "jeopardy": (None, None, None),
    "boolq": ("boolq", None, "validation"),
    "squad": ("squad", None, "validation"),
    "coqa": ("coqa", None, "validation"),
    "bigbench_language_identification": (None, None, None),
    "bigbench_qa_wikidata": (None, None, None),
    "openbook_qa": ("openbookqa", "main", "validation"),
    "piqa": ("ybisk/piqa", None, "validation"),
    "arc_easy": ("ai2_arc", "ARC-Easy", "validation"),
    "arc_challenge": ("ai2_arc", "ARC-Challenge", "validation"),
    "commonsense_qa": ("commonsense_qa", None, "validation"),
    "agi_eval_lsat_ar": (None, None, None),
    "bigbench_dyck_languages": (None, None, None),
    "bigbench_cs_algorithms": (None, None, None),
    "bigbench_operators": (None, None, None),
    "bigbench_repeat_copy_logic": (None, None, None),
}


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


def compare_distributions(train_stats, test_stats):
    """Compare two distributions and return a similarity score."""
    if not train_stats or not test_stats:
        return None
    
    metrics = ["avg_chars", "std_chars", "avg_words", "vocab_per_sample"]
    diffs = []
    
    for metric in metrics:
        if metric in train_stats and metric in test_stats:
            train_val = train_stats[metric]
            test_val = test_stats[metric]
            if train_val > 0:
                rel_diff = abs(train_val - test_val) / train_val
                diffs.append(rel_diff)
    
    if not diffs:
        return None
    
    avg_diff = sum(diffs) / len(diffs)
    return avg_diff


def main():
    core_yaml_path = os.path.join(EVAL_BUNDLE_DIR, "core.yaml")
    with open(core_yaml_path, "r") as f:
        core_config = yaml.safe_load(f)
    
    task_map = {}
    for task in core_config["icl_tasks"]:
        label = task["label"]
        if label not in task_map:
            task_map[label] = task
    
    print("=" * 80)
    print("Train/Test Distribution Analysis")
    print("=" * 80)
    
    results = []
    
    for label in sorted(HF_DATASET_MAP.keys()):
        hf_info = HF_DATASET_MAP[label]
        hf_name, hf_config, hf_split = hf_info
        
        print(f"\n{label}")
        print("-" * 80)
        
        if hf_name is None:
            print("  No HuggingFace dataset available (BIG-bench or custom)")
            results.append({
                "task": label,
                "status": "no_hf_dataset",
                "recommendation": "skip"
            })
            continue
        
        try:
            print(f"  Loading: {hf_name} ({hf_config or 'default'})")
            
            if hf_config:
                ds = load_dataset(hf_name, hf_config)
            else:
                ds = load_dataset(hf_name)
            
            print(f"  Available splits: {list(ds.keys())}")
            
            train_split = "train" if "train" in ds else None
            test_split = hf_split if hf_split in ds else None
            
            if not train_split or not test_split:
                print(f"  Missing train or test split")
                results.append({
                    "task": label,
                    "status": "missing_split",
                    "recommendation": "skip"
                })
                continue
            
            train_data = ds[train_split]
            test_data = ds[test_split]
            
            print(f"  Train samples: {len(train_data)}")
            print(f"  Test samples: {len(test_data)}")
            
            train_texts = []
            test_texts = []
            
            for sample in train_data:
                if "text" in sample:
                    train_texts.append(sample["text"])
                elif "context" in sample and "continuation" in sample:
                    train_texts.append(sample["context"] + " " + sample["continuation"])
                elif "question" in sample and "answer" in sample:
                    train_texts.append(sample["question"] + " " + str(sample["answer"]))
                elif "sentence" in sample:
                    train_texts.append(sample["sentence"])
                else:
                    train_texts.append(str(sample))
            
            for sample in test_data:
                if "text" in sample:
                    test_texts.append(sample["text"])
                elif "context" in sample and "continuation" in sample:
                    test_texts.append(sample["context"] + " " + sample["continuation"])
                elif "question" in sample and "answer" in sample:
                    test_texts.append(sample["question"] + " " + str(sample["answer"]))
                elif "sentence" in sample:
                    test_texts.append(sample["sentence"])
                else:
                    test_texts.append(str(sample))
            
            train_stats = compute_text_stats(train_texts)
            test_stats = compute_text_stats(test_texts)
            
            print(f"\n  Train stats:")
            print(f"    Avg chars: {train_stats['avg_chars']:.1f} ± {train_stats['std_chars']:.1f}")
            print(f"    Avg words: {train_stats['avg_words']:.1f}")
            print(f"    Vocab size: {train_stats['vocab_size']}")
            print(f"    Vocab/sample: {train_stats['vocab_per_sample']:.2f}")
            
            print(f"\n  Test stats:")
            print(f"    Avg chars: {test_stats['avg_chars']:.1f} ± {test_stats['std_chars']:.1f}")
            print(f"    Avg words: {test_stats['avg_words']:.1f}")
            print(f"    Vocab size: {test_stats['vocab_size']}")
            print(f"    Vocab/sample: {test_stats['vocab_per_sample']:.2f}")
            
            diff_score = compare_distributions(train_stats, test_stats)
            
            if diff_score is not None:
                print(f"\n  Distribution difference: {diff_score:.3f}")
                
                if diff_score < 0.1:
                    recommendation = "merge"
                    print(f"  ✓ Very similar - SAFE to merge train+test")
                elif diff_score < 0.3:
                    recommendation = "merge_with_caution"
                    print(f"  ~ Moderate difference - Can merge with caution")
                else:
                    recommendation = "keep_separate"
                    print(f"  ✗ Large difference - Keep separate")
            else:
                recommendation = "unknown"
                print(f"\n  Could not compute distribution difference")
            
            results.append({
                "task": label,
                "status": "ok",
                "train_samples": len(train_data),
                "test_samples": len(test_data),
                "diff_score": diff_score,
                "recommendation": recommendation
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "task": label,
                "status": "error",
                "error": str(e),
                "recommendation": "skip"
            })
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    merge_tasks = [r for r in results if r.get("recommendation") == "merge"]
    caution_tasks = [r for r in results if r.get("recommendation") == "merge_with_caution"]
    separate_tasks = [r for r in results if r.get("recommendation") == "keep_separate"]
    skip_tasks = [r for r in results if r.get("recommendation") == "skip"]
    
    print(f"\n✓ Safe to merge ({len(merge_tasks)} tasks):")
    for r in merge_tasks:
        print(f"  - {r['task']}: diff={r['diff_score']:.3f}, train={r['train_samples']}, test={r['test_samples']}")
    
    print(f"\n~ Merge with caution ({len(caution_tasks)} tasks):")
    for r in caution_tasks:
        print(f"  - {r['task']}: diff={r['diff_score']:.3f}, train={r['train_samples']}, test={r['test_samples']}")
    
    print(f"\n✗ Keep separate ({len(separate_tasks)} tasks):")
    for r in separate_tasks:
        print(f"  - {r['task']}: diff={r['diff_score']:.3f}, train={r['train_samples']}, test={r['test_samples']}")
    
    print(f"\n? Skip ({len(skip_tasks)} tasks):")
    for r in skip_tasks:
        print(f"  - {r['task']}: {r.get('status', 'unknown')}")
    
    output_path = os.path.join(OUTPUT_DIR, "train_test_distribution_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
