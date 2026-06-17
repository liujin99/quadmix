#!/usr/bin/env python3
"""
Complete train/test sample comparison for remaining tasks.
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import yaml
from datasets import load_dataset
import random

random.seed(42)

EVAL_BUNDLE_DIR = "/tmp/opencode/eval_bundle"

HF_DATASET_MAP = {
    "openbook_qa": ("openbookqa", "main", "train"),
    "piqa": ("ybisk/piqa", None, "train"),
    "arc_easy": ("ai2_arc", "ARC-Easy", "train"),
    "arc_challenge": ("ai2_arc", "ARC-Challenge", "train"),
    "commonsense_qa": ("commonsense_qa", None, "train"),
}


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


def format_hf_sample(task_label, sample):
    if task_label == "openbook_qa":
        question_stem = sample.get("question_stem", "")
        choices = sample.get("choices", {})
        answer_key = sample.get("answerKey", "")
        texts = choices.get("text", [])
        labels = choices.get("label", [])
        ans = ""
        for i, lbl in enumerate(labels):
            if lbl == answer_key and i < len(texts):
                ans = texts[i]
                break
        return f"{question_stem} → {ans}"
    
    elif task_label in ["arc_easy", "arc_challenge"]:
        question = sample.get("question", "")
        choices = sample.get("choices", {})
        answer_key = sample.get("answerKey", "")
        texts = choices.get("text", [])
        labels = choices.get("label", [])
        ans = ""
        for i, lbl in enumerate(labels):
            if lbl == answer_key and i < len(texts):
                ans = texts[i]
                break
        return f"Question: {question} → {ans}"
    
    elif task_label == "commonsense_qa":
        question = sample.get("question", "")
        choices = sample.get("choices", {})
        answer_key = sample.get("answerKey", "")
        texts = choices.get("text", [])
        labels = choices.get("label", [])
        ans = ""
        for i, lbl in enumerate(labels):
            if lbl == answer_key and i < len(texts):
                ans = texts[i]
                break
        return f"Question: {question} Choices: {texts} → {ans}"
    
    elif task_label == "piqa":
        goal = sample.get("goal", "")
        sol1 = sample.get("sol1", "")
        sol2 = sample.get("sol2", "")
        label = sample.get("label", 0)
        ans = sol1 if label == 0 else sol2
        return f"Goal: {goal} → {ans}"
    
    return str(sample)


def main():
    core_yaml_path = os.path.join(EVAL_BUNDLE_DIR, "core.yaml")
    with open(core_yaml_path, "r") as f:
        core_config = yaml.safe_load(f)
    
    task_map = {}
    for task in core_config["icl_tasks"]:
        label = task["label"]
        if label not in task_map:
            task_map[label] = task
    
    for label in HF_DATASET_MAP.keys():
        if label not in task_map:
            continue
        
        task = task_map[label]
        task_type = task["icl_task_type"]
        dataset_uri = task["dataset_uri"]
        filepath = os.path.join(EVAL_BUNDLE_DIR, "eval_data", dataset_uri)
        
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")
        
        items = load_jsonl(filepath)
        test_samples = random.sample(items, min(2, len(items)))
        
        print(f"\n  [TEST] ({len(items)} samples)")
        for i, item in enumerate(test_samples):
            text = item_to_text(item, task_type, label)
            print(f"    #{i+1}: {text[:200]}")
        
        hf_info = HF_DATASET_MAP.get(label)
        if hf_info and hf_info[0]:
            hf_name, hf_config, hf_split = hf_info
            try:
                if hf_config:
                    ds = load_dataset(hf_name, hf_config, split=hf_split, streaming=True)
                else:
                    ds = load_dataset(hf_name, split=hf_split, streaming=True)
                
                train_samples = []
                for i, sample in enumerate(ds):
                    if i >= 2:
                        break
                    train_samples.append(sample)
                
                print(f"\n  [TRAIN] (HuggingFace: {hf_name})")
                for i, sample in enumerate(train_samples):
                    text = format_hf_sample(label, sample)
                    print(f"    #{i+1}: {text[:200]}")
            except Exception as e:
                print(f"\n  [TRAIN] Error: {e}")
        else:
            print(f"\n  [TRAIN] No HuggingFace train split available")


if __name__ == "__main__":
    main()
