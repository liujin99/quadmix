#!/usr/bin/env python3
"""
Complete train/test sample comparison for all 21 tasks.
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
    "hellaswag_zeroshot": ("hellaswag", None, "train"),
    "lambada_openai": (None, None, None),
    "winogrande": ("winogrande", "winogrande_xl", "train"),
    "winograd": (None, None, None),
    "copa": ("super_glue", "copa", "train"),
    "jeopardy": (None, None, None),
    "boolq": ("boolq", None, "train"),
    "squad": ("squad", None, "train"),
    "coqa": ("coqa", None, "train"),
    "bigbench_language_identification": (None, None, None),
    "bigbench_qa_wikidata": (None, None, None),
    "openbook_qa": ("openbookqa", "main", "train"),
    "piqa": ("ybisk/piqa", None, "train"),
    "arc_easy": ("ai2_arc", "ARC-Easy", "train"),
    "arc_challenge": ("ai2_arc", "ARC-Challenge", "train"),
    "commonsense_qa": ("commonsense_qa", None, "train"),
    "agi_eval_lsat_ar": (None, None, None),
    "bigbench_dyck_languages": (None, None, None),
    "bigbench_cs_algorithms": (None, None, None),
    "bigbench_operators": (None, None, None),
    "bigbench_repeat_copy_logic": (None, None, None),
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
    if task_label == "hellaswag_zeroshot":
        ctx = sample.get("ctx", "") or (sample.get("ctx_a", "") + " " + sample.get("ctx_b", ""))
        endings = sample.get("endings", [])
        label = sample.get("label", "0")
        ans = endings[int(label)] if endings and label.isdigit() else ""
        return f"{ctx} → {ans}"
    
    elif task_label == "winogrande":
        sentence = sample.get("sentence", "")
        opt1 = sample.get("option1", "")
        opt2 = sample.get("option2", "")
        answer = sample.get("answer", "")
        ans = opt1 if answer == "1" else opt2
        return f"{sentence} → {ans}"
    
    elif task_label == "copa":
        premise = sample.get("premise", "")
        question = sample.get("question", "")
        choice1 = sample.get("choice1", "")
        choice2 = sample.get("choice2", "")
        label = sample.get("label", 0)
        ans = choice1 if label == 0 else choice2
        return f"{premise} {question} → {ans}"
    
    elif task_label == "boolq":
        question = sample.get("question", "")
        passage = sample.get("passage", "")
        answer = sample.get("answer", False)
        return f"Passage: {passage[:150]}... Q: {question} A: {answer}"
    
    elif task_label == "squad":
        context = sample.get("context", "")
        question = sample.get("question", "")
        answers = sample.get("answers", {})
        ans_text = answers.get("text", [""])[0] if answers.get("text") else ""
        return f"Context: {context[:150]}... Q: {question} A: {ans_text}"
    
    elif task_label == "coqa":
        story = sample.get("story", "")
        questions = sample.get("questions", [])
        answers = sample.get("answers", [])
        if questions and answers:
            return f"Story: {story[:150]}... Q: {questions[0]} A: {answers[0]}"
        return f"Story: {story[:150]}"
    
    elif task_label == "openbook_qa":
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
        
        print(f"\n  [判断] ", end="")
        if hf_info and hf_info[0]:
            print("需要人工对比train/test样本")
        else:
            print("无train split，无法合并")


if __name__ == "__main__":
    main()
