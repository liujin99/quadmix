#!/usr/bin/env python3
"""
Show train samples from HuggingFace for comparison with test samples.
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
import random

random.seed(42)

HF_DATASET_MAP = {
    "hellaswag_zeroshot": ("hellaswag", None),
    "winogrande": ("winogrande", "winogrande_xl"),
    "copa": ("super_glue", "copa"),
    "boolq": ("boolq", None),
    "squad": ("squad", None),
    "coqa": ("coqa", None),
    "openbook_qa": ("openbookqa", "main"),
    "arc_easy": ("ai2_arc", "ARC-Easy"),
    "arc_challenge": ("ai2_arc", "ARC-Challenge"),
    "commonsense_qa": ("commonsense_qa", None),
}


def format_sample(task_label, sample):
    if task_label == "hellaswag_zeroshot":
        ctx = sample.get("ctx", "") or sample.get("ctx_a", "") + " " + sample.get("ctx_b", "")
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
        return f"Passage: {passage[:200]}... Q: {question} A: {answer}"
    
    elif task_label == "squad":
        context = sample.get("context", "")
        question = sample.get("question", "")
        answers = sample.get("answers", {})
        ans_text = answers.get("text", [""])[0] if answers.get("text") else ""
        return f"Context: {context[:200]}... Q: {question} A: {ans_text}"
    
    elif task_label == "coqa":
        story = sample.get("story", "")
        questions = sample.get("questions", [])
        answers = sample.get("answers", [])
        if questions and answers:
            return f"Story: {story[:200]}... Q: {questions[0]} A: {answers[0]}"
        return f"Story: {story[:200]}"
    
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
    
    return str(sample)


def main():
    for label in HF_DATASET_MAP.keys():
        hf_name, hf_config = HF_DATASET_MAP[label]
        
        print(f"\n{'='*80}")
        print(f"  {label}  (TRAIN)")
        print(f"{'='*80}")
        
        try:
            if hf_config:
                ds = load_dataset(hf_name, hf_config, split="train", streaming=True)
            else:
                ds = load_dataset(hf_name, split="train", streaming=True)
            
            samples = []
            for i, sample in enumerate(ds):
                if i >= 3:
                    break
                samples.append(sample)
            
            print(f"  Loaded {len(samples)} train samples")
            
            for i, sample in enumerate(samples):
                text = format_sample(label, sample)
                print(f"\n  [TRAIN #{i+1}]")
                print(f"  {text[:300]}")
        
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
