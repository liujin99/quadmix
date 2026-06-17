#!/usr/bin/env python3
"""
完整的21个task train/test样本对比分析
- 从HuggingFace加载有train split的数据集
- 从GitHub下载BIG-bench数据
- 从eval bundle加载test数据
"""

import os
import json
import yaml
import requests
from datasets import load_dataset

EVAL_BUNDLE_DIR = "/tmp/opencode/eval_bundle"

# HuggingFace数据集映射
HF_DATASETS = {
    "hellaswag_zeroshot": ("hellaswag", None, "train"),
    "winogrande": ("winogrande", "winogrande_xl", "train"),
    "copa": ("super_glue", "copa", "train"),
    "boolq": ("boolq", None, "train"),
    "squad": ("squad", None, "train"),
    "coqa": ("coqa", None, "train"),
    "openbook_qa": ("openbookqa", "main", "train"),
    "arc_easy": ("ai2_arc", "ARC-Easy", "train"),
    "arc_challenge": ("ai2_arc", "ARC-Challenge", "train"),
    "commonsense_qa": ("commonsense_qa", None, "train"),
}

# BIG-bench任务（从GitHub下载）
BIGBENCH_TASKS = {
    "bigbench_language_identification": "language_identification",
    "bigbench_qa_wikidata": "qa_wikidata",
    "bigbench_dyck_languages": "dyck_languages",
    "bigbench_cs_algorithms": "cs_algorithms",
    "bigbench_operators": "operators",
    "bigbench_repeat_copy_logic": "repeat_copy_logic",
}

# 无train split的任务
NO_TRAIN_TASKS = [
    "lambada_openai",
    "winograd",
    "jeopardy",
    "piqa",
    "agi_eval_lsat_ar",
]


def load_eval_bundle_data(task_name):
    """从eval bundle加载test数据"""
    core_yaml_path = os.path.join(EVAL_BUNDLE_DIR, "core.yaml")
    with open(core_yaml_path, 'r') as f:
        core_config = yaml.safe_load(f)
    
    task_map = {}
    for task in core_config["icl_tasks"]:
        if task["label"] == task_name:
            task_map = task
            break
    
    if not task_map:
        return None
    
    dataset_uri = task_map["dataset_uri"]
    jsonl_path = os.path.join(EVAL_BUNDLE_DIR, "eval_data", dataset_uri)
    
    if not os.path.exists(jsonl_path):
        return None
    
    items = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            items.append(json.loads(line))
    
    return items


def load_hf_train(task_name, num_samples=3):
    """从HuggingFace加载train数据"""
    if task_name not in HF_DATASETS:
        return None
    
    dataset_name, config_name, split = HF_DATASETS[task_name]
    
    try:
        if config_name:
            ds = load_dataset(dataset_name, config_name, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)
        
        samples = []
        for i, sample in enumerate(ds):
            if i >= num_samples:
                break
            samples.append(sample)
        
        return samples
    except Exception as e:
        print(f"  HuggingFace加载失败: {e}")
        return None


def load_bigbench_train(task_name, num_samples=3):
    """从GitHub下载BIG-bench数据"""
    if task_name not in BIGBENCH_TASKS:
        return None
    
    subtask = BIGBENCH_TASKS[task_name]
    url = f"https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/{subtask}/task.json"
    
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            examples = data.get("examples", [])
            return examples[:num_samples]
        else:
            print(f"  GitHub下载失败: status={r.status_code}")
            return None
    except Exception as e:
        print(f"  GitHub下载失败: {e}")
        return None


def format_sample(task_name, sample, is_train=False):
    """格式化样本用于显示"""
    if is_train:
        # HuggingFace或BIG-bench的train样本
        if task_name in BIGBENCH_TASKS:
            input_text = sample.get("input", "")[:150]
            target = sample.get("target", [])
            target_text = str(target[0]) if target else ""
            return f"input=\"{input_text}\" target=\"{target_text[:50]}\""
        else:
            # HuggingFace样本，根据不同task格式化
            if task_name == "hellaswag_zeroshot":
                ctx = sample.get("ctx", "") or (sample.get("ctx_a", "") + " " + sample.get("ctx_b", ""))
                endings = sample.get("endings", [])
                label = sample.get("label", "0")
                ans = endings[int(label)] if endings and label.isdigit() else ""
                return f"ctx=\"{ctx[:100]}\" → {ans[:50]}"
            elif task_name == "winogrande":
                sentence = sample.get("sentence", "")
                opt1 = sample.get("option1", "")
                opt2 = sample.get("option2", "")
                answer = sample.get("answer", "")
                ans = opt1 if answer == "1" else opt2
                return f"\"{sentence[:100]}\" → {ans}"
            elif task_name == "copa":
                premise = sample.get("premise", "")
                question = sample.get("question", "")
                choice1 = sample.get("choice1", "")
                choice2 = sample.get("choice2", "")
                label = sample.get("label", 0)
                ans = choice1 if label == 0 else choice2
                return f"\"{premise}\" {question} → {ans}"
            elif task_name == "boolq":
                question = sample.get("question", "")
                passage = sample.get("passage", "")
                answer = sample.get("answer", False)
                return f"Passage: {passage[:100]}... Q: {question} A: {answer}"
            elif task_name == "squad":
                context = sample.get("context", "")
                question = sample.get("question", "")
                answers = sample.get("answers", {})
                ans_text = answers.get("text", [""])[0] if answers.get("text") else ""
                return f"Context: {context[:100]}... Q: {question} A: {ans_text}"
            elif task_name == "coqa":
                story = sample.get("story", "")
                questions = sample.get("questions", [])
                answers = sample.get("answers", {})
                ans_text = answers.get("input_text", [""])[0] if isinstance(answers, dict) and answers.get("input_text") else ""
                if questions and ans_text:
                    return f"Story: {story[:100]}... Q: {questions[0]} A: {ans_text}"
                return f"Story: {story[:100]}"
            elif task_name == "openbook_qa":
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
            elif task_name in ["arc_easy", "arc_challenge"]:
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
            elif task_name == "commonsense_qa":
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
            else:
                return str(sample)[:200]
    else:
        # eval bundle的test样本
        task_type = sample.get("task_type", "")
        if task_type == "schema":
            ctx_opts = sample.get("context_options", [])
            gold = sample.get("gold", 0)
            cont = sample.get("continuation", "")
            if ctx_opts and 0 <= gold < len(ctx_opts):
                return f"\"{ctx_opts[gold][:100]}\" {cont[:50]}"
            return ""
        elif task_type == "language_modeling":
            ctx = sample.get("context", "").strip()
            cont = sample.get("continuation", "")
            return f"\"{ctx[:100]}\" {cont[:50]}"
        elif task_type == "multiple_choice":
            query = sample.get("query", "")
            choices = sample.get("choices", [])
            gold = sample.get("gold", 0)
            ans = choices[gold] if choices and 0 <= gold < len(choices) else ""
            return f"\"{query[:100]}\" → {ans[:50]}"
        return str(sample)[:200]


def main():
    print("=" * 80)
    print("完整的21个task train/test样本对比分析")
    print("=" * 80)
    
    all_tasks = (
        list(HF_DATASETS.keys()) +
        list(BIGBENCH_TASKS.keys()) +
        NO_TRAIN_TASKS
    )
    
    results = []
    
    for task_name in all_tasks:
        print(f"\n{'='*80}")
        print(f"Task: {task_name}")
        print(f"{'='*80}")
        
        # 加载test数据
        test_items = load_eval_bundle_data(task_name)
        if test_items:
            print(f"\n[TEST] {len(test_items)} samples (from eval bundle)")
            for i, item in enumerate(test_items[:3]):
                text = format_sample(task_name, item, is_train=False)
                print(f"  #{i+1}: {text}")
        else:
            print(f"\n[TEST] 无法加载")
        
        # 加载train数据
        if task_name in HF_DATASETS:
            train_samples = load_hf_train(task_name, num_samples=3)
            if train_samples:
                print(f"\n[TRAIN] {len(train_samples)} samples (from HuggingFace)")
                for i, sample in enumerate(train_samples):
                    text = format_sample(task_name, sample, is_train=True)
                    print(f"  #{i+1}: {text}")
                results.append({
                    "task": task_name,
                    "status": "can_merge",
                    "source": "huggingface",
                    "train_count": "available",
                    "test_count": len(test_items) if test_items else 0
                })
            else:
                results.append({
                    "task": task_name,
                    "status": "load_failed",
                    "source": "huggingface"
                })
        elif task_name in BIGBENCH_TASKS:
            train_samples = load_bigbench_train(task_name, num_samples=3)
            if train_samples:
                print(f"\n[TRAIN] {len(train_samples)} samples (from GitHub)")
                for i, sample in enumerate(train_samples):
                    text = format_sample(task_name, sample, is_train=True)
                    print(f"  #{i+1}: {text}")
                results.append({
                    "task": task_name,
                    "status": "can_merge",
                    "source": "github",
                    "train_count": len(train_samples),
                    "test_count": len(test_items) if test_items else 0
                })
            else:
                results.append({
                    "task": task_name,
                    "status": "load_failed",
                    "source": "github"
                })
        else:
            print(f"\n[TRAIN] 无train split")
            results.append({
                "task": task_name,
                "status": "no_train",
                "test_count": len(test_items) if test_items else 0
            })
    
    # 输出总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    can_merge = [r for r in results if r["status"] == "can_merge"]
    no_train = [r for r in results if r["status"] == "no_train"]
    load_failed = [r for r in results if r["status"] == "load_failed"]
    
    print(f"\n可以合并train+test: {len(can_merge)}个")
    for r in can_merge:
        print(f"  - {r['task']} ({r['source']})")
    
    print(f"\n无train split: {len(no_train)}个")
    for r in no_train:
        print(f"  - {r['task']}")
    
    print(f"\n加载失败: {len(load_failed)}个")
    for r in load_failed:
        print(f"  - {r['task']} ({r['source']})")
    
    # 保存结果
    output_path = "/tmp/opencode/train_test_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
