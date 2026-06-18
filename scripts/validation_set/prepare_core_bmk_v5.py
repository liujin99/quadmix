#!/usr/bin/env python3
"""
Prepare CORE-BMK v5 validation set for QuaDMix proxy model.

Changes from v4.3:
  - 11 tasks now load from HuggingFace (train+test+val merged) instead of eval_bundle
  - 10 tasks remain on eval_bundle (unchanged)
  - commonsense_qa uses all 5 choices (A-E) from HF instead of 4 (answer-only loss unaffected)
  - Loss strategy (full-seq/answer-only) unchanged for all tasks

HF-loaded tasks (11):
  hellaswag_zeroshot, winogrande, copa, boolq, squad, coqa,
  openbook_qa, arc_easy, arc_challenge, commonsense_qa, piqa

Eval-bundle tasks (10, unchanged):
  lambada_openai, winograd, jeopardy, agi_eval_lsat_ar,
  bigbench_language_identification, bigbench_qa_wikidata,
  bigbench_dyck_languages, bigbench_cs_algorithms, bigbench_operators,
  bigbench_repeat_copy_logic

Usage:
  HF_ENDPOINT=https://hf-mirror.com python scripts/validation_set/prepare_core_bmk_v5.py

Output:
  data/core_bmk_21tasks_v5_tokenized.pt
  data/core_bmk_21tasks_v5.parquet
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
    "piqa",
    "arc_easy",
    "arc_challenge",
    "commonsense_qa",
    "agi_eval_lsat_ar",
    "bigbench_operators",
]

ANSWER_ONLY_TASKS = [
    "bigbench_dyck_languages",
    "bigbench_cs_algorithms",
    "bigbench_repeat_copy_logic",
]

ALL_V5_TASKS = FULL_SEQ_TASKS + ANSWER_ONLY_TASKS



MC_FULLSEQ_ANSWER_SEP_TASKS = {"commonsense_qa", "agi_eval_lsat_ar"}

MC_PER_SAMPLE_TASKS = {"openbook_qa", "piqa", "arc_easy", "arc_challenge"}

QUESTION_WORDS = {"how", "what", "why", "which", "where", "when", "who", "can", "do", "does", "is", "are", "should"}

HF_TASKS = {
    "hellaswag_zeroshot": {
        "repo": "Rowan/hellaswag",
        "config": None,
        "splits": ["train", "validation"],
        "task_type": "multiple_choice",
    },
    "winogrande": {
        "repo": "allenai/winogrande",
        "config": "winogrande_xl",
        "splits": ["train", "test", "validation"],
        "task_type": "schema",
    },
    "copa": {
        "repo": "aps/super_glue",
        "config": "copa",
        "splits": ["train", "test", "validation"],
        "task_type": "multiple_choice",
    },
    "boolq": {
        "repo": "google/boolq",
        "config": None,
        "splits": ["train", "validation"],
        "task_type": "multiple_choice",
    },
    "squad": {
        "repo": "rajpurkar/squad",
        "config": None,
        "splits": ["train", "validation"],
        "task_type": "language_modeling",
    },
    "coqa": {
        "repo": "coqa",
        "config": None,
        "splits": ["train", "validation"],
        "task_type": "language_modeling",
    },
    "openbook_qa": {
        "repo": "allenai/openbookqa",
        "config": "main",
        "splits": ["train", "test", "validation"],
        "task_type": "multiple_choice",
    },
    "arc_easy": {
        "repo": "allenai/ai2_arc",
        "config": "ARC-Easy",
        "splits": ["train", "test", "validation"],
        "task_type": "multiple_choice",
    },
    "arc_challenge": {
        "repo": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "splits": ["train", "test", "validation"],
        "task_type": "multiple_choice",
    },
    "commonsense_qa": {
        "repo": "tau/commonsense_qa",
        "config": None,
        "splits": ["train", "test", "validation"],
        "task_type": "multiple_choice",
    },
    "piqa": {
        "repo": "baber/piqa",
        "config": None,
        "splits": ["train", "validation"],
        "task_type": "multiple_choice",
    },
}





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
    elif task_label in ("commonsense_qa", "agi_eval_lsat_ar"):
        text = text.rstrip()
        if text.endswith("Answer:"):
            text = text[:-7].rstrip()
    return text


def extract_answer(item, task_label):
    choices = item.get("choices", [])
    gold = item.get("gold", 0)
    if not choices or gold < 0 or gold >= len(choices):
        return ""
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


def extract_pairs(task_type, task_label, items, continuation_delimiter=" "):
    is_full_seq = task_label in FULL_SEQ_TASKS
    pairs = []

    if task_type == "schema":
        for item in items:
            context_options = item.get("context_options", [])
            gold = item.get("gold", 0)
            continuation = item.get("continuation", "")
            if not continuation.strip() or continuation.strip() in {".", ",", "!", "?"}:
                continue
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
                if is_full_seq:
                    continuation = continuation_delimiter + continuation
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
                if task_label in MC_PER_SAMPLE_TASKS:
                    has_prefix = question.startswith("Question: ")
                    core = question[len("Question: "):] if has_prefix else question
                    first_word = core.split()[0].lower() if core.split() else ""
                    if core.endswith("?") or first_word in QUESTION_WORDS:
                        question = "Question: " + core
                        sep = "\nAnswer: "
                    else:
                        question = core
                        sep = " "
                elif continuation_delimiter != " ":
                    sep = continuation_delimiter
                elif task_label in MC_FULLSEQ_ANSWER_SEP_TASKS:
                    sep = "\nAnswer: "
                else:
                    sep = "" if question and question[-1].isspace() else " "
                pairs.append((question, sep + answer, is_full_seq))
            else:
                pairs.append((question, answer, is_full_seq))
        return pairs

    return pairs


# ============================================================
# HF → eval_bundle item conversion (verified in Step 1)
# ============================================================

def _convert_hellaswag(s):
    return {
        "query": s["ctx"],
        "choices": s["endings"],
        "gold": int(s["label"]),
    }


def _convert_winogrande(s):
    if not s["answer"]:
        return None
    sentence = s["sentence"]
    parts = sentence.split("_", 1)
    prefix = parts[0]
    suffix = parts[1].lstrip() if len(parts) > 1 else ""
    return {
        "context_options": [prefix + s["option1"], prefix + s["option2"]],
        "continuation": suffix,
        "gold": int(s["answer"]) - 1,
    }


def _convert_copa(s):
    gold = s["label"]
    if gold < 0:
        return None
    premise = s["premise"]
    if premise.endswith("."):
        premise = premise[:-1]
    connective = "therefore" if s["question"] == "effect" else "because"
    c1 = s["choice1"]
    c2 = s["choice2"]
    return {
        "query": f"{premise}, {connective}",
        "choices": [
            c1[0].lower() + c1[1:] if c1 else c1,
            c2[0].lower() + c2[1:] if c2 else c2,
        ],
        "gold": gold,
    }


def _convert_boolq(s):
    return {
        "query": f"Passage: {s['passage']}\nQuestion: {s['question']}",
        "choices": ["no", "yes"],
        "gold": 1 if s["answer"] else 0,
    }


def _convert_squad(s):
    answer = s["answers"]["text"][0]
    return {
        "context": f"Context: {s['context']}\nQuestion: {s['question']}\nAnswer: ",
        "continuation": answer,
    }


def _convert_coqa_story(s):
    story = s["story"]
    questions = s["questions"]
    answers = s["answers"]["input_text"]
    header = "Below is a story followed by a series of related questions. Please answer the final question by referring to the story and the previous questions."
    items = []
    for i in range(len(questions)):
        if i == 0:
            context = f"{header}\nStory: {story}\n\nFinal question:\nQuestion: {questions[i]}\nAnswer: "
        else:
            preceding = ""
            for j in range(i):
                preceding += f"\nQuestion: {questions[j]}\nAnswer: {answers[j]}"
            context = f"{header}\nStory: {story}\nPreceding questions:{preceding}\n\nFinal question:\nQuestion: {questions[i]}\nAnswer: "
        items.append({"context": context, "continuation": answers[i]})
    return items


def _convert_openbook_qa(s):
    texts = s["choices"]["text"]
    labels = s["choices"]["label"]
    gold_idx = labels.index(s["answerKey"])
    return {
        "query": s["question_stem"],
        "choices": texts,
        "gold": gold_idx,
    }


def _convert_arc(s):
    texts = s["choices"]["text"]
    labels = s["choices"]["label"]
    gold_idx = labels.index(s["answerKey"])
    return {
        "query": "Question: " + s["question"],
        "choices": texts,
        "gold": gold_idx,
    }


def _convert_commonsense_qa(s):
    ak = s["answerKey"]
    if not ak:
        return None
    texts = s["choices"]["text"]
    labels = s["choices"]["label"]
    query = f"Question: {s['question']}\nChoices:\n"
    for label, text in zip(labels, texts):
        query += f"{label}. {text}\n"
    gold_idx = labels.index(ak)
    return {
        "query": query,
        "choices": labels,
        "gold": gold_idx,
    }


def _convert_piqa(s):
    if s["label"] < 0:
        return None
    return {
        "query": "Question: " + s["goal"] + "\n",
        "choices": [s["sol1"], s["sol2"]],
        "gold": s["label"],
    }


_CONVERTERS = {
    "hellaswag_zeroshot": _convert_hellaswag,
    "winogrande": _convert_winogrande,
    "copa": _convert_copa,
    "boolq": _convert_boolq,
    "squad": _convert_squad,
    "openbook_qa": _convert_openbook_qa,
    "arc_easy": _convert_arc,
    "arc_challenge": _convert_arc,
    "commonsense_qa": _convert_commonsense_qa,
    "piqa": _convert_piqa,
}


def load_hf_task(task_label, hf_config):
    from datasets import load_dataset

    repo = hf_config["repo"]
    config = hf_config["config"]
    splits = hf_config["splits"]

    all_items = []
    split_counts = {}

    for split in splits:
        try:
            if config:
                ds = load_dataset(repo, config, split=split, trust_remote_code=True)
            else:
                ds = load_dataset(repo, split=split, trust_remote_code=True)
        except Exception as e:
            print(f"    WARNING: failed to load {repo}/{config} split={split}: {e}")
            continue

        if task_label == "coqa":
            items = []
            for s in ds:
                items.extend(_convert_coqa_story(s))
        else:
            converter = _CONVERTERS[task_label]
            items = []
            for s in ds:
                item = converter(s)
                if item is not None:
                    items.append(item)

        split_counts[split] = len(items)
        all_items.extend(items)

    return all_items, split_counts


def main():
    parser = argparse.ArgumentParser(
        description="Create CORE-BMK v5 validation set for QuaDMix"
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

    hf_count = len(HF_TASKS)
    eb_count = len(ALL_V5_TASKS) - hf_count
    print(f"\nv5 data sources:")
    print(f"  HuggingFace tasks ({hf_count}): {list(HF_TASKS.keys())}")
    print(f"  Eval-bundle tasks ({eb_count}): {[t for t in ALL_V5_TASKS if t not in HF_TASKS]}")

    all_samples = []
    task_meta = []

    for label in ALL_V5_TASKS:
        if label not in task_map:
            print(f"  WARNING: {label} not found in core.yaml")
            continue

        task = task_map[label]
        task_type = task["icl_task_type"]
        dataset_uri = task["dataset_uri"]
        is_full_seq = label in FULL_SEQ_TASKS
        loss_strategy = "full_sequence" if is_full_seq else "answer_only"
        is_hf = label in HF_TASKS

        print(f"\nLoading task: {label} (type={task_type}, loss={loss_strategy}, source={'HF' if is_hf else 'eval_bundle'})")

        if is_hf:
            hf_cfg = HF_TASKS[label]
            items, split_counts = load_hf_task(label, hf_cfg)
            for split_name, count in split_counts.items():
                print(f"  HF {split_name}: {count} items")
            print(f"  Total items (merged): {len(items)}")
        else:
            items = load_task_items(args.eval_bundle, dataset_uri)
            print(f"  Total items: {len(items)}")

        cont_delim = task.get("continuation_delimiter", " ")
        pairs = extract_pairs(task_type, label, items, continuation_delimiter=cont_delim)
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
            "data_source": "huggingface" if is_hf else "eval_bundle",
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
    
    contexts = [ctx for ctx, _, _ in all_samples]
    continuations = [cont for _, cont, _ in all_samples]
    is_full_seq_flags = [is_f for _, _, is_f in all_samples]
    
    print(f"  Batch encoding {len(contexts)} contexts...")
    ctx_encoded = tokenizer(contexts, add_special_tokens=False, verbose=False)
    ctx_ids_list = ctx_encoded["input_ids"]
    
    print(f"  Batch encoding {len(continuations)} continuations...")
    cont_encoded = tokenizer(continuations, add_special_tokens=False, verbose=False)
    cont_ids_list = cont_encoded["input_ids"]
    
    print(f"  Building tensors...")
    n_samples = len(all_samples)
    token_ids = torch.zeros(n_samples, block_size, dtype=torch.long)
    loss_mask = torch.zeros(n_samples, block_size, dtype=torch.bool)
    
    for idx in range(n_samples):
        context_ids = ctx_ids_list[idx]
        cont_ids = cont_ids_list[idx]
        full_ids = context_ids + cont_ids
        seq_len = min(len(full_ids), block_size)
        
        if seq_len > 0:
            token_ids[idx, :seq_len] = torch.tensor(full_ids[:seq_len], dtype=torch.long)
        
        if is_full_seq_flags[idx]:
            loss_mask[idx, :seq_len] = True
        else:
            ans_start = min(len(context_ids), seq_len)
            loss_mask[idx, ans_start:seq_len] = True
        
        if (idx + 1) % 10000 == 0:
            print(f"  {idx+1}/{n_samples}")

    task_labels = []
    for tm in task_meta:
        for _ in range(tm["sampled"]):
            task_labels.append(tm["label"])

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
            masked = loss_mask[_offset + i].sum().item()
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
            "loss_strategy": f"mixed (full_sequence={len(FULL_SEQ_TASKS)} tasks, answer_only={len(ANSWER_ONLY_TASKS)} tasks)",
            "source": f"CORE-BMK-{len(task_meta)}tasks-v5",
            "eval_bundle": args.eval_bundle,
            "num_samples_per_task": args.num_samples_per_task,
            "seed": seed,
            "description": (
                f"CORE-BMK v5: {len(task_meta)} tasks (all CORE benchmarks, deduped). "
                f"11 tasks loaded from HuggingFace (train+test+val merged), 10 from eval_bundle. "
                f"Full-seq tasks ({len(FULL_SEQ_TASKS)}): context+continuation = natural text, all tokens in loss. "
                f"Answer-only tasks ({len(ANSWER_ONLY_TASKS)}): context is Q+A/SFT/symbolic, only answer tokens in loss. "
                "piqa/arc_easy/arc_challenge use '\\nAnswer: ' separator for natural Q&A format. "
                "SFT artifacts cleaned for full-seq tasks (Passage:/Context:/instruction prefixes). "
                "MC-embedded tasks (commonsense_qa, agi_eval_lsat_ar) use letter labels as answers. "
                "commonsense_qa uses all 5 choices (A-E) from HF."
            ),
            "tasks": task_meta,
        }
    }

    save_path = os.path.join(output_dir, "core_bmk_21tasks_v5_tokenized.pt")
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
    parquet_path = os.path.join(output_dir, "core_bmk_21tasks_v5.parquet")
    pq.write_table(table, parquet_path)
    pq_size = os.path.getsize(parquet_path) / 1024**2
    print(f"  Saved: {parquet_path} ({pq_size:.1f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
