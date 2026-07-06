#!/usr/bin/env python3
"""
Prepare CAP v1 (Capability-Aligned Proxy) validation set for QuaDMix.

CAP uses capability-aligned TRAINING data as validation set, instead of
benchmark test data (BMK) or SFT dialogue data (OH).

Key design principles:
  - 70% external training data (proven effective in literature) + 30% benchmark train
  - Benchmark train portion uses equal-ratio sampling per benchmark
  - Rich text format (long context, explanations, reasoning)
  - Full-sequence loss for all samples (strong signal)
  - 5 capability clusters (not 21 per-task or 1 aggregate)

5 capability clusters:
  1. language_understanding: NaturalReasoning (70%) + HellaSwag/WinoGrande/LAMBADA (30%)
  2. common_sense_reasoning: OpenOrca CoT (70%) + CommonsenseQA/PIQA/COPA/OpenBookQA (30%)
  3. world_knowledge: NaturalReasoning science (70%) + ARC-Easy/ARC-Challenge (30%)
  4. reading_comprehension: HotpotQA + QASPER (70%) + SQuAD/BoolQ/CoQA (30%)
  5. symbol_logic: Orca-Math + MetaMathQA + NuminaMath-CoT (70%) + GSM8K/synthetic (30%)

Usage:
  HF_ENDPOINT=https://huggingface.co python scripts/validation_set/prepare_cap_v1.py

Output:
  data/cap_v1_tokenized.pt
  data/cap_v1.parquet
"""

import os
import random
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer


OUTPUT_DIR = "data"
SEED = 42
BLOCK_SIZE = 2048

EXTERNAL_RATIO = 0.70
BENCHMARK_RATIO = 0.30

CLUSTER_CONFIG = {
    "language_understanding": {
        "target_samples": 8000,
        "external_sources": ["natural_reasoning_lang"],
        "benchmark_sources": ["hellaswag", "winogrande", "lambada"],
    },
    "common_sense_reasoning": {
        "target_samples": 8000,
        "external_sources": ["openorca_cot"],
        "benchmark_sources": ["commonsense_qa", "piqa", "copa", "openbook_qa"],
    },
    "world_knowledge": {
        "target_samples": 8000,
        "external_sources": ["natural_reasoning_sci"],
        "benchmark_sources": ["arc_easy", "arc_challenge"],
    },
    "reading_comprehension": {
        "target_samples": 8000,
        "external_sources": ["hotpotqa", "qasper"],
        "benchmark_sources": ["squad", "boolq", "coqa"],
    },
    "symbol_logic": {
        "target_samples": 8000,
        "external_sources": ["orca_math", "metamathqa", "numinamath_cot"],
        "benchmark_sources": ["gsm8k", "synthetic_dyck", "synthetic_operators", "synthetic_repeat_copy"],
    },
}


def _load_natural_reasoning(category_keywords=None, max_samples=None):
    from datasets import load_dataset
    ds = load_dataset("facebook/natural_reasoning", split="train", streaming=True)
    texts = []
    for s in ds:
        question = s.get("question", "")
        ref_answer = s.get("reference_answer", "")
        responses = s.get("responses", [])
        if not question:
            continue
        answer = ""
        if responses and isinstance(responses, list) and len(responses) > 0:
            answer = responses[0].get("response", "") if isinstance(responses[0], dict) else str(responses[0])
        if not answer and ref_answer:
            answer = ref_answer
        if not answer:
            continue
        if category_keywords is not None:
            q_lower = question.lower()
            if not any(kw in q_lower for kw in category_keywords):
                continue
        text = f"Question: {question}\nAnswer: {answer}"
        if len(text) < 20:
            continue
        texts.append(text)
        if max_samples and len(texts) >= max_samples:
            break
    return texts


def _load_natural_reasoning_lang():
    lang_keywords = [
        "language", "grammar", "sentence", "word", "meaning", "translate",
        "paraphrase", "synonym", "antonym", "text", "reading", "comprehension",
        "passage", "context", "interpret", "summarize", "summary", "rewrite",
        "analogy", "metaphor", "idiom", "pronoun", "verb", "noun", "adjective",
        "semantics", "syntax", "linguistic", "etymology", "definition",
        "what does", "what is the meaning", "explain the phrase",
    ]
    return _load_natural_reasoning(category_keywords=lang_keywords, max_samples=20000)


def _load_natural_reasoning_sci():
    sci_keywords = [
        "physics", "chemistry", "biology", "science", "energy", "force",
        "mass", "atom", "molecule", "cell", "organism", "evolution",
        "gravity", "electron", "proton", "neutron", "quantum", "thermodynamic",
        "entropy", "velocity", "acceleration", "momentum", "wavelength",
        "frequency", "reaction", "catalyst", "enzyme", "protein", "dna",
        "rna", "photosynthesis", "mitosis", "ecosystem", "climate",
        "geology", "astronomy", "planet", "star", "galaxy", "orbit",
        "equation", "derive", "prove", "theorem", "calculate",
        "voltage", "current", "resistance", "magnetic", "electric",
        "optics", "refraction", "diffraction", "spectrum",
    ]
    return _load_natural_reasoning(category_keywords=sci_keywords, max_samples=20000)


def _load_openorca_cot():
    import pandas as pd
    local_path = "/tmp/opencode/cap_data/1M-GPT4-Augmented.parquet"
    if os.path.exists(local_path):
        print(f"    Loading from local: {local_path}")
        df = pd.read_parquet(local_path)
        cot = df[df["id"].str.startswith("cot.")]
        texts = []
        for _, row in cot.iterrows():
            q = row.get("question", "")
            r = row.get("response", "")
            if not q or not r:
                continue
            text = f"Question: {q}\nAnswer: {r}"
            if len(text) < 20:
                continue
            texts.append(text)
            if len(texts) >= 20000:
                break
        return texts
    from datasets import load_dataset
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    texts = []
    for s in ds:
        sid = s.get("id", "")
        if not sid.startswith("cot."):
            continue
        question = s.get("question", "")
        response = s.get("response", "")
        if not question or not response:
            continue
        text = f"Question: {question}\nAnswer: {response}"
        if len(text) < 20:
            continue
        texts.append(text)
        if len(texts) >= 20000:
            break
    return texts


def _load_hotpotqa():
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="train")
    texts = []
    for s in ds:
        question = s.get("question", "")
        answer = s.get("answer", "")
        context = s.get("context", {})
        if not question or not answer:
            continue
        ctx_titles = context.get("title", [])
        ctx_sentences = context.get("sentences", [])
        ctx_parts = []
        for title, sents in zip(ctx_titles, ctx_sentences):
            if isinstance(sents, list):
                ctx_parts.append(f"{title}: {' '.join(sents)}")
            else:
                ctx_parts.append(f"{title}: {sents}")
        ctx_text = "\n".join(ctx_parts[:3])
        if ctx_text:
            text = f"{ctx_text}\nQuestion: {question}\nAnswer: {answer}"
        else:
            text = f"Question: {question}\nAnswer: {answer}"
        if len(text) < 20:
            continue
        texts.append(text)
        if len(texts) >= 10000:
            break
    return texts


def _load_qasper():
    from datasets import load_dataset
    ds = load_dataset("hulki/allenai_qasper", split="train")
    texts = []
    for s in ds:
        context = s.get("context", "")
        questions = s.get("questions", [])
        answers = s.get("answers", [])
        if not context or not questions:
            continue
        for i, q in enumerate(questions):
            if i >= len(answers):
                break
            ans_list = answers[i]
            if not q or not ans_list:
                continue
            ans = ans_list[0] if isinstance(ans_list, list) else str(ans_list)
            if not ans:
                continue
            ctx_truncated = context[:2000]
            text = f"Paper: {ctx_truncated}\nQuestion: {q}\nAnswer: {ans}"
            if len(text) < 20:
                continue
            texts.append(text)
    return texts


def _load_orca_math():
    from datasets import load_dataset
    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True)
    texts = []
    for s in ds:
        question = s.get("question", "")
        answer = s.get("answer", "")
        if not question or not answer:
            continue
        text = f"Question: {question}\nSolution: {answer}"
        if len(text) < 20:
            continue
        texts.append(text)
        if len(texts) >= 10000:
            break
    return texts


def _load_metamathqa():
    import json
    local_path = "/tmp/opencode/cap_data/MetaMathQA-395K.json"
    if os.path.exists(local_path):
        print(f"    Loading from local: {local_path}")
        with open(local_path) as f:
            data = json.load(f)
        texts = []
        for s in data:
            query = s.get("query", "")
            response = s.get("response", "")
            if not query or not response:
                continue
            text = f"Question: {query}\nSolution: {response}"
            if len(text) < 20:
                continue
            texts.append(text)
            if len(texts) >= 10000:
                break
        return texts
    from datasets import load_dataset
    ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    texts = []
    for s in ds:
        query = s.get("query", "")
        response = s.get("response", "")
        if not query or not response:
            continue
        text = f"Question: {query}\nSolution: {response}"
        if len(text) < 20:
            continue
        texts.append(text)
        if len(texts) >= 10000:
            break
    return texts


def _load_numinamath_cot():
    from datasets import load_dataset
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    texts = []
    for s in ds:
        problem = s.get("problem", "")
        solution = s.get("solution", "")
        if not problem or not solution:
            continue
        text = f"Problem: {problem}\nSolution: {solution}"
        if len(text) < 20:
            continue
        texts.append(text)
        if len(texts) >= 10000:
            break
    return texts


def _load_hellaswag():
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="train")
    texts = []
    for s in ds:
        ctx = s["ctx"]
        endings = s["endings"]
        label = int(s["label"])
        if not ctx or not endings or label < 0 or label >= len(endings):
            continue
        correct_ending = endings[label]
        text = f"{ctx} {correct_ending}"
        if len(text) < 20:
            continue
        texts.append(text)
    return texts


def _load_commonsense_qa():
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa", split="train")
    texts = []
    for s in ds:
        question = s.get("question", "")
        choices = s.get("choices", {})
        answer_key = s.get("answerKey", "")
        if not question or not choices or not answer_key:
            continue
        choice_texts = choices.get("text", [])
        choice_labels = choices.get("label", [])
        if answer_key not in choice_labels:
            continue
        idx = choice_labels.index(answer_key)
        if idx >= len(choice_texts):
            continue
        correct = choice_texts[idx]
        if not correct:
            continue
        text = f"Question: {question}\nAnswer: {correct}"
        if len(text) < 20:
            continue
        texts.append(text)
    return texts


def _load_piqa():
    from datasets import load_dataset
    ds = load_dataset("baber/piqa", split="train")
    texts = []
    for s in ds:
        goal = s.get("goal", "")
        sol1 = s.get("sol1", "")
        sol2 = s.get("sol2", "")
        label = s.get("label", -1)
        if not goal or label < 0:
            continue
        solutions = [sol1, sol2]
        if label >= len(solutions):
            continue
        correct = solutions[label]
        if not correct:
            continue
        text = f"Goal: {goal}\nSolution: {correct}"
        if len(text) < 20:
            continue
        texts.append(text)
    return texts


def _load_arc(config_name):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", config_name, split="train")
    texts = []
    for s in ds:
        question = s.get("question", "")
        choices = s.get("choices", {})
        answer_key = s.get("answerKey", "")
        if not question or not choices or not answer_key:
            continue
        choice_texts = choices.get("text", [])
        choice_labels = choices.get("label", [])
        if answer_key not in choice_labels:
            continue
        idx = choice_labels.index(answer_key)
        if idx >= len(choice_texts):
            continue
        correct = choice_texts[idx]
        if not correct:
            continue
        text = f"Question: {question}\nAnswer: {correct}"
        if len(text) < 20:
            continue
        texts.append(text)
    return texts


def _load_squad():
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="train")
    texts = []
    for s in ds:
        context = s.get("context", "")
        question = s.get("question", "")
        answers = s.get("answers", {})
        answer_texts = answers.get("text", [])
        if not context or not question or not answer_texts:
            continue
        answer = answer_texts[0]
        if not answer:
            continue
        text = f"{context}\nQuestion: {question}\nAnswer: {answer}"
        if len(text) < 20:
            continue
        texts.append(text)
    return texts


def _load_coqa():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/coqa", split="train")
    texts = []
    for s in ds:
        story = s.get("story", "")
        questions = s.get("questions", [])
        answers = s.get("answers", {})
        answer_texts = answers.get("input_text", [])
        if not story or not questions:
            continue
        for i, q in enumerate(questions):
            if i >= len(answer_texts):
                break
            a = answer_texts[i]
            if not q or not a:
                continue
            text = f"Story: {story}\nQuestion: {q}\nAnswer: {a}"
            if len(text) < 20:
                continue
            texts.append(text)
    return texts


def _load_gsm8k():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    texts = []
    for s in ds:
        question = s.get("question", "")
        answer = s.get("answer", "")
        if not question or not answer:
            continue
        text = f"Question: {question}\nSolution: {answer}"
        if len(text) < 20:
            continue
        texts.append(text)
    return texts


def _load_winogrande():
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    texts = []
    for s in ds:
        sentence = s.get("sentence", "")
        option1 = s.get("option1", "")
        option2 = s.get("option2", "")
        answer = s.get("answer", "")
        if not sentence or not answer:
            continue
        try:
            ans_idx = int(answer) - 1
        except (ValueError, TypeError):
            continue
        options = [option1, option2]
        if ans_idx < 0 or ans_idx >= len(options):
            continue
        correct = options[ans_idx]
        if not correct:
            continue
        filled = sentence.replace("_", correct)
        text = f"Sentence: {sentence}\nAnswer: {correct}\nCompleted: {filled}"
        if len(text) < 20:
            continue
        texts.append(text)
    return texts


def _load_lambada():
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    texts = []
    for s in ds:
        text = s.get("text", "").strip()
        if len(text) < 20:
            continue
        texts.append(text)
    return texts


def _load_copa():
    from datasets import load_dataset
    ds = load_dataset("pkavumba/balanced-copa", split="train")
    texts = []
    for s in ds:
        premise = s.get("premise", "")
        question = s.get("question", "")
        choice1 = s.get("choice1", "")
        choice2 = s.get("choice2", "")
        label = s.get("label", -1)
        if not premise or not question or label < 0:
            continue
        choices = [choice1, choice2]
        if label >= len(choices):
            continue
        correct = choices[label]
        if not correct:
            continue
        connective = "therefore" if question == "effect" else "because"
        premise_clean = premise.rstrip(".")
        correct_clean = correct[0].lower() + correct[1:] if correct else correct
        text = f"{premise_clean}, {connective} {correct_clean}"
        if len(text) < 20:
            continue
        texts.append(text)
    return texts


def _load_openbook_qa():
    from datasets import load_dataset
    ds = load_dataset("allenai/openbookqa", "main", split="train")
    texts = []
    for s in ds:
        question = s.get("question_stem", "")
        choices = s.get("choices", {})
        answer_key = s.get("answerKey", "")
        if not question or not choices or not answer_key:
            continue
        choice_texts = choices.get("text", [])
        choice_labels = choices.get("label", [])
        if answer_key not in choice_labels:
            continue
        idx = choice_labels.index(answer_key)
        if idx >= len(choice_texts):
            continue
        correct = choice_texts[idx]
        if not correct:
            continue
        text = f"Question: {question}\nAnswer: {correct}"
        if len(text) < 20:
            continue
        texts.append(text)
    return texts


def _load_boolq():
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="train")
    texts = []
    for s in ds:
        passage = s.get("passage", "")
        question = s.get("question", "")
        answer = s.get("answer", None)
        if not passage or not question or answer is None:
            continue
        ans_text = "yes" if answer else "no"
        text = f"Passage: {passage}\nQuestion: {question}\nAnswer: {ans_text}"
        if len(text) < 20:
            continue
        texts.append(text)
    return texts


def _generate_synthetic_dyck(n_samples=5000, seed=42):
    rng = random.Random(seed)
    bracket_pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    texts = []
    for _ in range(n_samples):
        n_types = rng.choice([1, 2, 3])
        depth = rng.randint(2, 8)
        seq_len = rng.randint(depth * 2, depth * 4)
        if seq_len % 2 != 0:
            seq_len += 1
        chosen = bracket_pairs[:n_types]
        stack = []
        seq = []
        for _ in range(seq_len):
            if not stack or (rng.random() < 0.5 and len(stack) < depth):
                open_b, close_b = rng.choice(chosen)
                seq.append(open_b)
                stack.append(close_b)
            elif stack:
                seq.append(stack.pop())
        while stack:
            seq.append(stack.pop())
        prompt = "".join(seq)
        text = f"Dyck language sequence: {prompt}\nThis is a valid balanced bracket sequence."
        texts.append(text)
    return texts


def _generate_synthetic_operators(n_samples=5000, seed=42):
    rng = random.Random(seed)
    texts = []
    for _ in range(n_samples):
        n_ops = rng.randint(1, 3)
        op_symbols = ["@", "#", "$", "%", "&", "~"]
        chosen_ops = rng.sample(op_symbols, n_ops)
        for sym in chosen_ops:
            op_type = rng.choice(["binary", "unary_prefix", "unary_suffix"])
            if op_type == "binary":
                a, b = rng.randint(1, 20), rng.randint(1, 20)
                result = rng.choice([a + b, a * b, a - b, a * b - 1, a + b + 1])
                expr = f"{a} {sym} {b}"
                text = f"Define: a {sym} b = {result}\nCompute: {expr} = {result}"
            else:
                a = rng.randint(1, 20)
                result = rng.choice([a * 2, a + 3, a * a, a - 1])
                expr = f"{sym}{a}" if op_type == "unary_prefix" else f"{a}{sym}"
                text = f"Define: {sym}x = {result} when x = {a}\nCompute: {expr} = {result}"
            texts.append(text)
    return texts


def _generate_synthetic_repeat_copy(n_samples=5000, seed=42):
    rng = random.Random(seed)
    words = [
        "apple", "banana", "cherry", "dog", "cat", "fish", "bird", "tree",
        "house", "car", "book", "pen", "sun", "moon", "star", "river",
        "mountain", "ocean", "cloud", "rain", "snow", "wind", "fire", "ice",
        "red", "blue", "green", "gold", "silver", "black", "white", "purple",
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        "hello", "world", "data", "python", "code", "test", "run", "stop",
    ]
    texts = []
    for _ in range(n_samples):
        n_instructions = rng.randint(1, 3)
        parts = []
        results = []
        for _ in range(n_instructions):
            word = rng.choice(words)
            count = rng.randint(2, 5)
            parts.append(f"say {word} {count} times")
            results.extend([word] * count)
        instruction = " and ".join(parts)
        if n_instructions > 1 and rng.random() < 0.5:
            all_parts = list(results)
            repeat_n = rng.randint(2, 3)
            instruction = f"{instruction}, then repeat all of this {repeat_n} times"
            results = all_parts * repeat_n
        output = " ".join(results)
        text = f"Q: {instruction}\nA: {output}"
        if len(text) < 20:
            continue
        texts.append(text)
    return texts


_SOURCE_LOADERS = {
    "natural_reasoning_lang": _load_natural_reasoning_lang,
    "natural_reasoning_sci": _load_natural_reasoning_sci,
    "openorca_cot": _load_openorca_cot,
    "hotpotqa": _load_hotpotqa,
    "qasper": _load_qasper,
    "orca_math": _load_orca_math,
    "metamathqa": _load_metamathqa,
    "numinamath_cot": _load_numinamath_cot,
    "hellaswag": _load_hellaswag,
    "winogrande": _load_winogrande,
    "lambada": _load_lambada,
    "commonsense_qa": _load_commonsense_qa,
    "piqa": _load_piqa,
    "copa": _load_copa,
    "openbook_qa": _load_openbook_qa,
    "arc_easy": lambda: _load_arc("ARC-Easy"),
    "arc_challenge": lambda: _load_arc("ARC-Challenge"),
    "boolq": _load_boolq,
    "squad": _load_squad,
    "coqa": _load_coqa,
    "gsm8k": _load_gsm8k,
    "synthetic_dyck": _generate_synthetic_dyck,
    "synthetic_operators": _generate_synthetic_operators,
    "synthetic_repeat_copy": _generate_synthetic_repeat_copy,
}


def _load_source(src_name):
    loader = _SOURCE_LOADERS.get(src_name)
    if loader is None:
        print(f"  WARNING: no loader for source '{src_name}'")
        return []
    print(f"  Loading {src_name}...")
    try:
        texts = loader()
    except Exception as e:
        print(f"  ERROR loading {src_name}: {e}")
        texts = []
    print(f"  {src_name}: {len(texts)} samples loaded")
    return texts


def _sample_equal_ratio(all_source_texts, target_count):
    """Sample equally from each source to reach target_count total.
    If some sources have fewer samples than their equal share,
    redistribute the deficit to sources with surplus capacity."""
    sources = {k: v for k, v in all_source_texts.items() if v}
    if not sources:
        return [], {}

    n_sources = len(sources)
    per_source = target_count // n_sources

    allocations = {}
    for src_name, texts in sources.items():
        allocations[src_name] = min(per_source, len(texts))

    total_allocated = sum(allocations.values())
    remaining = target_count - total_allocated
    if remaining > 0:
        surplus = {k: len(v) - allocations[k] for k, v in sources.items() if len(v) > allocations[k]}
        total_surplus = sum(surplus.values())
        if total_surplus > 0:
            for src_name in surplus:
                extra = int(remaining * surplus[src_name] / total_surplus)
                extra = min(extra, surplus[src_name])
                allocations[src_name] += extra

    total_allocated = sum(allocations.values())
    remaining = target_count - total_allocated
    if remaining > 0:
        for src_name, texts in sources.items():
            can_add = len(texts) - allocations[src_name]
            add = min(remaining, can_add)
            if add > 0:
                allocations[src_name] += add
                remaining -= add
            if remaining <= 0:
                break

    sampled = []
    source_counts = {}
    for src_name, texts in sources.items():
        n = allocations[src_name]
        if len(texts) > n:
            picked = random.sample(texts, n)
        else:
            picked = texts
        sampled.extend(picked)
        source_counts[src_name] = len(picked)
        print(f"    {src_name}: picked {len(picked)} ({'all' if len(texts) <= n else f'{n} of {len(texts)}'})")

    return sampled, source_counts


def main():
    parser = argparse.ArgumentParser(
        description="Create CAP v1 validation set for QuaDMix"
    )
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help=f"Block size for tokenization (default: {BLOCK_SIZE})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed (default: {SEED})")
    parser.add_argument("--samples-per-cluster", type=int, default=0,
                        help="Override target samples per cluster (0=use config defaults)")
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
    print(f"Mix ratio: external={EXTERNAL_RATIO:.0%}, benchmark={BENCHMARK_RATIO:.0%}")

    all_texts = []
    all_labels = []
    cluster_meta = {}

    for cluster_name, cfg in CLUSTER_CONFIG.items():
        target = args.samples_per_cluster if args.samples_per_cluster > 0 else cfg["target_samples"]
        ext_target = int(target * EXTERNAL_RATIO)
        bmk_target = target - ext_target
        ext_sources = cfg["external_sources"]
        bmk_sources = cfg["benchmark_sources"]

        print(f"\n{'='*60}")
        print(f"Cluster: {cluster_name}")
        print(f"  Target: {target} (external={ext_target}, benchmark={bmk_target})")
        print(f"  External sources: {ext_sources}")
        print(f"  Benchmark sources: {bmk_sources}")

        cluster_texts = []
        ext_counts = {}
        bmk_counts = {}

        print(f"\n  --- External data ({ext_target} samples) ---")
        ext_texts = {}
        for src in ext_sources:
            texts = _load_source(src)
            ext_texts[src] = texts

        ext_sampled, ext_counts = _sample_equal_ratio(ext_texts, ext_target)
        cluster_texts.extend(ext_sampled)
        print(f"  External total: {len(ext_sampled)}")

        print(f"\n  --- Benchmark train data ({bmk_target} samples, equal-ratio) ---")
        bmk_texts = {}
        for src in bmk_sources:
            texts = _load_source(src)
            bmk_texts[src] = texts

        bmk_sampled, bmk_counts = _sample_equal_ratio(bmk_texts, bmk_target)
        cluster_texts.extend(bmk_sampled)
        print(f"  Benchmark total: {len(bmk_sampled)}")

        if not cluster_texts:
            print(f"  SKIPPED: no samples loaded")
            cluster_meta[cluster_name] = {
                "external_sources": ext_sources,
                "benchmark_sources": bmk_sources,
                "external_counts": ext_counts,
                "benchmark_counts": bmk_counts,
                "loaded": 0,
                "sampled": 0,
            }
            continue

        avg_chars = sum(len(t) for t in cluster_texts) / len(cluster_texts)
        min_chars = min(len(t) for t in cluster_texts)
        max_chars = max(len(t) for t in cluster_texts)

        print(f"\n  Cluster total: {len(cluster_texts)}")
        print(f"  Chars: avg={avg_chars:.0f}, min={min_chars}, max={max_chars}")
        if cluster_texts:
            print(f"  Example: \"{cluster_texts[0][:150]}\"")

        all_texts.extend(cluster_texts)
        all_labels.extend([cluster_name] * len(cluster_texts))

        cluster_meta[cluster_name] = {
            "external_sources": ext_sources,
            "benchmark_sources": bmk_sources,
            "external_counts": ext_counts,
            "benchmark_counts": bmk_counts,
            "total": len(cluster_texts),
            "external_total": len(ext_sampled),
            "benchmark_total": len(bmk_sampled),
            "avg_chars": round(avg_chars, 1),
            "min_chars": min_chars,
            "max_chars": max_chars,
        }

    print(f"\n{'='*60}")
    print(f"Total samples: {len(all_texts)}")
    for cn in CLUSTER_CONFIG:
        count = sum(1 for l in all_labels if l == cn)
        print(f"  {cn}: {count}")

    if not all_texts:
        print("ERROR: no samples collected, aborting")
        return

    print(f"\nTokenizing (block_size={block_size})...")
    print(f"  Batch encoding {len(all_texts)} texts...")
    encoded = tokenizer(all_texts, add_special_tokens=False, verbose=False)
    ids_list = encoded["input_ids"]

    print(f"  Building tensors...")
    n_samples = len(all_texts)
    token_ids = torch.zeros(n_samples, block_size, dtype=torch.long)
    loss_mask = torch.zeros(n_samples, block_size, dtype=torch.bool)

    for idx in range(n_samples):
        doc_ids = ids_list[idx]
        seq_len = min(len(doc_ids), block_size)
        if seq_len > 0:
            token_ids[idx, :seq_len] = torch.tensor(doc_ids[:seq_len], dtype=torch.long)
            loss_mask[idx, :seq_len] = True
        if (idx + 1) % 10000 == 0:
            print(f"  {idx+1}/{n_samples}")

    total_tokens = token_ids.numel()
    non_padding = (token_ids != tokenizer.pad_token_id).sum().item()
    masked_tokens = loss_mask.sum().item()
    print(f"\n  Tokenized: {token_ids.shape}")
    print(f"  Non-padding tokens: {non_padding:,} / {total_tokens:,} ({non_padding/total_tokens*100:.1f}%)")
    print(f"  Loss tokens (full-seq): {masked_tokens:,} ({masked_tokens/max(non_padding,1)*100:.1f}% of non-padding)")

    truncated = sum(1 for ids in ids_list if len(ids) > block_size)
    print(f"  Truncated (>block_size): {truncated}/{n_samples} ({truncated/n_samples*100:.1f}%)")

    output = {
        "token_ids": token_ids,
        "loss_mask": loss_mask,
        "task_labels": all_labels,
        "metadata": {
            "num_docs": len(token_ids),
            "block_size": block_size,
            "tokenizer": "gpt-neox-20b",
            "tokenizer_vocab": tokenizer.vocab_size,
            "model_vocab": 50432,
            "loss_strategy": "full_sequence (all clusters)",
            "source": "CAP-v1 (Capability-Aligned Proxy, external+benchmark mix)",
            "seed": seed,
            "mix_ratio": f"external={EXTERNAL_RATIO:.0%}, benchmark={BENCHMARK_RATIO:.0%}",
            "description": (
                f"CAP v1: {len(CLUSTER_CONFIG)} capability clusters. "
                f"70% external training data (proven effective) + 30% benchmark train (equal-ratio). "
                f"All clusters use full-sequence loss. "
                f"Clusters: {list(CLUSTER_CONFIG.keys())}. "
                f"Total samples: {len(token_ids)}. "
                f"External sources: Orca-Math, MetaMathQA, NuminaMath-CoT, NaturalReasoning, "
                f"OpenOrca CoT, HotpotQA, QASPER. "
                f"Designed to align proxy optimization with downstream reasoning capabilities."
            ),
            "clusters": cluster_meta,
        },
    }

    save_path = os.path.join(output_dir, "cap_v1_tokenized.pt")
    torch.save(output, save_path)
    file_size = os.path.getsize(save_path) / 1024**2
    print(f"\n  Saved: {save_path} ({file_size:.0f} MB)")

    print(f"\nExporting Parquet...")
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_rows = {
        "cluster": [],
        "text": [],
        "char_len": [],
        "token_len": [],
    }
    for i in range(n_samples):
        parquet_rows["cluster"].append(all_labels[i])
        parquet_rows["text"].append(all_texts[i])
        parquet_rows["char_len"].append(len(all_texts[i]))
        parquet_rows["token_len"].append(len(ids_list[i]))

    table = pa.table(parquet_rows)
    parquet_path = os.path.join(output_dir, "cap_v1.parquet")
    pq.write_table(table, parquet_path)
    pq_size = os.path.getsize(parquet_path) / 1024**2
    print(f"  Saved: {parquet_path} ({pq_size:.1f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
