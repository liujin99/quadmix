#!/usr/bin/env python3
"""Post-hoc quartile budget allocation — validation script.

Uses existing optimal_parameters.json + the full data pool to produce a
sampled_dataset where quality ranking (Eq.2) is computed normally
(non-stratified), but the sampling budget is redistributed across
length quartiles within each domain.

Default mode: equal allocation (N_domain/K per quartile), char-based.

With --match-benchmark-lengths: the allocation across quartiles is shaped
to match the token-length distribution of STEM benchmark questions
(arc, gsm8k, mmlu, gpqa, math). The full STEM corpus is tokenized with
the nanochat tokenizer for accurate token-based quartile boundaries.

Algorithm:
  1. Eq.1 (merge) + Eq.2 (compute_quality_ranks, non-stratified) + Eq.3
     (compute_sampling_values) → S(r) per document.
  2. Run baseline selection (_select_documents_vectorized) → record
     N_domain (selected count per domain).  Domain proportions are
     therefore identical to the QuadMix baseline.
  3. For each domain, split pool into K=4 quartiles by length
     (char_count by default, token_count with --match-benchmark-lengths).
     Scale S(r) within each quartile so that Σ scaled_S(r) = budget_per_q.
     Uniform mode: budget_per_q = N_domain/K.
     Benchmark mode: budget_per_q = N_domain * target_dist[k].
  4. Re-select with scaled S(r) → sampled_dataset.

Standalone script: no pipeline code is modified.

Usage:
  # Uniform allocation (char-based, default)
  python scripts/validate_stratified_selection.py \
      --preprocessed-dir /path/to/stem/parquets \
      --params-file result/quadmix_20260728_201517/optimal_parameters.json \
      --schema configs/schema_stem.yaml \
      --output result/stratified_validation

  # Benchmark-matched allocation (token-based, matches STEM eval question lengths)
  python scripts/validate_stratified_selection.py \
      --preprocessed-dir /path/to/stem/parquets \
      --params-file result/quadmix_20260728_201517/optimal_parameters.json \
      --schema configs/schema_stem.yaml \
      --output result/stratified_benchmark_matched \
      --match-benchmark-lengths \
      --tokenizer-path /path/to/tokenizer.pkl
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import time

os.environ.setdefault('MALLOC_ARENA_MAX', '4')

try:
    import quadmix  # noqa: F401
except ImportError:
    sys.path.insert(
        0,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'),
    )

import numpy as np

from quadmix.core.types import ParameterSet
from quadmix.data.metadata_manager import ShardMetadataManager
from quadmix.data.dataset_schema import DatasetSchema
from quadmix.core.quality_merger import compute_merged_quality_scores
from quadmix.core.quality_rank import compute_quality_ranks
from quadmix.core.sampler import compute_sampling_values
from quadmix.sampling.batch_sampler import (
    _select_documents_vectorized,
    save_sampled_dataset,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)


def reconstruct_params_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return ParameterSet.from_dict(data["quality_weights"], data["sampling_params"])


def resolve_schema_path(schema_arg):
    if os.path.isabs(schema_arg):
        return schema_arg
    return os.path.join(_PROJECT_ROOT, schema_arg)


def parse_args():
    p = argparse.ArgumentParser(
        description="Post-hoc quartile budget allocation validation",
    )
    p.add_argument("--preprocessed-dir", required=True,
                   help="Directory with parquet shards (same as pipeline --preprocessed-dir)")
    p.add_argument("--params-file", required=True,
                   help="Path to optimal_parameters.json from a prior QuaDMix run")
    p.add_argument("--schema", required=True,
                   help="Schema YAML path (relative to project root or absolute)")
    p.add_argument("--output", "-o", default=None,
                   help="Output directory (default: result/stratified_<timestamp>)")
    p.add_argument("--n-strata", type=int, default=4,
                   help="Number of length quartiles per domain (default: 4)")
    p.add_argument("--target-tokens", type=float, default=0.0,
                   help="Target tokens in billions (0 = no limit)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--match-benchmark-lengths", action="store_true",
                   help="Shape length distribution to match STEM benchmark question token lengths")
    p.add_argument("--tokenizer-path", default=None,
                   help="nanochat tokenizer.pkl or its dir "
                        "(default: $NANOCHAT_MODEL_DIR/tokenizer)")
    return p.parse_args()


def _compute_quartile_ids(values, K):
    """Assign quartile IDs (0..K-1) based on value quantiles."""
    quantiles = np.linspace(0, 1, K + 1)
    boundaries = np.quantile(values, quantiles)
    stratum_ids = np.zeros(len(values), dtype=np.int64)
    for k in range(1, K):
        stratum_ids[values > boundaries[k]] = k
    return stratum_ids, boundaries


def _resolve_tokenizer_path(args):
    if args.tokenizer_path:
        return args.tokenizer_path
    model_dir = os.environ.get(
        "NANOCHAT_MODEL_DIR", "/home/ma-user/work/nanochat_model_dir")
    default_tok = os.path.join(model_dir, "tokenizer")
    default_pkl = os.path.join(default_tok, "tokenizer.pkl")
    if os.path.isfile(default_pkl):
        print(f"  [info] tokenizer auto-detected: {default_pkl}")
        return default_tok
    return default_tok


def _load_tokenizer(tokenizer_path):
    pkl = tokenizer_path if os.path.isfile(tokenizer_path) \
        else os.path.join(tokenizer_path, "tokenizer.pkl")
    if not os.path.isfile(pkl):
        print(f"  WARNING: tokenizer not found at {pkl}")
        return None
    try:
        with open(pkl, "rb") as f:
            enc = pickle.load(f)
        print(f"  Tokenizer loaded from {pkl}")
        return enc
    except Exception as e:
        print(f"  WARNING: failed to load tokenizer: {e}")
        return None


_MMLU_STEM_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy",
    "college_biology", "college_chemistry", "college_computer_science",
    "college_mathematics", "college_physics", "computer_security",
    "conceptual_physics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "machine_learning",
    "medical_genetics", "virology",
]

_MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


def _load_benchmark_token_lengths(enc):
    """Load 6 STEM benchmarks from HuggingFace, tokenize questions, return per-benchmark token counts."""
    from datasets import load_dataset
    import random as _rng

    results = {}

    def _tok_batch(texts):
        if not texts:
            return []
        ids_batch = enc.encode_ordinary_batch(texts, num_threads=8)
        return [len(ids) for ids in ids_batch]

    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        texts = [f"Question: {s['question']}\nAnswer: {s['answer']}" for s in ds]
        results["gsm8k_cot"] = _tok_batch(texts)
        print(f"    gsm8k_cot: {len(results['gsm8k_cot'])} questions")
    except Exception as e:
        print(f"    WARNING: failed to load gsm8k: {e}")

    try:
        all_texts = []
        for subject in _MMLU_STEM_SUBJECTS:
            ds = load_dataset("cais/mmlu", subject, split="test")
            for s in ds:
                question = s.get("question", "")
                choices = s.get("choices", [])
                answer = s.get("answer", -1)
                if isinstance(answer, str):
                    answer = {"A": 0, "B": 1, "C": 2, "D": 3}.get(answer.upper(), -1)
                if answer < 0 or answer >= len(choices):
                    continue
                choice_str = "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(choices))
                text = f"Question: {question}\nChoices:\n{choice_str}\nAnswer: {chr(65+answer)}"
                all_texts.append(text)
        results["mmlu_stem"] = _tok_batch(all_texts)
        print(f"    mmlu_stem: {len(results['mmlu_stem'])} questions")
    except Exception as e:
        print(f"    WARNING: failed to load mmlu: {e}")

    for arc_name, arc_config in [("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge")]:
        try:
            ds = load_dataset("allenai/ai2_arc", arc_config, split="test")
            all_texts = []
            for s in ds:
                question = s.get("question", "")
                choices = s.get("choices", {})
                answer_key = s.get("answerKey", "")
                choice_texts = choices.get("text", [])
                choice_labels = choices.get("label", [])
                if not all([question, choice_texts, choice_labels, answer_key]):
                    continue
                choice_str = "\n".join(f"  {lbl}. {t}" for lbl, t in zip(choice_labels, choice_texts))
                text = f"Question: {question}\nChoices:\n{choice_str}\nAnswer: {answer_key}"
                all_texts.append(text)
            results[arc_name] = _tok_batch(all_texts)
            print(f"    {arc_name}: {len(results[arc_name])} questions")
        except Exception as e:
            print(f"    WARNING: failed to load {arc_name}: {e}")

    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        rng = _rng.Random(42)
        all_texts = []
        for s in ds:
            question = s.get("Question", "")
            correct = s.get("Correct Answer", "")
            incorrect = [s.get("Incorrect Answer 1", ""), s.get("Incorrect Answer 2", ""), s.get("Incorrect Answer 3", "")]
            if not all([question, correct] + incorrect):
                continue
            choices = [correct, incorrect[0], incorrect[1], incorrect[2]]
            indices = list(range(4))
            rng.shuffle(indices)
            shuffled = [choices[i] for i in indices]
            correct_idx = indices.index(0)
            choice_str = "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(shuffled))
            text = f"Question: {question}\nChoices:\n{choice_str}\nAnswer: {chr(65+correct_idx)}"
            all_texts.append(text)
        results["gpqa_diamond"] = _tok_batch(all_texts)
        print(f"    gpqa_diamond: {len(results['gpqa_diamond'])} questions")
    except Exception as e:
        print(f"    WARNING: failed to load gpqa: {e}")

    try:
        all_texts = []
        for subject in _MATH_SUBJECTS:
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
            for s in ds:
                problem = s.get("problem", "")
                solution = s.get("solution", "")
                if not problem or not solution:
                    continue
                text = f"Question: {problem}\nSolution: {solution}"
                all_texts.append(text)
        results["math_cot_500"] = _tok_batch(all_texts)
        print(f"    math_cot_500: {len(results['math_cot_500'])} questions")
    except Exception as e:
        print(f"    WARNING: failed to load math: {e}")

    return results


def _tokenize_corpus(mm, enc, chunk_size=50000):
    """Read all STEM doc texts and tokenize to get accurate token counts."""
    n_docs = mm.num_docs
    token_counts = np.zeros(n_docs, dtype=np.int64)
    all_indices = np.arange(n_docs, dtype=np.int64)

    t0 = time.time()
    for start in range(0, n_docs, chunk_size):
        end = min(start + chunk_size, n_docs)
        chunk_indices = all_indices[start:end]
        texts = mm.read_texts(chunk_indices, verbose=False)
        ids_batch = enc.encode_ordinary_batch(texts, num_threads=8)
        for i, ids in enumerate(ids_batch):
            token_counts[start + i] = len(ids)
        if start % (chunk_size * 5) == 0 or end == n_docs:
            elapsed = time.time() - t0
            pct = end / n_docs * 100
            eta = elapsed / end * (n_docs - end) if end > 0 else 0
            print(f"  Tokenized {end:,}/{n_docs:,} ({pct:.0f}%) — {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    print(f"  Tokenization complete: {n_docs:,} docs in {time.time()-t0:.1f}s")
    return token_counts


def _compute_target_dist(benchmark_token_counts, stem_token_counts, K):
    """Compute target distribution from benchmark token lengths mapped to global STEM quartiles.

    Each benchmark contributes equally (per-benchmark distribution averaged).
    Returns (target_dist, global_boundaries).
    """
    quantiles = np.linspace(0, 1, K + 1)
    global_boundaries = np.quantile(stem_token_counts, quantiles)

    per_benchmark_dists = []
    for name, counts in benchmark_token_counts.items():
        counts_arr = np.array(counts, dtype=np.float64)
        stratum_ids = np.zeros(len(counts_arr), dtype=np.int64)
        for k in range(1, K):
            stratum_ids[counts_arr > global_boundaries[k]] = k
        dist = np.zeros(K)
        for k in range(K):
            dist[k] = float(np.sum(stratum_ids == k))
        if dist.sum() > 0:
            dist = dist / dist.sum()
        per_benchmark_dists.append(dist)
        print(f"    {name}: {dist.tolist()}")

    target_dist = np.mean(per_benchmark_dists, axis=0)
    target_dist = target_dist / target_dist.sum()

    return target_dist, global_boundaries


def main():
    args = parse_args()

    output_dir = args.output or os.path.join(
        _PROJECT_ROOT, f"result/stratified_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)

    K = args.n_strata

    print("=" * 70)
    print("  Post-hoc Quartile Budget Allocation Validation")
    print(f"  Data:    {args.preprocessed_dir}")
    print(f"  Params:  {args.params_file}")
    print(f"  Schema:  {args.schema}")
    print(f"  Output:  {output_dir}")
    print(f"  Quartiles: {K}")
    print(f"  Seed:    {args.seed}")
    if args.match_benchmark_lengths:
        print(f"  Mode:    benchmark-length-matched (token-based)")
    else:
        print(f"  Mode:    uniform (char-based)")
    print("=" * 70)

    t_start = time.time()

    # ── Stage 1: Load metadata ──────────────────────────────
    _t = time.time()
    print(f"\n[Stage 1] Loading metadata...")
    schema = DatasetSchema.from_yaml(resolve_schema_path(args.schema))
    mm = ShardMetadataManager(args.preprocessed_dir, schema=schema)
    domain_names = mm.detected_domain_names
    quality_names = mm.detected_quality_names
    domain_labels = mm.domain_labels
    quality_scores = mm.quality_scores
    token_counts = mm.estimate_token_counts()
    char_counts = mm.doc_char_counts
    n_docs = mm.num_docs

    if char_counts is None or len(char_counts) == 0:
        print("ERROR: doc_char_counts not available in metadata.")
        return 1

    print(f"  {n_docs:,} docs across {mm.num_shards} shards")
    print(f"  {mm.num_domains} domains, {mm.num_quality_criteria} quality criteria")
    print(f"  Stage 1: {time.time()-_t:.1f}s")

    # ── Stage 1b: Benchmark length matching (optional) ─────
    target_dist = None
    length_values = char_counts
    benchmark_stats = None

    if args.match_benchmark_lengths:
        _t = time.time()
        print(f"\n[Stage 1b] Benchmark length matching...")
        tokenizer_path = _resolve_tokenizer_path(args)
        enc = _load_tokenizer(tokenizer_path)
        if enc is not None:
            print(f"  Tokenizing {n_docs:,} STEM docs for accurate token counts...")
            stem_token_counts = _tokenize_corpus(mm, enc)
            token_counts = stem_token_counts
            length_values = stem_token_counts

            print(f"  Loading benchmark questions from HuggingFace...")
            benchmark_token_counts = _load_benchmark_token_lengths(enc)
            if benchmark_token_counts:
                print(f"  Computing target distribution (K={K}, equal-weight per benchmark)...")
                target_dist, global_boundaries = _compute_target_dist(
                    benchmark_token_counts, stem_token_counts, K,
                )
                print(f"  Target distribution: {[round(x, 4) for x in target_dist]}")
                print(f"  Global token quartile boundaries: {[int(b) for b in global_boundaries]}")
                benchmark_stats = {
                    name: {"count": len(counts),
                           "mean": round(float(np.mean(counts)), 1),
                           "median": round(float(np.median(counts)), 1),
                           "min": int(min(counts)), "max": int(max(counts))}
                    for name, counts in benchmark_token_counts.items()
                }
            else:
                print(f"  WARNING: no benchmarks loaded, falling back to uniform distribution")
            print(f"  Stage 1b: {time.time()-_t:.1f}s")
        else:
            print(f"  WARNING: tokenizer unavailable, falling back to char_count + uniform")

    # ── Stage 2: Load optimal parameters ────────────────────
    _t = time.time()
    print(f"\n[Stage 2] Loading optimal parameters...")
    optimal_params = reconstruct_params_from_json(args.params_file)
    for m, sc in enumerate(optimal_params.sampling_configs):
        name = domain_names[m] if m < len(domain_names) else f"D{m}"
        print(f"  [{m}] {name}: lambda={sc.lambda_:.2f}, omega={sc.omega:.6f}, "
              f"eta={sc.eta:.4f}, epsilon={sc.epsilon:.6f}")
    print(f"  Stage 2: {time.time()-_t:.1f}s")

    # ── Stage 3: Eq.1 — Merge quality scores ────────────────
    _t = time.time()
    print(f"\n[Stage 3] Merging quality scores (Eq.1)...")
    merged = compute_merged_quality_scores(
        quality_scores, domain_labels, optimal_params.merge_config, n_jobs=-1,
    )
    print(f"  Merged: [{merged.min():.4f}, {merged.max():.4f}]")
    print(f"  Stage 3: {time.time()-_t:.1f}s")

    # ── Stage 4: Eq.2 (non-stratified) — Quality ranks ─────
    _t = time.time()
    print(f"\n[Stage 4] Computing quality ranks (Eq.2, non-stratified)...")
    final_ranks = compute_quality_ranks(
        merged, domain_labels, token_counts, seed=args.seed, n_jobs=-1,
    )
    print(f"  Ranks: [{final_ranks.min():.4f}, {final_ranks.max():.4f}]")
    print(f"  Stage 4: {time.time()-_t:.1f}s")

    # ── Stage 5: Eq.3 — S(r) + baseline selection ─────────
    _t = time.time()
    print(f"\n[Stage 5] Computing S(r) and baseline selection (Eq.3)...")
    sampling_values = compute_sampling_values(
        final_ranks, domain_labels, optimal_params,
    )
    rng = np.random.default_rng(args.seed)
    baseline_indices, _ = _select_documents_vectorized(sampling_values, rng)

    num_domains = optimal_params.num_domains
    baseline_domain_counts = np.bincount(
        domain_labels[baseline_indices][domain_labels[baseline_indices] >= 0],
        minlength=num_domains,
    )

    print(f"  Baseline selected: {len(baseline_indices):,} docs")
    print(f"  Domain counts (baseline):")
    for m in range(num_domains):
        if baseline_domain_counts[m] > 0:
            name = domain_names[m] if m < len(domain_names) else f"D{m}"
            print(f"    [{m}] {name:>10s}: {baseline_domain_counts[m]:>7,}")
    print(f"  Stage 5: {time.time()-_t:.1f}s")

    # ── Stage 6: Post-hoc quartile budget allocation ────────
    _t = time.time()
    print(f"\n[Stage 6] Post-hoc quartile budget allocation ({K} quartiles/domain)...")
    if target_dist is not None:
        print(f"  Scaling S(r) within each (domain x quartile) to match")
        print(f"  benchmark-derived target distribution: {[round(x, 4) for x in target_dist]}")
    else:
        print(f"  Scaling S(r) within each (domain x quartile) so that")
        print(f"  each quartile gets N_domain/{K} budget (equal allocation).")

    # Collect per-quartile stats for reporting
    per_quartile_stats = {}

    for m in range(num_domains):
        if baseline_domain_counts[m] == 0:
            continue
        name = domain_names[m] if m < len(domain_names) else f"D{m}"
        N_domain = int(baseline_domain_counts[m])

        domain_mask = domain_labels == m
        domain_indices = np.where(domain_mask)[0]
        domain_lengths = length_values[domain_mask].astype(np.float64)
        domain_sv = sampling_values[domain_mask]

        stratum_ids, boundaries = _compute_quartile_ids(domain_lengths, K)

        q_stats = []
        for k in range(K):
            s_mask = stratum_ids == k
            if not s_mask.any():
                continue
            if target_dist is not None:
                budget_per_q = N_domain * target_dist[k]
            else:
                budget_per_q = N_domain / K
            original_sum = float(domain_sv[s_mask].sum())
            if original_sum < 1e-10:
                continue
            scale = budget_per_q / original_sum
            scaled = domain_sv[s_mask] * scale
            sc = optimal_params.sampling_configs[m]
            s_cap = 2.0 ** sc.eta + sc.epsilon
            scaled = np.clip(scaled, 0, s_cap)
            sampling_values[domain_indices[s_mask]] = scaled
            clipped_sum = float(scaled.sum())
            q_stats.append({
                "length_range": [int(boundaries[k]), int(boundaries[k + 1])],
                "pool_docs": int(s_mask.sum()),
                "original_S_sum": round(original_sum, 1),
                "budget": round(budget_per_q, 1),
                "scale": round(scale, 6),
                "scaled_S_sum": round(clipped_sum, 1),
                "clipped": clipped_sum < budget_per_q * 0.99,
            })

        per_quartile_stats[name] = q_stats
        n_clipped = sum(1 for q in q_stats if q["clipped"])
        clip_msg = f"  ({n_clipped}/{len(q_stats)} quartiles clipped to 2x cap)" if n_clipped else ""
        if target_dist is not None:
            print(f"  [{m}] {name:>10s}: N_domain={N_domain:,}, "
                  f"target_dist={[round(x,3) for x in target_dist]}{clip_msg}")
        else:
            print(f"  [{m}] {name:>10s}: N_domain={N_domain:,}, "
                  f"budget/quartile={N_domain/K:.0f}{clip_msg}")

    # Re-select with scaled S(r)
    rng_alloc = np.random.default_rng(args.seed + 1000)
    selected_indices, _ = _select_documents_vectorized(
        sampling_values, rng_alloc,
    )

    print(f"\n  Stratified selected: {len(selected_indices):,} docs")
    print(f"  Stage 6: {time.time()-_t:.1f}s")

    # ── Stage 7: Target token adjustment ────────────────────
    target_tokens = int(args.target_tokens * 1e9) if args.target_tokens > 0 else 0
    if target_tokens > 0:
        actual_tokens = float(np.sum(token_counts[selected_indices]))
        print(f"\n[Stage 7] Target token adjustment:")
        print(f"  Selection produces: {actual_tokens/1e9:.2f}B tokens")
        print(f"  Target:             {target_tokens/1e9:.1f}B tokens")
        if actual_tokens > target_tokens:
            keep_prob = target_tokens / actual_tokens
            rng_discard = np.random.default_rng(args.seed + 1)
            keep_mask = rng_discard.random(len(selected_indices)) < keep_prob
            selected_indices = selected_indices[keep_mask]
            final_tokens = float(np.sum(token_counts[selected_indices]))
            print(f"  Uniform discard (keep_prob={keep_prob:.4f}) -> "
                  f"{final_tokens/1e9:.2f}B tokens")
        elif actual_tokens < target_tokens * 0.95:
            print(f"  [WARN] selection produces less than target")
        else:
            print(f"  Accept result (within tolerance)")

    # ── Distribution analysis ───────────────────────────────
    print(f"\n{'─' * 70}")
    print("Distribution Analysis")
    print(f"{'─' * 70}")

    orig_dist = np.bincount(domain_labels[domain_labels >= 0],
                            minlength=num_domains)
    sel_dist = np.bincount(
        domain_labels[selected_indices][domain_labels[selected_indices] >= 0],
        minlength=num_domains,
    )

    print(f"\n  Domain distribution (baseline vs stratified):")
    print(f"    {'domain':>10s}  {'baseline':>10s}  {'stratified':>10s}  {'delta':>8s}")
    for m in range(num_domains):
        if orig_dist[m] > 0:
            name = domain_names[m] if m < len(domain_names) else f"D{m}"
            b = int(baseline_domain_counts[m])
            s = int(sel_dist[m])
            delta_pct = (s - b) / b * 100 if b > 0 else 0.0
            print(f"    {name:>10s}  {b:>10,}  {s:>10,}  {delta_pct:>+7.1f}%")

    # ── Per-quartile selection breakdown ───────────────────
    print(f"\n  Per-quartile selection breakdown ({K} quartiles per domain):")
    per_stratum = {}
    for m in range(num_domains):
        if orig_dist[m] == 0:
            continue
        name = domain_names[m] if m < len(domain_names) else f"D{m}"
        domain_indices = np.where(domain_labels == m)[0]
        domain_lengths = length_values[domain_indices].astype(np.float64)
        stratum_ids, boundaries = _compute_quartile_ids(domain_lengths, K)
        domain_selected = np.isin(domain_indices, selected_indices)

        print(f"\n    domain={name}:")
        print(f"      {'quartile':>8s}  {'length_range':>22s}  "
              f"{'pool':>10s}  {'selected':>10s}  {'pct':>6s}")
        strata = []
        for k in range(K):
            s_mask = stratum_ids == k
            pool_n = int(s_mask.sum())
            sel_n = int((s_mask & domain_selected).sum())
            pct = sel_n / pool_n * 100 if pool_n > 0 else 0.0
            lo = int(boundaries[k])
            hi = int(boundaries[k + 1])
            rng_str = f"[{lo}, {hi})"
            print(f"      {'Q' + str(k):>8s}  {rng_str:>22s}  "
                  f"{pool_n:>10,}  {sel_n:>10,}  {pct:>5.1f}%")
            strata.append({
                "length_range": [lo, hi],
                "pool": pool_n,
                "selected": sel_n,
                "pct": round(pct, 2),
            })
        per_stratum[name] = strata

    sel_lengths = length_values[selected_indices]
    total_tokens_est = float(np.sum(token_counts[selected_indices]))
    unique_indices = np.unique(selected_indices)

    length_metric = "tokens" if target_dist is not None else "chars"
    print(f"\n  Length distribution (selected docs, unit={length_metric}):")
    print(f"    {length_metric:>10s}  mean={sel_lengths.mean():.0f}  "
          f"median={np.median(sel_lengths):.0f}")
    print(f"                p25={np.percentile(sel_lengths, 25):.0f}  "
          f"p75={np.percentile(sel_lengths, 75):.0f}  "
          f"p90={np.percentile(sel_lengths, 90):.0f}")
    print(f"                min={sel_lengths.min()}  max={sel_lengths.max()}")
    print(f"    Total tokens:  {total_tokens_est/1e9:.2f}B")
    print(f"    Total docs:    {len(selected_indices):,}")
    print(f"    Unique docs:   {len(unique_indices):,}")
    print(f"    Repeat rate:   "
          f"{(1 - len(unique_indices)/len(selected_indices))*100:.1f}%")

    # ── Stage 8: Save outputs ────────────────────────────────
    _t = time.time()
    print(f"\n[Stage 8] Saving outputs...")

    sampled_path = os.path.join(output_dir, "sampled_dataset")

    quality_dict = None
    if quality_scores is not None and quality_names:
        quality_dict = {
            name: quality_scores[:, i]
            for i, name in enumerate(quality_names)
        }

    save_sampled_dataset(
        get_text_fn=mm.read_texts,
        num_total_docs=n_docs,
        selected_indices=selected_indices,
        output_path=sampled_path,
        domain_labels=domain_labels,
        quality_ranks=final_ranks,
        sampling_values=sampling_values,
        format="parquet",
        text_col=schema.text_col,
        domain_col=schema.domain_col,
        quality_scores=quality_dict,
    )

    shutil.copy2(args.params_file,
                 os.path.join(output_dir, "optimal_parameters.json"))

    summary = {
        "method": "post_hoc_quartile_allocation",
        "params_file": args.params_file,
        "preprocessed_dir": args.preprocessed_dir,
        "n_strata": K,
        "seed": args.seed,
        "match_benchmark_lengths": args.match_benchmark_lengths,
        "length_metric": "tokens" if target_dist is not None else "chars",
        "target_distribution": [round(float(x), 6) for x in target_dist] if target_dist is not None else None,
        "benchmark_length_stats": benchmark_stats,
        "num_original_docs": n_docs,
        "num_baseline_selected": len(baseline_indices),
        "num_selected_docs": len(selected_indices),
        "num_unique_docs": len(unique_indices),
        "sampling_ratio": len(selected_indices) / n_docs,
        "estimated_tokens": total_tokens_est,
        "estimated_tokens_billions": round(total_tokens_est / 1e9, 3),
        "length_stats": {
            "metric": "tokens" if target_dist is not None else "chars",
            "mean": float(sel_lengths.mean()),
            "median": float(np.median(sel_lengths)),
            "p25": float(np.percentile(sel_lengths, 25)),
            "p75": float(np.percentile(sel_lengths, 75)),
            "p90": float(np.percentile(sel_lengths, 90)),
        },
        "baseline_domain_counts": {
            domain_names[m] if m < len(domain_names) else f"D{m}": int(baseline_domain_counts[m])
            for m in range(num_domains) if baseline_domain_counts[m] > 0
        },
        "domain_distribution": {
            domain_names[m] if m < len(domain_names) else f"D{m}": {
                "original": int(orig_dist[m]),
                "selected": int(sel_dist[m]),
                "ratio": round(sel_dist[m] / orig_dist[m], 4)
                         if orig_dist[m] > 0 else 0,
            }
            for m in range(num_domains) if orig_dist[m] > 0
        },
        "per_quartile_scaling": per_quartile_stats,
        "per_quartile_distribution": per_stratum,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    summary_path = os.path.join(output_dir, "stratified_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating,))
                  else int(x) if isinstance(x, (np.integer,)) else x)

    print(f"\n{'=' * 70}")
    print(f"  Quartile Allocation Complete! ({time.time()-t_start:.1f}s)")
    print(f"  Baseline:  {len(baseline_indices):,} docs")
    print(f"  Selected:  {len(selected_indices):,} docs "
          f"({total_tokens_est/1e9:.2f}B tokens)")
    print(f"  Output: {output_dir}/")
    print(f"    ├── sampled_dataset/")
    print(f"    ├── optimal_parameters.json")
    print(f"    └── stratified_summary.json")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
