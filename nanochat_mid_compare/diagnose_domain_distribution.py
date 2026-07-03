"""
Diagnose domain distribution across QuadMix, Random, and Quality-top-k baselines.

Reads:
  - sampled_dataset.parquet (QuadMix, has domain column)
  - preprocessed_*.parquet (upstream shards with domain + quality scores)

Replicates the selection logic from prepare_data.py to determine which
documents each baseline would select, then compares domain distributions.

Usage:
    python diagnose_domain_distribution.py \
        --quadmix-sampled-data /path/to/sampled_dataset.parquet \
        --preprocessed-data-dir /path/to/preprocessed \
        [--quality-methods dclm,fineweb_edu] \
        [--max-shards 500] \
        [--seed 42]
"""

import os
import sys
import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quadmix.constants import DOMAIN_NAMES, DOMAIN_SHORT_NAMES, NUM_DOMAINS

QUALITY_SCORE_MAP = {
    "dclm": "qs_dclm",
    "fineweb_edu": "qs_fineweb_edu_approx",
    "english": "qs_english",
    "math_general": "qs_eai_general_math",
    "math_openweb": "qs_eai_open_web_math",
}


def scan_shards_metadata(preprocessed_data_dir, max_shards=500,
                         max_chars=1000000, max_char_repeat_ratio=0.3):
    shard_files = sorted(Path(preprocessed_data_dir).glob("preprocessed_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No preprocessed_*.parquet in {preprocessed_data_dir}")
    if max_shards is not None:
        shard_files = shard_files[:max_shards]

    all_docs = []
    for shard_id, sf in enumerate(tqdm(shard_files, desc="Scanning shards")):
        cols = ["doc_char_count", "domain"]
        table = pq.read_table(str(sf), columns=cols)
        char_counts = table["doc_char_count"].to_pylist()
        domains = table["domain"].to_pylist()
        for doc_id in range(len(char_counts)):
            cc = char_counts[doc_id]
            if not cc or cc > max_chars:
                continue
            all_docs.append((shard_id, doc_id, cc, domains[doc_id]))
    return shard_files, all_docs


def select_random(all_docs, total_tokens, seed=42):
    rng = random.Random(seed)
    candidates = [(sid, did, cc // 4, dom) for sid, did, cc, dom in all_docs]
    rng.shuffle(candidates)
    selected = []
    accumulated = 0
    target = int(total_tokens * 1.1)
    for sid, did, est_tok, dom in candidates:
        if accumulated >= target:
            break
        selected.append((sid, did, dom))
        accumulated += est_tok
    return selected


def select_quality_topk(all_docs, shard_files, quality_col, total_tokens):
    shard_id_set = set(sid for sid, _, _, _ in all_docs)
    score_cache = {}
    for shard_id in tqdm(sorted(shard_id_set), desc=f"Reading {quality_col}"):
        table = pq.read_table(str(shard_files[shard_id]), columns=[quality_col])
        score_cache[shard_id] = table[quality_col].to_pylist()

    scored = []
    for shard_id, doc_id, cc, domain in all_docs:
        score = score_cache[shard_id][doc_id]
        scored.append((shard_id, doc_id, cc, domain, float(score)))

    scored.sort(key=lambda x: x[4], reverse=True)

    selected = []
    accumulated = 0
    target = int(total_tokens * 1.1)
    for sid, did, cc, dom, score in scored:
        if accumulated >= target:
            break
        selected.append((sid, did, dom))
        accumulated += cc // 4
    return selected


def compute_distribution(selected, label):
    counter = Counter(dom for _, _, dom in selected)
    total = sum(counter.values())
    dist = {}
    for i in range(NUM_DOMAINS):
        count = counter.get(i, 0)
        dist[DOMAIN_SHORT_NAMES[i]] = {
            "count": count,
            "pct": count / total * 100 if total > 0 else 0,
        }
    return dist, total


def compute_stats(dist):
    pcts = sorted([dist[d]["pct"] for d in DOMAIN_SHORT_NAMES], reverse=True)
    top3 = sum(pcts[:3])
    top5 = sum(pcts[:5])
    nonzero = sum(1 for p in pcts if p > 0.1)
    max_min = pcts[0] / pcts[-1] if pcts[-1] > 0 else float('inf')
    hhi = sum(p * p for p in pcts) / 100
    return {"top3": top3, "top5": top5, "nonzero": nonzero,
            "max_min": max_min, "hhi": hhi}


def print_comparison(methods_data):
    labels = list(methods_data.keys())
    col_w = 16

    header = f"{'Domain':<14}"
    for label in labels:
        header += f" | {label:>{col_w}}"
    print(header)
    print("-" * len(header))

    for d in DOMAIN_SHORT_NAMES:
        row = f"{d:<14}"
        for label in labels:
            dist, total = methods_data[label]
            info = dist.get(d, {"count": 0, "pct": 0})
            row += f" | {info['count']:>8,} {info['pct']:>5.1f}%"
        print(row)

    print("-" * len(header))
    row = f"{'TOTAL':<14}"
    for label in labels:
        dist, total = methods_data[label]
        row += f" | {total:>8,} {'100.0':>5}%"
    print(row)

    print()
    print("=== Concentration Metrics ===")
    col_w2 = 16
    hdr = f"{'Metric':<24}"
    for label in labels:
        hdr += f" | {label:>{col_w2}}"
    print(hdr)
    print("-" * len(hdr))

    stats_all = {label: compute_stats(methods_data[label][0]) for label in labels}
    metrics = [
        ("Top-3 concentration %", "top3", ".1f"),
        ("Top-5 concentration %", "top5", ".1f"),
        ("Domains with >0.1%", "nonzero", "d"),
        ("Max/Min ratio", "max_min", ".1f"),
        ("HHI (Herfindahl)", "hhi", ".0f"),
    ]
    for name, key, fmt in metrics:
        row = f"{name:<24}"
        for label in labels:
            val = stats_all[label][key]
            row += f" | {val:>{col_w2}{fmt}}"
        print(row)


def plot_comparison(methods_data, output_path):
    labels = list(methods_data.keys())
    n_methods = len(labels)
    domains = DOMAIN_SHORT_NAMES
    n_domains = len(domains)

    x = np.arange(n_domains)
    width = 0.8 / n_methods
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, label in enumerate(labels):
        dist, total = methods_data[label]
        pcts = [dist[d]["pct"] for d in domains]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, pcts, width, label=label, color=colors[i % len(colors)])
        ax.bar_label(bars, fmt="%.1f", padding=1, fontsize=6, rotation=90)

    ax.set_xlabel("Domain")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Domain Distribution Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose domain distribution")
    parser.add_argument("--quadmix-sampled-data", type=str, required=True)
    parser.add_argument("--preprocessed-data-dir", type=str, required=True)
    parser.add_argument("--quality-methods", type=str, default="dclm,fineweb_edu")
    parser.add_argument("--max-shards", type=int, default=500,
                        help="Max preprocessed shards to scan for all baselines (default: 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save the plot (default: current directory)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Domain Distribution Diagnostic")
    print("=" * 60)

    print(f"\n[1/4] Reading QuadMix domain distribution...")
    qm_table = pq.read_table(args.quadmix_sampled_data, columns=["domain"])
    qm_domains = qm_table["domain"].to_pylist()
    qm_selected = [(0, i, d) for i, d in enumerate(qm_domains)]
    print(f"  QuadMix docs: {len(qm_domains):,}")

    qm_total_tokens_est = len(qm_domains) * 2300

    print(f"\n[2/4] Scanning preprocessed shards...")
    shard_files, all_docs = scan_shards_metadata(
        args.preprocessed_data_dir, max_shards=args.max_shards)
    print(f"  Scanned {len(shard_files)} shards, {len(all_docs):,} candidate docs")

    methods_data = {}
    qm_dist, qm_total = compute_distribution(qm_selected, "QuadMix")
    methods_data["QuadMix"] = (qm_dist, qm_total)

    print(f"\n[3/4] Simulating Random selection...")
    random_selected = select_random(all_docs, qm_total_tokens_est, seed=args.seed)
    random_dist, random_total = compute_distribution(random_selected, "Random")
    methods_data["Random"] = (random_dist, random_total)
    print(f"  Random selected: {random_total:,} docs")

    quality_methods = [m.strip() for m in args.quality_methods.split(",") if m.strip()]
    print(f"\n[4/4] Simulating Quality-top-k selection...")
    for qm in quality_methods:
        if qm not in QUALITY_SCORE_MAP:
            print(f"  WARNING: Unknown quality method '{qm}', skipping")
            continue
        quality_col = QUALITY_SCORE_MAP[qm]
        print(f"  Simulating quality-topk ({quality_col})...")
        q_selected = select_quality_topk(
            all_docs, shard_files, quality_col, qm_total_tokens_est)
        q_dist, q_total = compute_distribution(q_selected, f"Quality({qm})")
        methods_data[f"Quality({qm})"] = (q_dist, q_total)
        print(f"  Quality({qm}) selected: {q_total:,} docs")

    print()
    print("=" * 60)
    print("  Domain Distribution Comparison")
    print("=" * 60)
    print()
    print_comparison(methods_data)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "domain_distribution_comparison.png"
    plot_comparison(methods_data, plot_path)


if __name__ == "__main__":
    main()
