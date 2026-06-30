"""
Analyze L2 domain distribution from preprocessed parquet shards.

Reads the 'domain' (int 0-22) and 'doc_char_count' columns from
preprocessed output, computes per-domain statistics, and produces
a bar chart + JSON report.

Usage:
    python scripts/analysis/analyze_l2_domain_distribution.py \
        --preprocessed-dir ~/.cache/QuaDMix/temp/preprocessed \
        --output-dir result/l2_analysis
"""

import argparse
import json
import os
import sys
import glob
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from quadmix.constants import DOMAIN_NAMES, NUM_DOMAINS


def _setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
    })


def _scan_shard(path: str) -> dict:
    try:
        df = pd.read_parquet(path, columns=["domain", "doc_char_count"])
    except Exception as e:
        return {"error": str(e), "path": path}
    domains = df["domain"].to_numpy(dtype=np.int32)
    chars = df["doc_char_count"].to_numpy(dtype=np.int64)
    counts = np.bincount(domains, minlength=NUM_DOMAINS)
    char_sums = np.zeros(NUM_DOMAINS, dtype=np.int64)
    for d in range(NUM_DOMAINS):
        mask = domains == d
        char_sums[d] = chars[mask].sum()
    return {"counts": counts.tolist(), "char_sums": char_sums.tolist(), "total_docs": len(df)}


def main():
    p = argparse.ArgumentParser(description="Analyze L2 domain distribution from preprocessed shards")
    p.add_argument("--preprocessed-dir",
                   default=os.path.join(os.path.expanduser("~"), ".cache", "QuaDMix", "temp", "preprocessed"),
                   help="Directory containing preprocessed parquet shards")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: same as preprocessed-dir/l2_analysis)")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.preprocessed_dir, "l2_analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    _setup_style()

    paths = sorted(glob.glob(os.path.join(args.preprocessed_dir, "preprocessed_*.parquet")))
    if not paths:
        print(f"Error: no preprocessed_*.parquet found in {args.preprocessed_dir}")
        return 1

    print(f"Scanning {len(paths)} preprocessed shards from {args.preprocessed_dir}")
    t0 = time.time()

    total_counts = np.zeros(NUM_DOMAINS, dtype=np.int64)
    total_chars = np.zeros(NUM_DOMAINS, dtype=np.int64)
    total_docs = 0
    errors = 0

    workers = min(args.workers, len(paths))
    if len(paths) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_scan_shard, p): p for p in paths}
            done = 0
            for future in as_completed(futures):
                r = future.result()
                done += 1
                if "error" in r:
                    errors += 1
                    continue
                total_counts += np.array(r["counts"], dtype=np.int64)
                total_chars += np.array(r["char_sums"], dtype=np.int64)
                total_docs += r["total_docs"]
                if done % 50 == 0 or done == len(paths):
                    print(f"  [{done}/{len(paths)}] shards scanned...")
    else:
        r = _scan_shard(paths[0])
        if "error" not in r:
            total_counts += np.array(r["counts"], dtype=np.int64)
            total_chars += np.array(r["char_sums"], dtype=np.int64)
            total_docs += r["total_docs"]

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s (errors: {errors})")

    total_tokens = total_chars // 4
    grand_tokens = int(total_tokens.sum())
    grand_docs = int(total_counts.sum())

    print(f"\n{'=' * 90}")
    print(f"  L2 Domain Distribution ({NUM_DOMAINS} domains, {len(paths)} shards)")
    print(f"  Total docs: {grand_docs:,}  |  Total tokens (est): {grand_tokens:,}")
    print(f"{'=' * 90}")
    print(f"  {'ID':>3}  {'Domain':<30}  {'Docs':>12}  {'%':>7}  {'Tokens(est)':>14}  {'%':>7}  {'Avg tok/doc':>11}")
    print(f"  {'─'*3}  {'─'*30}  {'─'*12}  {'─'*7}  {'─'*14}  {'─'*7}  {'─'*11}")

    sorted_ids = np.argsort(-total_counts)
    rows = []
    for d in sorted_ids:
        d = int(d)
        n_docs = int(total_counts[d])
        n_tok = int(total_tokens[d])
        pct_docs = n_docs / max(grand_docs, 1) * 100
        pct_tok = n_tok / max(grand_tokens, 1) * 100
        avg_tok = n_tok / max(n_docs, 1)
        name = DOMAIN_NAMES[d] if d < len(DOMAIN_NAMES) else f"Unknown_{d}"
        print(f"  {d:>3}  {name:<30}  {n_docs:>12,}  {pct_docs:>6.2f}%  {n_tok:>14,}  {pct_tok:>6.2f}%  {avg_tok:>11.0f}")
        rows.append({
            "domain_id": d,
            "domain_name": name,
            "num_docs": n_docs,
            "pct_docs": round(pct_docs, 4),
            "num_tokens_est": n_tok,
            "pct_tokens": round(pct_tok, 4),
            "avg_tokens_per_doc": round(avg_tok, 1),
        })

    print(f"  {'─'*3}  {'─'*30}  {'─'*12}  {'─'*7}  {'─'*14}  {'─'*7}  {'─'*11}")
    print(f"  {'':>3}  {'TOTAL':<30}  {grand_docs:>12,}  {'100.00':>6}%  {grand_tokens:>14,}  {'100.00':>6}%")

    max_pct = rows[0]["pct_docs"]
    min_pct = rows[-1]["pct_docs"]
    print(f"\n  Max/Min ratio: {max_pct/max(min_pct, 0.001):.1f}x")
    print(f"  Top-5 domains: {sum(r['pct_docs'] for r in rows[:5]):.1f}%")
    print(f"  Top-10 domains: {sum(r['pct_docs'] for r in rows[:10]):.1f}%")
    print(f"  Bottom-5 domains: {sum(r['pct_docs'] for r in rows[-5:]):.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sorted_names = [DOMAIN_NAMES[int(d)] if int(d) < len(DOMAIN_NAMES) else f"Unknown_{d}" for d in sorted_ids]
    sorted_pcts = [total_counts[int(d)] / max(grand_docs, 1) * 100 for d in sorted_ids]
    colors = ["#4472C4" if p >= 1.0 else "#A5A5A5" for p in sorted_pcts]

    axes[0].barh(range(len(sorted_names)), sorted_pcts, color=colors)
    axes[0].set_yticks(range(len(sorted_names)))
    axes[0].set_yticklabels(sorted_names, fontsize=8)
    axes[0].set_xlabel("Percentage (%)")
    axes[0].set_title("L2 Domain Distribution (by doc count)")
    axes[0].invert_yaxis()
    for i, (pct, name) in enumerate(zip(sorted_pcts, sorted_names)):
        if pct >= 0.5:
            axes[0].text(pct + 0.2, i, f"{pct:.1f}%", va="center", fontsize=7)

    sorted_tok_pcts = [total_tokens[int(d)] / max(grand_tokens, 1) * 100 for d in sorted_ids]
    axes[1].barh(range(len(sorted_names)), sorted_tok_pcts, color=colors)
    axes[1].set_yticks(range(len(sorted_names)))
    axes[1].set_yticklabels(sorted_names, fontsize=8)
    axes[1].set_xlabel("Percentage (%)")
    axes[1].set_title("L2 Domain Distribution (by token count)")
    axes[1].invert_yaxis()
    for i, pct in enumerate(sorted_tok_pcts):
        if pct >= 0.5:
            axes[1].text(pct + 0.2, i, f"{pct:.1f}%", va="center", fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "l2_domain_distribution.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"\n  [Figure] Saved: {fig_path}")

    report = {
        "num_shards": len(paths),
        "total_docs": grand_docs,
        "total_tokens_est": grand_tokens,
        "num_domains": NUM_DOMAINS,
        "max_min_ratio": round(max_pct / max(min_pct, 0.001), 1),
        "top5_pct": round(sum(r["pct_docs"] for r in rows[:5]), 2),
        "top10_pct": round(sum(r["pct_docs"] for r in rows[:10]), 2),
        "domains": rows,
    }
    report_path = os.path.join(args.output_dir, "l2_distribution_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [Report] Saved: {report_path}")

    print(f"\n{'=' * 90}")
    print(f"  All outputs saved to: {args.output_dir}")
    print(f"{'=' * 90}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
