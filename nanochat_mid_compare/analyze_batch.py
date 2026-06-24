"""
Deep analysis of captured batch data to identify crash-inducing patterns.

Looks at per-row, per-document, and token-level patterns beyond basic stats.

Usage:
    python analyze_batch.py \
        --crash-batch=/path/to/step329_capture/batches_rank0.pt \
        --normal-batch=/path/to/step328_capture/batches_rank0.pt \
        [--all-ranks-dir=/path/to/step329_capture/] \
        [--output-dir=/path/to/analysis_output]
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
import json
from collections import Counter


def load_batch(batch_path, device="cpu"):
    return torch.load(batch_path, map_location=device)


def analyze_per_row(batch_data, name="batch"):
    results = []
    for bi, batch in enumerate(batch_data):
        x = batch["x"].numpy()  # (B, T)
        for row_idx in range(x.shape[0]):
            row = x[row_idx]
            unique = np.unique(row)
            counts = Counter(row.tolist())
            top5 = counts.most_common(5)
            max_run = 1
            cur_run = 1
            for i in range(1, len(row)):
                if row[i] == row[i-1]:
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 1
            results.append({
                "batch_idx": bi,
                "row_idx": row_idx,
                "unique_tokens": len(unique),
                "top5_tokens": top5,
                "max_repeat_run": max_run,
                "token_mean": float(row.mean()),
                "token_std": float(row.std()),
                "token_min": int(row.min()),
                "token_max": int(row.max()),
            })
    return results


def find_bos_positions(batch_data, bos_token_id=1):
    results = []
    for bi, batch in enumerate(batch_data):
        x = batch["x"].numpy()
        for row_idx in range(x.shape[0]):
            row = x[row_idx]
            bos_positions = np.where(row == bos_token_id)[0].tolist()
            doc_lengths = []
            for i in range(len(bos_positions)):
                start = bos_positions[i]
                end = bos_positions[i+1] if i+1 < len(bos_positions) else len(row)
                doc_lengths.append(end - start)
            results.append({
                "batch_idx": bi,
                "row_idx": row_idx,
                "num_docs": len(bos_positions),
                "doc_lengths": doc_lengths,
                "doc_lengths_min": min(doc_lengths) if doc_lengths else 0,
                "doc_lengths_max": max(doc_lengths) if doc_lengths else 0,
                "doc_lengths_mean": float(np.mean(doc_lengths)) if doc_lengths else 0,
            })
    return results


def find_unique_tokens(crash_data, normal_data):
    crash_tokens = set()
    normal_tokens = set()
    for batch in crash_data:
        crash_tokens.update(batch["x"].flatten().tolist())
    for batch in normal_data:
        normal_tokens.update(batch["x"].flatten().tolist())
    only_crash = crash_tokens - normal_tokens
    only_normal = normal_tokens - crash_tokens
    return only_crash, only_normal


def compare_all_ranks(crash_dir):
    rank_data = {}
    for rank in range(8):
        path = os.path.join(crash_dir, f"batches_rank{rank}.pt")
        if os.path.exists(path):
            data = load_batch(path)
            x_all = np.concatenate([b["x"].flatten().numpy() for b in data])
            rank_data[rank] = {
                "unique_tokens": len(np.unique(x_all)),
                "token_mean": float(x_all.mean()),
                "token_std": float(x_all.std()),
                "token_min": int(x_all.min()),
                "token_max": int(x_all.max()),
            }
    return rank_data


def main():
    parser = argparse.ArgumentParser(description="Deep batch analysis for crash investigation")
    parser.add_argument("--crash-batch", type=str, required=True)
    parser.add_argument("--normal-batch", type=str, default=None)
    parser.add_argument("--all-ranks-dir", type=str, default=None,
                        help="Directory with batches_rank{0-7}.pt for cross-rank analysis")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  Deep Batch Analysis")
    print("=" * 60)

    crash_data = load_batch(args.crash_batch)
    print(f"\n[1] Per-row analysis (crash batch, {len(crash_data)} micro_steps x 8 rows):")

    crash_rows = analyze_per_row(crash_data, "crash")
    max_runs = [r["max_repeat_run"] for r in crash_rows]
    unique_counts = [r["unique_tokens"] for r in crash_rows]
    print(f"  Max repeat run: min={min(max_runs)}, max={max(max_runs)}, mean={np.mean(max_runs):.1f}")
    print(f"  Unique tokens/row: min={min(unique_counts)}, max={max(unique_counts)}, mean={np.mean(unique_counts):.1f}")

    outlier_rows = [r for r in crash_rows if r["max_repeat_run"] > 50 or r["unique_tokens"] < 100]
    if outlier_rows:
        print(f"  OUTLIER ROWS (repeat>50 or unique<100): {len(outlier_rows)}")
        for r in outlier_rows[:5]:
            print(f"    batch={r['batch_idx']} row={r['row_idx']} unique={r['unique_tokens']} max_run={r['max_repeat_run']} top5={r['top5_tokens'][:3]}")

    print(f"\n[2] BOS/document structure (crash batch):")
    crash_bos = find_bos_positions(crash_data)
    num_docs = [b["num_docs"] for b in crash_bos]
    doc_lens = [l for b in crash_bos for l in b["doc_lengths"]]
    print(f"  Docs per row: min={min(num_docs)}, max={max(num_docs)}, mean={np.mean(num_docs):.1f}")
    if doc_lens:
        print(f"  Doc lengths: min={min(doc_lens)}, max={max(doc_lens)}, mean={np.mean(doc_lens):.1f}, median={np.median(doc_lens):.1f}")
        short_docs = sum(1 for l in doc_lens if l < 10)
        long_docs = sum(1 for l in doc_lens if l > 1000)
        print(f"  Short docs (<10 tokens): {short_docs}")
        print(f"  Long docs (>1000 tokens): {long_docs}")

    if args.normal_batch:
        normal_data = load_batch(args.normal_batch)
        print(f"\n[3] Per-row analysis (normal batch):")
        normal_rows = analyze_per_row(normal_data, "normal")
        n_max_runs = [r["max_repeat_run"] for r in normal_rows]
        n_unique_counts = [r["unique_tokens"] for r in normal_rows]
        print(f"  Max repeat run: min={min(n_max_runs)}, max={max(n_max_runs)}, mean={np.mean(n_max_runs):.1f}")
        print(f"  Unique tokens/row: min={min(n_unique_counts)}, max={max(n_unique_counts)}, mean={np.mean(n_unique_counts):.1f}")

        print(f"\n[4] BOS/document structure (normal batch):")
        normal_bos = find_bos_positions(normal_data)
        n_num_docs = [b["num_docs"] for b in normal_bos]
        n_doc_lens = [l for b in normal_bos for l in b["doc_lengths"]]
        print(f"  Docs per row: min={min(n_num_docs)}, max={max(n_num_docs)}, mean={np.mean(n_num_docs):.1f}")
        if n_doc_lens:
            print(f"  Doc lengths: min={min(n_doc_lens)}, max={max(n_doc_lens)}, mean={np.mean(n_doc_lens):.1f}, median={np.median(n_doc_lens):.1f}")

        print(f"\n[5] Token overlap analysis:")
        only_crash, only_normal = find_unique_tokens(crash_data, normal_data)
        print(f"  Tokens only in crash: {len(only_crash)}")
        print(f"  Tokens only in normal: {len(only_normal)}")
        if only_crash:
            print(f"  Crash-only tokens (first 20): {sorted(only_crash)[:20]}")

    if args.all_ranks_dir:
        print(f"\n[6] Cross-rank analysis (crash step):")
        rank_stats = compare_all_ranks(args.all_ranks_dir)
        for rank, stats in sorted(rank_stats.items()):
            print(f"  Rank {rank}: unique={stats['unique_tokens']}, mean={stats['token_mean']:.1f}, "
                  f"std={stats['token_std']:.1f}, range=[{stats['token_min']}, {stats['token_max']}]")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "crash_per_row": crash_rows,
            "crash_bos": crash_bos,
        }
        if args.normal_batch:
            results["normal_per_row"] = normal_rows
            results["normal_bos"] = normal_bos
            only_crash_list = sorted(only_crash) if only_crash else []
            only_normal_list = sorted(only_normal) if only_normal else []
            results["unique_tokens"] = {
                "only_crash": only_crash_list,
                "only_normal": only_normal_list,
            }
        if args.all_ranks_dir:
            results["cross_rank"] = {str(k): v for k, v in rank_stats.items()}
        with open(output_dir / "deep_analysis.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_dir / 'deep_analysis.json'}")

    print("=" * 60)


if __name__ == "__main__":
    main()
