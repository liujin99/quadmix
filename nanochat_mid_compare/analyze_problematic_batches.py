"""
Analyze training batches around a specific step to find problematic data.

Checks for:
- Unusual token distributions
- Very long/short sequences
- High variance between ranks
- Special token patterns

Usage:
    python analyze_problematic_batches.py \
        --data-dir /path/to/quality_data_fineweb_edu \
        --tokenizer-dir /path/to/tokenizer \
        --target-step 350
"""

import os
import pickle
import argparse
from collections import Counter

import numpy as np
import pyarrow.parquet as pq


def load_tokenizer(tokenizer_dir):
    pkl_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(pkl_path, "rb") as f:
        enc = pickle.load(f)
    bos_token_id = enc.encode_single_token("<|bos|>")
    return enc, bos_token_id


def list_train_parquets(data_dir):
    files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    ])
    if len(files) < 2:
        return [os.path.join(data_dir, f) for f in files]
    return [os.path.join(data_dir, f) for f in files[:-1]]


def document_batches_for_rank(parquet_paths, rank, world_size, tokenizer_batch_size):
    for pq_idx, filepath in enumerate(parquet_paths):
        pf = pq.ParquetFile(filepath)
        rg_idx = rank
        while rg_idx < pf.num_row_groups:
            rg = pf.read_row_group(rg_idx)
            batch = rg.column('text').to_pylist()
            for i in range(0, len(batch), tokenizer_batch_size):
                yield batch[i:i + tokenizer_batch_size], pq_idx, rg_idx
            rg_idx += world_size


def analyze_batches(args):
    enc, bos_token_id = load_tokenizer(args.tokenizer_dir)
    parquet_paths = list_train_parquets(args.data_dir)

    print(f"Analyzing batches for all {args.num_npu} ranks up to step {args.target_step}")
    print(f"Each step = {args.device_batch_size} rows x {args.grad_accum} micro-batches")
    print()

    rank_stats = {}

    for rank in range(args.num_npu):
        print(f"Rank {rank}...", end=" ", flush=True)

        batches = document_batches_for_rank(
            parquet_paths, rank, args.num_npu, args.tokenizer_batch_size
        )

        step = 0
        micro_step = 0
        total_docs = 0
        total_tokens = 0
        token_counter = Counter()
        doc_lengths = []
        problem_batches = []

        try:
            for batch_idx, (doc_batch, pq_idx, rg_idx) in enumerate(batches):
                token_lists = enc.encode_ordinary_batch(doc_batch, num_threads=args.tokenizer_threads)

                for tokens, text in zip(token_lists, doc_batch):
                    tokens.insert(0, bos_token_id)
                    total_docs += 1
                    total_tokens += len(tokens)
                    doc_lengths.append(len(tokens))
                    token_counter.update(tokens)

                    if len(tokens) > args.seq_len * 2:
                        problem_batches.append({
                            "step": step,
                            "micro_step": micro_step,
                            "issue": "very_long_doc",
                            "length": len(tokens),
                            "pq_idx": pq_idx,
                            "rg_idx": rg_idx,
                            "text_preview": text[:200]
                        })

                micro_step += 1
                if micro_step >= args.grad_accum:
                    micro_step = 0
                    step += 1

                if step > args.target_step + 5:
                    break

        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if doc_lengths:
            rank_stats[rank] = {
                "total_docs": total_docs,
                "total_tokens": total_tokens,
                "avg_doc_len": np.mean(doc_lengths),
                "max_doc_len": max(doc_lengths),
                "min_doc_len": min(doc_lengths),
                "std_doc_len": np.std(doc_lengths),
                "unique_tokens": len(token_counter),
                "problem_batches": problem_batches,
            }
            print(f"OK | docs={total_docs} tokens={total_tokens} "
                  f"avg_len={np.mean(doc_lengths):.0f} max={max(doc_lengths)} "
                  f"problems={len(problem_batches)}")

    print()
    print("=" * 70)
    print("Summary:")
    print("=" * 70)

    for rank in sorted(rank_stats.keys()):
        stats = rank_stats[rank]
        print(f"\nRank {rank}:")
        print(f"  Total docs: {stats['total_docs']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Doc length: avg={stats['avg_doc_len']:.0f}, "
              f"min={stats['min_doc_len']}, max={stats['max_doc_len']}, "
              f"std={stats['std_doc_len']:.0f}")
        print(f"  Unique tokens: {stats['unique_tokens']}")

        if stats['problem_batches']:
            print(f"  Problem batches ({len(stats['problem_batches'])}):")
            for pb in stats['problem_batches'][:5]:
                print(f"    Step {pb['step']} micro={pb['micro_step']}: "
                      f"{pb['issue']} len={pb['length']} "
                      f"pq={pb['pq_idx']} rg={pb['rg_idx']}")
                print(f"      Preview: {pb['text_preview'][:100]}...")

    print()
    print("=" * 70)

    if args.target_step > 0:
        print(f"\nChecking step {args.target_step} specifically:")
        for rank in sorted(rank_stats.keys()):
            stats = rank_stats[rank]
            problems_at_target = [
                pb for pb in stats['problem_batches']
                if abs(pb['step'] - args.target_step) <= 2
            ]
            if problems_at_target:
                print(f"  Rank {rank}: {len(problems_at_target)} problems near step {args.target_step}")
                for pb in problems_at_target:
                    print(f"    Step {pb['step']}: {pb['issue']} len={pb['length']}")
            else:
                print(f"  Rank {rank}: no problems near step {args.target_step}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True,
                        default="/home/ma-user/work/nanochat_model_dir/tokenizer")
    parser.add_argument("--target-step", type=int, default=350)
    parser.add_argument("--num-npu", type=int, default=8)
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    parser.add_argument("--tokenizer-threads", type=int, default=16)
    args = parser.parse_args()

    analyze_batches(args)


if __name__ == "__main__":
    main()
