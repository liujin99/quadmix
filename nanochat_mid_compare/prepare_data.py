"""
Prepare two nanochat-compatible training datasets for comparison:
  1. QuadMix-selected subset from essential-web
  2. Random subset from essential-web (token-count aligned)

Both datasets are written as sharded parquet files with a single "text" column,
compatible with nanochat's dataloader (last shard = validation).

A shared validation shard is used for fair comparison.

Usage:
    python prepare_data.py \
        --quadmix-dataset /path/to/sampled_dataset.parquet \
        --essential-web-dir /path/to/essential-web-v1 \
        --output-dir /path/to/experiment/data \
        --tokenizer-pkl /path/to/nanochat/tokenizer/tokenizer.pkl \
        [--shard-size 10000] \
        [--val-ratio 0.05] \
        [--seed 42]
"""

import os
import argparse
import random
import json
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm


def load_tokenizer(tokenizer_pkl_path):
    if not tokenizer_pkl_path or not os.path.exists(tokenizer_pkl_path):
        return None
    try:
        with open(tokenizer_pkl_path, "rb") as f:
            enc = pickle.load(f)
        if hasattr(enc, "encode_ordinary"):
            return enc
        return None
    except Exception as e:
        print(f"  WARNING: Failed to load tokenizer: {e}")
        return None


def count_tokens_single(text, enc):
    return len(enc.encode_ordinary(text))


def estimate_tokens(text):
    return len(text) // 4


def read_essential_web_shard(shard_path, enc=None):
    df = pq.read_table(shard_path).to_pandas()
    texts = df["text"].tolist() if "text" in df.columns else []
    docs = []
    for text in texts:
        if not text or len(text) < 100:
            continue
        tok = count_tokens_single(text, enc) if enc else estimate_tokens(text)
        docs.append({
            "text": text,
            "char_count": len(text),
            "token_count": tok,
        })
    return docs


def write_shard(docs, output_path):
    texts = [d["text"] for d in docs]
    table = pa.table({"text": texts})
    pq.write_table(table, output_path, row_group_size=1024)


def main():
    parser = argparse.ArgumentParser(description="Prepare comparison datasets for nanochat mid-training")
    parser.add_argument("--quadmix-dataset", type=str, required=True,
                        help="Path to QuadMix sampled_dataset.parquet")
    parser.add_argument("--essential-web-dir", type=str, required=True,
                        help="Path to essential-web-v1 raw parquet shards directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for the two datasets")
    parser.add_argument("--tokenizer-pkl", type=str, default=None,
                        help="Path to nanochat tokenizer.pkl for accurate token counting. "
                             "Falls back to char_count//4 if not provided.")
    parser.add_argument("--shard-size", type=int, default=10000,
                        help="Documents per output shard (default: 10000)")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fraction of documents reserved for shared validation (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--max-random-scan", type=int, default=500,
                        help="Max number of essential-web shards to scan for random baseline (default: 500)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    enc = load_tokenizer(args.tokenizer_pkl)
    token_method = "nanochat tokenizer" if enc else "char_count // 4 (estimate)"

    output_dir = Path(args.output_dir)
    quadmix_dir = output_dir / "quadmix_data"
    random_dir = output_dir / "random_data"
    quadmix_dir.mkdir(parents=True, exist_ok=True)
    random_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Nanochat Comparison Dataset Preparation")
    print("=" * 60)
    print(f"  Token counting: {token_method}")

    print(f"\n[1/5] Reading QuadMix selected dataset...")
    quadmix_df = pd.read_parquet(args.quadmix_dataset)
    quadmix_docs = []
    total_tokens = 0
    texts = quadmix_df["text"].tolist()
    for text in tqdm(texts, desc="  Counting QuadMix tokens"):
        if not text or len(text) < 100:
            continue
        tok = count_tokens_single(text, enc) if enc else estimate_tokens(text)
        quadmix_docs.append({"text": text, "token_count": tok})
        total_tokens += tok

    print(f"  QuadMix docs: {len(quadmix_docs):,}")
    print(f"  Tokens ({token_method}): {total_tokens:,}")

    print(f"\n[2/5] Scanning essential-web shards for random sampling...")
    shard_files = sorted(Path(args.essential_web_dir).glob("shard_*.parquet"))
    shard_files = shard_files[:args.max_random_scan]
    print(f"  Scanning {len(shard_files)} shards...")

    all_candidates = []
    for sf in tqdm(shard_files, desc="  Reading shards"):
        docs = read_essential_web_shard(str(sf), enc)
        all_candidates.extend(docs)

    print(f"  Total candidate docs: {len(all_candidates):,}")

    print(f"\n[3/5] Random sampling (target: {total_tokens:,} tokens)...")
    random.shuffle(all_candidates)
    random_docs = []
    accumulated_tokens = 0
    for doc in all_candidates:
        if accumulated_tokens >= total_tokens:
            break
        random_docs.append(doc)
        accumulated_tokens += doc["token_count"]

    print(f"  Random docs: {len(random_docs):,}")
    print(f"  Tokens ({token_method}): {accumulated_tokens:,}")

    print(f"\n[4/5] Splitting train/val (val_ratio={args.val_ratio})...")
    n_val = max(1, int(len(quadmix_docs) * args.val_ratio))

    random.shuffle(quadmix_docs)
    random.shuffle(random_docs)

    val_docs = quadmix_docs[:n_val]
    quadmix_train = quadmix_docs[n_val:]
    random_train = random_docs[n_val:]

    print(f"  QuadMix train: {len(quadmix_train):,}, val: {len(val_docs):,}")
    print(f"  Random  train: {len(random_train):,}, val: {len(val_docs):,} (shared)")

    print(f"\n[5/5] Writing sharded parquet files...")

    def write_dataset(docs, data_dir, name):
        n_shards = max(1, (len(docs) + args.shard_size - 1) // args.shard_size)
        for i in range(n_shards):
            start = i * args.shard_size
            end = min(start + args.shard_size, len(docs))
            shard_docs = docs[start:end]
            out_path = data_dir / f"shard_{i:05d}.parquet"
            write_shard(shard_docs, str(out_path))
        val_path = data_dir / f"shard_{n_shards:05d}.parquet"
        write_shard(val_docs, str(val_path))
        print(f"  {name}: {n_shards} train shards + 1 val shard -> {data_dir}")

    write_dataset(quadmix_train, quadmix_dir, "QuadMix")
    write_dataset(random_train, random_dir, "Random")

    stats = {
        "quadmix": {
            "train_docs": len(quadmix_train),
            "val_docs": len(val_docs),
            "tokens": sum(d["token_count"] for d in quadmix_train),
            "shards": max(1, (len(quadmix_train) + args.shard_size - 1) // args.shard_size),
        },
        "random": {
            "train_docs": len(random_train),
            "val_docs": len(val_docs),
            "tokens": sum(d["token_count"] for d in random_train),
            "shards": max(1, (len(random_train) + args.shard_size - 1) // args.shard_size),
        },
        "config": {
            "seed": args.seed,
            "shard_size": args.shard_size,
            "val_ratio": args.val_ratio,
            "token_method": token_method,
            "quadmix_source": args.quadmix_dataset,
            "essential_web_source": args.essential_web_dir,
        }
    }
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved to: {stats_path}")

    print("\n" + "=" * 60)
    print("  Done! Two datasets ready for nanochat mid-training:")
    print(f"    QuadMix: {quadmix_dir}")
    print(f"    Random:  {random_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
