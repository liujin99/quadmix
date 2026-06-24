"""
Analyze document length distribution in preprocessed shards.

Shows how many documents would be filtered at different max_chars thresholds.

Usage:
    python3 analyze_doc_lengths.py \
        --preprocessed-data-dir /path/to/preprocessed \
        [--thresholds 50000,100000,200000,500000]
"""

import argparse
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm


def scan_shard(args):
    idx, path = args
    table = pq.read_table(path, columns=["doc_char_count"])
    return idx, table["doc_char_count"].to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-data-dir", required=True)
    parser.add_argument("--thresholds", default="50000,100000,200000,500000",
                        help="Comma-separated char count thresholds")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    thresholds = [int(t) for t in args.thresholds.split(",")]
    num_workers = args.num_workers or min(32, __import__("os").cpu_count() or 1)

    data_dir = Path(args.preprocessed_data_dir)
    shard_files = sorted(data_dir.glob("*.parquet"))
    print(f"Scanning {len(shard_files)} parquet files from {data_dir}...")

    tasks = [(i, str(p)) for i, p in enumerate(shard_files)]
    all_counts = []

    with Pool(num_workers) as pool:
        for idx, counts in tqdm(pool.imap_unordered(scan_shard, tasks, chunksize=1),
                                total=len(tasks)):
            all_counts.extend(counts.tolist())

    all_counts = np.array(all_counts)
    valid = all_counts[all_counts >= 100]

    print(f"\nTotal documents: {len(all_counts):,}")
    print(f"Valid documents (>=100 chars): {len(valid):,}")
    print(f"\nLength distribution (valid docs):")
    print(f"  min={valid.min():,}  median={int(np.median(valid)):,}  "
          f"mean={int(np.mean(valid)):,}  max={valid.max():,}")
    print(f"  p90={int(np.percentile(valid, 90)):,}  "
          f"p95={int(np.percentile(valid, 95)):,}  "
          f"p99={int(np.percentile(valid, 99)):,}")

    print(f"\nFiltering impact at different thresholds:")
    print(f"{'Threshold':>12} {'Filtered':>10} {'Remaining':>12} {'% Kept':>8} {'Tokens Lost':>14}")
    print("-" * 60)
    for t in thresholds:
        kept = valid[valid <= t]
        filtered = len(valid) - len(kept)
        tokens_lost = int((valid[valid > t].sum()) // 4)
        print(f"{t:>12,} {filtered:>10,} {len(kept):>12,} {len(kept)/len(valid)*100:>7.2f}% "
              f"{tokens_lost:>14,}")


if __name__ == "__main__":
    main()
