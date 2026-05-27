#!/usr/bin/env python3
"""
Preprocess essential-web-v1.0: extract domain labels and quality signals,
outputting one parquet per shard (multi-shard mode).

Usage:
  python scripts/preprocess_essential_web_v1_sharded.py \
      --input-dir data/essential-web-v1 \
      --output-dir temp/preprocessed
"""

import argparse, json, os, time, glob
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_QUADMIX_DIR = os.path.dirname(_SCRIPT_DIR)

FASTTEXT_FIELDS = ["dclm", "fineweb_edu_approx", "english",
                   "eai_general_math", "eai_open_web_math"]

DOMAIN_MAP = {
    "Industrial arts, Technology, and Engineering": 0,
    "Social sciences": 1,
    "Science and Natural history": 2,
    "Religion": 3,
    "Philology; or, Language and languages": 4,
    "Literature": 5,
    "History and Geography": 6,
    "General works, books and libraries, information sciences": 7,
    "Philosophy and psychology": 8,
    "Arts": 9,
}


def extract_domain_level_1(eai_taxonomy):
    if isinstance(eai_taxonomy, str):
        try:
            eai_taxonomy = json.loads(eai_taxonomy)
        except (json.JSONDecodeError, ValueError):
            return -1
    if not isinstance(eai_taxonomy, dict):
        return -1
    fdc = eai_taxonomy.get("free_decimal_correspondence", {})
    if not isinstance(fdc, dict):
        return -1
    primary = fdc.get("primary", {})
    if not isinstance(primary, dict):
        return -1
    labels = primary.get("labels", {})
    if not isinstance(labels, dict):
        return -1
    level_1 = labels.get("level_1")
    if isinstance(level_1, str) and level_1 in DOMAIN_MAP:
        return DOMAIN_MAP[level_1]
    return -1


def extract_quality_signals(quality_signals):
    if not isinstance(quality_signals, dict):
        return np.zeros(len(FASTTEXT_FIELDS), dtype=np.float32)
    fasttext = quality_signals.get("fasttext", {})
    if not isinstance(fasttext, dict):
        return np.zeros(len(FASTTEXT_FIELDS), dtype=np.float32)
    return np.array([
        fasttext.get(field, 0.0) or 0.0
        for field in FASTTEXT_FIELDS
    ], dtype=np.float32)


QUALITY_COLUMNS = [
    "qs_dclm", "qs_fineweb_edu_approx", "qs_english",
    "qs_eai_general_math", "qs_eai_open_web_math",
]


def process_shard(shard_path: str, shard_idx: int, output_dir: str) -> dict:
    """Process one raw shard, save preprocessed version. Returns stats dict."""
    t0 = time.time()
    df = pd.read_parquet(shard_path,
                         columns=["text", "eai_taxonomy", "quality_signals"])
    n = len(df)

    # Extract domain labels
    domains = df["eai_taxonomy"].apply(extract_domain_level_1)

    # Extract quality signals
    quality = df["quality_signals"].apply(extract_quality_signals)
    quality_matrix = np.stack(quality.to_numpy())

    # Build output
    output = pd.DataFrame({
        "text": df["text"],
        "doc_char_count": df["text"].str.len().to_numpy(dtype=np.int64),
        "domain": domains,
        "shard_idx": np.full(n, shard_idx, dtype=np.int64),
        "row_in_shard": np.arange(n, dtype=np.int64),
        QUALITY_COLUMNS[0]: quality_matrix[:, 0],
        QUALITY_COLUMNS[1]: quality_matrix[:, 1],
        QUALITY_COLUMNS[2]: quality_matrix[:, 2],
        QUALITY_COLUMNS[3]: quality_matrix[:, 3],
        QUALITY_COLUMNS[4]: quality_matrix[:, 4],
    })

    # Save
    out_name = f"preprocessed_{shard_idx:05d}.parquet"
    out_path = os.path.join(output_dir, out_name)
    output.to_parquet(out_path, index=False)

    valid_domains = (domains >= 0).sum()
    elapsed = time.time() - t0
    print(f"  [{shard_idx:05d}] {out_name}: {n:,} docs, "
          f"{valid_domains}/{n} valid domains, {elapsed:.1f}s")

    return {
        "shard_idx": shard_idx,
        "file": out_name,
        "path": out_path,
        "num_docs": n,
        "valid_domains": int(valid_domains),
        "elapsed_seconds": elapsed,
    }


def parse_shard_idx_from_path(shard_path: str) -> int:
    """Extract shard index from filename like 'shard_00000.parquet'."""
    basename = os.path.basename(shard_path)
    # Handle both 'shard_00000.parquet' and 'shard_00000' patterns
    name = basename.replace(".parquet", "")
    if name.startswith("shard_"):
        return int(name.replace("shard_", ""))
    # Fallback: use enumeration index
    return -1


def main():
    p = argparse.ArgumentParser(
        description="Preprocess essential-web-v1 in multi-shard mode")
    p.add_argument("--input-dir",
                   default=os.path.join(_QUADMIX_DIR, "data/essential-web-v1"),
                   help="Directory containing raw parquet shards")
    p.add_argument("--output-dir",
                   default=os.path.join(_QUADMIX_DIR, "temp/preprocessed"),
                   help="Output directory for preprocessed shards")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of shards to process (for testing)")
    p.add_argument("--force", action="store_true",
                   help="Force reprocess even if output already exists")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover shards
    shard_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if not shard_paths:
        print(f"Error: no parquet files found in {args.input_dir}")
        return 1

    if args.limit:
        shard_paths = shard_paths[:args.limit]

    print(f"Found {len(shard_paths)} shards in {args.input_dir}")

    # ── Incremental: check existing preprocessed files ──
    existing_preprocessed = set()
    if not args.force:
        for fname in os.listdir(args.output_dir):
            if fname.startswith("preprocessed_") and fname.endswith(".parquet"):
                # Parse shard_idx from preprocessed_00000.parquet
                idx_str = fname.replace("preprocessed_", "").replace(".parquet", "")
                try:
                    existing_preprocessed.add(int(idx_str))
                except ValueError:
                    pass

    if existing_preprocessed and not args.force:
        print(f"  Already preprocessed: {len(existing_preprocessed)} shards")

    # ── Detect missing shards (deleted mid-shards) ──
    needed_shard_indices = set()
    for sp in shard_paths:
        shard_idx = parse_shard_idx_from_path(sp)
        if shard_idx >= 0:
            needed_shard_indices.add(shard_idx)

    missing_shards = needed_shard_indices - existing_preprocessed
    if missing_shards and not args.force:
        print(f"  Missing {len(missing_shards)} shards (will reprocess): {sorted(missing_shards)[:5]}{'...' if len(missing_shards) > 5 else ''}")

    # Process each shard with ORIGINAL shard_idx (from filename)
    t_start = time.time()
    shard_index = []
    skipped = 0
    for sp in shard_paths:
        shard_idx = parse_shard_idx_from_path(sp)
        if shard_idx < 0:
            # Fallback to enumeration if filename doesn't match pattern
            shard_idx = len(shard_index)

        # Skip if already processed (incremental mode)
        if shard_idx in existing_preprocessed and shard_idx not in missing_shards and not args.force:
            skipped += 1
            # Still add to shard_index (read stats from existing file)
            out_name = f"preprocessed_{shard_idx:05d}.parquet"
            out_path = os.path.join(args.output_dir, out_name)
            df = pd.read_parquet(out_path, columns=["domain"])
            n = len(df)
            valid_domains = (df["domain"] >= 0).sum()
            shard_index.append({
                "shard_idx": shard_idx,
                "file": out_name,
                "path": out_path,
                "num_docs": n,
                "valid_domains": int(valid_domains),
                "elapsed_seconds": 0.0,  # skipped
            })
            continue

        stats = process_shard(sp, shard_idx, args.output_dir)
        shard_index.append(stats)

    if skipped > 0:
        print(f"  Skipped {skipped} already-preprocessed shards")

    # Save shard index
    index_path = os.path.join(args.output_dir, "shard_index.json")
    total_docs = sum(s["num_docs"] for s in shard_index)
    total_valid = sum(s["valid_domains"] for s in shard_index)
    index_data = {
        "num_shards": len(shard_index),
        "total_docs": total_docs,
        "total_valid_domains": total_valid,
        "shards": shard_index,
    }
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Preprocessing complete!")
    print(f"  Shards:     {len(shard_index)}")
    print(f"  Total docs: {total_docs:,}")
    print(f"  Valid dom:  {total_valid:,} ({total_valid/total_docs*100:.1f}%)")
    print(f"  Index:      {index_path}")
    print(f"  Duration:   {elapsed:.1f}s")
    print(f"  Output:     {args.output_dir}/")
    print(f"  [Next] Use --preprocessed-dir {args.output_dir} for run script")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
