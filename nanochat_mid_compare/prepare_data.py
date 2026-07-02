"""
Prepare nanochat-compatible training datasets for comparison:
  1. QuadMix-selected subset
  2. Random subset from preprocessed shards (token-count aligned)
  3. Quality-Only Top-K subsets (token-count aligned, optional, multiple methods)

All datasets are written as sharded parquet files with a single "text" column,
compatible with nanochat's dataloader (last shard = validation).

Usage:
    python prepare_data.py \
        --quadmix-dataset /path/to/sampled_dataset.parquet \
        --preprocessed-data-dir /path/to/preprocessed \
        --output-dir /path/to/experiment/data \
        --tokenizer-pkl /path/to/nanochat/tokenizer/tokenizer.pkl \
        [--quality-method dclm,fineweb_edu] \
        [--shard-size 10000] \
        [--val-ratio 0.05] \
        [--seed 42]
"""

import os
import sys
import gc
import argparse
import random
from collections import Counter
import json
import pickle
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

_SPAWN_CTX = mp.get_context("spawn")


QUALITY_SCORE_MAP = {
    "dclm": "qs_dclm",
    "fineweb_edu": "qs_fineweb_edu_approx",
    "english": "qs_english",
    "math_general": "qs_eai_general_math",
    "math_openweb": "qs_eai_open_web_math",
}

_worker_tokenizer = None


def _init_worker(tokenizer_pkl_path):
    global _worker_tokenizer
    with open(tokenizer_pkl_path, "rb") as f:
        _worker_tokenizer = pickle.load(f)


def _worker_encode_batch(texts):
    if hasattr(_worker_tokenizer, "encode_ordinary_batch"):
        return [len(ids) for ids in _worker_tokenizer.encode_ordinary_batch(texts, num_threads=4)]
    return [len(_worker_tokenizer.encode_ordinary(t)) for t in texts]


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


def count_tokens_mp(texts, tokenizer_pkl_path, num_workers=None, chunk_timeout=600):
    if num_workers is None:
        num_workers = min(mp.cpu_count() // 4, 48) or 1
    chunk_size = max(1, len(texts) // (num_workers * 4))
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    enc = load_tokenizer(tokenizer_pkl_path)
    
    with _SPAWN_CTX.Pool(num_workers, initializer=_init_worker, initargs=(tokenizer_pkl_path,)) as pool:
        async_results = [(i, pool.apply_async(_worker_encode_batch, (chunk,)))
                         for i, chunk in enumerate(chunks)]
        results = [None] * len(chunks)
        pbar = tqdm(total=len(chunks), desc=f"  Tokenizing ({num_workers} processes x 4 rust threads)")
        for i, ar in async_results:
            try:
                results[i] = ar.get(timeout=chunk_timeout)
            except mp.TimeoutError:
                chunk = chunks[i]
                lengths = [len(t) for t in chunk]
                print(f"\n  WARNING: chunk {i} timed out after {chunk_timeout}s")
                print(f"    Chunk stats: {len(chunk)} docs, min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")
                print(f"    Retrying in main process...")
                try:
                    if hasattr(enc, "encode_ordinary_batch"):
                        results[i] = [len(ids) for ids in enc.encode_ordinary_batch(chunks[i], num_threads=1)]
                    else:
                        results[i] = [len(enc.encode_ordinary(t)) for t in chunks[i]]
                    print(f"  Retry succeeded for chunk {i}")
                except Exception as e2:
                    print(f"  Retry failed: {e2}, using estimates")
                    results[i] = [len(t) // 4 for t in chunks[i]]
            except Exception as e:
                print(f"\n  WARNING: chunk {i} failed: {e}, retrying in main process...")
                try:
                    if hasattr(enc, "encode_ordinary_batch"):
                        results[i] = [len(ids) for ids in enc.encode_ordinary_batch(chunks[i], num_threads=1)]
                    else:
                        results[i] = [len(enc.encode_ordinary(t)) for t in chunks[i]]
                    print(f"  Retry succeeded for chunk {i}")
                except Exception as e2:
                    print(f"  Retry failed: {e2}, using estimates")
                    results[i] = [len(t) // 4 for t in chunks[i]]
            pbar.update(1)
        pbar.close()
    return [c for batch in results for c in batch]


def estimate_tokens(text):
    return len(text) // 4


def _has_char_repetition(text, max_ratio=0.3):
    if len(text) < 1000:
        return False
    non_ws = [c for c in text if not c.isspace()]
    if len(non_ws) < 1000:
        return False
    counts = Counter(non_ws)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(non_ws) > max_ratio


def _filter_docs_chunk(args):
    chunk, max_chars, max_char_repeat_ratio = args
    valid = []
    n_empty = 0
    n_too_long = 0
    n_repeat = 0
    for t in chunk:
        if not t:
            n_empty += 1
        elif len(t) > max_chars:
            n_too_long += 1
        elif _has_char_repetition(t, max_char_repeat_ratio):
            n_repeat += 1
        else:
            valid.append(t)
    return valid, n_empty, n_too_long, n_repeat


def _read_docs_from_shard_tagged(args):
    shard_id, shard_path, doc_indices = args
    table = pq.read_table(shard_path, columns=["text"])
    texts = table["text"].to_pylist()
    return shard_id, [
        {"text": texts[i], "char_count": len(texts[i]), "token_count": len(texts[i]) // 4}
        for i in doc_indices
    ]


def read_docs_from_shards(shard_paths, selections, num_workers=None, desc=None):
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 128) or 1
    shard_to_docs = {}
    for shard_id, doc_id in selections:
        if shard_id not in shard_to_docs:
            shard_to_docs[shard_id] = []
        shard_to_docs[shard_id].append(doc_id)
    tasks = [(sid, str(shard_paths[sid]), indices) for sid, indices in shard_to_docs.items()]
    with _SPAWN_CTX.Pool(num_workers) as pool:
        unordered = list(tqdm(
            pool.imap_unordered(_read_docs_from_shard_tagged, tasks, chunksize=1),
            total=len(tasks),
            desc=desc or f"  Reading selected docs ({num_workers} processes)",
        ))
    shard_result_map = {sid: docs for sid, docs in unordered}
    shard_cursors = {sid: 0 for sid in shard_to_docs}
    result = []
    for shard_id, _ in selections:
        idx = shard_cursors[shard_id]
        result.append(shard_result_map[shard_id][idx])
        shard_cursors[shard_id] = idx + 1
    return result


def _scan_preprocessed_shard_indexed(args):
    idx, shard_path, max_chars, max_char_repeat_ratio = args
    table = pq.read_table(shard_path, columns=["doc_char_count", "text"])
    char_counts = table["doc_char_count"].to_pylist()
    texts = table["text"].to_pylist()
    valid = []
    filtered_long = 0
    filtered_repeat = 0
    for i, cc in enumerate(char_counts):
        if not cc:
            continue
        if max_chars is not None and cc > max_chars:
            filtered_long += 1
            continue
        if _has_char_repetition(texts[i], max_char_repeat_ratio):
            filtered_repeat += 1
            continue
        valid.append((i, cc))
    return idx, valid, filtered_long, filtered_repeat


def scan_preprocessed_shards(preprocessed_data_dir, num_workers=None, max_shards=None,
                             max_chars=1000000, max_char_repeat_ratio=0.3):
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 128) or 1
    shard_files = sorted(Path(preprocessed_data_dir).glob("preprocessed_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No preprocessed_*.parquet files found in {preprocessed_data_dir}")
    if max_shards is not None:
        shard_files = shard_files[:max_shards]
    tasks = [(i, str(p), max_chars, max_char_repeat_ratio) for i, p in enumerate(shard_files)]
    with _SPAWN_CTX.Pool(num_workers) as pool:
        results = [None] * len(shard_files)
        total_filtered_long = 0
        total_filtered_repeat = 0
        for idx, docs, fl, fr in tqdm(
            pool.imap_unordered(_scan_preprocessed_shard_indexed, tasks, chunksize=1),
            total=len(tasks),
            desc=f"  Scanning preprocessed shards ({num_workers} processes)",
        ):
            results[idx] = docs
            total_filtered_long += fl
            total_filtered_repeat += fr
    if total_filtered_long > 0 or total_filtered_repeat > 0:
        print(f"  Filtered {total_filtered_long:,} docs (>{max_chars:,} chars), "
              f"{total_filtered_repeat:,} docs (single char >{max_char_repeat_ratio*100:.0f}% repetition)")
    return shard_files, results


def write_shard(docs, output_path, num_npu=8):
    texts = [d["text"] for d in docs]
    table = pa.table({"text": texts})
    rg_size = max(1, len(docs) // (num_npu * 2))
    pq.write_table(table, output_path, row_group_size=rg_size)


def trim_docs_to_target(docs, target_tokens):
    trimmed = []
    trim_tokens = 0
    for doc in docs:
        if trim_tokens + doc["token_count"] > target_tokens:
            break
        trimmed.append(doc)
        trim_tokens += doc["token_count"]
    return trimmed, trim_tokens


def select_quality_topk(prep_files, prep_metadata, quality_method, total_tokens,
                        tokenizer_pkl=None, num_workers=None, enc=None):
    quality_col = QUALITY_SCORE_MAP[quality_method]
    print(f"  Using {len(prep_files)} preprocessed shards (pre-scanned)")

    print(f"  Reading quality scores ({quality_col})...")
    all_quality_docs = []
    for shard_id, docs in enumerate(tqdm(prep_metadata, desc=f"  Reading quality scores")):
        if not docs:
            continue
        df = pq.read_table(str(prep_files[shard_id]), columns=[quality_col]).to_pandas()
        scores = df[quality_col].to_numpy()
        for doc_id, char_count in docs:
            all_quality_docs.append((shard_id, doc_id, float(scores[doc_id]), char_count // 4))
    print(f"  Total candidate docs: {len(all_quality_docs):,}")

    all_quality_docs.sort(key=lambda x: x[2], reverse=True)

    target_with_buffer = int(total_tokens * 1.1)
    q_selected = []
    q_accumulated = 0
    for shard_id, doc_id, score, est_tokens in all_quality_docs:
        if q_accumulated >= target_with_buffer:
            break
        q_selected.append((shard_id, doc_id))
        q_accumulated += est_tokens
    cutoff_idx = min(len(q_selected) - 1, len(all_quality_docs) - 1)
    print(f"  Selected {len(q_selected):,} top-quality docs (estimated ~{q_accumulated:,} tokens)")
    print(f"  Quality score range: {all_quality_docs[0][2]:.4f} (best) -> {all_quality_docs[cutoff_idx][2]:.4f} (cutoff)")

    quality_docs = read_docs_from_shards(
        prep_files, q_selected, num_workers=num_workers,
        desc=f"  Reading quality docs ({num_workers or 'auto'} processes)")

    del all_quality_docs
    gc.collect()

    quality_tokens = 0
    if enc and tokenizer_pkl:
        print(f"  Re-counting tokens for {len(quality_docs):,} docs (exact)...")
        q_texts = [d["text"] for d in quality_docs]
        q_exact = count_tokens_mp(q_texts, tokenizer_pkl, num_workers=num_workers)
        for doc, tc in zip(quality_docs, q_exact):
            doc["token_count"] = tc
        del q_texts, q_exact
        gc.collect()
        quality_tokens = sum(d["token_count"] for d in quality_docs)
        print(f"  Exact tokens before trim: {quality_tokens:,} (target: {total_tokens:,})")

        if quality_tokens > total_tokens:
            n_before = len(quality_docs)
            quality_docs, quality_tokens = trim_docs_to_target(quality_docs, total_tokens)
            print(f"  Trimmed {n_before - len(quality_docs):,} docs to match target")
    else:
        quality_tokens = sum(d["token_count"] for d in quality_docs)

    print(f"  Quality docs: {len(quality_docs):,}")
    print(f"  Tokens: {quality_tokens:,}")
    return quality_docs, quality_tokens


def main():
    parser = argparse.ArgumentParser(description="Prepare comparison datasets for nanochat mid-training")
    parser.add_argument("--quadmix-sampled-data", type=str, required=True,
                        help="Path to QuadMix sampled_dataset.parquet")
    parser.add_argument("--preprocessed-data-dir", type=str, required=True,
                        help="Path to preprocessed shards directory (preprocessed_*.parquet)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for the datasets")
    parser.add_argument("--tokenizer-pkl", type=str, default=None,
                        help="Path to nanochat tokenizer.pkl for accurate token counting. "
                             "Falls back to char_count//4 if not provided.")
    parser.add_argument("--quality-method", type=str, default="dclm",
                        help="Comma-separated quality score methods for top-k selection. "
                             f"Options: {', '.join(QUALITY_SCORE_MAP.keys())} (default: dclm)")
    parser.add_argument("--shard-size", type=int, default=10000,
                        help="Documents per output shard (default: 10000)")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fraction of documents reserved for shared validation (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--max-random-scan", type=int, default=500,
                        help="Max number of essential-web shards to scan for random baseline (default: 500)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers. Tokenize: processes (each uses 4 rust threads). "
                             "Shard read: threads. Default: auto")
    parser.add_argument("--num-npu", type=int, default=8,
                        help="Number of NPUs for DDP (ensures enough row groups per shard)")
    parser.add_argument("--max-chars", type=int, default=1000000,
                        help="Skip documents longer than this (safety net). Default: 1000000")
    parser.add_argument("--max-char-repeat-ratio", type=float, default=0.3,
                        help="Skip documents where any single char exceeds this ratio "
                             "(filters corrupted/binary data). Default: 0.3")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    enc = load_tokenizer(args.tokenizer_pkl)
    token_method = "nanochat tokenizer" if enc else "char_count // 4 (estimate)"

    quality_methods = [m.strip() for m in args.quality_method.split(",") if m.strip()]
    for m in quality_methods:
        if m not in QUALITY_SCORE_MAP:
            print(f"ERROR: Unknown quality method '{m}'. Options: {', '.join(QUALITY_SCORE_MAP.keys())}")
            sys.exit(1)
    do_quality = len(quality_methods) > 0

    if not os.path.isdir(args.preprocessed_data_dir):
        print(f"ERROR: Preprocessed directory not found: {args.preprocessed_data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    quadmix_dir = output_dir / "quadmix_data"
    random_dir = output_dir / "random_data"
    quality_dirs = {}
    if do_quality:
        for m in quality_methods:
            qd = output_dir / f"quality_data_{m}"
            qd.mkdir(parents=True, exist_ok=True)
            quality_dirs[m] = qd
    quadmix_dir.mkdir(parents=True, exist_ok=True)
    random_dir.mkdir(parents=True, exist_ok=True)

    baselines = ["quadmix", "random"]
    if do_quality:
        for m in quality_methods:
            baselines.append(f"quality_{m}")

    print("=" * 60)
    print("  Nanochat Comparison Dataset Preparation")
    print("=" * 60)
    print(f"  Token counting: {token_method}")
    print(f"  Baselines: {', '.join(baselines)}")
    if do_quality:
        for m in quality_methods:
            print(f"  Quality method: {m} ({QUALITY_SCORE_MAP[m]})")

    print(f"\n[1/6] Reading QuadMix selected dataset...")
    quadmix_table = pq.read_table(args.quadmix_sampled_data, columns=["text"])
    texts = quadmix_table["text"].to_pylist()
    del quadmix_table
    max_chars = args.max_chars
    max_char_repeat_ratio = args.max_char_repeat_ratio
    num_workers = args.num_workers or min(mp.cpu_count(), 128) or 1
    chunk_size = max(1, len(texts) // (num_workers * 4))
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    filter_tasks = [(c, max_chars, max_char_repeat_ratio) for c in chunks]
    valid_texts = []
    n_empty = 0
    n_too_long = 0
    n_repeat = 0
    with _SPAWN_CTX.Pool(num_workers) as pool:
        for valid, ne, tl, tr in tqdm(
            pool.imap_unordered(_filter_docs_chunk, filter_tasks, chunksize=1),
            total=len(filter_tasks),
            desc=f"  Filtering QuadMix docs ({num_workers} processes)",
        ):
            valid_texts.extend(valid)
            n_empty += ne
            n_too_long += tl
            n_repeat += tr
    n_filtered = n_empty + n_too_long + n_repeat
    if n_filtered > 0:
        print(f"  Filtered {n_filtered:,} docs: {n_empty:,} empty, "
              f"{n_too_long:,} > {max_chars:,} chars, {n_repeat:,} repetitive")
    print(f"  Counting tokens for {len(valid_texts):,} docs...")
    if enc and args.tokenizer_pkl:
        token_counts = count_tokens_mp(valid_texts, args.tokenizer_pkl, num_workers=args.num_workers)
    else:
        token_counts = [estimate_tokens(t) for t in valid_texts]
    quadmix_docs = [
        {"text": t, "token_count": tc}
        for t, tc in zip(valid_texts, token_counts)
    ]
    total_tokens = sum(token_counts)

    del texts, valid_texts, token_counts
    gc.collect()

    print(f"  QuadMix docs: {len(quadmix_docs):,}")
    print(f"  Tokens ({token_method}): {total_tokens:,}")

    print(f"\n[2/6] Scanning preprocessed shards metadata...")
    prep_files, prep_metadata = scan_preprocessed_shards(
        args.preprocessed_data_dir, num_workers=args.num_workers, max_shards=args.max_random_scan,
        max_chars=args.max_chars, max_char_repeat_ratio=args.max_char_repeat_ratio)
    print(f"  Scanning {len(prep_files)} shards (metadata only)...")

    all_candidates = []
    for shard_id, docs in enumerate(prep_metadata):
        for doc_id, char_count in docs:
            all_candidates.append((shard_id, doc_id, char_count // 4))
    print(f"  Total candidate docs: {len(all_candidates):,}")

    print(f"\n[3/6] Random sampling (target: {total_tokens:,} tokens)...")
    random.shuffle(all_candidates)
    selected = []
    accumulated_tokens = 0
    target_with_buffer = int(total_tokens * 1.1)
    for shard_id, doc_id, est_tokens in all_candidates:
        if accumulated_tokens >= target_with_buffer:
            break
        selected.append((shard_id, doc_id))
        accumulated_tokens += est_tokens
    print(f"  Selected {len(selected):,} docs (estimated ~{accumulated_tokens:,} tokens, with 10% buffer)")

    print(f"\n[3.5/6] Reading selected documents...")
    random_docs = read_docs_from_shards(prep_files, selected, num_workers=args.num_workers,
                                        desc=f"  Reading random docs ({args.num_workers or 'auto'} processes)")

    if enc and args.tokenizer_pkl:
        print(f"  Re-counting tokens for {len(random_docs):,} docs (exact)...")
        random_texts = [d["text"] for d in random_docs]
        exact_counts = count_tokens_mp(random_texts, args.tokenizer_pkl, num_workers=args.num_workers)
        for doc, tc in zip(random_docs, exact_counts):
            doc["token_count"] = tc
        del random_texts, exact_counts
        gc.collect()
        accumulated_tokens = sum(d["token_count"] for d in random_docs)
        print(f"  Exact tokens before trim: {accumulated_tokens:,} (target: {total_tokens:,})")

        if accumulated_tokens > total_tokens:
            n_before = len(random_docs)
            random_docs, accumulated_tokens = trim_docs_to_target(random_docs, total_tokens)
            print(f"  Trimmed {n_before - len(random_docs):,} docs to match target")

    del all_candidates, selected
    gc.collect()

    print(f"  Random docs: {len(random_docs):,}")
    print(f"  Tokens ({token_method}): {accumulated_tokens:,}")

    import tempfile
    _tmp_dir = tempfile.mkdtemp(prefix="prepare_data_")
    print(f"\n  Saving docs to temp dir to free memory: {_tmp_dir}")
    _qm_tmp = os.path.join(_tmp_dir, "quadmix.parquet")
    _rd_tmp = os.path.join(_tmp_dir, "random.parquet")
    pq.write_table(pa.table({"text": [d["text"] for d in quadmix_docs],
                              "token_count": [d["token_count"] for d in quadmix_docs]}), _qm_tmp)
    pq.write_table(pa.table({"text": [d["text"] for d in random_docs],
                              "token_count": [d["token_count"] for d in random_docs]}), _rd_tmp)
    del quadmix_docs, random_docs
    gc.collect()
    print(f"  Freed quadmix_docs + random_docs from memory")

    quality_datasets = {}
    if do_quality:
        for mi, method in enumerate(quality_methods):
            print(f"\n[4.{mi+1}/6] Quality-Only Top-K selection ({method})...")
            q_docs, q_tokens = select_quality_topk(
                prep_files, prep_metadata, method, total_tokens,
                tokenizer_pkl=args.tokenizer_pkl, num_workers=args.num_workers, enc=enc)
            quality_datasets[method] = (q_docs, q_tokens)
    else:
        print(f"\n[4/6] Quality baseline skipped (no quality methods specified)")

    print(f"\n  Reloading docs from temp dir...")
    _qm_df = pq.read_table(_qm_tmp).to_pandas()
    quadmix_docs = [{"text": t, "char_count": len(t), "token_count": tc}
                    for t, tc in zip(_qm_df["text"], _qm_df["token_count"])]
    del _qm_df
    _rd_df = pq.read_table(_rd_tmp).to_pandas()
    random_docs = [{"text": t, "char_count": len(t), "token_count": tc}
                   for t, tc in zip(_rd_df["text"], _rd_df["token_count"])]
    del _rd_df
    gc.collect()
    import shutil
    shutil.rmtree(_tmp_dir, ignore_errors=True)
    print(f"  Reloaded {len(quadmix_docs):,} quadmix + {len(random_docs):,} random docs")

    print(f"\n[5/6] Splitting train/val (val_ratio={args.val_ratio})...")
    n_val = int(len(quadmix_docs) * args.val_ratio) if args.val_ratio > 0 else 0

    random.shuffle(quadmix_docs)
    random.shuffle(random_docs)
    for method in quality_datasets:
        random.shuffle(quality_datasets[method][0])

    if n_val > 0:
        val_docs = quadmix_docs[:n_val]
        quadmix_train = quadmix_docs[n_val:]
        random_train = random_docs[n_val:]
        quality_trains = {m: d[n_val:] for m, (d, _) in quality_datasets.items()}
    else:
        val_docs = []
        quadmix_train = quadmix_docs
        random_train = random_docs
        quality_trains = {m: d for m, (d, _) in quality_datasets.items()}

    print(f"  QuadMix train: {len(quadmix_train):,}, val: {len(val_docs):,}")
    print(f"  Random  train: {len(random_train):,}, val: {len(val_docs):,}")
    for m, qt in quality_trains.items():
        print(f"  Quality ({m}) train: {len(qt):,}, val: {len(val_docs):,}")

    print(f"\n[6/6] Writing sharded parquet files...")

    def write_dataset(docs, data_dir, name):
        n_shards = max(1, (len(docs) + args.shard_size - 1) // args.shard_size)
        for i in range(n_shards):
            start = i * args.shard_size
            end = min(start + args.shard_size, len(docs))
            shard_docs = docs[start:end]
            out_path = data_dir / f"shard_{i:05d}.parquet"
            write_shard(shard_docs, str(out_path), args.num_npu)
        val_path = data_dir / f"shard_{n_shards:05d}.parquet"
        if val_docs:
            write_shard(val_docs, str(val_path), args.num_npu)
        else:
            dummy_val = [{"text": "dummy"}]
            write_shard(dummy_val, str(val_path), args.num_npu)
        val_label = f"{len(val_docs)} val" if val_docs else "1 dummy val"
        print(f"  {name}: {n_shards} train shards + {val_label} -> {data_dir}")

    write_dataset(quadmix_train, quadmix_dir, "QuadMix")
    write_dataset(random_train, random_dir, "Random")
    for m, qt in quality_trains.items():
        write_dataset(qt, quality_dirs[m], f"Quality ({m})")

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
            "quadmix_source": args.quadmix_sampled_data,
            "preprocessed_source": args.preprocessed_data_dir,
            "baselines": baselines,
        }
    }
    if quality_trains:
        stats["config"]["quality_methods"] = quality_methods
        for m, qt in quality_trains.items():
            stats[f"quality_{m}"] = {
                "train_docs": len(qt),
                "val_docs": len(val_docs),
                "tokens": sum(d["token_count"] for d in qt),
                "shards": max(1, (len(qt) + args.shard_size - 1) // args.shard_size),
                "method": m,
                "quality_column": QUALITY_SCORE_MAP[m],
            }

    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved to: {stats_path}")

    print("\n" + "=" * 60)
    print(f"  Done! {len(baselines)} datasets ready for nanochat mid-training:")
    print(f"    QuadMix: {quadmix_dir}")
    print(f"    Random:  {random_dir}")
    for m in quality_datasets:
        print(f"    Quality ({m}): {quality_dirs[m]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
