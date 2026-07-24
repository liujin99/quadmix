"""
Prepare nanochat-compatible training datasets for comparison:
  1. QuadMix-selected subset (token-budget capped)
  2. Random subset from source shards (token-count aligned)
  3. Manual Ratio subset from source shards (domain-proportional, token-count aligned)
  4. Quality-Only Top-K subsets (token-count aligned, optional, multiple methods)

All datasets are written as sharded parquet files with a single "text" column,
compatible with nanochat's dataloader (last shard = validation).

Token budget logic:
  target_tokens = data_ratio x num_scaling_params
  budget_cap    = target_tokens x 1.1
  All baselines prepare data up to budget_cap to ensure fair comparison.

Usage (essential-web, backward compatible):
    python prepare_data.py \
        --quadmix-sampled-data /path/to/sampled_dataset.parquet \
        --preprocessed-data-dir /path/to/preprocessed \
        --output-dir /path/to/experiment/data \
        --tokenizer-pkl /path/to/nanochat/tokenizer/tokenizer.pkl \
        [--quality-method dclm,fineweb_edu]

Usage (STEM):
    python prepare_data.py \
        --quadmix-sampled-data /path/to/stem_sampled_dataset.parquet \
        --preprocessed-data-dir /path/to/100B_stem_parquet_filtered \
        --output-dir /path/to/experiment/data \
        --schema configs/schema_stem.yaml \
        --file-pattern "*.parquet" \
        --manual-ratio "数学=60:物理=15:化学=12.5:生物学=12.5" \
        --data-ratio 0.5 \
        --num-scaling-params 730000000 \
        --tokenizer-pkl /path/to/nanochat/tokenizer/tokenizer.pkl
"""

import os
import sys
import gc
import argparse
import random
from collections import Counter, defaultdict
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

_io_pool = None
_token_pool = None


def _get_io_pool(num_workers=None):
    global _io_pool
    if _io_pool is None:
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 256) or 1
        _io_pool = _SPAWN_CTX.Pool(num_workers)
    return _io_pool


def _get_token_pool(tokenizer_pkl_path, num_workers=None):
    global _token_pool
    if _token_pool is None:
        if num_workers is None:
            num_workers = min(mp.cpu_count() // 4, 48) or 1
        _token_pool = _SPAWN_CTX.Pool(
            num_workers, initializer=_init_worker, initargs=(tokenizer_pkl_path,))
    return _token_pool


def _cleanup_pools():
    global _io_pool, _token_pool
    if _io_pool is not None:
        _io_pool.close()
        _io_pool.join()
        _io_pool = None
    if _token_pool is not None:
        _token_pool.close()
        _token_pool.join()
        _token_pool = None

QUALITY_SCORE_MAP = {
    "dclm": "qs_dclm",
    "fineweb_edu": "qs_fineweb_edu_approx",
    "english": "qs_english",
    "math_general": "qs_eai_general_math",
    "math_openweb": "qs_eai_open_web_math",
}

ESSENTIAL_WEB_DEFAULTS = {
    "file_pattern": "preprocessed_*.parquet",
    "domain_col": "domain",
    "domain_names": None,
    "char_count_col": "doc_char_count",
    "text_col": "text",
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
    pool = _get_token_pool(tokenizer_pkl_path, num_workers)
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
    shard_id, shard_path, doc_indices, text_col = args
    table = pq.read_table(shard_path, columns=[text_col])
    taken = table.take(pa.array(doc_indices))
    texts = taken[text_col].to_pylist()
    return shard_id, [
        {"text": t, "char_count": len(t), "token_count": len(t) // 4}
        for t in texts
    ]


def read_docs_from_shards(shard_paths, selections, num_workers=None, desc=None, text_col="text",
                         max_char_repeat_ratio=0):
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 256) or 1
    shard_to_docs = {}
    for shard_id, doc_id in selections:
        if shard_id not in shard_to_docs:
            shard_to_docs[shard_id] = []
        shard_to_docs[shard_id].append(doc_id)
    tasks = [(sid, str(shard_paths[sid]), indices, text_col) for sid, indices in shard_to_docs.items()]
    pool = _get_io_pool(num_workers)
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
    if max_char_repeat_ratio > 0:
        n_before = len(result)
        result = [d for d in result if not _has_char_repetition(d["text"], max_char_repeat_ratio)]
        n_filtered = n_before - len(result)
        if n_filtered > 0:
            print(f"  Filtered {n_filtered:,} docs (single char >{max_char_repeat_ratio*100:.0f}% repetition)")
    return result


def _scan_shard_indexed(args):
    idx, shard_path, domain_col, domain_names, char_count_col, text_col, max_chars, max_char_repeat_ratio = args
    shard_schema = pq.read_schema(shard_path)
    shard_col_set = set(shard_schema.names)
    columns_to_read = []
    if domain_col and domain_col in shard_col_set:
        columns_to_read.append(domain_col)
    if char_count_col and char_count_col in shard_col_set:
        columns_to_read.append(char_count_col)
    need_text = (not char_count_col) or (char_count_col and char_count_col not in shard_col_set)
    if need_text and text_col in shard_col_set:
        columns_to_read.append(text_col)
    columns_to_read = list(set(columns_to_read))
    table = pq.read_table(shard_path, columns=columns_to_read)
    n = len(table)
    domain_data = None
    if domain_col and domain_col in table.column_names:
        domain_raw = table.column(domain_col).to_pylist()
        if domain_names is not None:
            cat_dtype = pd.CategoricalDtype(categories=domain_names, ordered=False)
            domain_series = pd.Series(domain_raw).astype(cat_dtype)
            domain_data = domain_series.cat.codes.to_numpy(dtype=np.int64)
        elif all(isinstance(v, (int, np.integer)) for v in domain_raw if v is not None):
            domain_data = np.array(domain_raw, dtype=np.int64)
        else:
            domain_data = np.array(domain_raw)
    char_count_data = None
    if char_count_col and char_count_col in table.column_names:
        char_count_data = table.column(char_count_col).to_pylist()
    text_data = None
    if text_col in table.column_names:
        text_data = table.column(text_col).to_pylist()
    valid_doc_ids = []
    valid_char_counts = []
    valid_domain_vals = []
    filtered_long = 0
    filtered_repeat = 0
    for i in range(n):
        cc = None
        if char_count_data is not None and i < len(char_count_data):
            cc = char_count_data[i]
        elif text_data is not None:
            cc = len(text_data[i]) if text_data[i] else 0
        if not cc or cc == 0:
            continue
        if max_chars is not None and cc > max_chars:
            filtered_long += 1
            continue
        if max_char_repeat_ratio > 0 and text_data is not None and text_data[i]:
            if _has_char_repetition(text_data[i], max_char_repeat_ratio):
                filtered_repeat += 1
                continue
        domain_val = -1
        if domain_data is not None:
            v = domain_data[i]
            domain_val = int(v) if isinstance(v, (int, np.integer)) else -1
        valid_doc_ids.append(i)
        valid_char_counts.append(cc)
        valid_domain_vals.append(domain_val)
    return idx, (np.array(valid_doc_ids, dtype=np.int64),
                 np.array(valid_char_counts, dtype=np.int64),
                 np.array(valid_domain_vals, dtype=np.int64)), filtered_long, filtered_repeat


def scan_shards(data_dir, file_pattern, domain_col=None, domain_names=None,
                char_count_col=None, text_col="text", num_workers=None, max_shards=None,
                max_chars=1000000, max_char_repeat_ratio=0.3):
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 256) or 1
    shard_files = sorted(Path(data_dir).glob(file_pattern))
    if not shard_files:
        raise FileNotFoundError(f"No {file_pattern} files found in {data_dir}")
    if max_shards is not None and max_shards > 0:
        shard_files = shard_files[:max_shards]
    tasks = [(i, str(p), domain_col, domain_names, char_count_col, text_col, max_chars, max_char_repeat_ratio)
             for i, p in enumerate(shard_files)]
    pool = _get_io_pool(num_workers)
    results = [None] * len(shard_files)
    total_filtered_long = 0
    total_filtered_repeat = 0
    for idx, docs, fl, fr in tqdm(
        pool.imap_unordered(_scan_shard_indexed, tasks, chunksize=1),
        total=len(tasks),
        desc=f"  Scanning shards ({num_workers} processes)",
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
                        tokenizer_pkl=None, num_workers=None, enc=None, text_col="text",
                        max_char_repeat_ratio=0):
    quality_col = QUALITY_SCORE_MAP[quality_method]
    print(f"  Using {len(prep_files)} shards (pre-scanned)")
    print(f"  Reading quality scores ({quality_col})...")
    q_doc_ids_list = []
    q_shard_ids_list = []
    q_est_tokens_list = []
    q_scores_list = []
    for shard_id, (doc_ids, char_counts, _domain_vals) in enumerate(
        tqdm(prep_metadata, desc=f"  Reading quality scores")
    ):
        if len(doc_ids) == 0:
            continue
        scores = pq.read_table(str(prep_files[shard_id]), columns=[quality_col]).to_pandas()[quality_col].to_numpy()
        q_scores_list.append(scores[doc_ids])
        q_doc_ids_list.append(doc_ids)
        q_shard_ids_list.append(np.full(len(doc_ids), shard_id, dtype=np.int32))
        q_est_tokens_list.append(char_counts // 4)
    q_doc_ids = np.concatenate(q_doc_ids_list)
    q_shard_ids = np.concatenate(q_shard_ids_list)
    q_est_tokens = np.concatenate(q_est_tokens_list)
    q_scores = np.concatenate(q_scores_list)
    del q_doc_ids_list, q_shard_ids_list, q_est_tokens_list, q_scores_list
    print(f"  Total candidate docs: {len(q_doc_ids):,}")
    order = np.argsort(-q_scores, kind='stable')
    q_scores = q_scores[order]
    q_doc_ids = q_doc_ids[order]
    q_shard_ids = q_shard_ids[order]
    q_est_tokens = q_est_tokens[order]
    cumsum = np.cumsum(q_est_tokens)
    if total_tokens <= 0 or len(cumsum) == 0:
        cutoff = 0
    else:
        cutoff = min(int(np.searchsorted(cumsum, total_tokens, side='left')) + 1, len(q_doc_ids))
    q_selected = list(zip(q_shard_ids[:cutoff].tolist(), q_doc_ids[:cutoff].tolist()))
    q_accumulated = int(cumsum[cutoff - 1]) if cutoff > 0 else 0
    if not q_selected:
        raise RuntimeError("No quality docs selected -- check total_tokens and quality thresholds")
    cutoff_idx = min(cutoff - 1, len(q_doc_ids) - 1)
    print(f"  Selected {len(q_selected):,} top-quality docs (estimated ~{q_accumulated:,} tokens)")
    print(f"  Quality score range: {q_scores[0]:.4f} (best) -> {q_scores[cutoff_idx]:.4f} (cutoff)")
    quality_docs = read_docs_from_shards(
        prep_files, q_selected, num_workers=num_workers,
        desc=f"  Reading quality docs ({num_workers or 'auto'} processes)", text_col=text_col,
        max_char_repeat_ratio=max_char_repeat_ratio)
    del q_doc_ids, q_shard_ids, q_est_tokens, q_scores, q_selected, order, cumsum
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


def parse_manual_ratio(manual_ratio_str, domain_names):
    if not manual_ratio_str:
        return None
    ratio_map = {}
    for part in manual_ratio_str.split(":"):
        if "=" not in part:
            raise ValueError(f"Invalid manual-ratio format: '{part}'. Expected domain=ratio (e.g. 数学=60)")
        domain, ratio = part.split("=", 1)
        domain = domain.strip()
        ratio = float(ratio.strip())
        if domain_names is not None and domain not in domain_names:
            raise ValueError(f"Domain '{domain}' not in schema domain_names {domain_names}")
        ratio_map[domain] = ratio
    total_ratio = sum(ratio_map.values())
    if total_ratio <= 0:
        raise ValueError(f"Total ratio must be > 0, got {total_ratio}")
    return ratio_map


def select_manual_ratio(prep_files, domain_candidates, manual_ratio_map, domain_names,
                        total_tokens, tokenizer_pkl=None, num_workers=None, enc=None, text_col="text",
                        max_char_repeat_ratio=0):
    if domain_names is None:
        raise ValueError("Manual Ratio requires domain_names in schema")
    name_to_id = {name: i for i, name in enumerate(domain_names)}
    total_ratio = sum(manual_ratio_map.values())
    domain_budgets = {}
    for domain_name, ratio in manual_ratio_map.items():
        domain_id = name_to_id[domain_name]
        domain_budgets[domain_id] = total_tokens * (ratio / total_ratio)

    print(f"  Manual Ratio allocation:")
    for domain_name, ratio in manual_ratio_map.items():
        domain_id = name_to_id[domain_name]
        pct = ratio / total_ratio * 100
        print(f"    {domain_name} (id={domain_id}): {pct:.1f}% -> ~{int(domain_budgets[domain_id]):,} est tokens")

    mr_selected = []
    mr_est_tokens = 0
    domain_selected_counts = {}
    for domain_id, budget in domain_budgets.items():
        if domain_id not in domain_candidates:
            print(f"    WARNING: No candidates for domain id={domain_id}, skipping")
            continue
        cand_shard_ids, cand_doc_ids, cand_est_tokens = domain_candidates[domain_id]
        if len(cand_doc_ids) == 0:
            print(f"    WARNING: No candidates for domain id={domain_id}, skipping")
            continue
        perm = np.random.permutation(len(cand_doc_ids))
        shuffled_est_tokens = cand_est_tokens[perm]
        cumsum = np.cumsum(shuffled_est_tokens)
        if budget <= 0 or len(cumsum) == 0:
            cutoff = 0
        else:
            cutoff = min(int(np.searchsorted(cumsum, budget, side='left')) + 1, len(cand_doc_ids))
        domain_selected = list(zip(
            cand_shard_ids[perm[:cutoff]].tolist(),
            cand_doc_ids[perm[:cutoff]].tolist(),
        ))
        accumulated = int(cumsum[cutoff - 1]) if cutoff > 0 else 0
        domain_selected_counts[domain_id] = len(domain_selected)
        pct_filled = accumulated / budget * 100 if budget > 0 else 0
        print(f"    domain id={domain_id}: selected {len(domain_selected):,} docs, "
              f"~{accumulated:,} est tokens ({pct_filled:.1f}% of budget)")
        mr_selected.extend(domain_selected)
        mr_est_tokens += accumulated

    print(f"  Total manual-ratio selected: {len(mr_selected):,} docs, ~{mr_est_tokens:,} est tokens")

    mr_docs = read_docs_from_shards(
        prep_files, mr_selected, num_workers=num_workers,
        desc=f"  Reading manual-ratio docs ({num_workers or 'auto'} processes)", text_col=text_col,
        max_char_repeat_ratio=max_char_repeat_ratio)

    if enc and tokenizer_pkl:
        print(f"  Re-counting tokens for {len(mr_docs):,} docs (exact)...")
        mr_texts = [d["text"] for d in mr_docs]
        mr_exact = count_tokens_mp(mr_texts, tokenizer_pkl, num_workers=num_workers)
        for doc, tc in zip(mr_docs, mr_exact):
            doc["token_count"] = tc
        del mr_texts, mr_exact
        gc.collect()
        mr_tokens = sum(d["token_count"] for d in mr_docs)
        print(f"  Exact tokens before trim: {mr_tokens:,} (target: {total_tokens:,})")
        if mr_tokens > total_tokens:
            n_before = len(mr_docs)
            random.shuffle(mr_docs)
            mr_docs, mr_tokens = trim_docs_to_target(mr_docs, total_tokens)
            print(f"  Trimmed {n_before - len(mr_docs):,} docs to match target (shuffled first for fair trimming)")
    else:
        mr_tokens = sum(d["token_count"] for d in mr_docs)

    print(f"  Manual Ratio docs: {len(mr_docs):,}")
    print(f"  Tokens: {mr_tokens:,}")
    return mr_docs, mr_tokens


def main():
    parser = argparse.ArgumentParser(description="Prepare comparison datasets for nanochat mid-training")
    parser.add_argument("--quadmix-sampled-data", type=str, required=True,
                        help="Path to QuadMix sampled_dataset.parquet")
    parser.add_argument("--preprocessed-data-dir", type=str, required=True,
                        help="Path to source shards directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for the datasets")
    parser.add_argument("--tokenizer-pkl", type=str, default=None,
                        help="Path to nanochat tokenizer.pkl for accurate token counting")
    parser.add_argument("--schema", type=str, default=None,
                        help="YAML schema config file. If not specified, uses essential-web defaults.")
    parser.add_argument("--file-pattern", type=str, default=None,
                        help="Parquet file glob pattern (default: preprocessed_*.parquet)")
    parser.add_argument("--manual-ratio", type=str, default=None,
                        help="Manual domain ratio: domain=ratio:domain=ratio "
                             "(e.g. 数学=60:物理=15:化学=12.5:生物学=12.5)")
    parser.add_argument("--data-ratio", type=float, default=None,
                        help="Target data:param ratio for token budget calculation")
    parser.add_argument("--num-scaling-params", type=int, default=None,
                        help="Number of scaling params. Used with --data-ratio to compute token budget.")
    parser.add_argument("--quality-method", type=str, default="",
                        help="Comma-separated quality score methods for top-k selection (empty = disabled)")
    parser.add_argument("--shard-size", type=int, default=10000,
                        help="Documents per output shard (default: 10000)")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fraction of documents reserved for validation (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="Max source shards to scan (0 = no limit)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers. Default: auto")
    parser.add_argument("--num-npu", type=int, default=8,
                        help="Number of NPUs for DDP (ensures enough row groups per shard)")
    parser.add_argument("--max-chars", type=int, default=1000000,
                        help="Skip documents longer than this (default: 1000000)")
    parser.add_argument("--max-char-repeat-ratio", type=float, default=0.3,
                        help="Skip documents where single char exceeds this ratio (default: 0.3)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    domain_col = ESSENTIAL_WEB_DEFAULTS["domain_col"]
    domain_names = ESSENTIAL_WEB_DEFAULTS["domain_names"]
    char_count_col = ESSENTIAL_WEB_DEFAULTS["char_count_col"]
    text_col = ESSENTIAL_WEB_DEFAULTS["text_col"]
    file_pattern = args.file_pattern or ESSENTIAL_WEB_DEFAULTS["file_pattern"]

    if args.schema:
        try:
            import yaml
        except ImportError:
            print("ERROR: --schema requires PyYAML. Install with: pip install pyyaml")
            sys.exit(1)
        with open(args.schema) as f:
            schema_config = yaml.safe_load(f)
        domain_col = schema_config.get("domain_col", domain_col)
        domain_names = schema_config.get("domain_names", domain_names)
        char_count_col = schema_config.get("char_count_col", char_count_col)
        text_col = schema_config.get("text_col", text_col)
        if file_pattern is None:
            file_pattern = "*.parquet"

    manual_ratio_map = parse_manual_ratio(args.manual_ratio, domain_names)

    enc = load_tokenizer(args.tokenizer_pkl)
    token_method = "nanochat tokenizer" if enc else "char_count // 4 (estimate)"

    quality_methods = [m.strip() for m in args.quality_method.split(",") if m.strip()]
    for m in quality_methods:
        if m not in QUALITY_SCORE_MAP:
            print(f"ERROR: Unknown quality method '{m}'. Options: {', '.join(QUALITY_SCORE_MAP.keys())}")
            sys.exit(1)
    do_quality = len(quality_methods) > 0
    do_manual_ratio = manual_ratio_map is not None

    if not os.path.isdir(args.preprocessed_data_dir):
        print(f"ERROR: Source directory not found: {args.preprocessed_data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    quadmix_dir = output_dir / "quadmix_data"
    random_dir = output_dir / "random_data"
    manual_ratio_dir = output_dir / "manual_ratio_data" if do_manual_ratio else None
    quality_dirs = {}
    quadmix_dir.mkdir(parents=True, exist_ok=True)
    random_dir.mkdir(parents=True, exist_ok=True)
    if do_manual_ratio:
        manual_ratio_dir.mkdir(parents=True, exist_ok=True)
    if do_quality:
        for m in quality_methods:
            qd = output_dir / f"quality_data_{m}"
            qd.mkdir(parents=True, exist_ok=True)
            quality_dirs[m] = qd

    baselines = ["quadmix", "random"]
    if do_manual_ratio:
        baselines.append("manual_ratio")
    if do_quality:
        for m in quality_methods:
            baselines.append(f"quality_{m}")

    print("=" * 60)
    print("  Nanochat Comparison Dataset Preparation")
    print("=" * 60)
    print(f"  Token counting: {token_method}")
    print(f"  Baselines: {', '.join(baselines)}")
    print(f"  Source dir: {args.preprocessed_data_dir}")
    print(f"  File pattern: {file_pattern}")
    print(f"  Domain col: {domain_col}")
    print(f"  Domain names: {domain_names}")
    print(f"  Char count col: {char_count_col}")
    print(f"  Text col: {text_col}")
    if args.data_ratio is not None:
        print(f"  Data ratio: {args.data_ratio}")
        print(f"  Num scaling params: {args.num_scaling_params}")
    if do_manual_ratio:
        mr_label = ": ".join(f"{d}={r}" for d, r in sorted(manual_ratio_map.items()))
        print(f"  Manual Ratio: {mr_label}")
    if do_quality:
        for m in quality_methods:
            print(f"  Quality method: {m} ({QUALITY_SCORE_MAP[m]})")

    print(f"\n[1/N] Reading QuadMix selected dataset...")
    quadmix_table = pq.read_table(args.quadmix_sampled_data, columns=["text"])
    texts = quadmix_table["text"].to_pylist()
    del quadmix_table
    max_chars = args.max_chars
    max_char_repeat_ratio = args.max_char_repeat_ratio
    num_workers = args.num_workers or min(mp.cpu_count(), 256) or 1
    chunk_size = max(1, len(texts) // (num_workers * 4))
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    filter_tasks = [(c, max_chars, max_char_repeat_ratio) for c in chunks]
    valid_texts = []
    n_empty = 0
    n_too_long = 0
    n_repeat = 0
    pool = _get_io_pool(num_workers)
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
        print(f"  Filtered {n_empty:,} docs (empty), "
              f"{n_too_long:,} docs (>{max_chars:,} chars), "
              f"{n_repeat:,} docs (single char >{max_char_repeat_ratio*100:.0f}% repetition)")
    print(f"  Counting tokens for {len(valid_texts):,} docs...")
    if enc and args.tokenizer_pkl:
        token_counts = count_tokens_mp(valid_texts, args.tokenizer_pkl, num_workers=args.num_workers)
    else:
        token_counts = [estimate_tokens(t) for t in valid_texts]
    quadmix_docs = [
        {"text": t, "token_count": tc}
        for t, tc in zip(valid_texts, token_counts)
    ]
    quadmix_total_tokens = sum(token_counts)

    del texts, valid_texts, token_counts
    gc.collect()

    print(f"  QuadMix docs: {len(quadmix_docs):,}")
    print(f"  QuadMix total tokens ({token_method}): {quadmix_total_tokens:,}")

    if args.data_ratio is not None and args.num_scaling_params is not None:
        target_tokens = int(args.data_ratio * args.num_scaling_params)
        budget_cap = int(target_tokens * 1.1)
        print(f"\n  Token budget: target={target_tokens:,}, "
              f"quadmix_total={quadmix_total_tokens:,}, "
              f"budget_cap={budget_cap:,}")
    else:
        budget_cap = int(quadmix_total_tokens * 1.1)
        target_tokens = quadmix_total_tokens
        print(f"\n  Token budget: no data-ratio specified, using quadmix_total={quadmix_total_tokens:,} "
              f"x 1.1 = {budget_cap:,}")

    random.shuffle(quadmix_docs)
    if sum(d["token_count"] for d in quadmix_docs) > budget_cap:
        quadmix_docs, quadmix_actual_tokens = trim_docs_to_target(quadmix_docs, budget_cap)
        print(f"  QuadMix capped to budget: {len(quadmix_docs):,} docs, {quadmix_actual_tokens:,} tokens")
    else:
        quadmix_actual_tokens = sum(d["token_count"] for d in quadmix_docs)
        print(f"  QuadMix within budget: {len(quadmix_docs):,} docs, {quadmix_actual_tokens:,} tokens")

    print(f"\n[2/N] Scanning source shards metadata...")
    prep_files, prep_metadata = scan_shards(
        args.preprocessed_data_dir, file_pattern,
        domain_col=domain_col, domain_names=domain_names,
        char_count_col=char_count_col, text_col=text_col,
        num_workers=args.num_workers, max_shards=args.max_shards,
        max_chars=args.max_chars, max_char_repeat_ratio=args.max_char_repeat_ratio)

    print("  Building candidate index...")
    all_doc_ids = np.concatenate([m[0] for m in prep_metadata])
    all_char_counts = np.concatenate([m[1] for m in prep_metadata])
    all_domain_vals = np.concatenate([m[2] for m in prep_metadata])
    all_shard_ids = np.concatenate([
        np.full(len(prep_metadata[sid][0]), sid, dtype=np.int32)
        for sid in range(len(prep_metadata))
    ])
    all_est_tokens = all_char_counts // 4
    print(f"  Total candidate docs: {len(all_doc_ids):,}")

    domain_candidates = None
    if do_manual_ratio:
        name_to_id = {name: i for i, name in enumerate(domain_names)}
        domain_budgets_keys = set(name_to_id[name] for name in manual_ratio_map)
        domain_candidates = {}
        for dv in domain_budgets_keys:
            mask = all_domain_vals == dv
            domain_candidates[dv] = (
                all_shard_ids[mask].copy(),
                all_doc_ids[mask].copy(),
                all_est_tokens[mask].copy(),
            )

    print(f"\n[3/N] Random sampling (target: {budget_cap:,} tokens)...")
    perm = np.random.permutation(len(all_doc_ids))
    shuffled_est_tokens = all_est_tokens[perm]
    cumsum = np.cumsum(shuffled_est_tokens)
    if budget_cap <= 0 or len(cumsum) == 0:
        cutoff = 0
    else:
        cutoff = min(int(np.searchsorted(cumsum, budget_cap, side='left')) + 1, len(all_doc_ids))
    selected = list(zip(
        all_shard_ids[perm[:cutoff]].tolist(),
        all_doc_ids[perm[:cutoff]].tolist(),
    ))
    accumulated_tokens = int(cumsum[cutoff - 1]) if cutoff > 0 else 0
    print(f"  Selected {len(selected):,} docs (estimated ~{accumulated_tokens:,} tokens)")

    print(f"\n  Reading selected documents...")
    random_docs = read_docs_from_shards(prep_files, selected, num_workers=args.num_workers,
                                         desc=f"  Reading random docs ({args.num_workers or 'auto'} processes)",
                                         text_col=text_col,
                                         max_char_repeat_ratio=args.max_char_repeat_ratio)

    if enc and args.tokenizer_pkl:
        print(f"  Re-counting tokens for {len(random_docs):,} docs (exact)...")
        random_texts = [d["text"] for d in random_docs]
        exact_counts = count_tokens_mp(random_texts, args.tokenizer_pkl, num_workers=args.num_workers)
        for doc, tc in zip(random_docs, exact_counts):
            doc["token_count"] = tc
        del random_texts, exact_counts
        gc.collect()
        accumulated_tokens = sum(d["token_count"] for d in random_docs)
        print(f"  Exact tokens before trim: {accumulated_tokens:,} (target: {budget_cap:,})")
        if accumulated_tokens > budget_cap:
            n_before = len(random_docs)
            random_docs, accumulated_tokens = trim_docs_to_target(random_docs, budget_cap)
            print(f"  Trimmed {n_before - len(random_docs):,} docs to match target")

    random_actual_tokens = sum(d["token_count"] for d in random_docs)
    del all_doc_ids, all_char_counts, all_domain_vals, all_shard_ids, all_est_tokens, selected
    gc.collect()

    print(f"  Random docs: {len(random_docs):,}")
    print(f"  Random tokens ({token_method}): {random_actual_tokens:,}")

    manual_ratio_docs = None
    manual_ratio_tokens = 0
    manual_ratio_label = ""
    if do_manual_ratio:
        mr_label_parts = [f"{d}={r}" for d, r in sorted(manual_ratio_map.items())]
        manual_ratio_label = "Manual Ratio (" + ", ".join(mr_label_parts) + ")"
        print(f"\n[4/N] Manual Ratio sampling ({manual_ratio_label})...")
        manual_ratio_docs, manual_ratio_tokens = select_manual_ratio(
            prep_files, domain_candidates, manual_ratio_map, domain_names,
            budget_cap, tokenizer_pkl=args.tokenizer_pkl,
            num_workers=args.num_workers, enc=enc, text_col=text_col,
            max_char_repeat_ratio=args.max_char_repeat_ratio)

    quality_datasets = {}
    step_num = 5 if do_manual_ratio else 4
    if do_quality:
        for mi, method in enumerate(quality_methods):
            print(f"\n[{step_num}.{mi+1}/N] Quality-Only Top-K selection ({method})...")
            q_docs, q_tokens = select_quality_topk(
                prep_files, prep_metadata, method, budget_cap,
                tokenizer_pkl=args.tokenizer_pkl, num_workers=args.num_workers, enc=enc,
                text_col=text_col,
                max_char_repeat_ratio=args.max_char_repeat_ratio)
            quality_datasets[method] = (q_docs, q_tokens)
    else:
        if not do_manual_ratio:
            print(f"\n[{step_num}/N] Quality baseline skipped")

    qm_count = len(quadmix_docs)
    rd_count = len(random_docs)
    mr_count = len(manual_ratio_docs) if manual_ratio_docs else 0
    print(f"  Docs: {qm_count:,} quadmix + {rd_count:,} random + {mr_count:,} manual_ratio")

    step_num = (6 if do_manual_ratio else 5) if do_quality else (5 if do_manual_ratio else 4)
    print(f"\n[{step_num}/N] Splitting train/val (val_ratio={args.val_ratio})...")

    random.shuffle(quadmix_docs)
    random.shuffle(random_docs)
    if manual_ratio_docs:
        random.shuffle(manual_ratio_docs)
    for method in quality_datasets:
        random.shuffle(quality_datasets[method][0])

    def split_train_val(docs, val_ratio):
        if val_ratio <= 0 or not docs:
            return docs, []
        n_val = max(1, int(len(docs) * val_ratio))
        return docs[n_val:], docs[:n_val]

    quadmix_train, quadmix_val = split_train_val(quadmix_docs, args.val_ratio)
    random_train, random_val = split_train_val(random_docs, args.val_ratio)
    manual_ratio_train, manual_ratio_val = split_train_val(manual_ratio_docs or [], args.val_ratio)
    quality_trains = {}
    quality_vals = {}
    for m, (d, _) in quality_datasets.items():
        qt, qv = split_train_val(d, args.val_ratio)
        quality_trains[m] = qt
        quality_vals[m] = qv

    print(f"  QuadMix train: {len(quadmix_train):,}, val: {len(quadmix_val):,}")
    print(f"  Random  train: {len(random_train):,}, val: {len(random_val):,}")
    if manual_ratio_train:
        print(f"  Manual Ratio train: {len(manual_ratio_train):,}, val: {len(manual_ratio_val):,}")
    for m, qt in quality_trains.items():
        print(f"  Quality ({m}) train: {len(qt):,}, val: {len(quality_vals[m]):,}")

    final_step = step_num + 1
    print(f"\n[{final_step}/N] Writing sharded parquet files...")

    def write_dataset(docs, data_dir, name, val_docs=None):
        n_shards = max(1, (len(docs) + args.shard_size - 1) // args.shard_size)
        for i in range(n_shards):
            start = i * args.shard_size
            end = min(start + args.shard_size, len(docs))
            shard_docs = docs[start:end]
            out_path = data_dir / f"shard_{i:05d}.parquet"
            write_shard(shard_docs, str(out_path), args.num_npu)
        val_path = data_dir / f"shard_{n_shards:05d}.parquet"
        if val_docs and len(val_docs) > 0:
            write_shard(val_docs, str(val_path), args.num_npu)
        else:
            dummy_val = [{"text": "dummy"}]
            write_shard(dummy_val, str(val_path), args.num_npu)
        val_label = f"{len(val_docs)} val" if val_docs and len(val_docs) > 0 else "1 dummy val"
        print(f"  {name}: {n_shards} train shards + {val_label} -> {data_dir}")

    write_dataset(quadmix_train, quadmix_dir, "QuadMix", quadmix_val)
    write_dataset(random_train, random_dir, "Random", random_val)
    if do_manual_ratio:
        write_dataset(manual_ratio_train, manual_ratio_dir, manual_ratio_label, manual_ratio_val)
    for m, qt in quality_trains.items():
        write_dataset(qt, quality_dirs[m], f"Quality ({m})", quality_vals[m])

    stats = {
        "quadmix": {
            "train_docs": len(quadmix_train),
            "val_docs": len(quadmix_val),
            "tokens": sum(d["token_count"] for d in quadmix_train),
            "shards": max(1, (len(quadmix_train) + args.shard_size - 1) // args.shard_size),
        },
        "random": {
            "train_docs": len(random_train),
            "val_docs": len(random_val),
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
            "file_pattern": file_pattern,
            "domain_col": domain_col,
            "domain_names": domain_names,
            "char_count_col": char_count_col,
            "text_col": text_col,
        }
    }
    if args.data_ratio is not None:
        stats["config"]["data_ratio"] = args.data_ratio
        stats["config"]["num_scaling_params"] = args.num_scaling_params
        stats["config"]["target_tokens"] = target_tokens
        stats["config"]["budget_cap"] = budget_cap

    if do_manual_ratio:
        mr_label_parts = [f"{d}={r}" for d, r in sorted(manual_ratio_map.items())]
        stats["manual_ratio"] = {
            "train_docs": len(manual_ratio_train),
            "val_docs": len(manual_ratio_val),
            "tokens": sum(d["token_count"] for d in manual_ratio_train),
            "shards": max(1, (len(manual_ratio_train) + args.shard_size - 1) // args.shard_size),
            "label": manual_ratio_label,
            "ratio_map": manual_ratio_map,
        }

    if quality_trains:
        stats["config"]["quality_methods"] = quality_methods
        for m, qt in quality_trains.items():
            stats[f"quality_{m}"] = {
                "train_docs": len(qt),
                "val_docs": len(quality_vals[m]),
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
    if do_manual_ratio:
        print(f"    Manual Ratio: {manual_ratio_dir}")
    for m in quality_datasets:
        print(f"    Quality ({m}): {quality_dirs[m]}")
    print("=" * 60)

    _cleanup_pools()


if __name__ == "__main__":
    main()
