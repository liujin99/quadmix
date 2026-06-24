"""
Find documents that cause tokenizer to hang on a given rank.

Uses concurrent workers with per-document timeout.

Usage:
    python3 find_hanging_document.py \
        --data-dir /path/to/quality_data_fineweb_edu \
        --tokenizer-dir /home/ma-user/work/nanochat_model_dir/tokenizer \
        --rank 5 \
        --timeout 10 \
        --num-workers 16
"""

import os
import pickle
import argparse
import multiprocessing as mp

import pyarrow.parquet as pq

_SPAWN_CTX = mp.get_context("spawn")

_worker_tokenizer = None


def _init_worker(tokenizer_dir):
    global _worker_tokenizer
    pkl_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(pkl_path, "rb") as f:
        _worker_tokenizer = pickle.load(f)


def _tokenize_one(task):
    pq_idx, rg_idx, doc_idx, text, timeout = task
    try:
        tokens = _worker_tokenizer.encode_ordinary(text)
        return (pq_idx, rg_idx, doc_idx, len(text), len(tokens), text[:500], None)
    except Exception as e:
        return (pq_idx, rg_idx, doc_idx, len(text), 0, text[:500], str(e))


def list_train_parquets(data_dir):
    files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    ])
    if len(files) < 2:
        return [os.path.join(data_dir, f) for f in files]
    return [os.path.join(data_dir, f) for f in files[:-1]]


def collect_docs_for_rank(parquet_paths, rank, world_size):
    docs = []
    for pq_idx, filepath in enumerate(parquet_paths):
        pf = pq.ParquetFile(filepath)
        rg_idx = rank
        while rg_idx < pf.num_row_groups:
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            for doc_idx, text in enumerate(texts):
                docs.append((pq_idx, rg_idx, doc_idx, text))
            rg_idx += world_size
    return docs


def find_hanging_document(args):
    parquet_paths = list_train_parquets(args.data_dir)

    print(f"Collecting documents for rank {args.rank}...")
    docs = collect_docs_for_rank(parquet_paths, args.rank, args.num_npu)
    print(f"  Total documents: {len(docs):,}")
    print(f"Scanning with {args.num_workers} workers, {args.timeout}s timeout per doc")
    print()

    tasks = [(pq_idx, rg_idx, doc_idx, text, args.timeout)
             for pq_idx, rg_idx, doc_idx, text in docs]

    hanging_docs = []
    long_docs = []
    processed = 0

    with _SPAWN_CTX.Pool(args.num_workers, initializer=_init_worker,
                         initargs=(args.tokenizer_dir,)) as pool:
        results = []
        for task in tasks:
            r = pool.apply_async(_tokenize_one, (task,))
            results.append(r)

        for i, r in enumerate(results):
            pq_idx, rg_idx, doc_idx, text = docs[i]
            try:
                result = r.get(timeout=args.timeout)
                _, _, _, char_len, tok_len, preview, error = result
                processed += 1

                if error:
                    print(f"\n[ERROR] pq={pq_idx} rg={rg_idx} doc={doc_idx}: {error}")
                elif tok_len > 100000:
                    long_docs.append(result)
                    print(f"\n[WARNING] Very long: {tok_len} tokens "
                          f"(pq={pq_idx} rg={rg_idx} doc={doc_idx}, {char_len} chars)")

            except mp.TimeoutError:
                hanging_docs.append({
                    "pq_idx": pq_idx,
                    "rg_idx": rg_idx,
                    "doc_idx": doc_idx,
                    "length": len(text),
                    "preview": text[:500]
                })
                print(f"\n[HANG] pq={pq_idx} rg={rg_idx} doc={doc_idx} "
                      f"({len(text):,} chars)")
                print(f"  Preview: {text[:300]}...")
                print()

                if len(hanging_docs) >= args.max_hanging:
                    print(f"Found {len(hanging_docs)} hanging documents, stopping")
                    pool.terminate()
                    break

            if processed % 5000 == 0 and processed > 0:
                print(f"  Processed {processed:,} docs, "
                      f"found {len(hanging_docs)} hanging...", end="\r", flush=True)

    print()
    print("=" * 70)
    print(f"Summary:")
    print(f"  Total documents: {len(docs):,}")
    print(f"  Processed: {processed:,}")
    print(f"  Hanging: {len(hanging_docs)}")
    print(f"  Very long (>100K tokens): {len(long_docs)}")
    print()

    if hanging_docs:
        print("Hanging documents:")
        for i, doc in enumerate(hanging_docs):
            print(f"\n{i+1}. pq={doc['pq_idx']} rg={doc['rg_idx']} doc={doc['doc_idx']}")
            print(f"   Length: {doc['length']:,} chars")
            print(f"   Preview: {doc['preview'][:200]}...")

    return hanging_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--num-npu", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=10,
                        help="Seconds to wait before declaring hang")
    parser.add_argument("--max-hanging", type=int, default=10,
                        help="Stop after finding this many hanging docs")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of concurrent worker processes")
    args = parser.parse_args()

    find_hanging_document(args)


if __name__ == "__main__":
    main()
