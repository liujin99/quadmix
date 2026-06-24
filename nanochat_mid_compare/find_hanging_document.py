"""
Find the specific document that causes tokenizer to hang on a given rank.

Processes documents one by one with timeout, reports which document hangs.

Usage:
    python3 find_hanging_document.py \
        --data-dir /path/to/quality_data_fineweb_edu \
        --tokenizer-dir /home/ma-user/work/nanochat_model_dir/tokenizer \
        --rank 5 \
        --timeout 10
"""

import os
import pickle
import argparse
import signal
from queue import Queue
from threading import Thread

import pyarrow.parquet as pq


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Tokenizer timeout")


def load_tokenizer(tokenizer_dir):
    pkl_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(pkl_path, "rb") as f:
        enc = pickle.load(f)
    return enc


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


def tokenize_with_timeout(enc, texts, timeout):
    """Tokenize in a thread with timeout."""
    result_queue = Queue()

    def worker():
        try:
            tokens = enc.encode_ordinary_batch(texts, num_threads=1)
            result_queue.put(("ok", tokens))
        except Exception as e:
            result_queue.put(("error", e))

    thread = Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"Tokenizer hung for {timeout}s")

    if result_queue.empty():
        raise TimeoutError("No result from tokenizer")

    status, result = result_queue.get()
    if status == "error":
        raise result
    return result


def find_hanging_document(args):
    enc = load_tokenizer(args.tokenizer_dir)
    parquet_paths = list_train_parquets(args.data_dir)

    print(f"Scanning rank {args.rank} documents with {args.timeout}s timeout per batch")
    print(f"Looking for documents that cause tokenizer to hang")
    print()

    batches = document_batches_for_rank(
        parquet_paths, args.rank, args.num_npu, args.tokenizer_batch_size
    )

    total_docs = 0
    hanging_docs = []

    for batch_idx, (doc_batch, pq_idx, rg_idx) in enumerate(batches):
        for doc_idx, text in enumerate(doc_batch):
            total_docs += 1

            if total_docs % 1000 == 0:
                print(f"  Processed {total_docs} docs, found {len(hanging_docs)} hanging...",
                      end="\r", flush=True)

            try:
                tokens = tokenize_with_timeout(enc, [text], args.timeout)
                token_count = len(tokens[0])

                if token_count > 100000:
                    print(f"\n[WARNING] Very long doc: {token_count} tokens "
                          f"(pq={pq_idx} rg={rg_idx} doc={doc_idx})")

            except TimeoutError as e:
                print(f"\n[HANG] Document caused tokenizer to hang!")
                print(f"  Location: pq={pq_idx} rg={rg_idx} doc={doc_idx}")
                print(f"  Length: {len(text)} chars")
                print(f"  Preview: {text[:500]}...")
                print()

                hanging_docs.append({
                    "pq_idx": pq_idx,
                    "rg_idx": rg_idx,
                    "doc_idx": doc_idx,
                    "length": len(text),
                    "preview": text[:500]
                })

                if len(hanging_docs) >= args.max_hanging:
                    print(f"Found {len(hanging_docs)} hanging documents, stopping")
                    break

            except Exception as e:
                print(f"\n[ERROR] Tokenizer error: {e}")
                print(f"  Location: pq={pq_idx} rg={rg_idx} doc={doc_idx}")

        if len(hanging_docs) >= args.max_hanging:
            break

    print()
    print("=" * 70)
    print(f"Summary:")
    print(f"  Total documents processed: {total_docs}")
    print(f"  Hanging documents found: {len(hanging_docs)}")
    print()

    if hanging_docs:
        print("Hanging documents:")
        for i, doc in enumerate(hanging_docs):
            print(f"\n{i+1}. pq={doc['pq_idx']} rg={doc['rg_idx']} doc={doc['doc_idx']}")
            print(f"   Length: {doc['length']} chars")
            print(f"   Preview: {doc['preview'][:200]}...")

    return hanging_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--num-npu", type=int, default=8)
    parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=10,
                        help="Seconds to wait before declaring hang")
    parser.add_argument("--max-hanging", type=int, default=10,
                        help="Stop after finding this many hanging docs")
    args = parser.parse_args()

    find_hanging_document(args)


if __name__ == "__main__":
    main()
