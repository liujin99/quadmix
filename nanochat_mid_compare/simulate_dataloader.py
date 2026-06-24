"""
Simulate the nanochat dataloader for ALL DDP ranks in parallel to find tokenizer hangs.

Replicates the exact data loading + tokenization + best-fit packing pipeline
from nanochat/dataloader.py, matching mid_train.py config exactly:
  - tokenizer_threads=16, tokenizer_batch_size=256, buffer_size=2000
  - device_batch_size=8, seq_len=2048, grad_accum=4 (524288 / (8*2048*8))

Each rank runs in a separate process. A per-batch timeout detects hangs.

Usage:
    DATA_DIR=/path/to/quality_data_fineweb_edu \
    bash nanochat_mid_compare/simulate_dataloader.sh
"""

import os
import sys
import time
import pickle
import argparse
import signal
import multiprocessing as mp
from pathlib import Path

import pyarrow.parquet as pq


class TokenizeTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TokenizeTimeout("tokenize batch timed out")


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


def run_rank(rank, args, result_queue):
    """Simulate one DDP rank's dataloader. Runs in a child process."""
    log_lines = []

    def log(msg):
        line = f"[rank{rank}] {msg}"
        log_lines.append(line)

    try:
        enc, bos_token_id = load_tokenizer(args.tokenizer_dir)
        parquet_paths = list_train_parquets(args.data_dir)

        batches = document_batches_for_rank(
            parquet_paths, rank, args.num_npu, args.tokenizer_batch_size
        )

        row_capacity = args.seq_len + 1
        B = args.device_batch_size
        doc_buffer = []
        total_rows = 0
        total_docs_tokenized = 0
        total_docs_consumed = 0
        step_num = 0
        slow_batches = []
        hang_info = None

        def refill_buffer():
            nonlocal total_docs_tokenized
            doc_batch, pq_idx, rg_idx = next(batches)
            t0 = time.time()

            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            old_alarm = signal.alarm(args.timeout)
            try:
                token_lists = enc.encode_ordinary_batch(doc_batch, num_threads=args.tokenizer_threads)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler if old_handler else signal.SIG_DFL)

            dt = time.time() - t0
            for tokens, text in zip(token_lists, doc_batch):
                tokens.insert(0, bos_token_id)
                doc_buffer.append((tokens, len(text), pq_idx, rg_idx))
            total_docs_tokenized += len(doc_batch)
            return dt, pq_idx, rg_idx, len(doc_batch), max(len(t) for t in doc_batch)

        rows_this_step = 0
        step_start_time = time.time()

        try:
            while step_num < args.target_step:
                for row_idx in range(B):
                    pos = 0
                    while pos < row_capacity:
                        while len(doc_buffer) < args.buffer_size:
                            tok_dt, pq_idx, rg_idx, n_docs, max_doc_tokens = refill_buffer()
                            if tok_dt > 5.0:
                                log(f"[SLOW] {tok_dt:.2f}s pq={pq_idx} rg={rg_idx} "
                                    f"docs={n_docs} max_tok={max_doc_tokens}")
                                slow_batches.append({
                                    "step": step_num, "time": tok_dt,
                                    "pq_idx": pq_idx, "rg_idx": rg_idx,
                                    "n_docs": n_docs, "max_tokens": max_doc_tokens
                                })

                        remaining = row_capacity - pos

                        best_idx = -1
                        best_len = 0
                        for i, (tokens, _, _, _) in enumerate(doc_buffer):
                            doc_len = len(tokens)
                            if doc_len <= remaining and doc_len > best_len:
                                best_idx = i
                                best_len = doc_len

                        if best_idx >= 0:
                            doc_buffer.pop(best_idx)
                            pos += best_len
                            total_docs_consumed += 1
                        else:
                            shortest_idx = min(range(len(doc_buffer)),
                                               key=lambda i: len(doc_buffer[i][0]))
                            doc_buffer.pop(shortest_idx)
                            pos += remaining
                            total_docs_consumed += 1

                    total_rows += 1
                    rows_this_step += 1

                    if rows_this_step >= B * args.grad_accum:
                        step_num += 1
                        step_dt = time.time() - step_start_time
                        if step_num % 50 == 0 or step_num <= 3 or step_num >= args.target_step - 3:
                            log(f"Step {step_num:04d}/{args.target_step:04d} "
                                f"dt={step_dt:.1f}s rows={total_rows} "
                                f"tok={total_docs_tokenized} used={total_docs_consumed} "
                                f"buf={len(doc_buffer)}")
                        if step_num >= args.target_step:
                            break
                        rows_this_step = 0
                        step_start_time = time.time()

        except StopIteration:
            log(f"RAN OUT OF DATA at step {step_num}")

        result_queue.put({
            "rank": rank,
            "status": "ok",
            "steps": step_num,
            "rows": total_rows,
            "docs_tokenized": total_docs_tokenized,
            "docs_consumed": total_docs_consumed,
            "buffer_remaining": len(doc_buffer),
            "slow_batches": slow_batches,
            "log": log_lines,
        })

    except TokenizeTimeout as e:
        result_queue.put({
            "rank": rank,
            "status": "HANG",
            "steps": step_num,
            "error": str(e),
            "log": log_lines,
        })
    except Exception as e:
        result_queue.put({
            "rank": rank,
            "status": "ERROR",
            "error": str(e),
            "log": log_lines,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--target-step", type=int, default=350)
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout per single tokenize batch call (seconds)")
    parser.add_argument("--num-npu", type=int, default=8)
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    parser.add_argument("--tokenizer-threads", type=int, default=16)
    args = parser.parse_args()

    print("=" * 70)
    print("  Dataloader Hang Simulation (All Ranks)")
    print("=" * 70)
    print(f"  Data dir:          {args.data_dir}")
    print(f"  Tokenizer:         {args.tokenizer_dir}")
    print(f"  Target step:       {args.target_step}")
    print(f"  Timeout/batch:     {args.timeout}s")
    print(f"  Ranks:             {args.num_npu}")
    print(f"  B={args.device_batch_size}, T={args.seq_len}, grad_accum={args.grad_accum}")
    print(f"  buffer_size:       {args.buffer_size}")
    print(f"  tokenizer_batch:   {args.tokenizer_batch_size}")
    print(f"  tokenizer_threads: {args.tokenizer_threads}")
    print(f"  total_batch_size:  {args.device_batch_size * args.seq_len * args.num_npu * args.grad_accum}")
    print("=" * 70)

    parquet_paths = list_train_parquets(args.data_dir)
    print(f"  Train shards: {len(parquet_paths)}")

    pf0 = pq.ParquetFile(parquet_paths[0])
    print(f"  Row groups per shard: {pf0.num_row_groups}")
    rgs_per_rank = pf0.num_row_groups // args.num_npu
    print(f"  Row groups per rank (per shard): ~{rgs_per_rank}")
    print()

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    t_start = time.time()

    for rank in range(args.num_npu):
        p = ctx.Process(target=run_rank, args=(rank, args, result_queue))
        p.start()
        processes.append(p)
        print(f"  Started rank {rank} (pid={p.pid})")

    print(f"\n  Waiting for all ranks to complete (or hang)...\n")

    results = []
    global_timeout = args.timeout * args.target_step + 300
    deadline = time.time() + global_timeout

    while len(results) < args.num_npu and time.time() < deadline:
        try:
            r = result_queue.get(timeout=10)
            results.append(r)
            elapsed = time.time() - t_start
            status = r["status"]
            steps = r.get("steps", "?")
            tag = f" steps={steps}" if status == "ok" else f" error={r.get('error', '?')}"
            print(f"  [{elapsed:6.1f}s] Rank {r['rank']} finished: {status}{tag}")
        except Exception:
            alive = sum(1 for p in processes if p.is_alive())
            print(f"  [{time.time()-t_start:6.1f}s] Waiting... {alive}/{args.num_npu} ranks still running")

    for p in processes:
        if p.is_alive():
            print(f"  Killing hung process pid={p.pid}")
            p.kill()
            p.join(timeout=5)

    total_time = time.time() - t_start

    print()
    print("=" * 70)
    print(f"  RESULTS (total time: {total_time:.1f}s)")
    print("=" * 70)

    all_ok = True
    for r in sorted(results, key=lambda x: x["rank"]):
        rank = r["rank"]
        status = r["status"]
        if status == "ok":
            print(f"  Rank {rank}: OK | steps={r['steps']} rows={r['rows']} "
                  f"tok={r['docs_tokenized']} used={r['docs_consumed']} "
                  f"buf={r['buffer_remaining']} slow={len(r['slow_batches'])}")
            for sb in r["slow_batches"]:
                print(f"         [SLOW step={sb['step']}] {sb['time']:.2f}s "
                      f"pq={sb['pq_idx']} rg={sb['rg_idx']} "
                      f"docs={sb['n_docs']} max_tok={sb['max_tokens']}")
        else:
            all_ok = False
            print(f"  Rank {rank}: {status} | steps={r.get('steps', '?')} error={r.get('error', '?')}")

        for line in r.get("log", []):
            print(f"    {line}")

    print()
    if all_ok and len(results) == args.num_npu:
        print("  ALL RANKS COMPLETED SUCCESSFULLY - no tokenizer hang detected")
    else:
        hung = [r["rank"] for r in results if r["status"] != "ok"]
        missing = set(range(args.num_npu)) - {r["rank"] for r in results}
        if hung:
            print(f"  HANG/ERROR on ranks: {hung}")
        if missing:
            print(f"  MISSING ranks (killed): {sorted(missing)}")

    print("=" * 70)


if __name__ == "__main__":
    main()
