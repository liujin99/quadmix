"""Parallel dispatch: worker loops, tokenize workers, and parallel orchestration.

Standalone functions used by EssentialWebProxyRunner for multi-NPU parallel training.
"""

import os
import time
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from quadmix.utils.perf_timer import PerfTimer
from quadmix.pipeline.shared_memory import shared_to_ndarray


_tokenizer_cache: Dict[str, "Tokenizer"] = {}


def _get_tokenizer(tokenizer_path: str) -> "Tokenizer":
    """Get or create a cached tokenizer instance (per-process cache)."""
    if tokenizer_path not in _tokenizer_cache:
        from tokenizers import Tokenizer
        _tokenizer_cache[tokenizer_path] = Tokenizer.from_pretrained(tokenizer_path)
    return _tokenizer_cache[tokenizer_path]


def _io_read_shard(
        sid: int,
        shard_path: str,
        miss_rows: List[int],
) -> Tuple[int, np.ndarray, List[str], float]:
    """Stage 1 worker: read one shard's parquet, return (sid, rows, texts, io_time)."""
    import pandas as pd
    io_t0 = time.time()
    df_shard = pd.read_parquet(
        shard_path,
        columns=["row_in_shard", "text"],
        filters=[("row_in_shard", "in", miss_rows)],
    )
    df_shard = df_shard.sort_values("row_in_shard")
    texts = df_shard["text"].astype(str).tolist()
    parsed_rows = df_shard["row_in_shard"].to_numpy(dtype=np.int64)
    io_time = time.time() - io_t0
    return (sid, parsed_rows, texts, io_time)


def _process_shard_full(
        sid: int,
        shard_path: str,
        miss_rows: List[int],
        tokenizer_path: str,
        block_size: int,
        threads_per_worker: int = 4,
) -> Tuple[int, np.ndarray, np.ndarray, float, float, float]:
    """Process one shard: IO + tokenize in sequence.

    This enables pipelining: as soon as one shard's IO completes, its tokenize starts
    immediately without waiting for other shards.

    Returns (sid, parsed_rows, tokens_array, io_time, tok_time, total_time).
    """
    io_t0 = time.time()
    import pandas as pd
    df_shard = pd.read_parquet(
        shard_path,
        columns=["row_in_shard", "text"],
        filters=[("row_in_shard", "in", miss_rows)],
    )
    df_shard = df_shard.sort_values("row_in_shard")
    texts = df_shard["text"].astype(str).tolist()
    parsed_rows = df_shard["row_in_shard"].to_numpy(dtype=np.int64)
    io_time = time.time() - io_t0

    tok_t0 = time.time()
    chunk = [(sid, idx, text) for idx, text in enumerate(texts)]
    meta, tokens_array = _tokenize_chunk_to_array(chunk, tokenizer_path, block_size, threads_per_worker)
    tok_time = time.time() - tok_t0

    return (sid, parsed_rows, tokens_array, io_time, tok_time, io_time + tok_time)


def _tokenize_shard_parallel(
        shard_tasks: List[Tuple[int, str, List[int]]],
        tokenizer_path: str,
        block_size: int,
) -> List[Tuple[int, np.ndarray, np.ndarray, float, float, float]]:
    """Parallel tokenize using ProcessPoolExecutor to bypass GIL.

    Each process handles one shard end-to-end (IO + tokenize) with its own
    Python interpreter and GIL, enabling true CPU parallelism.

    Config:
      TOKENIZE_WORKERS env var controls process count (default: min(48, cpu_count))
      RAYON_NUM_THREADS=4 per process → total threads = workers × 4

    Returns list of (sid, rows, tokens_int32, io_time, tok_time, total_time).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import threading

    n_shards = len(shard_tasks)
    num_cpus = mp.cpu_count() or 8

    threads_per_worker = 4
    os.environ["RAYON_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)

    env_workers = int(os.environ.get("TOKENIZE_WORKERS", "0"))
    if env_workers >= 1:
        n_workers = env_workers
    else:
        n_workers = min(48, num_cpus)
        n_workers = max(4, n_workers)

    print(f"  [ParallelTokenize] {n_shards} shards, {n_workers} processes "
          f"× {threads_per_worker} Rust threads = {n_workers * threads_per_worker} total threads")

    results = []
    t0 = time.time()

    completed = [0]
    lock = threading.Lock()

    def on_done(fut):
        with lock:
            completed[0] += 1
            c = completed[0]
        if c % 10 == 0 or c == n_shards:
            elapsed = time.time() - t0
            speed = c / elapsed if elapsed > 0 else 0
            eta = (n_shards - c) / speed if speed > 0 else 0
            print(f"  [Tokenize Progress] {c}/{n_shards} shards "
                  f"({c*100//n_shards}%), "
                  f"{speed:.1f} shards/s, ETA {eta:.0f}s")

    with PerfTimer.section("parallel_tokenize", "parallel_tokenize"):
        from quadmix.pipeline.tokenize_worker import _process_shard_full as _worker_process_shard
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futs = []
            for sid, shard_path, miss_rows in shard_tasks:
                fut = executor.submit(
                    _worker_process_shard,
                    sid, shard_path, miss_rows,
                    tokenizer_path, block_size, threads_per_worker,
                )
                fut.add_done_callback(on_done)
                futs.append(fut)

            for fut in futs:
                try:
                    sid, parsed_rows, tokens_array, io_time, tok_time, total_time = fut.result()
                    results.append((sid, parsed_rows, tokens_array, io_time, tok_time, total_time))
                except Exception as e:
                    print(f"  [Tokenize Error] {e}")
                    import traceback
                    traceback.print_exc()

    total_time = time.time() - t0
    total_docs = sum(len(r[1]) for r in results)
    print(f"  [ParallelTokenize] {total_docs:,} docs in {total_time:.1f}s "
          f"({total_docs / total_time:.0f} docs/s)")

    return results


def _tokenize_chunk_with_meta(
        chunk: List[Tuple[int, int, str]],
        tokenizer_path: str,
        block_size: int,
) -> List[Tuple[int, int, List[int]]]:
    """Tokenize a chunk of (sid, idx, text) items. Returns (sid, idx, token_ids).

    Uses the Rust-based tokenizers library directly for max throughput.
    """
    tok = _get_tokenizer(tokenizer_path)
    texts = [item[2] for item in chunk]
    encodings = tok.encode_batch(texts)

    PAD_TOKEN = 50256

    results = []
    for (sid, idx, _), enc in zip(chunk, encodings):
        ids = list(enc.ids)
        if len(ids) > block_size:
            ids = ids[:block_size]
        if len(ids) < block_size:
            ids = ids + [PAD_TOKEN] * (block_size - len(ids))
        results.append((sid, idx, ids))
    return results


def _tokenize_chunk_to_array(
        chunk: List[Tuple[int, int, str]],
        tokenizer_path: str,
        block_size: int,
        threads_per_worker: int = 4,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Tokenize a chunk, returning compact numpy array instead of per-doc lists.

    Returns ((sid, idx) pairs, np.array[N x block_size, dtype=int32]).
    """
    os.environ["RAYON_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)

    tok = _get_tokenizer(tokenizer_path)
    texts = [item[2] for item in chunk]
    encodings = tok.encode_batch(texts)

    PAD_TOKEN = 50256
    N = len(chunk)
    tokens_array = np.full((N, block_size), PAD_TOKEN, dtype=np.int32)

    meta = []
    for (i, (sid, idx, _)), enc in zip(enumerate(chunk), encodings):
        ids = list(enc.ids)
        n = min(len(ids), block_size)
        tokens_array[i, :n] = ids[:n]
        meta.append((sid, idx))

    return meta, tokens_array


def _worker_dynamic_loop(
        worker_id: int,
        device_type: str,
        config_dict: dict,
        task_queue,
        result_queue,
):
    """Worker loop for dynamic mode: pull task → run → push result → repeat.

    Uses shared memory for metadata arrays when available, avoiding
    per-worker 15GB+ parquet reload. Falls back to disk loading gracefully.
    """
    from quadmix.pipeline.essential_proxy_runner import EssentialWebProxyRunner
    from quadmix.core.types import ParameterSet

    with open(f"/tmp/worker_{worker_id}_entry.log", "w") as ef:
        ef.write(f"[Worker {worker_id}] FUNCTION ENTERED at {time.time()}")
    try:
        from quadmix.data.metadata_manager import ShardMetadataManager

        shared_dl = config_dict.get("shared_domain_labels")
        shared_qs = config_dict.get("shared_quality_scores")
        shared_cc = config_dict.get("shared_doc_char_counts")
        shared_nq = config_dict.get("shared_normalized_quality")
        per_shard_info = config_dict.get("per_shard_info")

        use_shared = (shared_dl is not None and shared_qs is not None
                      and shared_cc is not None and per_shard_info is not None)

        if use_shared:
            t0 = time.time()
            domain_labels = shared_to_ndarray(shared_dl)
            quality_scores = shared_to_ndarray(shared_qs)
            doc_char_counts = shared_to_ndarray(shared_cc)

            shard_starts_arr = np.array(
                [s["start_idx"] for s in per_shard_info], dtype=np.int64
            )

            mgr = ShardMetadataManager.from_shared(
                domain_labels=domain_labels,
                quality_scores=quality_scores,
                doc_char_counts=doc_char_counts,
                per_shard_info=per_shard_info,
                shard_starts=shard_starts_arr,
                preprocessed_dir=config_dict.get("preprocessed_dir", ""),
            )
            print(f"[Worker {worker_id}] Mapped {len(domain_labels):,} docs from shared memory "
                  f"({time.time() - t0:.1f}s vs ~60s disk reload)")
        else:
            if config_dict.get("preprocessed_dir"):
                mgr = ShardMetadataManager(config_dict["preprocessed_dir"])
            else:
                mgr = None

        runner = EssentialWebProxyRunner(
            config=config_dict["config"],
            metadata_manager=mgr,
            val_data_path=config_dict["val_data_path"],
            output_dir=config_dict["output_dir"],
            device_type=device_type,
            npu_device_id=worker_id,
            model_variant=config_dict["model_variant"],
            global_batch_size=config_dict["global_batch_size"],
            micro_batch_size=config_dict["micro_batch_size"],
            max_step=config_dict["max_step"],
            warmup_fraction=config_dict.get("warmup_fraction", 0.04),
            learning_rate=config_dict["learning_rate"],
            weight_decay=config_dict["weight_decay"],
            grad_clip=config_dict["grad_clip"],
            tiny_steps=config_dict["tiny_steps"],
            doc_limit=config_dict["doc_limit"],
            test_block_size=config_dict["test_block_size"],
            rank_ref_size=config_dict["rank_ref_size"],
            token_cache_dir=config_dict["token_cache_dir"],
            checkpoint_interval=config_dict.get("checkpoint_interval", 1000),
        )

        completed = 0
        while True:
            task = task_queue.get()

            if task is None:
                print(f"[Worker {worker_id}] Shutdown, completed {completed} experiments")
                break

            exp_id, params, selected_idx, shm_info = task[:4]
            sampled_doc_count = task[4] if len(task) > 4 else None
            print(f"[Worker {worker_id}] Running exp {exp_id}")

            r = runner.run_experiment(params, experiment_id=exp_id, selected_idx=selected_idx,
                                      shm_info=shm_info,
                                      sampled_doc_count=sampled_doc_count,
                                      checkpoint_interval=config_dict.get("checkpoint_interval", 1000))
            result_queue.put(r)
            completed += 1

            import gc
            gc.collect()
            if device_type == "npu":
                torch.npu.empty_cache()

            if shm_info is not None:
                from multiprocessing.shared_memory import SharedMemory
                try:
                    shm = SharedMemory(name=shm_info[0])
                    shm.close()
                    shm.unlink()
                    print(f"[Worker {worker_id}] Released SharedMemory for exp {exp_id}")
                except Exception as e:
                    print(f"[Worker {worker_id}] SharedMemory cleanup failed for exp {exp_id}: {e}")
            else:
                exp_token_path = runner._get_exp_token_path(exp_id)
                if os.path.exists(exp_token_path):
                    os.remove(exp_token_path)
                    print(f"[Worker {worker_id}] Cleaned temp file for exp {exp_id}")

        result_queue.put(None)

    except Exception as top_err:
        import sys, traceback
        err_path = f"/tmp/worker_{worker_id}_error.log"
        try:
            with open(err_path, "w") as ef:
                ef.write(f"[Worker {worker_id}] CRASH: {top_err}\n")
                ef.write(traceback.format_exc())
            print(f"[Worker {worker_id}] ERROR -> {err_path}", flush=True)
        except:
            pass
        try:
            result_queue.put(None)
        except:
            pass
        sys.exit(1)
