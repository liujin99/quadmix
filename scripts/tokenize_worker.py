"""Tokenize worker functions - NO torch import to avoid CANN initialization overhead."""
import time
import numpy as np
from typing import List, Tuple


def _get_tokenizer(tokenizer_path: str):
    """Lazy load tokenizer."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(tokenizer_path)


def _tokenize_chunk_to_array(
        chunk: List[Tuple[int, int, str]],
        tokenizer_path: str,
        block_size: int,
        threads_per_worker: int = 4,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Tokenize a chunk, returning compact numpy array."""
    import os as _os
    _os.environ["RAYON_NUM_THREADS"] = str(threads_per_worker)
    _os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)

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


def _process_shard_full(
        sid: int,
        shard_path: str,
        miss_rows: List[int],
        tokenizer_path: str,
        block_size: int,
        threads_per_worker: int = 4,
) -> Tuple[int, np.ndarray, np.ndarray, float, float, float]:
    """Process one shard: IO + tokenize in sequence."""
    import os as _os
    import sys
    
    print(f"[Worker {sid}] tokenizer_path={tokenizer_path}", flush=True)
    print(f"[Worker {sid}] exists={_os.path.exists(tokenizer_path)}", flush=True)
    print(f"[Worker {sid}] cwd={_os.getcwd()}", flush=True)
    print(f"[Worker {sid}] python={sys.executable}", flush=True)
    
    if not _os.path.exists(tokenizer_path):
        parent_dir = _os.path.dirname(tokenizer_path)
        print(f"[Worker {sid}] parent_dir={parent_dir}", flush=True)
        print(f"[Worker {sid}] parent_dir exists={_os.path.exists(parent_dir)}", flush=True)
        if _os.path.exists(parent_dir):
            try:
                contents = _os.listdir(parent_dir)[:10]
                print(f"[Worker {sid}] parent_dir contents={contents}", flush=True)
            except Exception as e:
                print(f"[Worker {sid}] cannot list parent_dir: {e}", flush=True)
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
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
