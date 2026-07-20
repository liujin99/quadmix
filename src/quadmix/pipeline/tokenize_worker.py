"""Tokenize worker functions - NO torch import to avoid CANN initialization overhead."""
import time
import numpy as np
from typing import List, Tuple


def _get_tokenizer(tokenizer_path: str):
    """Lazy load tokenizer."""
    from tokenizers import Tokenizer
    import os
    if os.path.exists(tokenizer_path):
        return Tokenizer.from_file(tokenizer_path)
    else:
        return Tokenizer.from_pretrained(tokenizer_path)


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
        text_col: str = "text",
        row_in_shard_col: str = "row_in_shard",
        has_row_in_shard: bool = True,
) -> Tuple[int, np.ndarray, np.ndarray, float, float, float]:
    """Process one shard: IO + tokenize in sequence."""
    io_t0 = time.time()
    import pandas as pd
    if has_row_in_shard:
        df_shard = pd.read_parquet(
            shard_path,
            columns=[row_in_shard_col, text_col],
            filters=[(row_in_shard_col, "in", miss_rows)],
        )
        df_shard = df_shard.sort_values(row_in_shard_col)
        texts = df_shard[text_col].astype(str).tolist()
        parsed_rows = df_shard[row_in_shard_col].to_numpy(dtype=np.int64)
    else:
        df_shard = pd.read_parquet(shard_path, columns=[text_col])
        texts = df_shard[text_col].astype(str).tolist()
        parsed_rows = np.arange(len(texts), dtype=np.int64)
    io_time = time.time() - io_t0

    tok_t0 = time.time()
    chunk = [(sid, idx, text) for idx, text in enumerate(texts)]
    meta, tokens_array = _tokenize_chunk_to_array(chunk, tokenizer_path, block_size, threads_per_worker)
    tok_time = time.time() - tok_t0

    return (sid, parsed_rows, tokens_array, io_time, tok_time, io_time + tok_time)
