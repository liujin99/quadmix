"""Shared tokenizer helpers for analysis scripts.

Extracted from analyze_pipeline_output.py and analyze_arm_data.py to
eliminate duplication of _init_tok_worker, _tokenize_lens, etc.
"""

import os
import pickle
import numpy as np

_ENC = None
_BOS_ID = None
_TOK_THREADS = 1


def init_tok_worker(tokenizer_path, tok_threads):
    global _ENC, _BOS_ID, _TOK_THREADS
    _TOK_THREADS = tok_threads
    if not tokenizer_path:
        return
    pkl = tokenizer_path if os.path.isfile(tokenizer_path) \
        else os.path.join(tokenizer_path, "tokenizer.pkl")
    with open(pkl, "rb") as f:
        _ENC = pickle.load(f)
    _BOS_ID = _ENC.encode_single_token("<|bos|>")


def tokenize_lens(texts):
    """Return numpy int64 array of token lengths (incl. +1 BOS the dataloader prepends)."""
    if _ENC is None:
        return None
    n = len(texts)
    out = np.empty(n, dtype=np.int64)
    bs = 256
    for s in range(0, n, bs):
        chunk = [t or "" for t in texts[s:s + bs]]
        try:
            encs = _ENC.encode_ordinary_batch(chunk, num_threads=_TOK_THREADS)
        except (AttributeError, TypeError):
            encs = [_ENC.encode_ordinary(t) for t in chunk]
        for j, ids in enumerate(encs):
            out[s + j] = len(ids) + 1
    return out


def tokenize_batch(task):
    """Worker: tokenize a batch of texts, return (batch_idx, token_lengths)."""
    idx, texts = task
    return idx, tokenize_lens(texts)


def resolve_tokenizer_path(args):
    """Resolve tokenizer path: --tokenizer arg, then $NANOCHAT_MODEL_DIR/tokenizer."""
    if args.tokenizer:
        pkl = args.tokenizer if os.path.isfile(args.tokenizer) \
            else os.path.join(args.tokenizer, "tokenizer.pkl")
        if os.path.isfile(pkl):
            return args.tokenizer
        print(f"  [warn] --tokenizer points to {args.tokenizer} but no tokenizer.pkl found")
        return None
    model_dir = os.environ.get(
        "NANOCHAT_MODEL_DIR", "/home/ma-user/work/nanochat_model_dir")
    default_tok = os.path.join(model_dir, "tokenizer")
    default_pkl = os.path.join(default_tok, "tokenizer.pkl")
    if os.path.isfile(default_pkl):
        print(f"  [info] tokenizer auto-detected: {default_pkl}")
        return default_tok
    return None
