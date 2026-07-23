"""Tokenizer loading utilities.

NO torch import — used in spawn subprocesses where CANN/torch initialization
is expensive.  Future maintainers: do NOT add any torch-dependent import here.
"""
import os
from typing import Dict

_cache: Dict[str, object] = {}


def get_tokenizer(tokenizer_path: str, use_cache: bool = True):
    """Load a tokenizer from a local file path or HuggingFace identifier.

    Args:
        tokenizer_path: Path to a local tokenizer.json file, or a HuggingFace
            model identifier (e.g. ``"gpt2"``).
        use_cache: If True, return a per-process cached instance.  In spawn
            subprocesses each process handles one shard, so the cache holds
            exactly one tokenizer — safe and efficient.
    """
    if use_cache and tokenizer_path in _cache:
        return _cache[tokenizer_path]
    from tokenizers import Tokenizer
    if os.path.exists(tokenizer_path):
        tok = Tokenizer.from_file(tokenizer_path)
    else:
        tok = Tokenizer.from_pretrained(tokenizer_path)
    if use_cache:
        _cache[tokenizer_path] = tok
    return tok
