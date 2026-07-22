"""Chunked loss utilities for memory-efficient training and adaptive val_bs."""

from typing import Tuple

import torch
import torch.nn.functional as F


def chunked_loss_from_hidden(model, hidden, targets, chunk_size=2048):
    """Compute CE loss chunk-by-chunk from hidden states, never materializing full logits.

    hidden: (B, T, C) from model.forward(return_hidden=True)
    Only chunk_logits (B, chunk, V) exists at any time — full (B, T, V) is never created.
    """
    B, T, _ = hidden.shape
    V = model.config.vocab_size
    total_loss = torch.tensor(0.0, device=hidden.device, dtype=torch.float32)
    n_tokens = 0
    for i in range(0, T, chunk_size):
        chunk_logits = model.lm_head(hidden[:, i:i + chunk_size])
        chunk_tgt = targets[:, i:i + chunk_size]
        n = chunk_tgt.numel()
        total_loss = total_loss + F.cross_entropy(
            chunk_logits.reshape(-1, V), chunk_tgt.reshape(-1)
        ) * n
        n_tokens += n
        del chunk_logits
    return total_loss / n_tokens


def chunked_loss_per_token_from_hidden(model, hidden, targets, chunk_size=2048):
    """Same as chunked_loss_from_hidden but returns per-token losses (reduction='none')."""
    B, T, _ = hidden.shape
    V = model.config.vocab_size
    per_token = torch.empty(B, T, device=hidden.device, dtype=torch.float32)
    for i in range(0, T, chunk_size):
        chunk_logits = model.lm_head(hidden[:, i:i + chunk_size])
        chunk_tgt = targets[:, i:i + chunk_size]
        per_token[:, i:i + chunk_size] = F.cross_entropy(
            chunk_logits.reshape(-1, V), chunk_tgt.reshape(-1), reduction="none"
        ).view(chunk_tgt.shape)
        del chunk_logits
    return per_token


def compute_val_batch_size(
    device: torch.device,
    vocab_size: int,
    seq_len: int,
    max_val_bs: int = 96,
) -> Tuple[int, int]:
    """Compute safe (val_bs, chunk_size) based on available device memory.

    Strategy: try large chunk_size first; if val_bs < 8, reduce chunk_size
    to lower per-sample peak memory, then recompute val_bs.

    Returns:
        (val_bs, chunk_size) — both guaranteed >= 1.
    """
    if device.type not in ("npu", "cuda"):
        return min(max_val_bs, 8), min(seq_len, 2048)

    api = torch.npu if device.type == "npu" else torch.cuda
    total_mem = api.get_device_properties(device).total_memory
    allocated = api.memory_allocated(device)
    safe_available = total_mem - allocated - 8 * 1024 ** 3

    if safe_available <= 0:
        return 4, 256

    for cs in (2048, 1024, 512, 256):
        if cs > seq_len:
            continue
        per_sample_peak = 6 * cs * vocab_size
        bs = max(4, min(max_val_bs, int(safe_available / per_sample_peak)))
        if bs >= 8:
            return bs, cs

    return 4, 256
