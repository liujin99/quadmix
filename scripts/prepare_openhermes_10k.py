#!/usr/bin/env python3
"""
openhermes-10k — Pre-tokenized validation set with assistant-only loss masking.

Following RegMix:
  1. Uses GPT-2 BPE tokenizer (same as GPT-NeoX) — matches paper
  2. Pre-tokenizes all 10K samples once
  3. Creates assistant-only loss mask (only compute loss on assistant tokens)
  4. Packs into blocks of block_size+1 for autoregressive evaluation

Output: openhermes_10k_tokenized.pt
  - token_ids:  LongTensor [num_docs, block_size]  padded
  - loss_mask:  BoolTensor  [num_docs, block_size]  True=assistant tokens
  - metadata:   dict  source/category info
"""

import os
import random
import json
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


DATA_DIR = "/home/liujin99/data/openhermes-2.5-1m"
OUTPUT_DIR = "/home/liujin99/data"
SEED = 42
BLOCK_SIZE = 2048  # default, matches tinyllama_1M; overridable via --block-size


def find_assistant_spans(text, assistant_prefix="<|im_start|>assistant"):
    """
    Find character ranges of assistant responses in the formatted text.
    Returns list of (start, end) char positions.
    """
    spans = []
    search_from = 0
    while True:
        pos = text.find(assistant_prefix, search_from)
        if pos == -1:
            break
        # Start of the assistant response (including marker? or after?)
        # For loss, we want to predict the assistant's RESPONSE, 
        # including the assistant marker itself
        # Actually: after <|im_start|>assistant\n, the response begins
        
        # Find the start of actual content (after \n)
        content_start = pos + len(assistant_prefix)
        if content_start < len(text) and text[content_start] == '\n':
            content_start += 1
        
        # Find end: next <|im_start|> or end of text
        next_marker = text.find("<|im_start|>", content_start)
        if next_marker == -1:
            end = len(text)
        else:
            end = next_marker
        
        spans.append((pos, end))  # include the marker itself
        search_from = end
    
    return spans


def format_and_tokenize(row, tokenizer, block_size):
    """Format conversation, tokenize, create loss mask."""
    conversations = row["conversations"]
    system_prompt = row.get("system_prompt")
    
    # Format
    parts = []
    if system_prompt and str(system_prompt).strip().lower() not in ("nan", "", "none"):
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
    for msg in conversations:
        role = msg["from"]
        value = msg["value"]
        if role == "human":
            parts.append(f"<|im_start|>user\n{value}<|im_end|>")
        elif role == "gpt":
            parts.append(f"<|im_start|>assistant\n{value}<|im_end|>")
    text = "\n".join(parts)
    
    # Find assistant character spans
    asst_spans = find_assistant_spans(text)
    
    # Tokenize
    token_ids = tokenizer.encode(text)
    ids_tensor = torch.LongTensor(token_ids)
    
    # Build loss mask: True for tokens that are part of assistant responses
    # (including the assistant marker itself, since the model should learn
    #  to predict the entire assistant turn)
    loss_mask = torch.zeros(len(token_ids), dtype=torch.bool)
    for start, end in asst_spans:
        # Find which token indices fall within this span
        char_to_token = None  # We'll compute this by repated encoding
        # Actually, find token indices by scanning
        # Simple approach: encode prefix up to start, then up to end
        prefix_to_start = tokenizer.encode(text[:start])
        prefix_to_end = tokenizer.encode(text[:end])
        token_start = len(prefix_to_start)
        token_end = len(prefix_to_end)
        loss_mask[token_start:token_end] = True
    
    # Pad/truncate to block_size
    if len(ids_tensor) > block_size:
        ids_tensor = ids_tensor[:block_size]
        loss_mask = loss_mask[:block_size]
    else:
        pad_len = block_size - len(ids_tensor)
        ids_tensor = torch.cat([ids_tensor, torch.zeros(pad_len, dtype=torch.long)])
        loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=torch.bool)])
    
    return ids_tensor, loss_mask


def main():
    parser = argparse.ArgumentParser(description="Create openhermes-10k validation set")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help=f"Block size for tokenization (default: {BLOCK_SIZE})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    block_size = args.block_size
    output_dir = args.output_dir
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Use pad_token=eos_token as standard practice
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer: GPT-NeoX-20B, vocab={tokenizer.vocab_size} (model uses {tokenizer.vocab_size} → padded to 50432)")
    
    # Load data
    print("Loading OpenHermes 2.5...")
    df = pd.read_parquet(os.path.join(DATA_DIR, "0000.parquet"))
    print(f"  Total: {len(df):,}")
    
    # Sample 10K
    sample = df.sample(n=10000, random_state=SEED)
    print(f"  Sampled: {len(sample):,}")
    
    # Tokenize with assistant mask
    print("Tokenizing with assistant loss masking...")
    all_ids = []
    all_masks = []
    categories = []
    sources = []
    
    for idx, (_, row) in enumerate(sample.iterrows()):
        ids, mask = format_and_tokenize(row, tokenizer, block_size)
        all_ids.append(ids)
        all_masks.append(mask)
        categories.append(str(row.get("category", "")))
        sources.append(str(row.get("source", "")))
        
        if (idx + 1) % 2000 == 0:
            print(f"  {idx+1}/{len(sample)}")
    
    token_ids = torch.stack(all_ids)       # [10000, 2048]
    loss_mask = torch.stack(all_masks)      # [10000, 2048]
    
    # Stats
    total_tokens = token_ids.numel()
    masked_tokens = loss_mask.sum().item()
    print(f"\n  Tokenized: {token_ids.shape}")
    print(f"  Assistant tokens: {masked_tokens:,} / {total_tokens:,} ({masked_tokens/total_tokens*100:.1f}%)")
    
    # Verify with one sample
    example_text = tokenizer.decode(token_ids[0].tolist())
    masked_count = loss_mask[0].sum().item()
    print(f"\n  Sample decoded (first 300 chars):")
    print(f"  {example_text[:300]}")
    print(f"  Assistant tokens in sample: {masked_count}/{BLOCK_SIZE}")
    
    # Save
    output = {
        "token_ids": token_ids,
        "loss_mask": loss_mask,
        "metadata": {
            "categories": categories,
            "sources": sources,
            "num_docs": len(token_ids),
            "block_size": block_size,
            "tokenizer": "gpt-neox-20b",
            "tokenizer_vocab": tokenizer.vocab_size,
            "model_vocab": 50432,
            "loss_strategy": "assistant_only",
        }
    }

    save_path = os.path.join(OUTPUT_DIR, "openhermes_10k_tokenized.pt")
    torch.save(output, save_path)
    file_size = os.path.getsize(save_path) / 1024**2
    print(f"\n  Saved: {save_path} ({file_size:.0f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
