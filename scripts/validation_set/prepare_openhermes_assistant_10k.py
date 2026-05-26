#!/usr/bin/env python3
"""
Prepare openhermes-10k validation set — assistant-only version.

Difference from the original prepare_openhermes_10k.py:
  1. Only extracts assistant response text (no system/user prompts)
  2. Loss mask = ALL non-padding tokens (since every token is assistant)
  3. Smaller file size, simpler processing

Usage (documentation only — end users download from Hugging Face):
  python scripts/validation_set/prepare_openhermes_assistant_10k.py

Output: openhermes_10k_assistant_tokenized.pt
  - token_ids:   LongTensor [num_docs, block_size]   padded
  - loss_mask:   BoolTensor  [num_docs, block_size]   True for all non-padding tokens
  - metadata:    dict  source info
"""

import os
import random
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


DATA_DIR = "/home/liujin99/data/openhermes-2.5-1m"
OUTPUT_DIR = "/home/liujin99/data"
SEED = 42
BLOCK_SIZE = 2048  # default, matches tinyllama_1M


def extract_assistant_text(conversations):
    """Extract only assistant response text from a conversation."""
    texts = []
    for msg in conversations:
        if msg.get("from") == "gpt":
            texts.append(msg.get("value", ""))
    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Create openhermes-10k assistant-only validation set"
    )
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help=f"Block size for tokenization (default: {BLOCK_SIZE})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--input-dir", type=str, default=DATA_DIR,
                        help=f"Input OpenHermes 2.5 data directory (default: {DATA_DIR})")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of assistant responses to sample (default: 10000)")
    args = parser.parse_args()

    block_size = args.block_size
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    # Tokenizer (same as RegMix / GPT-NeoX)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer: GPT-NeoX-20B, vocab={tokenizer.vocab_size}")

    # Load data
    print(f"Loading OpenHermes 2.5 from: {args.input_dir}")
    df = pd.read_parquet(os.path.join(args.input_dir, "0000.parquet"))
    print(f"  Total conversations: {len(df):,}")

    # Extract all assistant responses (each response = one sample)
    all_responses = []
    for _, row in df.iterrows():
        responses = extract_assistant_text(row["conversations"])
        all_responses.extend(responses)
    print(f"  Total assistant responses extracted: {len(all_responses):,}")

    # Sample 10K
    sampled = random.sample(all_responses, min(args.num_samples, len(all_responses)))
    print(f"  Sampled: {len(sampled):,} assistant responses")

    # Tokenize
    print(f"Tokenizing (block_size={block_size})...")
    all_ids = []
    all_masks = []

    for idx, text in enumerate(sampled):
        token_ids = tokenizer.encode(text)
        ids_tensor = torch.LongTensor(token_ids)

        # Truncate or pad to block_size
        if len(ids_tensor) > block_size:
            ids_tensor = ids_tensor[:block_size]
        else:
            pad_len = block_size - len(ids_tensor)
            ids_tensor = torch.cat([ids_tensor, torch.zeros(pad_len, dtype=torch.long)])

        # Loss mask: ALL non-padding tokens are assistant tokens
        loss_mask = ids_tensor != tokenizer.pad_token_id

        all_ids.append(ids_tensor)
        all_masks.append(loss_mask)

        if (idx + 1) % 2000 == 0:
            print(f"  {idx+1}/{len(sampled)}")

    token_ids = torch.stack(all_ids)       # [10000, 2048]
    loss_mask = torch.stack(all_masks)      # [10000, 2048]

    # Stats
    total_tokens = token_ids.numel()
    masked_tokens = loss_mask.sum().item()
    print(f"\n  Tokenized: {token_ids.shape}")
    print(f"  Assistant tokens: {masked_tokens:,} / {total_tokens:,} ({masked_tokens/total_tokens*100:.1f}%)")

    # Save
    output = {
        "token_ids": token_ids,
        "loss_mask": loss_mask,
        "metadata": {
            "num_docs": len(token_ids),
            "block_size": block_size,
            "tokenizer": "gpt-neox-20b",
            "tokenizer_vocab": tokenizer.vocab_size,
            "model_vocab": 50432,
            "loss_strategy": "assistant_only_all_tokens",
            "source": "OpenHermes-2.5-1M",
            "description": "Only assistant response text; all non-padding tokens have loss_mask=True",
        }
    }

    save_path = os.path.join(output_dir, "openhermes_10k_assistant_tokenized.pt")
    torch.save(output, save_path)
    file_size = os.path.getsize(save_path) / 1024**2
    print(f"\n  Saved: {save_path} ({file_size:.0f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
