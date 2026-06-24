"""
Decode captured batches to see actual text content.

Usage:
    python decode_batch.py \
        --batch-file=/path/to/batches_rank0.pt \
        --tokenizer-pkl=/path/to/tokenizer.pkl \
        [--num-samples=5]
"""

import argparse
import torch
import pickle


def decode_batch(batch_file, tokenizer_pkl, num_samples=5, bos_token_id=32759):
    print(f"Loading batch: {batch_file}")
    data = torch.load(batch_file, map_location="cpu")
    
    print(f"Loading tokenizer: {tokenizer_pkl}")
    with open(tokenizer_pkl, "rb") as f:
        tokenizer = pickle.load(f)
    
    print(f"\nDecoding first {num_samples} rows from first micro_step:\n")
    print("=" * 80)
    
    batch = data[0]
    x = batch["x"].numpy()  # (B, T)
    
    for row_idx in range(min(num_samples, x.shape[0])):
        row = x[row_idx]
        
        bos_positions = [i for i, t in enumerate(row) if t == bos_token_id]
        bos_positions.append(len(row))
        
        print(f"\n[Row {row_idx}] - {len(bos_positions)-1} documents:")
        print("-" * 80)
        
        for doc_idx in range(len(bos_positions) - 1):
            start = bos_positions[doc_idx]
            end = bos_positions[doc_idx + 1]
            doc_tokens = row[start:end].tolist()
            
            try:
                text = tokenizer.decode(doc_tokens)
                preview = text[:500] if len(text) > 500 else text
                print(f"\n  Doc {doc_idx} (tokens {start}-{end}, len={end-start}):")
                print(f"  {preview}")
                if len(text) > 500:
                    print(f"  ... [truncated, total {len(text)} chars]")
            except Exception as e:
                print(f"\n  Doc {doc_idx}: DECODE ERROR - {e}")
                print(f"  Tokens: {doc_tokens[:20]}...")


def main():
    parser = argparse.ArgumentParser(description="Decode batch to text")
    parser.add_argument("--batch-file", type=str, required=True)
    parser.add_argument("--tokenizer-pkl", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--bos-token-id", type=int, default=32759)
    args = parser.parse_args()
    
    decode_batch(args.batch_file, args.tokenizer_pkl, args.num_samples, args.bos_token_id)


if __name__ == "__main__":
    main()
