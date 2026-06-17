#!/usr/bin/env python3
"""
Show train samples for key low-R² tasks only.
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
import random

random.seed(42)


def main():
    tasks = [
        ("commonsense_qa", None, "commonsense_qa"),
        ("ai2_arc", "ARC-Easy", "arc_easy"),
        ("super_glue", "copa", "copa"),
    ]
    
    for hf_name, hf_config, label in tasks:
        print(f"\n{'='*80}")
        print(f"  {label}  (TRAIN)")
        print(f"{'='*80}")
        
        try:
            if hf_config:
                ds = load_dataset(hf_name, hf_config, split="train", streaming=True)
            else:
                ds = load_dataset(hf_name, split="train", streaming=True)
            
            samples = []
            for i, sample in enumerate(ds):
                if i >= 3:
                    break
                samples.append(sample)
            
            print(f"  Loaded {len(samples)} train samples")
            
            for i, sample in enumerate(samples):
                print(f"\n  [TRAIN #{i+1}]")
                print(f"  {sample}")
        
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
