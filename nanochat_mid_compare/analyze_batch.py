"""
Analyze captured batch data to identify characteristics of crash-inducing batches.

Usage:
    python analyze_batch.py \
        --crash-batch=/path/to/step329_dataloader_capture/batches_rank0.pt \
        --normal-batch=/path/to/step328_dataloader_capture/batches_rank0.pt \
        [--output-dir=/path/to/analysis_output]
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
import json


def load_batch(batch_path, device="cpu"):
    """Load batch data from .pt file"""
    data = torch.load(batch_path, map_location=device)
    return data


def analyze_batch_stats(batch_data, name="batch"):
    """Compute detailed statistics for a batch"""
    stats = {
        "name": name,
        "num_batches": len(batch_data),
    }
    
    all_x_lengths = []
    all_y_lengths = []
    all_x_values = []
    all_y_values = []
    
    for i, batch in enumerate(batch_data):
        x = batch["x"]
        y = batch["y"]
        
        all_x_lengths.append(x.shape[1])
        all_y_lengths.append(y.shape[1])
        
        all_x_values.append(x.flatten().numpy())
        all_y_values.append(y.flatten().numpy())
    
    all_x_values = np.concatenate(all_x_values)
    all_y_values = np.concatenate(all_y_values)
    
    stats["x"] = {
        "shape": batch_data[0]["x"].shape,
        "total_tokens": len(all_x_values),
        "lengths": {
            "min": int(min(all_x_lengths)),
            "max": int(max(all_x_lengths)),
            "mean": float(np.mean(all_x_lengths)),
            "std": float(np.std(all_x_lengths)),
        },
        "token_stats": {
            "min": int(all_x_values.min()),
            "max": int(all_x_values.max()),
            "mean": float(all_x_values.mean()),
            "std": float(all_x_values.std()),
            "unique_tokens": len(np.unique(all_x_values)),
        },
        "special_tokens": {
            "zeros": int((all_x_values == 0).sum()),
            "negatives": int((all_x_values < 0).sum()),
            "high_values": int((all_x_values > 50000).sum()),
        }
    }
    
    stats["y"] = {
        "shape": batch_data[0]["y"].shape,
        "total_tokens": len(all_y_values),
        "lengths": {
            "min": int(min(all_y_lengths)),
            "max": int(max(all_y_lengths)),
            "mean": float(np.mean(all_y_lengths)),
            "std": float(np.std(all_y_lengths)),
        },
        "token_stats": {
            "min": int(all_y_values.min()),
            "max": int(all_y_values.max()),
            "mean": float(all_y_values.mean()),
            "std": float(all_y_values.std()),
            "unique_tokens": len(np.unique(all_y_values)),
        },
        "special_tokens": {
            "zeros": int((all_y_values == 0).sum()),
            "negatives": int((all_y_values < 0).sum()),
            "high_values": int((all_y_values > 50000).sum()),
        }
    }
    
    return stats


def compare_batches(crash_stats, normal_stats):
    """Compare crash batch vs normal batch"""
    comparison = {
        "x_length_diff": {
            "crash_mean": crash_stats["x"]["lengths"]["mean"],
            "normal_mean": normal_stats["x"]["lengths"]["mean"],
            "diff": crash_stats["x"]["lengths"]["mean"] - normal_stats["x"]["lengths"]["mean"],
        },
        "x_token_diff": {
            "crash_unique": crash_stats["x"]["token_stats"]["unique_tokens"],
            "normal_unique": normal_stats["x"]["token_stats"]["unique_tokens"],
            "diff": crash_stats["x"]["token_stats"]["unique_tokens"] - normal_stats["x"]["token_stats"]["unique_tokens"],
        },
        "special_token_diff": {
            "crash_high_values": crash_stats["x"]["special_tokens"]["high_values"],
            "normal_high_values": normal_stats["x"]["special_tokens"]["high_values"],
        }
    }
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Analyze batch data characteristics")
    parser.add_argument("--crash-batch", type=str, required=True,
                        help="Path to crash batch .pt file (e.g., batches_rank0.pt)")
    parser.add_argument("--normal-batch", type=str, default=None,
                        help="Path to normal batch .pt file for comparison")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for analysis results")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Batch Data Analysis")
    print("=" * 60)
    
    print(f"\nLoading crash batch: {args.crash_batch}")
    crash_data = load_batch(args.crash_batch)
    crash_stats = analyze_batch_stats(crash_data, name="crash_batch")
    
    print(f"  Batches: {crash_stats['num_batches']}")
    print(f"  Shape: {crash_stats['x']['shape']}")
    print(f"  X token range: [{crash_stats['x']['token_stats']['min']}, {crash_stats['x']['token_stats']['max']}]")
    print(f"  X unique tokens: {crash_stats['x']['token_stats']['unique_tokens']}")
    print(f"  X high values (>50k): {crash_stats['x']['special_tokens']['high_values']}")
    
    if args.normal_batch:
        print(f"\nLoading normal batch: {args.normal_batch}")
        normal_data = load_batch(args.normal_batch)
        normal_stats = analyze_batch_stats(normal_data, name="normal_batch")
        
        print(f"  Batches: {normal_stats['num_batches']}")
        print(f"  Shape: {normal_stats['x']['shape']}")
        print(f"  X token range: [{normal_stats['x']['token_stats']['min']}, {normal_stats['x']['token_stats']['max']}]")
        print(f"  X unique tokens: {normal_stats['x']['token_stats']['unique_tokens']}")
        print(f"  X high values (>50k): {normal_stats['x']['special_tokens']['high_values']}")
        
        comparison = compare_batches(crash_stats, normal_stats)
        print(f"\nComparison:")
        print(f"  X length diff: {comparison['x_length_diff']['diff']:.2f}")
        print(f"  X unique token diff: {comparison['x_token_diff']['diff']}")
        print(f"  X high value diff: {comparison['special_token_diff']['crash_high_values'] - comparison['special_token_diff']['normal_high_values']}")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "crash_batch": crash_stats,
        }
        if args.normal_batch:
            results["normal_batch"] = normal_stats
            results["comparison"] = comparison
        
        results_path = output_dir / "batch_analysis.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
