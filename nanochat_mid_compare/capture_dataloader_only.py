"""
Capture dataloader batches at a specific step without running the model.
Only initializes the dataloader and advances to the target step, then saves the batches.

Usage:
    torchrun --standalone --nproc_per_node=8 -m capture_dataloader_only -- \
        --data-dir=/path/to/quality_data_fineweb_edu \
        --target-step=329 \
        --output-dir=/path/to/output
"""

import os
import argparse
import torch
import torch.distributed as dist
import torch_npu

from nanochat.common import compute_init, print0, get_base_dir
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit


def main():
    parser = argparse.ArgumentParser(description="Capture dataloader batches at target step")
    parser.add_argument("--data-dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--target-step", type=int, required=True, help="Step number to capture (0-indexed)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for captured batches")
    parser.add_argument("--device-batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")
    args = parser.parse_args()

    device_type = "npu"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    print0(f"Capturing batches at step {args.target_step} from {args.data_dir}")
    print0(f"  device_batch_size={args.device_batch_size}, max_seq_len={args.max_seq_len}, grad_accum={args.grad_accum_steps}")
    print0(f"  world_size={ddp_world_size}, rank={ddp_rank}")

    tokenizer = get_tokenizer()

    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="train",
        device=device, tokenizer_threads=16, tokenizer_batch_size=256,
        buffer_size=2000, data_dir=args.data_dir)

    x, y, dataloader_state_dict = next(train_loader)
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    total_micro_steps = args.target_step * args.grad_accum_steps
    print0(f"  Advancing {total_micro_steps} micro_steps (step {args.target_step} x grad_accum {args.grad_accum_steps})...")

    captured_batches = []
    for micro_step in range(total_micro_steps):
        x, y, dataloader_state_dict = next(train_loader)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if micro_step >= total_micro_steps - args.grad_accum_steps:
            captured_batches.append({"x": x.cpu(), "y": y.cpu()})
        if (micro_step + 1) % 100 == 0:
            print0(f"  micro_step {micro_step + 1}/{total_micro_steps}")

    print0(f"  Captured {len(captured_batches)} batches")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"batches_rank{ddp_rank}.pt")
    torch.save(captured_batches, output_path)
    print0(f"  Saved to: {output_path}")

    if dist.is_initialized():
        dist.barrier()
    print0("  Done!")


if __name__ == "__main__":
    main()
