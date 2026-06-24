"""
Capture dataloader batches for a range of steps.

Mirrors mid_train.py's batch consumption pattern exactly:
  1. Initial prefetch before training loop (used as first batch of step 0)
  2. Each step: grad_accum_steps micro-batches, last prefetch is first batch of next step

Usage:
    torchrun --standalone --nproc_per_node=8 -m capture_multi_steps -- \
        --data-dir=/path/to/quality_data_fineweb_edu \
        --start-step=320 \
        --end-step=330 \
        --output-dir=/path/to/output
"""

import os
import argparse
import torch
import torch.distributed as dist
import torch_npu

from nanochat.common import compute_init, print0
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit


def main():
    parser = argparse.ArgumentParser(description="Capture dataloader batches for a range of steps")
    parser.add_argument("--data-dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--start-step", type=int, required=True, help="Start step (inclusive)")
    parser.add_argument("--end-step", type=int, required=True, help="End step (inclusive)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device-batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")
    args = parser.parse_args()

    device_type = "npu"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    print0(f"Capturing batches for steps {args.start_step}-{args.end_step} from {args.data_dir}")
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

    print0(f"  Advancing to step {args.start_step}...")

    all_batches = {}

    for step in range(args.end_step + 1):
        step_batches = [{"x": x.cpu(), "y": y.cpu()}]
        for micro in range(args.grad_accum_steps - 1):
            x, y, dataloader_state_dict = next(train_loader)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            step_batches.append({"x": x.cpu(), "y": y.cpu()})

        if step >= args.start_step:
            all_batches[step] = step_batches

        x, y, dataloader_state_dict = next(train_loader)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if step % 50 == 0 or step == args.end_step:
            print0(f"  step {step}/{args.end_step}")

    print0(f"  Captured {len(all_batches)} steps")

    os.makedirs(args.output_dir, exist_ok=True)
    for step, batches in all_batches.items():
        output_path = os.path.join(args.output_dir, f"step{step}_rank{ddp_rank}.pt")
        torch.save(batches, output_path)
        print0(f"  Saved step {step} ({len(batches)} batches) to: {output_path}")

    if dist.is_initialized():
        dist.barrier()
    print0("  Done!")


if __name__ == "__main__":
    main()
