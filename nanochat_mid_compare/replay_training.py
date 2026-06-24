"""
Replay training using pre-captured batches to diagnose NPU crashes.

Usage:
    torchrun --standalone --nproc_per_node=8 -m replay_training -- \
        --batch-dir=/path/to/captured_batches \
        --start-step=320 \
        --end-step=330 \
        --model-tag=base \
        --model-step=0
"""

import os
import argparse
import time
import math
import torch
import torch.distributed as dist
from contextlib import nullcontext

import torch_npu

from nanochat.gpt import GPT
from nanochat.common import compute_init, print0, get_base_dir, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model, load_optimizer_state


def main():
    parser = argparse.ArgumentParser(description="Replay training with pre-captured batches")
    parser.add_argument("--batch-dir", type=str, required=True, help="Directory containing captured batches")
    parser.add_argument("--start-step", type=int, required=True, help="Original start step")
    parser.add_argument("--end-step", type=int, required=True, help="Original end step")
    parser.add_argument("--model-tag", type=str, default=None, help="Model tag to load from")
    parser.add_argument("--model-step", type=int, default=None, help="Model step to load from")
    parser.add_argument("--device-batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--total-batch-size", type=int, default=524288, help="Total batch size in tokens")
    parser.add_argument("--embedding-lr", type=float, default=0.3, help="Adam LR for embeddings")
    parser.add_argument("--unembedding-lr", type=float, default=0.008, help="Adam LR for unembeddings")
    parser.add_argument("--matrix-lr", type=float, default=0.02, help="Muon LR for matrices")
    parser.add_argument("--weight-decay", type=float, default=0.28, help="Weight decay")
    args = parser.parse_args()

    device_type = "npu"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    print0(f"Replay training from {args.batch_dir}")
    print0(f"  Original steps: {args.start_step}-{args.end_step}")
    print0(f"  world_size={ddp_world_size}, rank={ddp_rank}")

    autocast_ctx = torch.npu.amp.autocast(dtype=COMPUTE_DTYPE)

    model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)
    orig_model = model

    pretrain_user_config = meta.get("user_config", {})
    for name, fallback, source in [
        ("max_seq_len", args.max_seq_len, meta),
        ("device_batch_size", args.device_batch_size, meta),
        ("total_batch_size", args.total_batch_size, meta),
        ("embedding_lr", args.embedding_lr, pretrain_user_config),
        ("unembedding_lr", args.unembedding_lr, pretrain_user_config),
        ("matrix_lr", args.matrix_lr, pretrain_user_config),
    ]:
        arg_val = getattr(args, name)
        pretrain_val = source.get(name)
        if arg_val is None:
            resolved = pretrain_val if pretrain_val is not None else fallback
            setattr(args, name, resolved)

    tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    total_batch_size = args.total_batch_size
    grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
    print0(f"Grad accum steps: {grad_accum_steps}")

    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay
    )

    num_iterations = args.end_step - args.start_step + 1
    print0(f"Will train for {num_iterations} steps (replay steps {args.start_step}-{args.end_step})")

    torch.npu.empty_cache()
    import gc
    gc.collect()

    step = 0
    for original_step in range(args.start_step, args.end_step + 1):
        batch_file = os.path.join(args.batch_dir, f"step{original_step}_rank{ddp_rank}.pt")
        if not os.path.exists(batch_file):
            print0(f"ERROR: Batch file not found: {batch_file}")
            return
        batches = torch.load(batch_file, map_location="cpu")
        print0(f"Step {step} (original {original_step}): loaded {len(batches)} batches")

        torch.npu.synchronize()
        t0 = time.time()
        train_loss_f = 0.0

        for micro_step_idx, batch in enumerate(batches):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            with autocast_ctx:
                loss = model(x, y)
            loss = loss / grad_accum_steps
            loss.backward()

            train_loss_f += loss.detach().item()

        lrm = 1.0
        for group in optimizer.param_groups:
            group["lr"] = group.get("initial_lr", group["lr"]) * lrm

        has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters())
        if dist.is_initialized():
            dev = next(model.parameters()).device
            nan_flag = torch.tensor([1.0 if has_nan else 0.0], device=dev)
            dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
            has_nan = nan_flag.item() > 0

        if has_nan:
            print0(f"[WARNING] NaN gradients at step {step} (original {original_step}), skipping optimizer.step()")
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.zero_grad(set_to_none=True)
        torch.npu.synchronize()
        dt = time.time() - t0

        print0(f"step {step:03d} (original {original_step}) | loss: {train_loss_f:.6f} | dt: {dt*1000:.2f}ms")

        step += 1

    print0("Replay training completed!")


if __name__ == "__main__":
    main()
