#!/usr/bin/env python3
"""
Generate replay_mid_train.py from mid_train.py with minimal modifications.

This ensures the replay training loop is IDENTICAL to the original mid_train.py,
only replacing the dataloader with pre-captured batches.

Usage:
    python3 generate_replay_script.py /path/to/nanochat_repo
    # produces /path/to/nanochat_repo/scripts/replay_mid_train.py
"""
import sys
import os


def main():
    nanochat_repo = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NANOCHAT_REPO", "/home/ma-user/work/nanochat_midtrain_326")
    mid_train_path = os.path.join(nanochat_repo, "scripts", "mid_train.py")
    output_path = os.path.join(nanochat_repo, "scripts", "replay_mid_train.py")

    with open(mid_train_path, "r") as f:
        source = f.read()

    replay_dataloader_code = '''
# ══════ REPLAY DATALOADER (auto-injected by generate_replay_script.py) ══════
_replay_dir = os.environ.get('REPLAY_DIR', '')
if _replay_dir:
    class _ReplayDataLoader:
        def __init__(self, replay_dir, start_step, end_step, rank):
            self.replay_dir = replay_dir
            self.start_step = start_step
            self.end_step = end_step
            self.rank = rank
            self.all_batches = []
            self.idx = 0
            self.epoch = 0
            for s in range(start_step, end_step + 1):
                path = os.path.join(replay_dir, f"step{s}_rank{rank}.pt")
                if os.path.exists(path):
                    batches = torch.load(path, map_location="cpu")
                    self.all_batches.extend(batches)
                else:
                    print0(f"[REPLAY] WARNING: missing {path}")
            print0(f"[REPLAY] Loaded {len(self.all_batches)} total batches from steps {start_step}-{end_step}")

        def __next__(self):
            if self.idx >= len(self.all_batches):
                self.epoch += 1
                self.idx = 0
            batch = self.all_batches[self.idx]
            self.idx += 1
            x, y = batch["x"], batch["y"]
            state = {"epoch": self.epoch, "pq_idx": self.idx, "rg_idx": 0}
            return x, y, state

    _replay_start = int(os.environ.get('REPLAY_START', '0'))
    _replay_end = int(os.environ.get('REPLAY_END', '0'))
    print0(f"[REPLAY] Using pre-captured batches from {_replay_dir} (steps {_replay_start}-{_replay_end})")
# ══════ END REPLAY DATALOADER ══════

'''

    injection_point = "x, y, dataloader_state_dict = next(train_loader)\nx = x.to(device, non_blocking=True)\ny = y.to(device, non_blocking=True)"
    assert injection_point in source, f"Injection point not found in mid_train.py"

    wrapped_injection = '''if not _replay_dir:
    x, y, dataloader_state_dict = next(train_loader)
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

# ══════ REPLACE DATALOADER FOR REPLAY (auto-injected) ══════
if _replay_dir:
    train_loader = _ReplayDataLoader(_replay_dir, _replay_start, _replay_end, ddp_rank)
x, y, dataloader_state_dict = next(train_loader)
x = x.to(device, non_blocking=True)
y = y.to(device, non_blocking=True)
# ══════ END REPLACE ══════
'''

    source = source.replace(
        injection_point,
        replay_dataloader_code + wrapped_injection
    )

    source = source.replace(
        'parser.add_argument("--data-dir"',
        'parser.add_argument("--replay-dir", type=str, default=None, help="[replay] unused, kept for compat")\nparser.add_argument("--data-dir"'
    )

    num_iter_point = 'print0(f"Total tokens: {total_tokens:,}, Steps: {num_iterations:,}")'
    assert num_iter_point in source, f"num_iterations print not found in mid_train.py"
    source = source.replace(
        num_iter_point,
        num_iter_point + '''

# ══════ OVERRIDE NUM_ITERATIONS FOR REPLAY (auto-injected) ══════
if _replay_dir:
    num_iterations = _replay_end - _replay_start + 1
    total_tokens = total_batch_size * num_iterations
    print0(f"[REPLAY] Overriding: Steps={num_iterations}, Total tokens={total_tokens:,}")
# ══════ END OVERRIDE ══════
'''
    )

    source = source.replace(
        'min_val_bpb = float("inf")',
        '''# ══════ DISABLE EVAL/SAMPLE/SAVE FOR REPLAY (auto-injected) ══════
if _replay_dir:
    args.eval_every = -1
    args.sample_every = -1
    args.core_metric_every = -1
    args.save_every = -1
# ══════ END DISABLE ══════

min_val_bpb = float("inf")'''
    )

    with open(output_path, "w") as f:
        f.write(source)

    print(f"Generated: {output_path}")
    print(f"  Source: {mid_train_path}")
    print(f"  Modifications: replay dataloader, disabled eval/sample/save, num_iterations override")


if __name__ == "__main__":
    main()
