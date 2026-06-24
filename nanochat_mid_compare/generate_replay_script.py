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
            self.current_step = start_step
            self.current_micro = 0
            self.batches = None
            self.epoch = 0
            self._load_step()

        def _load_step(self):
            path = os.path.join(self.replay_dir, f"step{self.current_step}_rank{self.rank}.pt")
            if os.path.exists(path):
                self.batches = torch.load(path, map_location="cpu")
            else:
                self.batches = None

        def __next__(self):
            if self.batches is None or self.current_micro >= len(self.batches):
                self.current_step += 1
                self.current_micro = 0
                if self.current_step > self.end_step:
                    self.epoch += 1
                    self.current_step = self.start_step
                self._load_step()
                if self.batches is None:
                    raise StopIteration
            batch = self.batches[self.current_micro]
            self.current_micro += 1
            x, y = batch["x"], batch["y"]
            state = {"epoch": self.epoch, "pq_idx": self.current_step, "rg_idx": self.current_micro}
            return x, y, state

    _replay_start = int(os.environ.get('REPLAY_START', '0'))
    _replay_end = int(os.environ.get('REPLAY_END', '0'))
    print0(f"[REPLAY] Using pre-captured batches from {_replay_dir} (steps {_replay_start}-{_replay_end})")
# ══════ END REPLAY DATALOADER ══════

'''

    injection_point = "x, y, dataloader_state_dict = next(train_loader)\nx = x.to(device, non_blocking=True)\ny = y.to(device, non_blocking=True)"
    assert injection_point in source, f"Injection point not found in mid_train.py"

    source = source.replace(
        injection_point,
        replay_dataloader_code + injection_point + '''

# ══════ REPLACE DATALOADER FOR REPLAY (auto-injected) ══════
if _replay_dir:
    train_loader = _ReplayDataLoader(_replay_dir, _replay_start, _replay_end, ddp_rank)
    x, y, dataloader_state_dict = next(train_loader)
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
# ══════ END REPLACE ══════
'''
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
