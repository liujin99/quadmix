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
    resume_step = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("--resume-step="):
            resume_step = int(arg.split("=")[1])
        elif arg.startswith("--resume-step") and i < len(sys.argv) - 1:
            resume_step = int(sys.argv[i + 1])
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

    modifications = ["replay dataloader", "disabled eval/sample/save"]

    if resume_step is not None:
        source = source.replace(
            'load_model("base",',
            'load_model("mid",'
        )
        source = source.replace(
            'load_optimizer_state("base",',
            'load_optimizer_state("mid",'
        )
        source = source.replace(
            '\nstep = 0\n',
            f'\nstep = {resume_step}\n'
        )
        modifications.append(f"resume from mid checkpoint at step {resume_step}")

    with open(output_path, "w") as f:
        f.write(source)

    print(f"Generated: {output_path}")
    print(f"  Source: {mid_train_path}")
    print(f"  Modifications: {', '.join(modifications)}")


if __name__ == "__main__":
    main()
