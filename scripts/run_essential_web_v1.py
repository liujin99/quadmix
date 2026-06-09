#!/usr/bin/env python3
"""
Run QuaDMix on essential-web-v1.0 with multi-shard preprocessing.

Loads preprocessed shards via ShardMetadataManager → skips feature extraction →
quality ranks → parameter sampling → real proxy experiments (on-demand text) →
LightGBM regression → optimal search → final sampling → save outputs.

Usage:
  # Quick validation (200 experiments, fast)
  python scripts/run_essential_web_v1.py --quick --preprocessed-dir .../preprocessed

  # Full production (3000 experiments, needs GPU)
  python scripts/run_essential_web_v1.py --full --preprocessed-dir .../preprocessed

  # Or use the auto-detected defaults:
  python scripts/run_essential_web_v1.py --quick

The validation set (openhermes_10k_assistant_tokenized.pt) will be
automatically downloaded from HuggingFace if not found locally.
Alternatively, use --val-set=core to use a CORE benchmark-based
validation set (auto-generated from eval_bundle).
"""

import argparse, os, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from quadmix import QuaDMixConfig
from quadmix.pipeline.real_pipeline import QuaDMixPipeline
from quadmix.data.metadata_manager import ShardMetadataManager

# ── Defaults ──────────────────────────────────────────────
QUADMIX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Temp/cache dir: override via QUADMIX_TEMP_DIR env var, defaults to ~/.cache/QuaDMix/temp/
QUADMIX_TEMP_DIR = os.environ.get(
    "QUADMIX_TEMP_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "QuaDMix", "temp"),
)
DEFAULT_PREPROCESSED_DIR = os.path.join(QUADMIX_TEMP_DIR, "preprocessed")
DEFAULT_VAL_DIR = os.path.join(QUADMIX_DIR, "data")
DEFAULT_VAL_PATH = os.path.join(DEFAULT_VAL_DIR, "openhermes_10k_assistant_tokenized.pt")

# HuggingFace dataset for validation set
HF_DATASET = "liujin99/quadmix-openhermes-10k"
HF_VAL_FILENAME = "openhermes_10k_assistant_tokenized.pt"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
HF_RESOLVE = f"{HF_ENDPOINT}/datasets/{{repo}}/resolve/main/{{file}}"

# CORE benchmark validation set
CORE_VAL_FILENAME = "core_22tasks_tokenized.pt"
HF_CORE_DATASET = "liujin99/quadmix-core-22tasks"
DEFAULT_EVAL_BUNDLE = os.environ.get(
    "EVAL_BUNDLE_DIR",
    "/home/ma-user/work/nanochat-master-multi/eval_bundle",
)


def _hf_remote_size(repo_id: str, filename: str) -> int:
    """Get remote file size from HuggingFace via HEAD request. Returns 0 if failed."""
    url = f"{HF_ENDPOINT}/datasets/{repo_id}/resolve/main/{filename}"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return int(resp.headers.get("Content-Length", 0))
    except Exception:
        pass
    return 0


def _download_hf_file(repo_id: str, filename: str, local_path: str) -> bool:
    """Download a file from HuggingFace. Returns True on success."""
    url = HF_RESOLVE.format(repo=repo_id, file=filename)
    if HF_ENDPOINT != "https://huggingface.co":
        print(f"[Setup] Using HF mirror: {HF_ENDPOINT}")
    print(f"[Setup] Downloading from:\n  {url}")
    try:
        urllib.request.urlretrieve(url, local_path)
        size_mb = os.path.getsize(local_path) / 1024**2
        print(f"[Setup] Downloaded: {local_path} ({size_mb:.0f} MB)")
        return True
    except Exception as e:
        print(f"[Setup] Download failed: {e}")
        return False


def ensure_val_data(val_path: str) -> str:
    """
    Ensure the validation set exists locally and is up-to-date.
    Downloads from HuggingFace if not found or version mismatch.
    Returns the path to the file.
    """
    os.makedirs(os.path.dirname(val_path), exist_ok=True)

    local_size = os.path.getsize(val_path) if os.path.exists(val_path) else 0
    remote_size = _hf_remote_size(HF_DATASET, HF_VAL_FILENAME)

    if remote_size == 0:
        if local_size > 0:
            print(f"\n[Setup] Warning: Cannot connect to HuggingFace to check {HF_VAL_FILENAME}")
            print(f"[Setup] Using local file ({local_size / 1024**2:.0f} MB)")
            print(f"[Setup] To force re-download, delete: {val_path}")
            return val_path
        else:
            raise RuntimeError(
                f"Cannot connect to HuggingFace and no local file.\n"
                f"  Please download manually from:\n"
                f"    https://huggingface.co/datasets/{HF_DATASET}\n"
                f"  And place at: {val_path}"
            )

    if local_size == remote_size:
        print(f"[Setup] Validation set up-to-date: {HF_VAL_FILENAME} ({local_size / 1024**2:.0f} MB)")
        return val_path

    if local_size > 0:
        print(f"\n[Setup] {HF_VAL_FILENAME} version mismatch")
        print(f"[Setup]   Local:  {local_size / 1024**2:.0f} MB")
        print(f"[Setup]   Remote: {remote_size / 1024**2:.0f} MB")
        print(f"[Setup]   Re-downloading...")
        os.remove(val_path)
    else:
        print(f"\n[Setup] Validation set not found at:\n  {val_path}")

    if not _download_hf_file(HF_DATASET, HF_VAL_FILENAME, val_path):
        raise RuntimeError(
            f"Failed to download validation set.\n"
            f"  You can manually download from:\n"
            f"    https://huggingface.co/datasets/{HF_DATASET}\n"
            f"  Or place the file at: {val_path}"
        )
    return val_path

def ensure_core_val_data(val_path: str, eval_bundle: str) -> str:
    """
    Ensure the CORE benchmark validation set exists and is up-to-date.
    Checks remote version, downloads from HuggingFace, or falls back to local generation.
    Returns the path to the file.
    """
    os.makedirs(os.path.dirname(val_path), exist_ok=True)

    local_size = os.path.getsize(val_path) if os.path.exists(val_path) else 0
    remote_size = _hf_remote_size(HF_CORE_DATASET, CORE_VAL_FILENAME)

    if remote_size == 0:
        if local_size > 0:
            print(f"\n[Setup] Warning: Cannot connect to HuggingFace to check {CORE_VAL_FILENAME}")
            print(f"[Setup] Using local file ({local_size / 1024**2:.0f} MB)")
            print(f"[Setup] To force re-download, delete: {val_path}")
            return val_path
    elif local_size == remote_size:
        print(f"[Setup] CORE validation set up-to-date: {CORE_VAL_FILENAME} ({local_size / 1024**2:.0f} MB)")
        return val_path
    elif local_size > 0:
        print(f"\n[Setup] {CORE_VAL_FILENAME} version mismatch")
        print(f"[Setup]   Local:  {local_size / 1024**2:.0f} MB")
        print(f"[Setup]   Remote: {remote_size / 1024**2:.0f} MB")
        print(f"[Setup]   Re-downloading...")
        os.remove(val_path)
    else:
        print(f"\n[Setup] CORE validation set not found at:\n  {val_path}")

    if remote_size > 0:
        if _download_hf_file(HF_CORE_DATASET, CORE_VAL_FILENAME, val_path):
            return val_path
        print(f"[Setup] Falling back to local generation...")
    else:
        print(f"[Setup] Trying local generation...")

    # Fall back to local generation
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prepare_script = os.path.join(script_dir, "validation_set", "prepare_core_val_set.py")

    if not os.path.exists(prepare_script):
        raise FileNotFoundError(
            f"CORE validation set not found at:\n  {val_path}\n"
            f"Preparation script also missing:\n  {prepare_script}\n"
            f"You can manually download from:\n"
            f"  https://huggingface.co/datasets/{HF_CORE_DATASET}"
        )

    print(f"[Setup] Auto-generating from eval_bundle: {eval_bundle}")

    import subprocess
    result = subprocess.run(
        [
            sys.executable, prepare_script,
            "--output-dir", os.path.dirname(val_path),
            "--eval-bundle", eval_bundle,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to generate CORE validation set.\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}\n"
            f"You can manually download from:\n"
            f"  https://huggingface.co/datasets/{HF_CORE_DATASET}"
        )
    print(result.stdout)
    if not os.path.exists(val_path):
        raise RuntimeError(
            f"CORE validation set generation completed but file not found:\n  {val_path}"
        )
    size_mb = os.path.getsize(val_path) / 1024**2
    print(f"[Setup] Generated: {val_path} ({size_mb:.0f} MB)")
    return val_path


DOMAIN_NAMES = [
    "Industrial arts, Technology, and Engineering",  # 0
    "Social sciences",                                 # 1
    "Science and Natural history",                     # 2
    "Religion",                                        # 3
    "Philology; or, Language and languages",           # 4
    "Literature",                                      # 5
    "History and Geography",                           # 6
    "General works, books and libraries...",           # 7
    "Philosophy and psychology",                       # 8
    "Arts",                                            # 9
]
QUALITY_NAMES = ["dclm", "fineweb_edu", "english", "math_general", "math_openweb"]
QUALITY_COLUMNS = ["qs_dclm", "qs_fineweb_edu_approx", "qs_english",
                   "qs_eai_general_math", "qs_eai_open_web_math"]


def build_parser():
    p = argparse.ArgumentParser(description="QuaDMix on essential-web-v1 (sharded mode)")
    p.add_argument("--preprocessed-dir", default=DEFAULT_PREPROCESSED_DIR,
                   help="Directory of preprocessed parquet shards")
    p.add_argument("--quick", action="store_true", help="Quick: 200 exp, 2000 search")
    p.add_argument("--full", action="store_true", help="Full: 3000 exp, 100K search")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--num-experiments", type=int, default=None)
    p.add_argument("--num-search", type=int, default=None)
    # Note: --doc-limit removed. Proxy experiments should use full data pool.
    # Use --target-tokens to control final output size instead.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--target-tokens", type=float, default=0.0,
                   help="Desired max tokens in output (in B). θ* may produce less; "
                        "if more, uniformly discard to preserve distribution. "
                        "Paper: 'more tokens not always good' (30B > 90B > 180B)")
    p.add_argument("--device-type", default="cpu",
                   choices=["cpu", "cuda", "npu"],
                   help="Device type for proxy model training")
    p.add_argument("--npu-devices", type=int, default=1,
                   help="Number of NPU devices (1=sequential, 8=8-way parallel)")

    # Proxy model config
    p.add_argument("--block-size", type=int, default=64,
                   help="Sequence length (tokens). Full paper: 2048")
    p.add_argument("--tiny-steps", type=int, default=3,
                   help="Proxy training steps (0 = use max_step=25000)")
    p.add_argument("--micro-batch-size", type=int, default=2,
                   help="Micro batch size per forward pass")
    p.add_argument("--global-batch-size", type=int, default=8,
                   help="Effective global batch size (grad acc = global/micro)")
    p.add_argument("--rank-ref-size", type=int, default=10000,
                   help="Reference subset size for rank estimation")
    p.add_argument("--checkpoint-steps", type=str, default=None,
                   help="[DEPRECATED] Use --checkpoint-interval instead.")
    p.add_argument("--checkpoint-interval", type=int, default=1000,
                   help="Record val_loss every N steps during proxy training "
                        "(default: 1000, 0 = disable). "
                        "Results saved to each exp dir as checkpoint_trajectory.json.")
    p.add_argument("--val-set", type=str, default="openhermes",
                   choices=["openhermes", "core"],
                   help="Validation set: 'openhermes' (default, auto-download) or "
                        "'core' (CORE benchmark 22-task, auto-generate from eval_bundle)")
    p.add_argument("--val-path", type=str, default=None,
                   help="Path to validation .pt file (overrides --val-set)")
    p.add_argument("--eval-bundle", type=str, default=DEFAULT_EVAL_BUNDLE,
                   help="Path to CORE eval bundle directory "
                        "(used when --val-set=core, default: $EVAL_BUNDLE_DIR)")
    return p


def create_proxy_runner(config, args, output_dir, metadata_manager):
    """Create an EssentialWebProxyRunner with sharded metadata manager."""
    from essential_proxy_runner import EssentialWebProxyRunner
    proxy_dir = os.path.join(output_dir, "proxy_experiments")

    # Resolve validation path
    if args.val_path:
        val_path = args.val_path
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation file not found: {val_path}")
    elif args.val_set == "core":
        val_path = os.path.join(DEFAULT_VAL_DIR, CORE_VAL_FILENAME)
        val_path = ensure_core_val_data(val_path, args.eval_bundle)
    else:
        val_path = os.path.join(DEFAULT_VAL_DIR, HF_VAL_FILENAME)
        val_path = ensure_val_data(val_path)

    # Parse checkpoint interval
    checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval else 1000

    runner = EssentialWebProxyRunner(
        config=config,
        metadata_manager=metadata_manager,
        val_data_path=val_path,
        output_dir=proxy_dir,
        device_type=args.device_type,
        npu_device_id=0,  # Main process uses card 0; workers use their own
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        tiny_steps=args.tiny_steps,
        doc_limit=None,  # Always use full data pool for proxy experiments
        test_block_size=args.block_size,
        rank_ref_size=args.rank_ref_size,
        token_cache_dir=os.path.join(QUADMIX_TEMP_DIR, "token_cache"),
        checkpoint_interval=checkpoint_interval,
    )
    return runner


def main():
    args = build_parser().parse_args()

    if args.quick:
        n_exp, n_search, top_k = 200, 2000, args.top_k
    elif args.full:
        n_exp, n_search, top_k = 3000, 100000, args.top_k
    else:
        n_exp = args.num_experiments or 500
        n_search = args.num_search or 5000
        top_k = args.top_k

    output_dir = args.output or os.path.join(
        QUADMIX_DIR, f"result/quadmix_{time.strftime('%Y%m%d_%H%M%S')}"
    )

    # ── Load metadata manager (reads only domain + quality from all shards) ──
    print(f"\n[Setup] Loading ShardMetadataManager from: {args.preprocessed_dir}")
    metadata_manager = ShardMetadataManager(args.preprocessed_dir)
    print(f"[Setup] {metadata_manager.num_docs:,} docs across "
          f"{metadata_manager.num_shards} shards")

    # ── Dataset size estimation ──
    total_tokens_est = metadata_manager.get_total_tokens_estimate()
    total_chars = metadata_manager.get_total_chars()
    print(f"\n════════════════════════════════════════════════════════")
    print(f"  数据集信息:")
    print(f"    总文档数:         {metadata_manager.num_docs:,}")
    print(f"    总字符数:         {total_chars:,}")
    print(f"    估算 token 数:    {total_tokens_est:,} ({total_tokens_est/1e9:.1f}B)")
    print(f"    (按 4 chars/token 估算，GPT-NeoX tokenizer)")
    print(f"════════════════════════════════════════════════════════")

    # ── omega range guidance ──
    # Paper: ω sampled from [0,1] → rescaled to [0, 0.1]
    # Simplified: sampling_ratio ≈ omega
    omega_min, omega_max = 0.0, 0.1  # Paper default range (after rescaling)
    tokens_min = int(total_tokens_est * omega_min)
    tokens_max = int(total_tokens_est * omega_max)

    print(f"\n  omega (ω) 参数与数据量估算:")
    print(f"    论文默认范围:     [{omega_min:.2f}, {omega_max:.2f}]")
    print(f"    预计数据量范围:   {tokens_min/1e9:.1f}B - {tokens_max/1e9:.1f}B tokens")
    print(f"    (简化估算: sampling_ratio ≈ ω，忽略 sigmoid 形态)")

    if args.target_tokens > 0:
        target_b = args.target_tokens
        print(f"\n  你设置的 target:   {target_b:.1f}B tokens")
        # Rough estimate: omega ≈ target / total (for reference only)
        target_omega_est = target_b * 1e9 / total_tokens_est
        print(f"    参考信息: 若 ω ≈ {target_omega_est:.3f}，可能产生约 {target_b:.1f}B 数据")
        print(f"    注意: 系统会全范围搜索最优参数，不限制 omega 范围")
        if target_omega_est > 0.1:
            print(f"    [提示] target {target_b:.1f}B 对应 ω > 0.1，可能超出论文推荐范围")
    print(f"════════════════════════════════════════════════════════")

    config = QuaDMixConfig(
        num_domains=10, num_quality_criteria=5,
        num_proxy_experiments=n_exp, num_search_points=n_search,
        top_k_average=top_k,
        target_tokens=int(args.target_tokens * 1e9) if args.target_tokens > 0 else 0,
    )

    pipeline = QuaDMixPipeline(config)
    proxy_runner = create_proxy_runner(config, args, output_dir, metadata_manager)

    print(f"\n[Setup] Proxy runner: {n_exp} experiments, "
         f"{args.tiny_steps} steps each, "
         f"val={args.val_set}"
         f"{', ' + str(args.npu_devices) + ' NPU devices' if args.npu_devices > 1 else ''}")

    pipeline.run(
        data_path=args.preprocessed_dir,
        output_dir=output_dir,
        precomputed=True,
        # load_precomputed_sharded will be triggered by passing metadata_manager
        # via **load_kwargs:
        metadata_manager=metadata_manager,
        doc_limit=None,  # Always use full data pool
        domain_names=DOMAIN_NAMES,
        quality_names=QUALITY_NAMES,
        proxy_runner=proxy_runner,
        parallel_workers=args.npu_devices,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())