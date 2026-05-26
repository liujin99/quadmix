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
"""

import argparse, os, sys, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quadmix import QuaDMixConfig
from quadmix.pipeline.real_pipeline import QuaDMixPipeline
from quadmix.data.metadata_manager import ShardMetadataManager

# ── Defaults ──────────────────────────────────────────────
QUADMIX_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PREPROCESSED_DIR = os.path.join(QUADMIX_DIR, "temp/preprocessed")
DEFAULT_VAL_DIR = os.path.join(QUADMIX_DIR, "data")
DEFAULT_VAL_PATH = os.path.join(DEFAULT_VAL_DIR, "openhermes_10k_assistant_tokenized.pt")

# HuggingFace dataset for validation set
HF_DATASET = "liujin99/quadmix-openhermes-10k"
HF_VAL_FILENAME = "openhermes_10k_assistant_tokenized.pt"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"


def ensure_val_data(val_path: str) -> str:
    """
    Ensure the validation set exists locally.
    Downloads from HuggingFace if not found.
    Returns the path to the file.
    """
    if os.path.exists(val_path):
        return val_path

    # Try downloading
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    url = HF_RESOLVE.format(repo=HF_DATASET, file=HF_VAL_FILENAME)
    print(f"\n[Setup] Validation set not found at:\n  {val_path}")
    print(f"[Setup] Downloading from:\n  {url}")
    try:
        urllib.request.urlretrieve(url, val_path)
        size_mb = os.path.getsize(val_path) / 1024**2
        print(f"[Setup] Downloaded: {val_path} ({size_mb:.0f} MB)")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download validation set from {url}.\n"
            f"  Error: {e}\n"
            f"  You can manually download from:\n"
            f"    https://huggingface.co/datasets/{HF_DATASET}\n"
            f"  Or place the file at: {val_path}"
        )
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
    p.add_argument("--doc-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--target-tokens", type=float, default=0.0,
                   help="Desired total tokens in output dataset (in B, e.g. 10 = 10B tokens)")
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
    p.add_argument("--val-limit", type=int, default=500,
                   help="Validation docs to evaluate")
    p.add_argument("--rank-ref-size", type=int, default=10000,
                   help="Reference subset size for rank estimation")
    p.add_argument("--val-path", type=str, default=None,
                   help="Path to validation .pt file "
                        "(default: auto-download from liujin99/quadmix-openhermes-10k to project data/)")
    return p


def create_proxy_runner(config, args, output_dir, metadata_manager):
    """Create an EssentialWebProxyRunner with sharded metadata manager."""
    from scripts.essential_proxy_runner import EssentialWebProxyRunner
    proxy_dir = os.path.join(output_dir, "proxy_experiments")

    # Ensure validation data exists (auto-download if needed)
    val_path = args.val_path or DEFAULT_VAL_PATH
    val_path = ensure_val_data(val_path)

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
        doc_limit=args.doc_limit,
        test_block_size=args.block_size,
        rank_ref_size=args.rank_ref_size,
        val_doc_limit=args.val_limit,
        token_cache_dir=os.path.join(QUADMIX_DIR, "temp/token_cache"),
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
          f"val_limit={args.val_limit} docs"
          f"{', ' + str(args.npu_devices) + ' NPU devices' if args.npu_devices > 1 else ''}")

    pipeline.run(
        data_path=args.preprocessed_dir,
        output_dir=output_dir,
        precomputed=True,
        # load_precomputed_sharded will be triggered by passing metadata_manager
        # via **load_kwargs:
        metadata_manager=metadata_manager,
        doc_limit=args.doc_limit,
        domain_names=DOMAIN_NAMES,
        quality_names=QUALITY_NAMES,
        proxy_runner=proxy_runner,
        parallel_workers=args.npu_devices,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
