#!/usr/bin/env python3
"""
Backward-compatible shim — re-exports from src/quadmix/pipeline/.

All implementation has moved to:
  - quadmix.pipeline.essential_proxy_runner (EssentialWebProxyRunner)
  - quadmix.utils.perf_timer (PerfTimer)
  - quadmix.pipeline.loss_utils (chunked loss functions)
  - quadmix.pipeline.shared_memory (SharedArrayInfo, helpers)
  - quadmix.pipeline.parallel_dispatch (worker functions)
"""

import sys, os
try:
    import quadmix
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from quadmix.pipeline.essential_proxy_runner import EssentialWebProxyRunner
from quadmix.utils.perf_timer import PerfTimer
from quadmix.pipeline.loss_utils import chunked_loss_from_hidden, chunked_loss_per_token_from_hidden
from quadmix.pipeline.shared_memory import SharedArrayInfo, ndarray_to_shared, shared_to_ndarray
from quadmix.pipeline.parallel_dispatch import (
    _worker_dynamic_loop,
    _tokenize_shard_parallel,
    _tokenize_chunk_to_array,
    _tokenize_chunk_with_meta,
    _process_shard_full,
    _get_tokenizer,
)
from quadmix.constants import DOMAIN_MAP, FASTTEXT_FIELDS, DEFAULT_TEMP_DIR, DEFAULT_TOKEN_CACHE_DIR

_SCRIPTS_DIR = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
_PROJECT_DIR = __import__("os").path.dirname(_SCRIPTS_DIR)

__all__ = [
    "EssentialWebProxyRunner",
    "PerfTimer",
    "chunked_loss_from_hidden",
    "chunked_loss_per_token_from_hidden",
    "SharedArrayInfo",
    "ndarray_to_shared",
    "shared_to_ndarray",
    "_worker_dynamic_loop",
    "_tokenize_shard_parallel",
    "_tokenize_chunk_to_array",
    "_tokenize_chunk_with_meta",
    "_process_shard_full",
    "_get_tokenizer",
    "DOMAIN_MAP",
    "FASTTEXT_FIELDS",
    "DEFAULT_TEMP_DIR",
    "DEFAULT_TOKEN_CACHE_DIR",
]

if __name__ == "__main__":
    from quadmix import QuaDMixConfig
    from quadmix.pipeline.param_sampler import ParameterSampler
    from quadmix.data.metadata_manager import ShardMetadataManager

    mgr = ShardMetadataManager(
        __import__("os").path.join(DEFAULT_TEMP_DIR, "preprocessed")
    )

    config = QuaDMixConfig(
        num_domains=10, num_quality_criteria=5, num_proxy_experiments=2
    )

    runner = EssentialWebProxyRunner(
        config=config,
        metadata_manager=mgr,
        val_data_path=__import__("os").path.join(_PROJECT_DIR, "data/openhermes_10k_assistant_tokenized.pt"),
        output_dir=__import__("os").path.join(DEFAULT_TEMP_DIR, "outputs/test_sharded"),
        device_type="cpu",
        micro_batch_size=2,
        global_batch_size=8,
        tiny_steps=5,
        doc_limit=5000,
        test_block_size=64,
        rank_ref_size=500,
    )

    params = ParameterSampler(config).sample_batch(2)
    print(f"\nRunning {len(params)} experiments (sharded)...")
    import time
    t0 = time.time()
    results = runner.run_batch(params)
    print(f"\n{'=' * 60}")
    print(f"  Test Complete! ({time.time() - t0:.1f}s)")
    for r in results:
        print(f"  Exp {r.metadata['experiment_id']:04d}: "
              f"val_loss={r.validation_loss:.4f}")
    runner.save_summary(results, __import__("os").path.join(runner.output_dir, "test_summary.json"))
    print("=" * 60)
