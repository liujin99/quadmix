"""
QuaDMix: Quality-Diversity Balanced Data Selection for Efficient LLM Pretraining

Implementation of the QuaDMix algorithm from:
  "QuaDMix: Quality-Diversity Balanced Data Selection for Efficient LLM Pretraining"
  Fengze Liu, Weidong Zhou, Binbin Liu, et al. (ByteDance, 2025)

Key reference: arXiv:2504.16511v2
"""

__version__ = "2.0.0"

from quadmix.core.types import (
    QualityScore, DomainLabel,
    MergedQualityConfig, SamplingConfig,
    QuaDMixConfig, ParameterSet, ProxyResult,
)
from quadmix.pipeline.param_sampler import ParameterSampler
from quadmix.pipeline.optimizer import QuaDMixOptimizer
from quadmix.pipeline.proxy_runner import BaseProxyRunner
from quadmix.pipeline.real_pipeline import QuaDMixPipeline
from quadmix.constants import DOMAIN_NAMES, QUALITY_NAMES, QUALITY_COLUMNS

__all__ = [
    "QualityScore", "DomainLabel",
    "MergedQualityConfig", "SamplingConfig",
    "QuaDMixConfig", "ParameterSet", "ProxyResult",
    "ParameterSampler", "QuaDMixOptimizer",
    "BaseProxyRunner", "QuaDMixPipeline",
    "DOMAIN_NAMES", "QUALITY_NAMES", "QUALITY_COLUMNS",
]
