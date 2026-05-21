"""Proxy experiment runner base class for QuaDMix.

The actual proxy runner is implemented in scripts/essential_proxy_runner.py.
This module provides the abstract interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable

from quadmix.core.types import ParameterSet, ProxyResult


class BaseProxyRunner(ABC):
    """Abstract base class for proxy experiment runners."""

    @abstractmethod
    def run_experiment(
        self,
        params: ParameterSet,
        experiment_id: int = 0,
    ) -> ProxyResult:
        """Run a single proxy experiment.

        Args:
            params: QuaDMix parameter configuration.
            experiment_id: Index for tracking.

        Returns:
            ProxyResult with validation loss.
        """
        ...

    def run_batch(
        self,
        params_list: List[ParameterSet],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[ProxyResult]:
        """Run multiple proxy experiments.

        Args:
            params_list: List of parameter configurations.
            progress_callback: Optional callback(current, total).

        Returns:
            List of ProxyResult instances.
        """
        results = []
        for i, params in enumerate(params_list):
            result = self.run_experiment(params, experiment_id=i)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(params_list))
        return results
