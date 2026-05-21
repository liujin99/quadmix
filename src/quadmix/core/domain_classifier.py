"""Domain classification abstraction for QuaDMix.

The paper uses a DeBERTa V3-based domain classifier to divide data
into 26 different domains (e.g., Health, Science, Sports, Computers).

This module provides an abstract base class. For NPU deployment,
the classifier model should be loaded on the target NPU device.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import numpy.typing as npt


class BaseDomainClassifier(ABC):
    """Abstract base class for domain classification.

    Assigns a domain label d_x in {0, ..., M-1} to each document.
    """

    def __init__(self, name: str, num_domains: int):
        self._name = name
        self._num_domains = num_domains

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_domains(self) -> int:
        return self._num_domains

    @abstractmethod
    def classify(self, documents: List[str]) -> npt.NDArray[np.int64]:
        """Classify documents into domains.

        Args:
            documents: List of document strings.

        Returns:
            Array of domain labels. Shape: (len(documents),)
            Values in {0, ..., num_domains - 1}.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._name}', num_domains={self._num_domains})"

    def __call__(self, documents: List[str]) -> npt.NDArray[np.int64]:
        return self.classify(documents)
