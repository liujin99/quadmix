"""Quality scoring abstraction for QuaDMix.

In the paper, three quality scorers are used:
  1. AskLLM — uses a prompted LLM to score document quality
  2. Fineweb-Edu Classifier — classifier trained on educational quality
  3. DCLM — fastText-based classifier

This module provides an abstract base class that can be subclassed
for any quality scoring method. For NPU deployment, the actual models
should be loaded on the target device.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import numpy.typing as npt


class BaseQualityScorer(ABC):
    """Abstract base class for quality scoring.

    Each scorer outputs a score where SMALLER values = BETTER quality.
    This convention matches the paper's notation where q_{n,x} represents
    quality and smaller is better.
    """

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def score(self, documents: List[str]) -> npt.NDArray[np.float64]:
        """Score a batch of documents.

        Args:
            documents: List of document strings to score.

        Returns:
            Array of quality scores. SMALLER = BETTER.
            Shape: (len(documents),)
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._name}')"

    def __call__(self, documents: List[str]) -> npt.NDArray[np.float64]:
        return self.score(documents)
