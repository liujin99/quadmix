"""
Base data adapter for QuaDMix — unified dataset loading.

All adapters produce a UnifiedData object with a consistent schema:
  - text: str            (required) — document content
  - doc_id: str          (generated) — unique document identifier
  - token_count: int     (optional) — precomputed token count
  - domain: int          (optional) — preassigned domain label
  - quality_scores: dict (optional) — precomputed quality scores
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import hashlib

import numpy as np
import pandas as pd


@dataclass
class UnifiedData:
    """Unified dataset representation after adapter processing."""

    # Core data: documents
    texts: List[str]
    # Auto-generated doc IDs (hash of text content)
    doc_ids: List[str]
    # Optional token counts
    token_counts: Optional[np.ndarray] = None
    # Optional domain labels
    domain_labels: Optional[np.ndarray] = None
    # Optional precomputed quality scores
    quality_scores: Optional[np.ndarray] = None
    # Metadata about the source
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def num_docs(self) -> int:
        return len(self.texts)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for downstream processing."""
        df = pd.DataFrame({"text": self.texts, "doc_id": self.doc_ids})
        if self.token_counts is not None:
            df["token_count"] = self.token_counts
        if self.domain_labels is not None:
            df["domain"] = self.domain_labels
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "UnifiedData":
        """Create from a DataFrame (reverse of to_dataframe)."""
        texts = df["text"].tolist() if "text" in df.columns else []
        doc_ids = df["doc_id"].tolist() if "doc_id" in df.columns else [
            hashlib.md5(t.encode()).hexdigest()[:12] for t in texts
        ]
        token_counts = df["token_count"].to_numpy(dtype=np.int64) if "token_count" in df.columns else None
        domain_labels = df["domain"].to_numpy(dtype=np.int64) if "domain" in df.columns else None
        return cls(
            texts=texts,
            doc_ids=doc_ids,
            token_counts=token_counts,
            domain_labels=domain_labels,
        )


class BaseDataAdapter(ABC):
    """
    Abstract base class for data adapters.
    
    Each adapter knows how to load one specific format and produce UnifiedData.
    """

    @abstractmethod
    def can_handle(self, path: str) -> bool:
        """Check if this adapter can handle the given file path."""
        ...

    @abstractmethod
    def load(self, path: str, **kwargs) -> UnifiedData:
        """
        Load data from the given path.
        
        Args:
            path: File path to load.
            **kwargs: Adapter-specific options.
            
        Returns:
            UnifiedData with at minimum texts and doc_ids populated.
        """
        ...

    @abstractmethod
    def save(self, data: UnifiedData, path: str, **kwargs):
        """Save UnifiedData back to the given path."""
        ...

    def _generate_doc_ids(self, texts: List[str]) -> List[str]:
        """Generate deterministic document IDs from text content."""
        return [hashlib.md5(t.encode("utf-8")).hexdigest()[:12] for t in texts]

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count (words * ~1.3 for subword tokens)."""
        word_count = len(text.split())
        return max(1, int(word_count * 1.3))
