"""
Plain text data adapter — reads text files as single-document or line-separated datasets.
"""

from typing import Optional
import os

import numpy as np

from quadmix.data.base import BaseDataAdapter, UnifiedData


class TxtDataAdapter(BaseDataAdapter):
    """
    Adapter for plain text files.
    
    Modes:
      - 'lines': Each line is a document (default)
      - 'single': Entire file is one document
    """

    def can_handle(self, path: str) -> bool:
        return path.endswith(".txt")

    def load(
        self,
        path: str,
        mode: str = "lines",
        max_docs: Optional[int] = None,
        encoding: str = "utf-8",
        **kwargs,
    ) -> UnifiedData:
        """Load a text file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Text file not found: {path}")

        with open(path, "r", encoding=encoding) as f:
            if mode == "single":
                texts = [f.read()]
            else:
                texts = [line.strip() for line in f if line.strip()]

        if max_docs is not None:
            texts = texts[:max_docs]

        doc_ids = self._generate_doc_ids(texts)
        token_counts = np.array(
            [self._estimate_token_count(t) for t in texts],
            dtype=np.int64,
        )

        return UnifiedData(
            texts=texts,
            doc_ids=doc_ids,
            token_counts=token_counts,
            metadata={
                "source": path,
                "format": "txt",
                "mode": mode,
                "num_docs": len(texts),
            },
        )

    def save(self, data: UnifiedData, path: str, mode: str = "lines", **kwargs):
        """Save UnifiedData to text file."""
        with open(path, "w", encoding="utf-8") as f:
            if mode == "single":
                f.write("\n".join(data.texts))
            else:
                for t in data.texts:
                    f.write(t + "\n")
