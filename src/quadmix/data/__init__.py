"""Data adapter module for QuaDMix — unified dataset loading from various formats."""

from quadmix.data.base import BaseDataAdapter, UnifiedData
from quadmix.data.parquet_adapter import ParquetDataAdapter
from quadmix.data.jsonl_adapter import JSONLDataAdapter
from quadmix.data.csv_adapter import CSVDataAdapter
from quadmix.data.txt_adapter import TxtDataAdapter
from quadmix.data.registry import get_adapter, auto_detect_adapter

__all__ = [
    "BaseDataAdapter",
    "UnifiedData",
    "ParquetDataAdapter",
    "JSONLDataAdapter",
    "CSVDataAdapter",
    "TxtDataAdapter",
    "get_adapter",
    "auto_detect_adapter",
]
