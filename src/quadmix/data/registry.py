"""Data adapter registry — auto-detects file format and selects the right adapter."""

from typing import Dict, List, Optional, Type

from quadmix.data.base import BaseDataAdapter, UnifiedData
from quadmix.data.parquet_adapter import ParquetDataAdapter
from quadmix.data.jsonl_adapter import JSONLDataAdapter
from quadmix.data.csv_adapter import CSVDataAdapter
from quadmix.data.txt_adapter import TxtDataAdapter


# Built-in adapters
_BUILTIN_ADAPTERS: List[Type[BaseDataAdapter]] = [
    ParquetDataAdapter,
    JSONLDataAdapter,
    CSVDataAdapter,
    TxtDataAdapter,
]

_EXTRA_ADAPTERS: Dict[str, Type[BaseDataAdapter]] = {}


def auto_detect_adapter(path: str) -> BaseDataAdapter:
    """Auto-detect the correct adapter for a given file path.

    Checks built-in adapters first, then registered extras.

    Args:
        path: File path to detect adapter for.

    Returns:
        An adapter instance that can handle the file.

    Raises:
        ValueError: If no adapter can handle the file.
    """
    all_adapters = _BUILTIN_ADAPTERS + list(_EXTRA_ADAPTERS.values())

    for adapter_cls in all_adapters:
        adapter = adapter_cls()
        if adapter.can_handle(path):
            return adapter

    raise ValueError(
        f"No adapter found for file: {path}\n"
        f"Supported formats: .parquet, .jsonl, .jsonlines, .ndjson, .csv, .txt"
    )


def get_adapter(path: str, adapter_type: Optional[str] = None) -> BaseDataAdapter:
    """Get an adapter for the given path, optionally specifying type.

    Args:
        path: File path.
        adapter_type: Optional override. One of: 'parquet', 'jsonl', 'csv', 'txt'.
                      If None, auto-detects from extension.

    Returns:
        An adapter instance.
    """
    if adapter_type:
        type_map = {
            "parquet": ParquetDataAdapter,
            "jsonl": JSONLDataAdapter,
            "csv": CSVDataAdapter,
            "txt": TxtDataAdapter,
        }
        if adapter_type not in type_map:
            raise ValueError(f"Unknown adapter type: {adapter_type}. Options: {list(type_map.keys())}")
        return type_map[adapter_type]()

    return auto_detect_adapter(path)
