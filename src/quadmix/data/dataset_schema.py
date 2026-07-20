"""
DatasetSchema — YAML-driven configuration that maps parquet column names
to algorithm concepts (domain, quality, text, char_count).

Decouples QuaDMix from Essential-Web's hardcoded schema so that arbitrary
datasets can be run with zero code changes.

Usage:
    # From YAML file
    schema = DatasetSchema.from_yaml("schema_stem.yaml")

    # Default (Essential-Web)
    schema = DatasetSchema()

    # Validate against a parquet file
    schema._validate(parquet_columns, parquet_dtypes)

    # Access parsed config
    schema.domain_col        # "category_name"
    schema.quality_cols      # ["category_score", "stem_relevance", ...]
    schema.quality_directions # [True, True, ..., False]  (higher_better)
    schema.text_col          # "text"
    schema.char_count_col    # None (compute from text)
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from quadmix.constants import QUALITY_COLUMNS, QUALITY_NAMES, DOMAIN_NAMES, NUM_DOMAINS


def _parse_quality_cols(raw: list) -> Tuple[List[str], List[bool]]:
    """Parse quality_cols from YAML, supporting both plain strings and
    {name: ..., higher_better: ...} dicts.

    Returns (col_names, higher_better_flags).
    """
    names: List[str] = []
    directions: List[bool] = []
    for entry in raw:
        if isinstance(entry, str):
            names.append(entry)
            directions.append(True)
        elif isinstance(entry, dict):
            names.append(entry["name"])
            directions.append(entry.get("higher_better", True))
        else:
            raise ValueError(f"quality_cols entry must be str or dict, got {type(entry)}: {entry}")
    return names, directions


@dataclass
class DatasetSchema:
    domain_col: str = "domain"
    quality_cols: List[str] = field(default_factory=lambda: list(QUALITY_COLUMNS))
    quality_directions: List[bool] = field(default_factory=lambda: [True] * len(QUALITY_COLUMNS))
    text_col: str = "text"
    char_count_col: Optional[str] = "doc_char_count"
    row_in_shard_col: Optional[str] = "row_in_shard"
    domain_names: Optional[List[str]] = field(default_factory=lambda: list(DOMAIN_NAMES))
    quality_names: Optional[List[str]] = field(default_factory=lambda: list(QUALITY_NAMES))

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetSchema":
        """Load DatasetSchema from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            return cls()

        raw_quality = data.get("quality_cols", list(QUALITY_COLUMNS))
        col_names, directions = _parse_quality_cols(raw_quality)

        return cls(
            domain_col=data.get("domain_col", "domain"),
            quality_cols=col_names,
            quality_directions=directions,
            text_col=data.get("text_col", "text"),
            char_count_col=data.get("char_count_col", "doc_char_count"),
            row_in_shard_col=data.get("row_in_shard_col", "row_in_shard"),
            domain_names=data.get("domain_names"),
            quality_names=data.get("quality_names"),
        )

    def _validate(
        self,
        parquet_columns: List[str],
        parquet_dtypes: Dict[str, str],
    ) -> None:
        """Check that all specified columns exist in the parquet with correct dtype.

        Raises ValueError with available columns if any mismatch.
        """
        missing = []
        dtype_mismatch = []

        required_cols = [self.domain_col] + self.quality_cols
        if self.char_count_col is not None:
            required_cols.append(self.char_count_col)

        for col in required_cols:
            if col not in parquet_columns:
                missing.append(col)
            elif col == self.domain_col:
                pass
            elif col in self.quality_cols:
                dt = parquet_dtypes.get(col, "")
                if not dt.startswith(("float", "double", "Float64", "Float32")):
                    dtype_mismatch.append((col, dt, "float"))
            elif col == self.char_count_col:
                dt = parquet_dtypes.get(col, "")
                if not dt.startswith(("int", "Int64", "Int32", "uint", "UInt64")):
                    dtype_mismatch.append((col, dt, "int"))

        row_warnings = []
        if self.row_in_shard_col is not None and self.row_in_shard_col not in parquet_columns:
            row_warnings.append(self.row_in_shard_col)

        if missing or dtype_mismatch:
            lines = ["Schema 校验失败:"]
            if missing:
                lines.append(f"  以下列不存在于 parquet 中:")
                for col in missing:
                    lines.append(f"    - {col} (缺失)")
            if dtype_mismatch:
                lines.append(f"  以下列 dtype 不匹配:")
                for col, actual, expected in dtype_mismatch:
                    lines.append(f"    - {col}: 实际={actual}, 期望={expected}")

            lines.append("")
            lines.append(f"可用列 ({len(parquet_columns)}):")
            for col in parquet_columns:
                dt = parquet_dtypes.get(col, "?")
                lines.append(f"  {col} ({dt})")
            lines.append("")
            lines.append("请创建 YAML 配置文件并使用 --schema 指定。")
            lines.append("示例:")
            lines.append(f"  domain_col: <one_of_available>")
            lines.append(f"  quality_cols: [<float_cols>]")
            lines.append(f"  text_col: <text_col_if_available>")
            raise ValueError("\n".join(lines))

        if row_warnings:
            import warnings
            for col in row_warnings:
                warnings.warn(
                    f"row_in_shard_col '{col}' 不存在于 parquet 中。"
                    f"将使用位置索引读取文本（较慢）。"
                    f"如无该列，请在 YAML 中设置 row_in_shard_col: null。",
                    UserWarning,
                )

    def needs_text_for_char_count(self) -> bool:
        """True when char_count_col is None and must be computed from text."""
        return self.char_count_col is None

    def needs_text_loading(self) -> bool:
        """True when text column must be loaded during metadata reading
        (either for char_count computation or because it's explicitly needed).
        """
        return self.needs_text_for_char_count()

    def metadata_read_columns(self) -> List[str]:
        """Columns to read from parquet for metadata loading."""
        cols = [self.domain_col] + list(self.quality_cols)
        if self.char_count_col is not None:
            cols.append(self.char_count_col)
        if self.needs_text_for_char_count():
            cols.append(self.text_col)
        return cols

    def text_read_columns(self) -> List[str]:
        """Columns to read from parquet for text/tokenize loading."""
        cols = [self.text_col]
        if self.row_in_shard_col is not None:
            cols.append(self.row_in_shard_col)
        return cols
