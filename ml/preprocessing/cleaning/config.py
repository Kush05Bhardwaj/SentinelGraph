"""Configuration dataclasses for the preprocessing/cleaning pipeline.

All settings are explicit and serializable so that preprocessing is fully
deterministic and reproducible across training and inference runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ColumnConfig:
    """Per-column transformation settings."""

    name: str
    dtype: Optional[str] = None  # target pandas/numpy dtype, e.g. "float64", "int64", "str"
    normalize_whitespace: bool = True  # strip string whitespace
    lowercase: bool = False  # lowercase string values
    treat_as_identifier: bool = False  # column is an ID; keep but do not transform
    treat_as_label: bool = False  # column is the target label; preserve separately


@dataclass
class MissingValueConfig:
    """Configuration for missing value handling."""

    strategy: str = "impute_median"  # drop | impute_mean | impute_median | impute_mode | impute_constant | none
    constant_value: Any = None
    threshold: float = 0.5  # columns with missing fraction above this are flagged, not silently dropped


@dataclass
class InfinityConfig:
    """Configuration for infinite value handling."""

    replace_with_nan: bool = True
    impute_strategy: str = "median"  # mean | median | constant | none
    constant_value: Any = None


@dataclass
class DuplicateConfig:
    """Configuration for duplicate investigation/handling."""

    investigate: bool = True  # report duplicates without dropping
    drop_exact: bool = False  # drop exact duplicate rows
    subset: Optional[List[str]] = None  # columns to consider for duplicates


@dataclass
class ScalingConfig:
    """Configuration for numerical scaling."""

    method: str = "standard"  # standard | minmax | robust | none
    with_mean: bool = True
    with_std: bool = True


@dataclass
class EncodingConfig:
    """Configuration for categorical encoding."""

    method: str = "label"  # label | onehot | ordinal | none
    handle_unknown: str = "error"  # error | ignore


@dataclass
class PreprocessingConfig:
    """Top-level preprocessing configuration.

    All fields are optional with sensible defaults. The config is deterministic:
    given the same training data and config, fit() produces identical artifacts.
    """

    columns: List[ColumnConfig] = field(default_factory=list)
    label_column: Optional[str] = None
    identifier_columns: List[str] = field(default_factory=list)
    missing_values: MissingValueConfig = field(default_factory=MissingValueConfig)
    infinity: InfinityConfig = field(default_factory=InfinityConfig)
    duplicates: DuplicateConfig = field(default_factory=DuplicateConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    normalize_column_names: bool = True  # strip whitespace from column names
    lowercase_column_names: bool = False
    random_seed: int = 42
    preserve_original_labels: bool = True  # keep a copy of the raw label column
