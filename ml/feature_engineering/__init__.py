"""Public API for the feature engineering package."""

from __future__ import annotations

from .config import (
    DurationConfig,
    FeatureEngineeringConfig,
    FlagConfig,
    FrequencyConfig,
    PortConfig,
    RateConfig,
)
from .engine import FeatureEngineer, FeatureResult
from .exceptions import (
    FeatureConfigError,
    FeatureEngineeringError,
    FeatureNotFittedError,
    FeatureSchemaError,
)
from .persistence import load_config, save_config
from .report import FeatureReport
from .schema import FeatureEntry, FeatureSchema

__all__ = [
    "FeatureEngineer",
    "FeatureResult",
    "FeatureEngineeringConfig",
    "DurationConfig",
    "RateConfig",
    "FlagConfig",
    "PortConfig",
    "FrequencyConfig",
    "FeatureSchema",
    "FeatureEntry",
    "FeatureReport",
    "FeatureEngineeringError",
    "FeatureNotFittedError",
    "FeatureSchemaError",
    "FeatureConfigError",
    "save_config",
    "load_config",
]
