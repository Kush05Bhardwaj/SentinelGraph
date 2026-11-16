"""Configuration for leakage-safe dataset splitting.

All settings are explicit and serializable so splits are deterministic and
reproducible across runs and environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .exceptions import SplitConfigError


@dataclass
class SplitConfig:
    """Dataset splitting configuration.

    Ratios must sum to 1.0 (train + validation + test). If ``test_ratio`` is
    0.0 the test split will be empty; if ``val_ratio`` is 0.0 the validation
    split will be empty.

    Attributes:
        train_ratio: Fraction of data for training (0.0 - 1.0).
        val_ratio: Fraction of data for validation (0.0 - 1.0).
        test_ratio: Fraction of data for testing (0.0 - 1.0).
        random_seed: Seed for reproducible random splits.
        strategy: Splitting strategy - ``"random"``, ``"group"``, or
            ``"temporal"``.
        group_column: Column name used for group-based splitting. When set,
            all rows sharing a group value are assigned to the same split.
        timestamp_column: Column name used for temporal splitting. When set
            (and strategy is ``"temporal"``), rows are sorted by timestamp
            before splitting.
        label_column: Column name of the target label, used for class
            distribution reporting and (optionally) stratified sampling.
        shuffle: Whether to shuffle within groups before splitting (random
            strategy only).
        min_group_size: Groups smaller than this may be merged into the train
            split to avoid empty val/test groups.
    """

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    strategy: str = "random"
    group_column: Optional[str] = None
    timestamp_column: Optional[str] = None
    label_column: Optional[str] = None
    shuffle: bool = True
    min_group_size: int = 1

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(total - 1.0) < 1e-9:
            raise SplitConfigError(
                f"Split ratios must sum to 1.0, got {total}"
            )
        if self.strategy not in ("random", "group", "temporal"):
            raise SplitConfigError(
                f"Unknown strategy '{self.strategy}'; use 'random', 'group', or 'temporal'"
            )
        if self.strategy == "group" and self.group_column is None:
            raise SplitConfigError(
                "group_column is required when strategy='group'"
            )
        if self.strategy == "temporal" and self.timestamp_column is None:
            raise SplitConfigError(
                "timestamp_column is required when strategy='temporal'"
            )
