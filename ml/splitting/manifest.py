"""Split manifest: a machine-readable record of a dataset split.

The manifest documents partition sizes, group assignments, class
distributions, and leakage-check results so that downstream consumers and
auditors can verify that no leakage occurred across splits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SplitManifest:
    """Complete record of a train/validation/test split."""

    strategy: str
    random_seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    n_train: int
    n_val: int
    n_test: int
    total_rows: int
    train_indices: List[int] = field(default_factory=list)
    val_indices: List[int] = field(default_factory=list)
    test_indices: List[int] = field(default_factory=list)
    train_groups: List[Any] = field(default_factory=list)
    val_groups: List[Any] = field(default_factory=list)
    test_groups: List[Any] = field(default_factory=list)
    class_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    leakage_checks: Dict[str, Any] = field(default_factory=dict)
    group_column: Optional[str] = None
    timestamp_column: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "random_seed": self.random_seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "total_rows": self.total_rows,
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "test_indices": self.test_indices,
            "train_groups": [str(g) for g in self.train_groups],
            "val_groups": [str(g) for g in self.val_groups],
            "test_groups": [str(g) for g in self.test_groups],
            "class_distribution": self.class_distribution,
            "leakage_checks": self.leakage_checks,
            "group_column": self.group_column,
            "timestamp_column": self.timestamp_column,
            "warnings": self.warnings,
        }
