"""Leakage-safe dataset splitting engine.

Provides :class:`Splitter`, a deterministic train/validation/test splitter
designed to prevent data leakage across partitions for network-flow data.

Design principles (SG-011):
* **No group leakage:** when a capture/session identifier is available, all
  rows sharing that identifier are assigned to the same split (group-based
  splitting).
* **Temporal ordering:** when timestamps are reliable, rows are split
  chronologically so the model never sees the future during training.
* **Duplicate overlap detection:** exact duplicate content is checked across
  splits and reported (optionally raised as an error).
* **Class distribution reporting:** per-split label counts are recorded in the
  manifest.
* **Reproducible seeds:** all random operations use a fixed seed.
* **Preprocessing boundary:** the splitter returns raw index-based partitions;
  preprocessing must be fitted only on the training split and applied to
  validation/test (see SG-009).

Typical usage::

    from sentinelgraph.ml.splitting import Splitter, SplitConfig

    cfg = SplitConfig(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        random_seed=42, strategy="group",
        group_column="Flow ID", label_column="Label",
    )
    splitter = Splitter(cfg)
    result = splitter.split(df)   # df: ingested/cleaned flow records

    train_df = result.train_df
    # Fit preprocessing ONLY on train_df (SG-009), then transform val/test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import SplitConfig
from .exceptions import SplitConfigError, SplitError, SplitLeakageError
from .manifest import SplitManifest


@dataclass
class SplitResult:
    """Outcome of a split call."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    manifest: SplitManifest


class Splitter:
    """Deterministic, leakage-safe dataset splitter."""

    def __init__(self, config: SplitConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, df: pd.DataFrame) -> SplitResult:
        """Split ``df`` into train/validation/test partitions.

        Parameters
        ----------
        df:
            Input dataframe (ingested and optionally cleaned flow records).

        Returns:
            SplitResult containing the three partitions and a manifest.
        """
        config = self.config
        df = df.reset_index(drop=True)

        if config.strategy == "group":
            train_idx, val_idx, test_idx = self._split_group(df)
        elif config.strategy == "temporal":
            train_idx, val_idx, test_idx = self._split_temporal(df)
        else:
            train_idx, val_idx, test_idx = self._split_random(df)

        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)
        test_df = df.loc[test_idx].reset_index(drop=True)

        manifest = self._build_manifest(
            df, train_idx, val_idx, test_idx, train_df, val_df, test_df
        )

        return SplitResult(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            manifest=manifest,
        )

    # ------------------------------------------------------------------
    # Splitting strategies
    # ------------------------------------------------------------------

    def _split_random(
        self, df: pd.DataFrame
    ) -> Tuple[List[int], List[int], List[int]]:
        config = self.config
        n = len(df)
        n_train = int(round(config.train_ratio * n))
        n_val = int(round(config.val_ratio * n))
        n_test = n - n_train - n_val

        if config.label_column and config.label_column in df.columns:
            return self._stratified_random_split(df, n_train, n_val, n_test)

        order = self._rng.permutation(n).tolist()
        train_idx = order[:n_train]
        val_idx = order[n_train : n_train + n_val]
        test_idx = order[n_train + n_val :]
        return train_idx, val_idx, test_idx

    def _stratified_random_split(
        self, df: pd.DataFrame, n_train: int, n_val: int, n_test: int
    ) -> Tuple[List[int], List[int], List[int]]:
        label_col = self.config.label_column
        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []

        for _, group in df.groupby(label_col, sort=False):
            indices = group.index.tolist()
            self._rng.shuffle(indices)
            g = len(indices)
            g_train = max(1, int(round(self.config.train_ratio * g))) if g > 0 else 0
            g_val = max(0, int(round(self.config.val_ratio * g)))
            g_test = g - g_train - g_val
            # Ensure non-negative.
            if g_test < 0:
                g_test = 0
                g_val = max(0, g - g_train)
            train_idx.extend(indices[:g_train])
            val_idx.extend(indices[g_train : g_train + g_val])
            test_idx.extend(indices[g_train + g_val :])

        return train_idx, val_idx, test_idx

    def _split_group(
        self, df: pd.DataFrame
    ) -> Tuple[List[int], List[int], List[int]]:
        config = self.config
        group_col = config.group_column
        if group_col not in df.columns:
            raise SplitConfigError(
                f"Group column '{group_col}' not found in dataframe"
            )

        # Map each group to its row indices.
        groups: Dict[Any, List[int]] = {}
        for idx, val in df[group_col].items():
            groups.setdefault(val, []).append(int(idx))

        group_keys = sorted(groups.keys(), key=lambda g: str(g))
        self._rng.shuffle(group_keys)

        n_groups = len(group_keys)
        n_train = max(1, int(round(config.train_ratio * n_groups))) if n_groups > 0 else 0
        n_val = max(0, int(round(config.val_ratio * n_groups)))
        n_test = n_groups - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = max(0, n_groups - n_train)

        train_groups = group_keys[:n_train]
        val_groups = group_keys[n_train : n_train + n_val]
        test_groups = group_keys[n_train + n_val :]

        train_idx = [i for g in train_groups for i in groups[g]]
        val_idx = [i for g in val_groups for i in groups[g]]
        test_idx = [i for g in test_groups for i in groups[g]]

        return train_idx, val_idx, test_idx

    def _split_temporal(
        self, df: pd.DataFrame
    ) -> Tuple[List[int], List[int], List[int]]:
        config = self.config
        ts_col = config.timestamp_column
        if ts_col not in df.columns:
            raise SplitConfigError(
                f"Timestamp column '{ts_col}' not found in dataframe"
            )

        # Sort by timestamp (stable), then by original index for determinism.
        df_sorted = df.sort_values(by=[ts_col], kind="mergesort")
        order = df_sorted.index.tolist()

        n = len(order)
        n_train = int(round(config.train_ratio * n))
        n_val = int(round(config.val_ratio * n))
        n_test = n - n_train - n_val

        train_idx = order[:n_train]
        val_idx = order[n_train : n_train + n_val]
        test_idx = order[n_train + n_val :]
        return train_idx, val_idx, test_idx

    # ------------------------------------------------------------------
    # Manifest / leakage checks
    # ------------------------------------------------------------------

    def _build_manifest(
        self,
        df: pd.DataFrame,
        train_idx: List[int],
        val_idx: List[int],
        test_idx: List[int],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> SplitManifest:
        config = self.config

        leakage = self._check_leakage(train_df, val_df, test_df)
        class_dist = self._class_distribution(train_df, val_df, test_df)

        manifest = SplitManifest(
            strategy=config.strategy,
            random_seed=config.random_seed,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            n_train=len(train_df),
            n_val=len(val_df),
            n_test=len(test_df),
            total_rows=len(df),
            train_indices=sorted(train_idx),
            val_indices=sorted(val_idx),
            test_indices=sorted(test_idx),
            train_groups=sorted(
                {str(g) for g in self._group_values(df, train_idx)},
                key=str,
            ),
            val_groups=sorted(
                {str(g) for g in self._group_values(df, val_idx)}, key=str
            ),
            test_groups=sorted(
                {str(g) for g in self._group_values(df, test_idx)}, key=str
            ),
            class_distribution=class_dist,
            leakage_checks=leakage,
            group_column=config.group_column,
            timestamp_column=config.timestamp_column,
            warnings=list(leakage.get("warnings", [])),
        )
        return manifest

    def _group_values(self, df: pd.DataFrame, indices: List[int]) -> List[Any]:
        config = self.config
        if config.group_column and config.group_column in df.columns:
            return df.loc[indices, config.group_column].tolist()
        return []

    def _check_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        config = self.config
        checks: Dict[str, Any] = {"passed": True, "warnings": []}

        # Group overlap: no group value in more than one split.
        if config.group_column and config.group_column in train_df.columns:
            train_groups = set(train_df[config.group_column].astype(str))
            val_groups = set(val_df[config.group_column].astype(str))
            test_groups = set(test_df[config.group_column].astype(str))

            train_val = train_groups & val_groups
            train_test = train_groups & test_groups
            val_test = val_groups & test_groups
            overlaps = {
                "train_val": sorted(train_val),
                "train_test": sorted(train_test),
                "val_test": sorted(val_test),
            }
            any_overlap = any(len(v) > 0 for v in overlaps.values())
            checks["group_overlap"] = overlaps
            checks["group_overlap_passed"] = not any_overlap
            if any_overlap:
                checks["passed"] = False
                checks["warnings"].append(
                    "Group identifiers appear in multiple splits (leakage)"
                )

        # Duplicate overlap: no exact row content in more than one split.
        dup_check = self._check_duplicate_overlap(train_df, val_df, test_df)
        checks.update(dup_check)

        return checks

    @staticmethod
    def _check_duplicate_overlap(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Detect rows with identical content appearing in multiple splits.

        Uses a hash of all column values (excluding the index)."""
        checks: Dict[str, Any] = {"duplicate_overlap_passed": True}

        def row_hash(frame: pd.DataFrame) -> set:
            if len(frame) == 0:
                return set()
            return set(
                pd.util.hash_pandas_object(frame, index=False).tolist()
            )

        train_hashes = row_hash(train_df)
        val_hashes = row_hash(val_df)
        test_hashes = row_hash(test_df)

        train_val = train_hashes & val_hashes
        train_test = train_hashes & test_hashes
        val_test = val_hashes & test_hashes

        checks["duplicate_overlap"] = {
            "train_val": len(train_val),
            "train_test": len(train_test),
            "val_test": len(val_test),
        }
        any_overlap = (
            len(train_val) > 0 or len(train_test) > 0 or len(val_test) > 0
        )
        checks["duplicate_overlap_passed"] = not any_overlap
        if any_overlap:
            checks["warnings"] = list(checks.get("warnings", [])) + [
                "Duplicate rows appear in multiple splits (leakage)"
            ]
            checks["passed"] = False
        return checks

    def _class_distribution(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, int]]:
        config = self.config
        label_col = config.label_column
        dist: Dict[str, Dict[str, int]] = {}
        if not label_col:
            return dist
        for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
            if label_col in frame.columns:
                dist[name] = {
                    str(k): int(v) for k, v in frame[label_col].value_counts().items()
                }
            else:
                dist[name] = {}
        return dist
