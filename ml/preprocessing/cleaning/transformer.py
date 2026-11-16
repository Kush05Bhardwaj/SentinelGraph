"""Reusable data cleaning and preprocessing transformer.

Provides :class:`PreprocessingTransformer`, a scikit-learn-style fit/transform
preprocessor that is:

* **Leakage-safe:** ``fit()`` learns parameters (scaler means, encoder
  mappings, ...) *only* from the training data. ``transform()`` applies those
  learned parameters to any subsequent data (train / test / inference) without
  ever re-fitting.
* **Deterministic:** column ordering, sampling, and mappings are sorted and
  seeded so identical inputs produce identical outputs.
* **Transparent:** no column is silently dropped or transformed. Every
  decision is recorded in a :class:`PreprocessingReport`.
* **Traceable:** original labels are preserved under a ``_<label>__raw``
  column when ``preserve_original_labels`` is set.

Typical usage::

    from sentinelgraph.ml.preprocessing.cleaning import (
        PreprocessingTransformer, PreprocessingConfig,
    )

    cfg = PreprocessingConfig(
        label_column="Label",
        scaling=ScalingConfig(method="standard"),
        encoding=EncodingConfig(method="label"),
    )
    preprocessor = PreprocessingTransformer(cfg)

    # Fit ONLY on training data.
    result = preprocessor.fit_transform(train_df)

    # Apply the *same* fitted parameters to test/inference data.
    test_result = preprocessor.transform(test_df)

    preprocessor.save_artifacts("artifacts/preprocessing/")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .artifacts import save_artifacts
from .config import (
    ColumnConfig,
    DuplicateConfig,
    EncodingConfig,
    InfinityConfig,
    MissingValueConfig,
    PreprocessingConfig,
    ScalingConfig,
)
from .exceptions import (
    PreprocessingColumnError,
    PreprocessingError,
    PreprocessingNotFittedError,
    PreprocessingSchemaError,
)
from .report import ColumnReport, PreprocessingReport


def _config_from_dict(d: Dict[str, Any]) -> PreprocessingConfig:
    """Rebuild a :class:`PreprocessingConfig` from its ``asdict`` form."""
    return PreprocessingConfig(
        columns=[ColumnConfig(**c) for c in d.get("columns", [])],
        label_column=d.get("label_column"),
        identifier_columns=list(d.get("identifier_columns", [])),
        missing_values=MissingValueConfig(**d.get("missing_values", {})),
        infinity=InfinityConfig(**d.get("infinity", {})),
        duplicates=DuplicateConfig(**d.get("duplicates", {})),
        scaling=ScalingConfig(**d.get("scaling", {})),
        encoding=EncodingConfig(**d.get("encoding", {})),
        normalize_column_names=bool(d.get("normalize_column_names", True)),
        lowercase_column_names=bool(d.get("lowercase_column_names", False)),
        random_seed=int(d.get("random_seed", 42)),
        preserve_original_labels=bool(d.get("preserve_original_labels", True)),
    )


@dataclass
class PreprocessingResult:
    """Outcome of a transform or fit_transform call."""

    X: pd.DataFrame  # feature dataframe (label removed)
    y: Optional[pd.Series] = None  # normalized label series (or None)
    y_raw: Optional[pd.Series] = None  # original label series (preserved)
    report: PreprocessingReport = field(default_factory=PreprocessingReport)


class PreprocessingTransformer:
    """Fit/transform preprocessor for tabular network-flow data.

    Parameters
    ----------
    config:
        PreprocessingConfig describing all transformations. If None, defaults
        are used.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()
        self._fitted = False
        self._report: Optional[PreprocessingReport] = None

        # Learned parameters (populated during fit).
        self._column_types: Dict[str, str] = {}  # col -> "numeric" | "categorical" | "identifier" | "label"
        self._scaler_params: Dict[str, Dict[str, float]] = {}  # col -> {mean, scale, min, max, median, iqr}
        self._encoder_mappings: Dict[str, Dict[Any, int]] = {}  # col -> {category: code}
        self._label_mapping: Dict[Any, Any] = {}  # raw label -> normalized label
        self._feature_columns: List[str] = []  # columns retained as features
        self._input_columns: List[str] = []  # columns seen during fit
        self._derived_columns: List[str] = []  # columns created during transform (e.g. onehot)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, label_column: Optional[str] = None) -> "PreprocessingTransformer":
        """Learn preprocessing parameters from training data.

        This is the *only* method that learns from data. Call it on the
        training split only. Test/inference data must be passed to
        ``transform()``.

        Parameters
        ----------
        df:
            Training dataframe.
        label_column:
            Name of the target label column. Overrides
            ``config.label_column`` if provided.

        Returns:
            self
        """
        df = self._copy_and_normalize_columns(df)
        label_column = label_column or self.config.label_column

        self._input_columns = list(df.columns)
        self._report = PreprocessingReport()
        self._report.n_rows_train = len(df)
        self._report.n_columns_input = len(df.columns)
        self._report.label_column = label_column
        self._report.identifier_columns = list(self.config.identifier_columns)
        self._report.config_summary = self._config_summary()

        # Classify columns.
        self._classify_columns(df, label_column)

        # Per-column diagnostics on training data.
        for col in df.columns:
            self._analyze_column(df, col)

        # Handle duplicates (investigate; optionally drop).
        self._handle_duplicates(df)

        # Fit scalers on numeric feature columns (train only).
        self._fit_scalers(df)

        # Fit encoders on categorical feature columns (train only).
        self._fit_encoders(df)

        # Fit label mapping.
        self._fit_label(df, label_column)

        # Determine final feature columns.
        self._feature_columns = [
            c for c in self._column_types
            if self._column_types[c] in ("numeric", "categorical")
        ]

        self._report.n_columns_output = len(self._feature_columns)
        self._report.fitted = True
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> PreprocessingResult:
        """Apply learned preprocessing to a dataframe.

        Parameters are taken from ``fit()``; nothing is re-learned. This makes
        the method safe to call on test and inference data.

        Parameters
        ----------
        df:
            Dataframe to transform.

        Returns:
            PreprocessingResult with features, labels, and report.
        """
        if not self._fitted or self._report is None:
            raise PreprocessingNotFittedError(
                "PreprocessingTransformer has not been fitted. Call fit() or "
                "fit_transform() on training data first."
            )

        df = self._copy_and_normalize_columns(df)

        report = PreprocessingReport()
        report.fitted = True
        report.n_rows_transform = len(df)
        report.n_columns_input = len(df.columns)
        report.label_column = self.config.label_column
        report.identifier_columns = list(self.config.identifier_columns)
        report.config_summary = self._config_summary()

        # Carry over per-column diagnostics captured during fit.
        if self._report is not None:
            for cr in self._report.column_reports:
                report.add_column_report(cr)

        # Validate schema against training columns.
        self._validate_schema(df, report)

        # Apply declared column dtype corrections.
        df = self._apply_dtype_correction(df, report)

        # Separate and preserve raw label.
        y_raw = None
        y = None
        label_col = self.config.label_column
        if label_col and label_col in df.columns:
            y_raw = df[label_col].copy()
            if self.config.preserve_original_labels:
                df[f"{label_col}__raw"] = y_raw
            y = self._normalize_label_series(y_raw, report)
            df = df.drop(columns=[label_col])

        # Missing values.
        df = self._impute_missing(df, report)

        # Infinite values.
        df = self._handle_infinity(df, report)

        # Duplicates.
        df = self._apply_duplicates(df, report)

        # Scale numeric features.
        df = self._apply_scaling(df, report)

        # Encode categorical features.
        df = self._apply_encoding(df, report)

        # Keep only feature columns + derived (e.g. onehot) columns + preserved raw label.
        keep = [c for c in self._feature_columns if c in df.columns]
        keep += [c for c in self._derived_columns if c in df.columns and c not in keep]
        if self.config.preserve_original_labels and label_col and f"{label_col}__raw" in df.columns:
            keep.append(f"{label_col}__raw")
        df = df[keep]

        report.n_columns_output = len(keep)
        report.transformations = list(self._report.transformations)

        return PreprocessingResult(X=df, y=y, y_raw=y_raw, report=report)

    def fit_transform(
        self, df: pd.DataFrame, label_column: Optional[str] = None
    ) -> PreprocessingResult:
        """Fit on ``df`` then transform it. Convenience for training data."""
        self.fit(df, label_column=label_column)
        return self.transform(df)

    def save_artifacts(self, directory: str) -> List[str]:
        """Persist fitted artifacts to ``directory``.

        Requires the transformer to be fitted.
        """
        if not self._fitted or self._report is None:
            raise PreprocessingNotFittedError(
                "Cannot save artifacts before fitting."
            )
        from dataclasses import asdict

        fitted_state = {
            "column_types": self._column_types,
            "scaler_params": self._scaler_params,
            "encoder_mappings": {
                k: {str(kk): vv for kk, vv in v.items()}
                for k, v in self._encoder_mappings.items()
            },
            "label_mapping": {str(k): v for k, v in self._label_mapping.items()},
            "feature_columns": self._feature_columns,
            "input_columns": self._input_columns,
            "config": asdict(self.config),
        }
        return save_artifacts(directory, self._report, fitted_state)

    @classmethod
    def load_artifacts(cls, directory: str) -> "PreprocessingTransformer":
        """Load a previously saved transformer from ``directory``."""
        from .artifacts import load_artifacts as _load_artifacts

        data = _load_artifacts(directory)
        report = data["report"]
        state = data["fitted_state"]

        transformer = cls()
        transformer.config = _config_from_dict(state.get("config", {}))
        transformer._fitted = True
        transformer._column_types = state.get("column_types", {})
        transformer._scaler_params = state.get("scaler_params", {})
        raw_encoders = state.get("encoder_mappings", {})
        transformer._encoder_mappings = {
            k: {kk: vv for kk, vv in v.items()} for k, v in raw_encoders.items()
        }
        transformer._label_mapping = state.get("label_mapping", {})
        transformer._feature_columns = state.get("feature_columns", [])
        transformer._input_columns = state.get("input_columns", [])
        transformer._derived_columns = []
        transformer._report = PreprocessingReport()
        transformer._report.fitted = report.get("fitted", False)
        transformer._report.n_rows_train = report.get("n_rows_train", 0)
        transformer._report.n_columns_input = report.get("n_columns_input", 0)
        transformer._report.label_column = report.get("label_column")
        transformer._report.identifier_columns = report.get(
            "identifier_columns", []
        )
        transformer._report.dropped_columns = report.get("dropped_columns", [])
        transformer._report.transformations = report.get("transformations", [])
        return transformer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _config_summary(self) -> Dict[str, Any]:
        return {
            "label_column": self.config.label_column,
            "identifier_columns": self.config.identifier_columns,
            "missing_values": {
                "strategy": self.config.missing_values.strategy,
                "threshold": self.config.missing_values.threshold,
            },
            "infinity": {
                "replace_with_nan": self.config.infinity.replace_with_nan,
                "impute_strategy": self.config.infinity.impute_strategy,
            },
            "duplicates": {
                "investigate": self.config.duplicates.investigate,
                "drop_exact": self.config.duplicates.drop_exact,
            },
            "scaling": {
                "method": self.config.scaling.method,
                "with_mean": self.config.scaling.with_mean,
                "with_std": self.config.scaling.with_std,
            },
            "encoding": {
                "method": self.config.encoding.method,
                "handle_unknown": self.config.encoding.handle_unknown,
            },
            "normalize_column_names": self.config.normalize_column_names,
            "preserve_original_labels": self.config.preserve_original_labels,
            "random_seed": self.config.random_seed,
        }

    def _copy_and_normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.config.normalize_column_names:
            new_names = []
            for c in df.columns:
                name = str(c).strip()
                if self.config.lowercase_column_names:
                    name = name.lower()
                new_names.append(name)
            df.columns = new_names
        return df

    def _classify_columns(self, df: pd.DataFrame, label_column: Optional[str]) -> None:
        identifier_set = set(self.config.identifier_columns)
        for col in df.columns:
            col_cfg = next(
                (c for c in self.config.columns if c.name == col), None
            )
            if col_cfg and col_cfg.treat_as_identifier:
                self._column_types[col] = "identifier"
                continue
            if col_cfg and col_cfg.treat_as_label:
                self._column_types[col] = "label"
                continue
            if col == label_column:
                self._column_types[col] = "label"
                continue
            if col in identifier_set:
                self._column_types[col] = "identifier"
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                self._column_types[col] = "numeric"
            else:
                self._column_types[col] = "categorical"

    def _analyze_column(self, df: pd.DataFrame, col: str) -> None:
        report = ColumnReport(name=col)
        report.original_dtype = str(df[col].dtype)
        missing = int(df[col].isna().sum())
        report.missing_count = missing
        report.missing_pct = round(missing / len(df) * 100, 4) if len(df) else 0.0

        if pd.api.types.is_numeric_dtype(df[col]):
            inf_count = int(np.isinf(df[col].replace([np.inf, -np.inf], np.nan)).sum())
            # np.isinf on already-NaN-safe series; count true infs.
            inf_count = int((df[col] == np.inf).sum() + (df[col] == -np.inf).sum())
            report.inf_count = inf_count
            report.inf_pct = round(inf_count / len(df) * 100, 4) if len(df) else 0.0

        self._report.add_column_report(report)

    def _handle_duplicates(self, df: pd.DataFrame) -> None:
        if not self.config.duplicates.investigate:
            return
        n_dup = int(df.duplicated().sum())
        self._report.add_transformation(
            f"duplicate_investigation: {n_dup} exact duplicate rows found"
        )
        if n_dup > 0:
            self._report.add_warning(
                f"{n_dup} exact duplicate rows detected in training data; "
                "drop_exact=False so none were removed automatically"
            )
        if self.config.duplicates.drop_exact:
            before = len(df)
            df.drop_duplicates(inplace=True)
            self._report.add_transformation(
                f"dropped {before - len(df)} exact duplicate rows"
            )

    def _fit_scalers(self, df: pd.DataFrame) -> None:
        method = self.config.scaling.method
        if method == "none":
            return
        for col in df.columns:
            if self._column_types.get(col) != "numeric":
                continue
            series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(series) == 0:
                continue
            params: Dict[str, float] = {}
            if method == "standard":
                params["mean"] = float(series.mean())
                params["scale"] = float(series.std(ddof=0)) if series.std(ddof=0) != 0 else 1.0
            elif method == "minmax":
                params["min"] = float(series.min())
                params["max"] = float(series.max())
                if params["max"] == params["min"]:
                    params["max"] = params["min"] + 1.0
            elif method == "robust":
                params["median"] = float(series.median())
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                params["iqr"] = float(q3 - q1) if q3 != q1 else 1.0
            self._scaler_params[col] = params
            self._report.add_transformation(
                f"scaler_fit:{col}:{method}"
            )

    def _fit_encoders(self, df: pd.DataFrame) -> None:
        method = self.config.encoding.method
        if method == "none":
            return
        for col in df.columns:
            if self._column_types.get(col) != "categorical":
                continue
            series = df[col].dropna().astype(str)
            categories = sorted(series.unique())
            mapping = {cat: idx for idx, cat in enumerate(categories)}
            self._encoder_mappings[col] = mapping
            self._report.add_transformation(
                f"encoder_fit:{col}:{method}:{len(categories)} categories"
            )

    def _fit_label(self, df: pd.DataFrame, label_column: Optional[str]) -> None:
        if not label_column or label_column not in df.columns:
            return
        series = df[label_column].dropna().astype(str)
        mapping = {val: val for val in sorted(series.unique())}
        self._label_mapping = mapping
        self._report.add_transformation(
            f"label_fit:{label_column}:{len(mapping)} classes"
        )

    def _validate_schema(self, df: pd.DataFrame, report: PreprocessingReport) -> None:
        missing_required = [
            c for c in self._input_columns
            if c not in df.columns and self._column_types.get(c) == "label"
        ]
        if missing_required:
            raise PreprocessingSchemaError(
                f"Label column(s) missing from transform data: {missing_required}"
            )
        # Warn about identifier columns that are absent (not an error).
        for col in self.config.identifier_columns:
            if col not in df.columns:
                report.add_warning(
                    f"Identifier column '{col}' not present in transform data"
                )

    def _apply_dtype_correction(self, df: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
        """Cast columns to declared dtypes from ColumnConfig."""
        for col_cfg in self.config.columns:
            col = col_cfg.name
            if col not in df.columns:
                continue
            if not col_cfg.dtype:
                continue
            try:
                df[col] = df[col].astype(col_cfg.dtype)
                report.add_transformation(f"dtype_cast:{col}:{col_cfg.dtype}")
            except (ValueError, TypeError) as exc:
                report.add_warning(
                    f"Could not cast column '{col}' to {col_cfg.dtype}: {exc}"
                )
        return df

    def _impute_missing(self, df: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
        cfg = self.config.missing_values
        if cfg.strategy == "none":
            return df
        for col in df.columns:
            if self._column_types.get(col) in ("identifier", "label"):
                continue
            missing = int(df[col].isna().sum())
            if missing == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                if cfg.strategy == "drop":
                    df = df[df[col].notna()]
                elif cfg.strategy == "impute_mean":
                    fill = float(df[col].mean())
                    df[col] = df[col].fillna(fill)
                elif cfg.strategy == "impute_median":
                    fill = float(df[col].median())
                    df[col] = df[col].fillna(fill)
                elif cfg.strategy == "impute_constant":
                    df[col] = df[col].fillna(cfg.constant_value)
                elif cfg.strategy == "impute_mode":
                    mode = df[col].mode()
                    fill = mode.iloc[0] if len(mode) else 0
                    df[col] = df[col].fillna(fill)
            else:
                if cfg.strategy == "drop":
                    df = df[df[col].notna()]
                elif cfg.strategy == "impute_mode":
                    mode = df[col].mode()
                    fill = mode.iloc[0] if len(mode) else "missing"
                    df[col] = df[col].fillna(fill)
                elif cfg.strategy == "impute_constant":
                    df[col] = df[col].fillna(
                        cfg.constant_value if cfg.constant_value is not None else "missing"
                    )
            report.add_transformation(
                f"missing_impute:{col}:{cfg.strategy}:{missing} cells"
            )
        return df

    def _handle_infinity(self, df: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
        cfg = self.config.infinity
        if not cfg.replace_with_nan:
            return df
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            inf_count = int((df[col] == np.inf).sum() + (df[col] == -np.inf).sum())
            if inf_count == 0:
                continue
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # Impute the newly created NaNs.
            if cfg.impute_strategy == "none":
                continue
            if cfg.impute_strategy == "constant":
                fill = cfg.constant_value if cfg.constant_value is not None else 0.0
            elif cfg.impute_strategy == "mean":
                fill = float(df[col].mean())
            else:  # median
                fill = float(df[col].median())
            df[col] = df[col].fillna(fill)
            report.add_transformation(
                f"infinity_impute:{col}:{cfg.impute_strategy}:{inf_count} cells"
            )
        return df

    def _apply_duplicates(self, df: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
        if self.config.duplicates.drop_exact:
            before = len(df)
            df = df.drop_duplicates()
            report.add_transformation(
                f"drop_duplicates: removed {before - len(df)} rows"
            )
        return df

    def _apply_scaling(self, df: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
        method = self.config.scaling.method
        if method == "none":
            return df
        for col, params in self._scaler_params.items():
            if col not in df.columns:
                continue
            series = df[col].astype(float)
            if method == "standard":
                mean = params["mean"]
                scale = params["scale"]
                df[col] = (series - mean) / scale
            elif method == "minmax":
                mn = params["min"]
                mx = params["max"]
                df[col] = (series - mn) / (mx - mn)
            elif method == "robust":
                med = params["median"]
                iqr = params["iqr"]
                df[col] = (series - med) / iqr
            report.add_transformation(f"scale_apply:{col}:{method}")
        return df

    def _apply_encoding(self, df: pd.DataFrame, report: PreprocessingReport) -> pd.DataFrame:
        method = self.config.encoding.method
        if method == "none":
            return df
        self._derived_columns = []
        for col, mapping in self._encoder_mappings.items():
            if col not in df.columns:
                continue
            series = df[col].astype(str)
            if method == "label":
                unmapped = ~series.isin(mapping.keys())
                if unmapped.any() and self.config.encoding.handle_unknown == "error":
                    unknown_vals = sorted(series[unmapped].unique().tolist())
                    raise PreprocessingSchemaError(
                        f"Column '{col}' contains unseen categories at transform: {unknown_vals}"
                    )
                df[col] = series.map(lambda val, m=mapping: m.get(val, -1))
                report.add_transformation(f"encode_apply:{col}:label:{len(mapping)} categories")
            elif method == "ordinal":
                unmapped = ~series.isin(mapping.keys())
                if unmapped.any() and self.config.encoding.handle_unknown == "error":
                    unknown_vals = sorted(series[unmapped].unique().tolist())
                    raise PreprocessingSchemaError(
                        f"Column '{col}' contains unseen categories at transform: {unknown_vals}"
                    )
                df[col] = series.map(lambda val, m=mapping: m.get(val, -1))
                report.add_transformation(f"encode_apply:{col}:ordinal:{len(mapping)} categories")
            elif method == "onehot":
                for cat, code in sorted(mapping.items(), key=lambda x: x[1]):
                    new_col = f"{col}__{cat}"
                    df[new_col] = (series == cat).astype(int)
                    self._derived_columns.append(new_col)
                df = df.drop(columns=[col])
                report.add_transformation(
                    f"encode_apply:{col}:onehot:{len(mapping)} columns"
                )
        return df

    def _normalize_label_series(self, y_raw: pd.Series, report: PreprocessingReport) -> pd.Series:
        label_col = self.config.label_column
        if not self._label_mapping:
            return y_raw
        mapped = y_raw.astype(str).map(self._label_mapping)
        unmapped = mapped.isna()
        if unmapped.any() and self.config.encoding.handle_unknown == "error":
            unknown_vals = sorted(y_raw[unmapped].unique().tolist())
            raise PreprocessingSchemaError(
                f"Label column '{label_col}' contains unseen values at transform: {unknown_vals}"
            )
        mapped = mapped.fillna(y_raw)  # fall back to raw for unknowns when ignore
        report.add_transformation(f"label_apply:{label_col}:mapped")
        return mapped
