"""Feature engineering engine for network flow data.

Provides :class:`FeatureEngineer`, a fit/transform preprocessor that derives
network-flow features from raw CSV ingest output.

Design principles (SG-010):
* **Raw vs engineered separation:** engineered features live in ``X``;
  raw/passthrough columns live in ``raw``. The two are never mixed silently.
* **No target leakage:** the label column and identifier columns are never
  used as predictive features unless explicitly opted in. Connection-frequency
  lookups are learned *only* during ``fit()`` on training data.
* **Zero-duration / division-by-zero safe:** rates use a configurable epsilon
  fallback when duration is 0 or missing; every division is guarded.
* **No invalid numerical values:** output is sanitized so no ``NaN``/``inf``
  leak into the feature matrix.
* **Stable schema:** the feature set and column order are fixed during
  ``fit()`` and reproduced exactly during ``transform()``.
* **Documented:** every feature has a derivation string in the schema.
* **Deterministic:** identical inputs produce identical outputs; lookups and
  encodings are sorted.

Typical usage::

    from sentinelgraph.ml.feature_engineering import (
        FeatureEngineer, FeatureEngineeringConfig,
    )

    engineer = FeatureEngineer()
    train_result = engineer.fit_transform(train_df)   # learn on train only
    test_result = engineer.transform(test_df)         # apply learned params
    engineer.save_config("artifacts/feature_engineering/")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    DurationConfig,
    FeatureEngineeringConfig,
    FlagConfig,
    FrequencyConfig,
    PortConfig,
    RateConfig,
)
from .exceptions import (
    FeatureEngineeringError,
    FeatureNotFittedError,
    FeatureSchemaError,
)
from .persistence import save_config
from .report import FeatureReport
from .schema import FeatureEntry, FeatureSchema


@dataclass
class FeatureResult:
    """Outcome of a feature-engineering fit/transform call."""

    X: pd.DataFrame  # engineered feature matrix (ML-ready)
    raw: pd.DataFrame  # raw/passthrough columns (identifiers, label, etc.)
    schema: FeatureSchema
    report: FeatureReport


# Candidate column names (CIC-IDS2017 style). The first match in a dataframe
# is used; missing columns are handled gracefully.
_DURATION_CANDIDATES = ["Flow Duration", "flow_duration", "duration"]
_FWD_PACKETS_CANDIDATES = ["Total Fwd Packets", "Total Fwd Packet", "fwd_packets"]
_BWD_PACKETS_CANDIDATES = ["Total Backward Packets", "Total Bwd Packet", "bwd_packets"]
_FWD_BYTES_CANDIDATES = [
    "Total Length of Fwd Packets",
    "Total Length of Fwd Packet",
    "fwd_bytes",
]
_BWD_BYTES_CANDIDATES = [
    "Total Length of Bwd Packets",
    "Total Length of Bwd Packet",
    "bwd_bytes",
]
_PROTOCOL_CANDIDATES = ["Protocol", "protocol"]
_DST_PORT_CANDIDATES = ["Dst Port", "dst_port", "destination_port"]
_SRC_IP_CANDIDATES = ["Src IP", "src_ip", "source_ip"]
_DST_IP_CANDIDATES = ["Dst IP", "dst_ip", "destination_ip"]
_TIMESTAMP_CANDIDATES = ["Timestamp", "timestamp"]
_FLOW_ID_CANDIDATES = ["Flow ID", "flow_id"]

_FLAG_CANDIDATES = {
    "fin": ["FIN Flag Count", "fin_flag_count"],
    "syn": ["SYN Flag Count", "syn_flag_count"],
    "rst": ["RST Flag Count", "rst_flag_count"],
    "psh": ["PSH Flag Count", "psh_flag_count"],
    "ack": ["ACK Flag Count", "ack_flag_count"],
    "urg": ["URG Flag Count", "urg_flag_count"],
}

_LABEL_CANDIDATES = ["Label", "label", "class"]


def _get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column present in ``df``, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_float(series: pd.Series) -> pd.Series:
    """Coerce a series to float, non-numeric -> NaN."""
    return pd.to_numeric(series, errors="coerce")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division guarding division-by-zero.

    Where denominator is 0 or NaN, result is 0. Infinite results are
    replaced with 0.
    """
    denom = denominator.replace(0, np.nan)
    out = numerator / denom
    out = out.replace([np.inf, -np.inf], 0.0)
    out = out.fillna(0.0)
    return out


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN/inf with 0 across all numeric columns."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


class FeatureEngineer:
    """Deterministic network-flow feature engineer.

    ``fit()`` learns connection-frequency and source/destination statistics
    *only* from the supplied training dataframe. ``transform()`` applies those
    learned parameters plus deterministic derivations to any dataframe
    (train / test / inference) without ever re-fitting.
    """

    def __init__(self, config: Optional[FeatureEngineeringConfig] = None) -> None:
        self.config = config or FeatureEngineeringConfig()
        self._fitted = False
        self._schema: Optional[FeatureSchema] = None
        self._report: Optional[FeatureReport] = None

        # Learned lookups (train-only).
        self._pair_freq: Dict[Tuple[str, str], int] = {}
        self._src_freq: Dict[str, int] = {}
        self._dst_ip_freq: Dict[str, int] = {}
        self._src_bytes_sum: Dict[str, float] = {}
        self._dst_bytes_sum: Dict[str, float] = {}
        self._protocol_mapping: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Learn feature configuration and lookups from training data.

        Parameters
        ----------
        df:
            Training dataframe (raw ingested flow records).

        Returns:
            self
        """
        df = df.copy()
        self._schema = FeatureSchema(
            version="1.0",
            random_seed=self.config.random_seed,
            input_columns=list(df.columns),
        )
        self._report = FeatureReport()
        self._report.n_rows = len(df)
        self._report.n_input_columns = len(df.columns)
        self._report.input_columns = list(df.columns)
        self._report.fitted = True

        # Resolve available source columns.
        duration_col = _get_col(df, _DURATION_CANDIDATES)
        fwd_pkts_col = _get_col(df, _FWD_PACKETS_CANDIDATES)
        bwd_pkts_col = _get_col(df, _BWD_PACKETS_CANDIDATES)
        fwd_bytes_col = _get_col(df, _FWD_BYTES_CANDIDATES)
        bwd_bytes_col = _get_col(df, _BWD_BYTES_CANDIDATES)
        protocol_col = _get_col(df, _PROTOCOL_CANDIDATES)
        dst_port_col = _get_col(df, _DST_PORT_CANDIDATES)
        src_ip_col = _get_col(df, _SRC_IP_CANDIDATES)
        dst_ip_col = _get_col(df, _DST_IP_CANDIDATES)

        # Identify label column to exclude from features.
        label_col = _get_col(df, _LABEL_CANDIDATES)

        # Record which expected inputs are missing (informational only).
        expected = [
            ("Flow Duration", _DURATION_CANDIDATES),
            ("Total Fwd Packets", _FWD_PACKETS_CANDIDATES),
            ("Total Backward Packets", _BWD_PACKETS_CANDIDATES),
            ("Total Length of Fwd Packets", _FWD_BYTES_CANDIDATES),
            ("Total Length of Bwd Packets", _BWD_BYTES_CANDIDATES),
            ("Protocol", _PROTOCOL_CANDIDATES),
            ("Dst Port", _DST_PORT_CANDIDATES),
            ("Src IP", _SRC_IP_CANDIDATES),
            ("Dst IP", _DST_IP_CANDIDATES),
        ]
        for name, cands in expected:
            if _get_col(df, cands) is None:
                self._report.missing_input_columns.append(name)

        # Build derived total packets / bytes columns (work columns, not output).
        df["__fwd_packets"] = _to_float(df[fwd_pkts_col]) if fwd_pkts_col else 0.0
        df["__bwd_packets"] = _to_float(df[bwd_pkts_col]) if bwd_pkts_col else 0.0
        df["__fwd_bytes"] = _to_float(df[fwd_bytes_col]) if fwd_bytes_col else 0.0
        df["__bwd_bytes"] = _to_float(df[bwd_bytes_col]) if bwd_bytes_col else 0.0
        df["__total_packets"] = df["__fwd_packets"] + df["__bwd_packets"]
        df["__total_bytes"] = df["__fwd_bytes"] + df["__bwd_bytes"]

        # Flow duration.
        df["__duration"], zero_count = self._derive_duration(df, duration_col)
        self._report.zero_duration_rows = int(zero_count)

        # Learn frequency lookups from training data ONLY.
        if self.config.frequency.enabled:
            self._learn_frequencies(
                df,
                src_ip_col=src_ip_col,
                dst_ip_col=dst_ip_col,
                dst_port_col=dst_port_col,
            )

        # Learn protocol encoding from training data ONLY.
        if protocol_col:
            self._learn_protocol_mapping(df[protocol_col])

        # Build the feature schema (documents every derived feature).
        self._build_schema(
            df,
            duration_col=duration_col,
            fwd_pkts_col=fwd_pkts_col,
            bwd_pkts_col=bwd_pkts_col,
            fwd_bytes_col=fwd_bytes_col,
            bwd_bytes_col=bwd_bytes_col,
            protocol_col=protocol_col,
            dst_port_col=dst_port_col,
            src_ip_col=src_ip_col,
            dst_ip_col=dst_ip_col,
            label_col=label_col,
        )

        # Raw/passthrough columns: identifiers (optionally included) + label.
        raw_cols: List[str] = []
        if self.config.include_raw_identifiers:
            for c in [_SRC_IP_CANDIDATES, _DST_IP_CANDIDATES,
                      _DST_PORT_CANDIDATES, _PROTOCOL_CANDIDATES]:
                col = _get_col(df, c)
                if col and col not in raw_cols:
                    raw_cols.append(col)
        if label_col and label_col not in raw_cols:
            raw_cols.append(label_col)
        self._schema.raw_columns = raw_cols

        # Identifier columns excluded from the feature matrix (leakage guard).
        self._schema.identifier_columns = [
            c for c in [src_ip_col, dst_ip_col, dst_port_col]
            if c and c not in raw_cols
        ]

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> FeatureResult:
        """Apply learned feature engineering to a dataframe.

        Parameters are taken from ``fit()``; nothing is re-learned.

        Parameters
        ----------
        df:
            Dataframe to engineer features for.

        Returns:
            FeatureResult with engineered feature matrix ``X``, raw columns
            ``raw``, the shared ``schema``, and a ``report``.
        """
        if not self._fitted or self._schema is None or self._report is None:
            raise FeatureNotFittedError(
                "FeatureEngineer has not been fitted. Call fit() or "
                "fit_transform() on training data first."
            )

        df = df.copy()
        report = FeatureReport()
        report.n_rows = len(df)
        report.n_input_columns = len(df.columns)
        report.input_columns = list(df.columns)
        report.fitted = True

        # Resolve source columns (same candidates as fit).
        duration_col = _get_col(df, _DURATION_CANDIDATES)
        fwd_pkts_col = _get_col(df, _FWD_PACKETS_CANDIDATES)
        bwd_pkts_col = _get_col(df, _BWD_PACKETS_CANDIDATES)
        fwd_bytes_col = _get_col(df, _FWD_BYTES_CANDIDATES)
        bwd_bytes_col = _get_col(df, _BWD_BYTES_CANDIDATES)
        protocol_col = _get_col(df, _PROTOCOL_CANDIDATES)
        dst_port_col = _get_col(df, _DST_PORT_CANDIDATES)
        src_ip_col = _get_col(df, _SRC_IP_CANDIDATES)
        dst_ip_col = _get_col(df, _DST_IP_CANDIDATES)
        label_col = _get_col(df, _LABEL_CANDIDATES)

        # Derived work columns.
        df["__fwd_packets"] = _to_float(df[fwd_pkts_col]) if fwd_pkts_col else 0.0
        df["__bwd_packets"] = _to_float(df[bwd_pkts_col]) if bwd_pkts_col else 0.0
        df["__fwd_bytes"] = _to_float(df[fwd_bytes_col]) if fwd_bytes_col else 0.0
        df["__bwd_bytes"] = _to_float(df[bwd_bytes_col]) if bwd_bytes_col else 0.0
        df["__total_packets"] = df["__fwd_packets"] + df["__bwd_packets"]
        df["__total_bytes"] = df["__fwd_bytes"] + df["__bwd_bytes"]

        # Flow duration (no re-learning).
        df["__duration"], zero_count = self._derive_duration(df, duration_col)
        report.zero_duration_rows = int(zero_count)

        # Build engineered feature matrix following the fixed schema order.
        engineered: Dict[str, pd.Series] = {}

        # 1. flow duration.
        engineered["flow_duration"] = df["__duration"]

        # 2-5. packet / byte counts.
        engineered["fwd_packets"] = df["__fwd_packets"]
        engineered["bwd_packets"] = df["__bwd_packets"]
        engineered["fwd_bytes"] = df["__fwd_bytes"]
        engineered["bwd_bytes"] = df["__bwd_bytes"]

        # 6-7. totals.
        engineered["total_packets"] = df["__total_packets"]
        engineered["total_bytes"] = df["__total_bytes"]

        # 8-9. rates (guarded against zero duration).
        duration_safe = df["__duration"].replace(0, self.config.duration.zero_duration_fallback)
        engineered["packets_per_sec"] = _safe_divide(
            df["__total_packets"], duration_safe
        )
        engineered["bytes_per_sec"] = _safe_divide(
            df["__total_bytes"], duration_safe
        )
        report.division_guards_applied += int((df["__duration"] == 0).sum())

        # 10. TCP flag counts.
        for flag_name, candidates in _FLAG_CANDIDATES.items():
            col = _get_col(df, candidates)
            engineered[f"flag_{flag_name}"] = (
                _to_float(df[col]) if col else 0.0
            )

        # 11. protocol representation.
        engineered["protocol_code"] = self._encode_protocol(
            df[protocol_col] if protocol_col else pd.Series([""] * len(df))
        )

        # 12-13. destination port.
        if dst_port_col:
            dst_port = _to_float(df[dst_port_col])
            engineered["dst_port"] = dst_port
            engineered["dst_port_category"] = dst_port.apply(self._port_category)
        else:
            engineered["dst_port"] = 0.0
            engineered["dst_port_category"] = 0.0

        # 14-18. connection frequency & communication statistics
        # (lookups learned at fit time; never re-learned here).
        engineered["src_conn_freq"] = self._map_freq(
            df[src_ip_col] if src_ip_col else pd.Series([""] * len(df)),
            self._src_freq,
        )
        engineered["dst_conn_freq"] = self._map_freq(
            df[dst_ip_col] if dst_ip_col else pd.Series([""] * len(df)),
            self._dst_ip_freq,
        )
        if src_ip_col and dst_ip_col:
            pairs = list(zip(df[src_ip_col].astype(str), df[dst_ip_col].astype(str)))
            engineered["pair_conn_freq"] = pd.Series(
                [self._pair_freq.get(p, 0) for p in pairs], index=df.index
            )
        else:
            engineered["pair_conn_freq"] = 0.0

        src_bytes = df["__total_bytes"]
        engineered["src_avg_bytes"] = _safe_divide(
            self._map_freq(df[src_ip_col] if src_ip_col else pd.Series([""] * len(df)),
                           self._src_bytes_sum),
            engineered["src_conn_freq"].replace(0, 1.0),
        )
        engineered["dst_avg_bytes"] = _safe_divide(
            self._map_freq(df[dst_ip_col] if dst_ip_col else pd.Series([""] * len(df)),
                           self._dst_bytes_sum),
            engineered["dst_conn_freq"].replace(0, 1.0),
        )

        X = pd.DataFrame(engineered, index=df.index)
        X = X[self._schema.feature_names()]  # enforce stable schema order
        X = _sanitize(X)  # no NaN/inf

        # Raw/passthrough columns.
        raw_cols = [c for c in self._schema.raw_columns if c in df.columns]
        raw = df[raw_cols].copy() if raw_cols else pd.DataFrame(index=df.index)

        report.n_engineered_features = len(self._schema.feature_names())
        report.transformations = list(self._schema.feature_names())
        report.missing_input_columns = list(self._schema.excluded_leakage_columns)

        return FeatureResult(X=X, raw=raw, schema=self._schema, report=report)

    def fit_transform(self, df: pd.DataFrame) -> FeatureResult:
        """Fit on ``df`` then transform it. Convenience for training data."""
        self.fit(df)
        return self.transform(df)

    def save_config(self, directory: str) -> List[str]:
        """Persist feature schema and learned lookups to ``directory``.

        Requires the engineer to be fitted.
        """
        if not self._fitted or self._schema is None:
            raise FeatureNotFittedError(
                "Cannot save config before fitting."
            )
        lookups = {
            "pair_freq": {f"{a}||{b}": v for (a, b), v in self._pair_freq.items()},
            "src_freq": self._src_freq,
            "dst_ip_freq": self._dst_ip_freq,
            "src_bytes_sum": self._src_bytes_sum,
            "dst_bytes_sum": self._dst_bytes_sum,
            "protocol_mapping": {str(k): v for k, v in self._protocol_mapping.items()},
        }
        return save_config(directory, self._schema, lookups)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_duration(
        self, df: pd.DataFrame, duration_col: Optional[str]
    ) -> Tuple[pd.Series, int]:
        """Derive flow duration; return (series, zero_duration_count)."""
        fallback = self.config.duration.zero_duration_fallback
        if duration_col and duration_col in df.columns:
            duration = _to_float(df[duration_col])
        else:
            duration = pd.Series(0.0, index=df.index)

        duration = duration.fillna(0.0)
        zero_count = int((duration == 0).sum())
        # Replace 0/missing with fallback so rates never divide by zero.
        duration = duration.replace(0, fallback).fillna(fallback)
        return duration, zero_count

    def _learn_frequencies(
        self,
        df: pd.DataFrame,
        src_ip_col: Optional[str],
        dst_ip_col: Optional[str],
        dst_port_col: Optional[str],
    ) -> None:
        """Learn connection-frequency lookups from training data only."""
        if src_ip_col:
            self._src_freq = df[src_ip_col].astype(str).value_counts().to_dict()
        if dst_ip_col:
            self._dst_ip_freq = df[dst_ip_col].astype(str).value_counts().to_dict()
        if src_ip_col and dst_ip_col:
            pairs = list(
                zip(df[src_ip_col].astype(str), df[dst_ip_col].astype(str))
            )
            self._pair_freq = pd.Series(pairs).value_counts().to_dict()
            self._pair_freq = {tuple(k): int(v) for k, v in self._pair_freq.items()}

        # Byte sums per src/dst (for avg-bytes features).
        if "__total_bytes" in df.columns:
            if src_ip_col:
                self._src_bytes_sum = (
                    df.groupby(df[src_ip_col].astype(str))["__total_bytes"]
                    .sum()
                    .to_dict()
                )
            if dst_ip_col:
                self._dst_bytes_sum = (
                    df.groupby(df[dst_ip_col].astype(str))["__total_bytes"]
                    .sum()
                    .to_dict()
                )

    def _learn_protocol_mapping(self, series: pd.Series) -> None:
        """Learn a sorted label-encoding for protocol values (train only)."""
        cats = sorted(series.dropna().astype(str).unique().tolist())
        self._protocol_mapping = {cat: idx for idx, cat in enumerate(cats)}

    def _encode_protocol(self, series: pd.Series) -> pd.Series:
        """Apply learned protocol encoding; unknown -> -1."""
        return series.astype(str).map(
            lambda v: self._protocol_mapping.get(v, -1)
        ).astype(float)

    @staticmethod
    def _port_category(value: float) -> float:
        """Categorize a destination port: 0 well-known, 1 registered, 2 dynamic."""
        try:
            v = int(value)
        except (ValueError, TypeError):
            return 0.0
        if v <= 1023:
            return 0.0
        if v <= 49151:
            return 1.0
        return 2.0

    @staticmethod
    def _map_freq(series: pd.Series, lookup: Dict[Any, int]) -> pd.Series:
        """Map a series through a frequency lookup; missing -> 0."""
        return series.astype(str).map(lambda v: lookup.get(v, 0)).astype(float)

    def _build_schema(
        self,
        df: pd.DataFrame,
        duration_col: Optional[str],
        fwd_pkts_col: Optional[str],
        bwd_pkts_col: Optional[str],
        fwd_bytes_col: Optional[str],
        bwd_bytes_col: Optional[str],
        protocol_col: Optional[str],
        dst_port_col: Optional[str],
        src_ip_col: Optional[str],
        dst_ip_col: Optional[str],
        label_col: Optional[str],
    ) -> None:
        """Document every engineered feature in the schema."""
        entries: List[FeatureEntry] = []

        entries.append(FeatureEntry(
            name="flow_duration",
            derivation=(
                f"From column '{duration_col}' if present, else 0. "
                "Zero/missing replaced with fallback for rate safety."
            ),
            source_columns=[duration_col] if duration_col else [],
            dtype="float64",
            notes="Zero-duration flows are flagged implicitly via rate guards.",
        ))

        entries.append(FeatureEntry(
            name="fwd_packets",
            derivation=f"From column '{fwd_pkts_col}' if present, else 0.",
            source_columns=[fwd_pkts_col] if fwd_pkts_col else [],
            dtype="float64",
        ))
        entries.append(FeatureEntry(
            name="bwd_packets",
            derivation=f"From column '{bwd_pkts_col}' if present, else 0.",
            source_columns=[bwd_pkts_col] if bwd_pkts_col else [],
            dtype="float64",
        ))
        entries.append(FeatureEntry(
            name="fwd_bytes",
            derivation=f"From column '{fwd_bytes_col}' if present, else 0.",
            source_columns=[fwd_bytes_col] if fwd_bytes_col else [],
            dtype="float64",
        ))
        entries.append(FeatureEntry(
            name="bwd_bytes",
            derivation=f"From column '{bwd_bytes_col}' if present, else 0.",
            source_columns=[bwd_bytes_col] if bwd_bytes_col else [],
            dtype="float64",
        ))
        entries.append(FeatureEntry(
            name="total_packets",
            derivation="fwd_packets + bwd_packets",
            source_columns=["fwd_packets", "bwd_packets"],
            dtype="float64",
        ))
        entries.append(FeatureEntry(
            name="total_bytes",
            derivation="fwd_bytes + bwd_bytes",
            source_columns=["fwd_bytes", "bwd_bytes"],
            dtype="float64",
        ))
        entries.append(FeatureEntry(
            name="packets_per_sec",
            derivation="total_packets / flow_duration (duration=0 guarded)",
            source_columns=["total_packets", "flow_duration"],
            dtype="float64",
            is_rate=True,
        ))
        entries.append(FeatureEntry(
            name="bytes_per_sec",
            derivation="total_bytes / flow_duration (duration=0 guarded)",
            source_columns=["total_bytes", "flow_duration"],
            dtype="float64",
            is_rate=True,
        ))

        for flag_name in ["fin", "syn", "rst", "psh", "ack", "urg"]:
            col = _get_col(df, _FLAG_CANDIDATES[flag_name])
            entries.append(FeatureEntry(
                name=f"flag_{flag_name}",
                derivation=f"From column '{col}' if present, else 0.",
                source_columns=[col] if col else [],
                dtype="float64",
            ))

        entries.append(FeatureEntry(
            name="protocol_code",
            derivation=(
                f"Label-encoded '{protocol_col}' learned at fit time; "
                "unknown -> -1."
            ),
            source_columns=[protocol_col] if protocol_col else [],
            dtype="float64",
        ))

        entries.append(FeatureEntry(
            name="dst_port",
            derivation=f"Raw '{dst_port_col}' if present, else 0.",
            source_columns=[dst_port_col] if dst_port_col else [],
            dtype="float64",
        ))
        entries.append(FeatureEntry(
            name="dst_port_category",
            derivation=(
                "0=well-known(0-1023), 1=registered(1024-49151), "
                "2=dynamic(49152-65535) of dst_port."
            ),
            source_columns=["dst_port"],
            dtype="float64",
        ))

        entries.append(FeatureEntry(
            name="src_conn_freq",
            derivation=(
                "Fit-time lookup: number of training flows from this src_ip; "
                "unknown -> 0."
            ),
            source_columns=[src_ip_col] if src_ip_col else [],
            dtype="float64",
            is_frequency=True,
        ))
        entries.append(FeatureEntry(
            name="dst_conn_freq",
            derivation=(
                "Fit-time lookup: number of training flows to this dst_ip; "
                "unknown -> 0."
            ),
            source_columns=[dst_ip_col] if dst_ip_col else [],
            dtype="float64",
            is_frequency=True,
        ))
        entries.append(FeatureEntry(
            name="pair_conn_freq",
            derivation=(
                "Fit-time lookup: number of training flows for (src_ip, "
                "dst_ip) pair; unknown -> 0."
            ),
            source_columns=[src_ip_col, dst_ip_col] if (src_ip_col and dst_ip_col) else [],
            dtype="float64",
            is_frequency=True,
        ))
        entries.append(FeatureEntry(
            name="src_avg_bytes",
            derivation=(
                "Fit-time lookup: src_bytes_sum / src_conn_freq; "
                "conn_freq=0 guarded."
            ),
            source_columns=[src_ip_col] if src_ip_col else [],
            dtype="float64",
            is_frequency=True,
        ))
        entries.append(FeatureEntry(
            name="dst_avg_bytes",
            derivation=(
                "Fit-time lookup: dst_bytes_sum / dst_conn_freq; "
                "conn_freq=0 guarded."
            ),
            source_columns=[dst_ip_col] if dst_ip_col else [],
            dtype="float64",
            is_frequency=True,
        ))

        self._schema.feature_entries = entries
        self._schema.engineered_columns = [e.name for e in entries]
