"""Dataset preparation pipeline orchestrator.

Combines ingestion, schema validation, cleaning, feature engineering, and
leakage-safe splitting into a single deterministic pipeline. Supports both
local execution and Kaggle execution through configurable input/output paths.

Pipeline stages:

1. **Ingest** — read and validate CSV files from the input directory.
2. **Split** — train/validation/test split using the configured strategy
   (random/group/temporal) to prevent data leakage.
3. **Clean** — fit preprocessing on training data only; transform all splits.
4. **Feature-engineer** — fit feature engineering on training data only;
   transform all splits.
5. **Export** — write processed datasets, split manifests, feature schemas,
   label mappings, preprocessing metadata, and quality reports.

Usage::

    from ml.pipeline import PipelineOrchestrator, PipelineConfig

    config = PipelineConfig(
        ingestion=IngestionConfig(input_dir="data/raw/cic-ids2017"),
        splitting=SplitConfig(
            strategy="group", group_column="capture_file", label_column="label"
        ),
        output_dir="artifacts/pipeline",
    )
    orchestrator = PipelineOrchestrator(config)
    result = orchestrator.run()
"""

from __future__ import annotations

import json
import os
import time
import copy as _copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ml.feature_engineering import (
    FeatureEngineer,
    FeatureEngineeringConfig,
    load_config as load_fe_config,
    save_config as save_fe_config,
)
from ml.preprocessing.cleaning import (
    PreprocessingTransformer,
    save_artifacts as save_cleaning_artifacts,
)
from ml.preprocessing.ingestion import CSVIngestor, Schema
from ml.splitting import SplitConfig, Splitter

from .config import IngestionConfig, PipelineConfig
from .exceptions import (
    PipelineArtifactError,
    PipelineConfigError,
    PipelineExecutionError,
)
from .report import PipelineReport, StageReport


@dataclass
class PipelineResult:
    """Outcome of a pipeline run."""

    train_df: pd.DataFrame  # processed training data (features + label)
    val_df: pd.DataFrame  # processed validation data
    test_df: pd.DataFrame  # processed test data
    manifest: Any  # SplitManifest
    feature_schema: Any  # FeatureSchema
    label_mapping: Dict[str, Any]  # raw label → normalized label
    report: PipelineReport


class PipelineOrchestrator:
    """Orchestrates the end-to-end dataset preparation pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._report = PipelineReport(config=config.to_dict())
        self._preprocessor: Optional[PreprocessingTransformer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        """Execute the full preparation pipeline.

        Returns:
            PipelineResult with processed splits and all artifacts.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Ingest
        stage = StageReport(name="ingest", status="completed")
        start = time.perf_counter()
        try:
            combined_df = self._ingest()
            stage.rows_in = len(combined_df)
            stage.rows_out = len(combined_df)
            self._report.add_stage(stage)
            self._report.total_rows = len(combined_df)
        except Exception as exc:  # pragma: no cover - depends on input
            stage.status = "failed"
            stage.error = str(exc)
            self._report.add_stage(stage)
            self._report.error = str(exc)
            raise PipelineExecutionError(f"Ingest stage failed: {exc}") from exc
        finally:
            stage.duration_seconds = time.perf_counter() - start

        # Stage 2: Split
        stage = StageReport(name="split", status="completed")
        start = time.perf_counter()
        try:
            split_result = self._split(combined_df)
            stage.rows_in = len(combined_df)
            stage.rows_out = (
                len(split_result.train_df)
                + len(split_result.val_df)
                + len(split_result.test_df)
            )
            self._report.add_stage(stage)
            self._report.train_rows = len(split_result.train_df)
            self._report.val_rows = len(split_result.val_df)
            self._report.test_rows = len(split_result.test_df)
        except Exception as exc:  # pragma: no cover - depends on input
            stage.status = "failed"
            stage.error = str(exc)
            self._report.add_stage(stage)
            self._report.error = str(exc)
            raise PipelineExecutionError(f"Split stage failed: {exc}") from exc
        finally:
            stage.duration_seconds = time.perf_counter() - start

        train_df = split_result.train_df
        val_df = split_result.val_df
        test_df = split_result.test_df

        # Stage 3: Clean (fit on train, transform all)
        stage = StageReport(name="clean", status="completed")
        start = time.perf_counter()
        try:
            (
                train_clean,
                val_clean,
                test_clean,
                cleaning_report,
                preprocessor,
            ) = self._clean(train_df, val_df, test_df)
            # Store preprocessor for label mapping extraction later.
            self._preprocessor = preprocessor
            stage.rows_in = len(train_df)
            stage.rows_out = len(train_clean.X)
            self._report.add_stage(stage)
        except Exception as exc:  # pragma: no cover - depends on input
            stage.status = "failed"
            stage.error = str(exc)
            self._report.add_stage(stage)
            self._report.error = str(exc)
            raise PipelineExecutionError(f"Clean stage failed: {exc}") from exc
        finally:
            stage.duration_seconds = time.perf_counter() - start

        # Stage 4: Feature-engineer (fit on train, transform all)
        stage = StageReport(name="feature_engineering", status="completed")
        start = time.perf_counter()
        try:
            (
                train_eng,
                val_eng,
                test_eng,
                feature_schema,
                label_mapping,
            ) = self._engineer_features(
                train_clean.X, val_clean.X, test_clean.X,
                train_clean.y_raw, val_clean.y_raw, test_clean.y_raw,
            )
            stage.rows_in = len(train_clean.X)
            stage.rows_out = len(train_eng)
            self._report.add_stage(stage)
        except Exception as exc:  # pragma: no cover - depends on input
            stage.status = "failed"
            stage.error = str(exc) or type(exc).__name__
            self._report.add_stage(stage)
            self._report.error = str(exc) or type(exc).__name__
            raise PipelineExecutionError(
                f"Feature engineering stage failed: {exc}"
            ) from exc
        finally:
            stage.duration_seconds = time.perf_counter() - start

        # Stage 5: Export
        stage = StageReport(name="export", status="completed")
        start = time.perf_counter()
        try:
            artifact_paths = self._export(
                output_dir,
                train_eng, val_eng, test_eng,
                split_result.manifest,
                feature_schema,
                label_mapping,
                cleaning_report,
            )
            for path in artifact_paths:
                self._report.add_artifact(path)
        except Exception as exc:  # pragma: no cover - depends on filesystem
            stage.status = "failed"
            stage.error = str(exc)
            self._report.add_stage(stage)
            self._report.error = str(exc)
            raise PipelineExecutionError(f"Export stage failed: {exc}") from exc
        finally:
            stage.duration_seconds = time.perf_counter() - start

        self._report.add_stage(stage)

        # Write pipeline report
        report_path = output_dir / "pipeline_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self._report.to_dict(), f, indent=2)
        self._report.add_artifact(str(report_path))

        return PipelineResult(
            train_df=train_eng,
            val_df=val_eng,
            test_df=test_eng,
            manifest=split_result.manifest,
            feature_schema=feature_schema,
            label_mapping=label_mapping,
            report=self._report,
        )

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _ingest(self) -> pd.DataFrame:
        """Ingest all CSVs from the input directory and combine them."""
        ingest_cfg = self.config.ingestion
        if not ingest_cfg.input_dir:
            raise PipelineConfigError(
                "ingestion.input_dir must be set for the pipeline"
            )

        schema = None
        if ingest_cfg.schema_required_columns:
            schema = Schema(required_columns=ingest_cfg.schema_required_columns)

        ingestor = CSVIngestor(
            input_dir=ingest_cfg.input_dir,
            schema=schema,
            encoding=ingest_cfg.encoding,
            fallback_encodings=ingest_cfg.fallback_encodings,
            on_malformed=ingest_cfg.on_malformed,
        )

        results, combined = ingestor.ingest_directory(
            pattern=ingest_cfg.pattern,
            recursive=ingest_cfg.recursive,
            combine=True,
        )

        if combined is None or len(combined) == 0:
            raise PipelineExecutionError(
                f"No data ingested from {ingest_cfg.input_dir}"
            )

        # Log per-file stats via the pipeline report.
        for result in results:
            self._report.add_warning(
                f"Ingested {result.file_name}: "
                f"{result.stats.rows_read} rows, "
                f"{result.stats.rows_malformed} malformed"
            )

        return combined

    def _split(self, df: pd.DataFrame) -> Any:
        """Split data into train/validation/test using the configured strategy."""
        splitter = Splitter(self.config.splitting)
        return splitter.split(df)

    def _clean(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple:
        """Clean data: fit preprocessing on train, transform all splits.

        Returns:
            (train_clean, val_clean, test_clean, cleaning_report, preprocessor)
            Each clean item is (X, y, y_raw).
        """
        # Make a mutable copy of the preprocessing config so we can add
        # identifier columns detected from the data without mutating the
        # caller's config.
        preprocessing_cfg = _copy.deepcopy(self.config.preprocessing)

        # Automatically detect and exclude identifier-like columns from
        # encoding/scaling so they are not treated as predictive features.
        identifier_candidates = [
            "Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port",
            "Timestamp", "flow_id", "src_ip", "dst_ip", "src_port",
            "dst_port", "timestamp",
        ]
        detected_identifiers = list(
            set(preprocessing_cfg.identifier_columns)
        )
        for cand in identifier_candidates:
            if cand in train_df.columns and cand not in detected_identifiers:
                detected_identifiers.append(cand)

        # Exclude splitting group/timestamp columns from features as well.
        for col in (
            self.config.splitting.group_column,
            self.config.splitting.timestamp_column,
        ):
            if col and col in train_df.columns and col not in detected_identifiers:
                detected_identifiers.append(col)

        preprocessing_cfg.identifier_columns = detected_identifiers

        preprocessor = PreprocessingTransformer(preprocessing_cfg)

        # Fit only on training data — no leakage.
        train_clean = preprocessor.fit_transform(
            train_df, label_column=preprocessing_cfg.label_column
        )
        val_clean = preprocessor.transform(val_df)
        test_clean = preprocessor.transform(test_df)

        return train_clean, val_clean, test_clean, train_clean.report, preprocessor

    def _engineer_features(
        self,
        train_X: pd.DataFrame,
        val_X: pd.DataFrame,
        test_X: pd.DataFrame,
        train_y_raw: pd.Series,
        val_y_raw: pd.Series,
        test_y_raw: pd.Series,
    ) -> tuple:
        """Engineer features: fit on train, transform all splits.

        Re-attaches the original labels to the engineered features.

        Returns:
            (train_final, val_final, test_final, feature_schema, label_mapping)
            Each final item is a DataFrame with engineered features + label.
        """
        engineer = FeatureEngineer(self.config.feature_engineering)

        # Fit only on training features — no leakage.
        train_fe = engineer.fit_transform(train_X)
        val_fe = engineer.transform(val_X)
        test_fe = engineer.transform(test_X)

        # Re-attach original labels (preserved from cleaning).
        label_col = self.config.preprocessing.label_column
        if label_col and train_y_raw is not None:
            train_final = pd.concat(
                [train_fe.X, train_y_raw.reset_index(drop=True)], axis=1
            )
            val_final = pd.concat(
                [val_fe.X, val_y_raw.reset_index(drop=True)], axis=1
            )
            test_final = pd.concat(
                [test_fe.X, test_y_raw.reset_index(drop=True)], axis=1
            )

            # Rename the label column consistently.
            train_final = train_final.rename(
                columns={train_y_raw.name or label_col: label_col}
            )
            val_final = val_final.rename(
                columns={val_y_raw.name or label_col: label_col}
            )
            test_final = test_final.rename(
                columns={test_y_raw.name or label_col: label_col}
            )
        else:
            train_final = train_fe.X.copy()
            val_final = val_fe.X.copy()
            test_final = test_fe.X.copy()

        # Build label mapping from the preprocessor's learned mapping.
        label_mapping: Dict[str, Any] = {}
        preprocessor = getattr(self, "_preprocessor", None)
        if preprocessor is not None and preprocessor._label_mapping:
            label_mapping = {
                str(k): v for k, v in preprocessor._label_mapping.items()
            }

        return (
            train_final,
            val_final,
            test_final,
            train_fe.schema,
            label_mapping,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(
        self,
        output_dir: Path,
        train_eng: tuple,
        val_eng: tuple,
        test_eng: tuple,
        manifest: Any,
        feature_schema: Any,
        label_mapping: Dict[str, Any],
        cleaning_report: Any,
    ) -> List[str]:
        """Export all pipeline artifacts.

        Returns the list of written file paths.
        """
        written: List[str] = []

        processed_dir = output_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        if self.config.export_processed_splits:
            train_path = processed_dir / "train.csv"
            val_path = processed_dir / "val.csv"
            test_path = processed_dir / "test.csv"

            train_eng.to_csv(train_path, index=False)
            val_eng.to_csv(val_path, index=False)
            test_eng.to_csv(test_path, index=False)

            written.extend([str(train_path), str(val_path), str(test_path)])

        # Split manifest.
        manifest_path = output_dir / "split_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        written.append(str(manifest_path))

        # Feature schema.
        schema_path = output_dir / "feature_schema.json"
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(feature_schema.to_dict(), f, indent=2)
        written.append(str(schema_path))

        # Label mapping.
        label_mapping_path = output_dir / "label_mapping.json"
        with open(label_mapping_path, "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, indent=2)
        written.append(str(label_mapping_path))

        # Preprocessing metadata (cleaning report + config).
        preproc_meta: Dict[str, Any] = {
            "config": asdict(self.config.preprocessing),
            "report": cleaning_report.to_dict()
            if hasattr(cleaning_report, "to_dict")
            else cleaning_report,
        }
        preproc_path = output_dir / "preprocessing_metadata.json"
        with open(preproc_path, "w", encoding="utf-8") as f:
            json.dump(preproc_meta, f, indent=2)
        written.append(str(preproc_path))

        # Dataset quality report.
        if self.config.export_quality_report:
            quality_report = self._build_quality_report(
                train_eng, val_eng, test_eng, manifest
            )
            quality_path = output_dir / "data_quality_report.json"
            with open(quality_path, "w", encoding="utf-8") as f:
                json.dump(quality_report, f, indent=2)
            written.append(str(quality_path))

        return written

    def _build_quality_report(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        manifest: Any,
    ) -> Dict[str, Any]:
        """Build a dataset quality report from the processed splits."""
        label_col = self.config.preprocessing.label_column

        report: Dict[str, Any] = {
            "total_rows": int(len(train_df) + len(val_df) + len(test_df)),
            "split_rows": {
                "train": int(len(train_df)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
            },
            "feature_count": int(train_df.shape[1]),
            "class_distribution": {},
            "missing_values": {},
            "leakage_checks": manifest.leakage_checks
            if hasattr(manifest, "leakage_checks")
            else {},
        }

        if label_col:
            for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
                if label_col in frame.columns:
                    report["class_distribution"][name] = {
                        str(k): int(v)
                        for k, v in frame[label_col].value_counts().items()
                    }

        for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
            missing = frame.isna().sum()
            report["missing_values"][name] = {
                "total_missing_cells": int(missing.sum()),
                "columns_with_missing": int((missing > 0).sum()),
            }

        return report
