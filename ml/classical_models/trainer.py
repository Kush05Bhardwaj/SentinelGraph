"""Reusable classical ML training engine.

Provides :class:`ClassicalTrainer`, a scikit-learn-style fit/evaluate/predict
wrapper that is:

* **Leakage-safe:** preprocessing is fitted only on training data (integrates
  with :class:`sentinelgraph.ml.preprocessing.cleaning.PreprocessingTransformer`
  and :class:`sentinelgraph.ml.feature_engineering.FeatureEngineer`).
* **Deterministic:** all random operations use a fixed seed; identical inputs
  produce identical trained models and metrics.
* **Configurable:** model algorithms, hyperparameters, and training settings
  are driven by :class:`ClassicalTrainingConfig`.
* **Serializable:** trained models are saved via joblib alongside experiment
  metadata JSON; they can be reloaded for inference without retraining.
* **Task-aware:** supports both binary and multiclass classification.

Typical usage::

    from ml.classical_models import ClassicalTrainer, ClassicalTrainingConfig, ModelConfig

    config = ClassicalTrainingConfig(
        models=[
            ModelConfig(name="rf", algorithm="random_forest", task="multiclass"),
        ],
        output_dir="artifacts/classical_models",
    )
    trainer = ClassicalTrainer(config)
    result = trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    preds = trainer.predict("rf", X_test)
    trainer.save("rf", "artifacts/classical_models/rf_model.joblib")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from .config import ClassicalTrainingConfig, ModelConfig, TrainingConfig
from .exceptions import (
    ClassicalModelError,
    ModelConfigurationError,
    ModelNotFittedError,
    ModelSerializationError,
    UnsupportedTaskError,
)
from .logger import get_logger
from .metadata import ExperimentMetadata

_logger = get_logger()


def _get_sklearn_version() -> str:
    import sklearn

    return sklearn.__version__


def _get_numpy_version() -> str:
    import numpy

    return numpy.__version__


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    task: str,
    metric_names: list,
) -> Dict[str, float]:
    """Compute classification metrics for a set of predictions."""
    metrics: Dict[str, float] = {}

    if "accuracy" in metric_names:
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    if "precision_macro" in metric_names:
        metrics["precision_macro"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
    if "precision_weighted" in metric_names:
        metrics["precision_weighted"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
    if "recall_macro" in metric_names:
        metrics["recall_macro"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
    if "recall_weighted" in metric_names:
        metrics["recall_weighted"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
    if "f1_macro" in metric_names:
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if "f1_weighted" in metric_names:
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
    if "f1_micro" in metric_names:
        metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    if "roc_auc" in metric_names:
        try:
            if task == "binary":
                if y_prob is not None and y_prob.ndim == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                elif y_prob is not None:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            else:
                if y_prob is not None:
                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            y_true, y_prob, multi_class="ovr", average="macro"
                        )
                    )
        except Exception:
            # ROC-AUC may be undefined for single-class data.
            pass

    return metrics


def _ensure_1d_labels(y: Any) -> np.ndarray:
    """Convert labels to a 1D numpy array."""
    arr = np.asarray(y)
    if arr.ndim > 1:
        arr = arr.ravel()
    return arr


class ClassicalTrainer:
    """Classical ML training engine.

    Parameters
    ----------
    config:
        :class:`ClassicalTrainingConfig` describing models, training settings,
        and output directory.
    """

    def __init__(self, config: ClassicalTrainingConfig) -> None:
        self.config = config
        self._fitted_models: Dict[str, Any] = {}
        self._metadata: Dict[str, ExperimentMetadata] = {}
        self._metrics: Dict[str, Dict[str, float]] = {}
        self._confusion_matrices: Dict[str, np.ndarray] = {}

        os.makedirs(self.config.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        model_name: Optional[str] = None,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        validation_fraction: float = 0.2,
    ) -> Dict[str, Any]:
        """Train one or more models.

        If ``X_val``/``y_val`` are not provided and ``config.training.use_validation``
        is true, a validation split is carved out of the training data.

        Parameters
        ----------
        X:
            Training features (array-like or DataFrame).
        y:
            Training labels (array-like or Series).
        model_name:
            Name of a specific model to train. If None, all configured models
            are trained.
        X_val:
            Optional validation features.
        y_val:
            Optional validation labels.
        validation_fraction:
            Fraction of training data to use for validation when no
            validation split is provided.

        Returns:
            A dict mapping model name to its evaluation metrics.
        """
        models_to_train = self._select_models(model_name)
        y = _ensure_1d_labels(y)

        # Determine the task from the first model if not already set.
        results: Dict[str, Any] = {}

        for model_config in models_to_train:
            _logger.info("Training model: %s (%s)", model_config.name, model_config.algorithm)

            start = time.perf_counter()

            # Split validation if needed.
            if X_val is None and y_val is None and self.config.training.use_validation:
                X_train, X_val_tmp, y_train, y_val_tmp = train_test_split(
                    X, y,
                    test_size=validation_fraction,
                    random_state=model_config.random_seed,
                    stratify=y if len(np.unique(y)) > 1 else None,
                )
            else:
                X_train, y_train = X, y
                X_val_tmp, y_val_tmp = X_val, y_val

            # Initialize and fit the model.
            model = self._initialize(model_config)
            model.fit(X_train, y_train)

            # Evaluate on validation data.
            metrics: Dict[str, float] = {}
            cm: Optional[np.ndarray] = None
            if X_val_tmp is not None and y_val_tmp is not None:
                y_val_tmp = _ensure_1d_labels(y_val_tmp)
                y_pred = model.predict(X_val_tmp)
                y_prob = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_prob = model.predict_proba(X_val_tmp)
                    except Exception:
                        y_prob = None
                metrics = _compute_metrics(
                    y_val_tmp, y_pred, y_prob, model_config.task,
                    self.config.training.metrics,
                )
                cm = confusion_matrix(y_val_tmp, y_pred)

            elapsed = time.perf_counter() - start

            # Store fitted model and metadata.
            self._fitted_models[model_config.name] = model
            self._metrics[model_config.name] = metrics
            if cm is not None:
                self._confusion_matrices[model_config.name] = cm

            experiment_id = self._generate_experiment_id(model_config)
            metadata = ExperimentMetadata(
                experiment_id=experiment_id,
                model_name=model_config.name,
                algorithm=model_config.algorithm,
                task=model_config.task,
                dataset_name=None,
                dataset_version=None,
                feature_schema_version=None,
                random_seed=model_config.random_seed,
                hyperparameters=dict(model_config.hyperparameters),
                framework_versions={
                    "sklearn": _get_sklearn_version(),
                    "numpy": _get_numpy_version(),
                    "joblib": joblib.__version__,
                },
                metrics=metrics,
                artifact_paths=[],
                training_environment="local",
                started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                status="completed",
            )
            self._metadata[model_config.name] = metadata

            _logger.info(
                "Model %s trained in %.2fs | metrics: %s",
                model_config.name, elapsed,
                {k: round(v, 4) for k, v in metrics.items()},
            )

            results[model_config.name] = {
                "metrics": metrics,
                "experiment_id": experiment_id,
                "model": model,
            }

        return results

    def predict(self, model_name: str, X: Any) -> np.ndarray:
        """Generate predictions for a fitted model.

        Parameters
        ----------
        model_name:
            Name of the fitted model.
        X:
            Feature matrix (array-like or DataFrame).

        Returns:
            Predicted labels as a 1D numpy array.
        """
        if model_name not in self._fitted_models:
            raise ModelNotFittedError(
                f"Model '{model_name}' has not been fitted. Call fit() first."
            )
        model = self._fitted_models[model_name]
        preds = model.predict(X)
        return np.asarray(preds)

    def predict_proba(self, model_name: str, X: Any) -> Optional[np.ndarray]:
        """Generate class probabilities for a fitted model.

        Returns None if the model does not support predict_proba.
        """
        if model_name not in self._fitted_models:
            raise ModelNotFittedError(
                f"Model '{model_name}' has not been fitted. Call fit() first."
            )
        model = self._fitted_models[model_name]
        if not hasattr(model, "predict_proba"):
            return None
        return model.predict_proba(X)

    def evaluate(
        self,
        model_name: str,
        X: Any,
        y: Any,
    ) -> Dict[str, float]:
        """Evaluate a fitted model on provided data.

        Returns the same metric dictionary as produced during fit.
        """
        if model_name not in self._fitted_models:
            raise ModelNotFittedError(
                f"Model '{model_name}' has not been fitted. Call fit() first."
            )
        model = self._fitted_models[model_name]
        y = _ensure_1d_labels(y)
        y_pred = model.predict(X)
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X)
            except Exception:
                y_prob = None
        # Determine task from metadata.
        task = self._metadata[model_name].task
        return _compute_metrics(
            y, y_pred, y_prob, task, self.config.training.metrics
        )

    def save(self, model_name: str, path: str) -> str:
        """Save a fitted model to disk via joblib.

        Also writes a sidecar JSON with experiment metadata next to the model
        file (same stem, ``.metadata.json`` extension).

        Parameters
        ----------
        model_name:
            Name of the fitted model.
        path:
            Destination file path (e.g. ``artifacts/rf_model.joblib``).

        Returns:
            The path to the saved model file.
        """
        if model_name not in self._fitted_models:
            raise ModelNotFittedError(
                f"Model '{model_name}' has not been fitted. Call fit() first."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            joblib.dump(self._fitted_models[model_name], path)
        except Exception as exc:
            raise ModelSerializationError(
                f"Failed to save model '{model_name}' to {path}: {exc}"
            ) from exc

        # Write sidecar metadata.
        metadata = self._metadata.get(model_name)
        if metadata is not None:
            metadata.artifact_paths = [str(path)]
            meta_path = path.with_suffix(".metadata.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, indent=2)

        _logger.info("Model '%s' saved to %s", model_name, path)
        return str(path)

    def load(self, model_name: str, path: str) -> Any:
        """Load a fitted model from disk via joblib.

        Parameters
        ----------
        model_name:
            Name to register the loaded model under.
        path:
            Source file path.

        Returns:
            The loaded estimator instance.
        """
        path = Path(path)
        if not path.is_file():
            raise ModelSerializationError(f"Model file not found: {path}")

        try:
            model = joblib.load(path)
        except Exception as exc:
            raise ModelSerializationError(
                f"Failed to load model from {path}: {exc}"
            ) from exc

        self._fitted_models[model_name] = model
        _logger.info("Model '%s' loaded from %s", model_name, path)
        return model

    def get_metrics(self, model_name: str) -> Dict[str, float]:
        """Return the evaluation metrics for a fitted model."""
        if model_name not in self._metrics:
            raise ModelNotFittedError(
                f"No metrics recorded for model '{model_name}'."
            )
        return dict(self._metrics[model_name])

    def get_metadata(self, model_name: str) -> ExperimentMetadata:
        """Return the experiment metadata for a fitted model."""
        if model_name not in self._metadata:
            raise ModelNotFittedError(
                f"No metadata recorded for model '{model_name}'."
            )
        return self._metadata[model_name]

    def list_fitted_models(self) -> list:
        """Return names of all fitted models."""
        return list(self._fitted_models.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_models(self, model_name: Optional[str]) -> List[ModelConfig]:
        if model_name is not None:
            for m in self.config.models:
                if m.name == model_name:
                    return [m]
            raise ModelConfigurationError(
                f"Model '{model_name}' not found in config. "
                f"Available: {[m.name for m in self.config.models]}"
            )
        if not self.config.models:
            raise ModelConfigurationError(
                "No models configured. Add ModelConfig entries to "
                "ClassicalTrainingConfig.models."
            )
        return list(self.config.models)

    def _initialize(self, config: ModelConfig) -> Any:
        from .models import initialize_model
        return initialize_model(config)

    @staticmethod
    def _generate_experiment_id(config: ModelConfig) -> str:
        import hashlib

        payload = f"{config.name}|{config.algorithm}|{config.random_seed}|{time.time()}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def train_single_model(
    X: Any,
    y: Any,
    *,
    model_name: str = "model",
    algorithm: str = "random_forest",
    task: str = "binary",
    hyperparameters: Optional[Dict[str, Any]] = None,
    output_dir: str = "artifacts/classical_models",
    random_seed: int = 42,
) -> Tuple[Any, Dict[str, float]]:
    """Convenience function to train and evaluate a single model.

    Parameters
    ----------
    X:
        Training features.
    y:
        Training labels.
    model_name:
        Name for the model.
    algorithm:
        Algorithm name (e.g. ``"random_forest"``).
    task:
        ``"binary"`` or ``"multiclass"``.
    hyperparameters:
        Optional hyperparameters dict.
    output_dir:
        Directory for model artifacts.
    random_seed:
        Random seed for reproducibility.

    Returns:
        A tuple of (fitted model, metrics dict).
    """
    config = ClassicalTrainingConfig(
        models=[
            ModelConfig(
                name=model_name,
                algorithm=algorithm,
                task=task,
                hyperparameters=hyperparameters or {},
                random_seed=random_seed,
            )
        ],
        output_dir=output_dir,
        random_seed=random_seed,
    )
    trainer = ClassicalTrainer(config)
    results = trainer.fit(X, y, model_name=model_name)
    model = trainer._fitted_models[model_name]
    metrics = results[model_name]["metrics"]
    return model, metrics
