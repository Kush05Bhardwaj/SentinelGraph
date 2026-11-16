"""SentinelGraph classical ML models package.

Provides a reusable, configuration-driven training pipeline for classical
machine learning classifiers (logistic regression, random forest, XGBoost,
etc.) supporting both binary and multiclass classification.

Core components:

* :class:`ClassicalTrainer` — fit/evaluate/predict/save/load engine.
* :class:`ClassicalTrainingConfig` — top-level training configuration.
* :class:`ModelConfig` — per-model configuration (algorithm, hyperparameters,
  task, random seed).
* :class:`TrainingConfig` — training loop configuration.
* :class:`ExperimentMetadata` — experiment metadata record.
* :func:`initialize_model` — model factory from a :class:`ModelConfig`.
* :func:`train_single_model` — convenience single-model training helper.
* Exceptions in :mod:`sentinelgraph.ml.classical_models.exceptions`.

Usage::

    from ml.classical_models import ClassicalTrainer, ClassicalTrainingConfig, ModelConfig

    config = ClassicalTrainingConfig(
        models=[
            ModelConfig(name="rf", algorithm="random_forest", task="multiclass"),
            ModelConfig(name="lr", algorithm="logistic_regression", task="binary"),
        ],
    )
    trainer = ClassicalTrainer(config)
    results = trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    preds = trainer.predict("rf", X_test)
    trainer.save("rf", "artifacts/rf_model.joblib")
"""

from __future__ import annotations

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
from .models import (
    MODEL_REGISTRY,
    initialize_model,
    list_available_algorithms,
)
from .trainer import ClassicalTrainer, train_single_model

__all__ = [
    "ClassicalTrainer",
    "ClassicalTrainingConfig",
    "ModelConfig",
    "TrainingConfig",
    "ExperimentMetadata",
    "initialize_model",
    "train_single_model",
    "MODEL_REGISTRY",
    "list_available_algorithms",
    "get_logger",
    "ClassicalModelError",
    "ModelConfigurationError",
    "ModelNotFittedError",
    "ModelSerializationError",
    "UnsupportedTaskError",
]
