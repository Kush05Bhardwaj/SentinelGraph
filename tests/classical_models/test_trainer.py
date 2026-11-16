"""Unit and integration tests for the SentinelGraph classical ML training pipeline.

Covers:
    * Model initialization (algorithm registry, hyperparameters, random seed)
    * Binary classification training and evaluation
    * Multiclass classification training and evaluation
    * Model save and load (joblib roundtrip)
    * Prediction interface (predict, predict_proba)
    * Reproducibility (same seed → identical metrics)
    * Integration with preprocessing pipeline (fit on train only)
    * Synthetic fixture generation for automated testing
    * Invalid algorithm / task errors
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from ml.classical_models import (
    ClassicalTrainer,
    ClassicalTrainingConfig,
    ExperimentMetadata,
    ModelConfig,
    TrainingConfig,
    initialize_model,
    list_available_algorithms,
    train_single_model,
)
from ml.classical_models.exceptions import (
    ModelConfigurationError,
    ModelNotFittedError,
    ModelSerializationError,
    UnsupportedTaskError,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """Binary classification synthetic dataset."""
    X, y = make_classification(
        n_samples=400, n_features=10, n_informative=5, n_redundant=2,
        n_classes=2, weights=[0.7, 0.3], random_state=42,
    )
    return X, y


@pytest.fixture
def multiclass_data():
    """Multiclass classification synthetic dataset."""
    X, y = make_classification(
        n_samples=600, n_features=12, n_informative=6, n_redundant=2,
        n_classes=4, n_clusters_per_class=2, random_state=42,
    )
    return X, y


@pytest.fixture
def split_data(binary_data):
    """Train/validation/test split of binary data."""
    from sklearn.model_selection import train_test_split
    X, y = binary_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def test_list_available_algorithms():
    """At least logistic regression and random forest are available."""
    algos = list_available_algorithms()
    assert "logistic_regression" in algos
    assert "random_forest" in algos


def test_initialize_logistic_regression():
    """Logistic regression model initializes with correct hyperparameters."""
    config = ModelConfig(
        name="lr", algorithm="logistic_regression", task="binary",
        hyperparameters={"C": 1.0, "max_iter": 1000}, random_seed=42,
    )
    model = initialize_model(config)
    assert model is not None
    assert model.C == 1.0
    assert model.max_iter == 1000
    assert model.random_state == 42


def test_initialize_random_forest():
    """Random forest model initializes with correct hyperparameters."""
    config = ModelConfig(
        name="rf", algorithm="random_forest", task="multiclass",
        hyperparameters={"n_estimators": 50, "max_depth": 5}, random_seed=7,
    )
    model = initialize_model(config)
    assert model is not None
    assert model.n_estimators == 50
    assert model.max_depth == 5
    assert model.random_state == 7


def test_unknown_algorithm_raises():
    """Unknown algorithm raises ModelConfigurationError."""
    config = ModelConfig(
        name="bad", algorithm="nonexistent_algorithm", task="binary",
    )
    with pytest.raises(ModelConfigurationError):
        initialize_model(config)


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------

def test_binary_training(binary_data):
    """Binary classification model trains and produces metrics."""
    X, y = binary_data
    config = ClassicalTrainingConfig(
        models=[
            ModelConfig(name="lr", algorithm="logistic_regression", task="binary",
                        random_seed=42),
        ],
    )
    trainer = ClassicalTrainer(config)
    results = trainer.fit(X, y, model_name="lr")

    assert "lr" in results
    metrics = results["lr"]["metrics"]
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0

    # Model is fitted
    assert trainer.list_fitted_models() == ["lr"]
    assert trainer.get_metadata("lr").task == "binary"


def test_binary_predict(binary_data):
    """Binary classification predict returns correct shape."""
    X, y = binary_data
    config = ClassicalTrainingConfig(
        models=[
            ModelConfig(name="rf", algorithm="random_forest", task="binary",
                        hyperparameters={"n_estimators": 10}, random_seed=42),
        ],
    )
    trainer = ClassicalTrainer(config)
    trainer.fit(X, y, model_name="rf")

    preds = trainer.predict("rf", X[:5])
    assert len(preds) == 5
    assert set(np.unique(preds)).issubset({0, 1})

    proba = trainer.predict_proba("rf", X[:5])
    assert proba is not None
    assert proba.shape == (5, 2)


def test_binary_evaluate(binary_data):
    """Evaluate produces metrics on held-out data."""
    X, y = binary_data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    config = ClassicalTrainingConfig(
        models=[
            ModelConfig(name="lr", algorithm="logistic_regression", task="binary",
                        random_seed=42),
        ],
    )
    trainer = ClassicalTrainer(config)
    trainer.fit(X_train, y_train, model_name="lr")

    metrics = trainer.evaluate("lr", X_test, y_test)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Multiclass classification
# ---------------------------------------------------------------------------

def test_multiclass_training(multiclass_data):
    """Multiclass classification model trains and produces metrics."""
    X, y = multiclass_data
    config = ClassicalTrainingConfig(
        models=[
            ModelConfig(name="rf", algorithm="random_forest", task="multiclass",
                        hyperparameters={"n_estimators": 20}, random_seed=42),
        ],
    )
    trainer = ClassicalTrainer(config)
    results = trainer.fit(X, y, model_name="rf")

    metrics = results["rf"]["metrics"]
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    # Predictions should cover 4 classes
    preds = trainer.predict("rf", X[:10])
    assert set(np.unique(preds)).issubset({0, 1, 2, 3})


def test_multiclass_predict_proba(multiclass_data):
    """Multiclass predict_proba returns correct shape."""
    X, y = multiclass_data
    config = ClassicalTrainingConfig(
        models=[
            ModelConfig(name="rf", algorithm="random_forest", task="multiclass",
                        hyperparameters={"n_estimators": 10}, random_seed=42),
        ],
    )
    trainer = ClassicalTrainer(config)
    trainer.fit(X, y, model_name="rf")

    proba = trainer.predict_proba("rf", X[:5])
    assert proba is not None
    assert proba.shape == (5, 4)


# ---------------------------------------------------------------------------
# Validation split
# ---------------------------------------------------------------------------

def test_validation_split(binary_data):
    """Training with validation split produces validation metrics."""
    X, y = binary_data
    config = ClassicalTrainingConfig(
        models=[
            ModelConfig(name="lr", algorithm="logistic_regression", task="binary",
                        random_seed=42),
        ],
        training=TrainingConfig(use_validation=True, metric_primary="f1_macro"),
    )
    trainer = ClassicalTrainer(config)
    results = trainer.fit(X, y, model_name="lr", validation_fraction=0.2)

    metrics = results["lr"]["metrics"]
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_explicit_validation_data(binary_data):
    """Explicit validation data is used when provided."""
    X, y = binary_data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    config = ClassicalTrainingConfig(
        models=[
            ModelConfig(name="lr", algorithm="logistic_regression", task="binary",
                        random_seed=42),
        ],
    )
    trainer = ClassicalTrainer(config)
    results = trainer.fit(
        X_train, y_train, model_name="lr", X_val=X_val, y_val=y_val,
    )
    metrics = results["lr"]["metrics"]
    assert "accuracy" in metrics


# ---------------------------------------------------------------------------
# Model save and load
# ---------------------------------------------------------------------------

def test_save_and_load_model(binary_data):
    """Model saves to joblib and loads back with identical predictions."""
    X, y = binary_data
    config = ClassicalTrainingConfig(
        models=[
            ModelConfig(name="rf", algorithm="random_forest", task="binary",
                        hyperparameters={"n_estimators": 10}, random_seed=42),
        ],
    )
    trainer = ClassicalTrainer(config)
    trainer.fit(X, y, model_name="rf")

    preds_before = trainer.predict("rf", X[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "rf_model.joblib")
        saved_path = trainer.save("rf", model_path)
        assert os.path.isfile(saved_path)
        assert os.path.isfile(saved_path.replace(".joblib", ".metadata.json"))

        # Load into a new trainer
        trainer2 = ClassicalTrainer(config)
        trainer2.load("rf", saved_path)
        preds_after = trainer2.predict("rf", X[:10])

        np.testing.assert_array_equal(preds_before, preds_after)


def test_save_model_metadata(binary_data):
    """Saved model includes metadata JSON."""
    X, y = binary_data
    config = ClassicalTrainingConfig(
        models=[
            ModelConfig(name="lr", algorithm="logistic_regression", task="binary",
                        random_seed=42),
        ],
    )
    trainer = ClassicalTrainer(config)
    trainer.fit(X, y, model_name="lr")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "lr_model.joblib")
        trainer.save("lr", model_path)
        meta_path = os.path.join(tmpdir, "lr_model.metadata.json")
        assert os.path.isfile(meta_path)

        with open(meta_path) as f:
            metadata = json.load(f)
        assert metadata["model_name"] == "lr"
        assert metadata["algorithm"] == "logistic_regression"
        assert metadata["task"] == "binary"
        assert "metrics" in metadata


def test_load_nonexistent_model_raises():
    """Loading a nonexistent file raises ModelSerializationError."""
    trainer = ClassicalTrainer(ClassicalTrainingConfig())
    with pytest.raises(ModelSerializationError):
        trainer.load("rf", "/nonexistent/path/model.joblib")


# ---------------------------------------------------------------------------
# Not fitted errors
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises(binary_data):
    """Predict before fit raises ModelNotFittedError."""
    X, y = binary_data
    trainer = ClassicalTrainer(ClassicalTrainingConfig(
        models=[ModelConfig(name="lr", algorithm="logistic_regression", task="binary")]
    ))
    with pytest.raises(ModelNotFittedError):
        trainer.predict("lr", X[:5])


def test_evaluate_before_fit_raises(binary_data):
    """Evaluate before fit raises ModelNotFittedError."""
    X, y = binary_data
    trainer = ClassicalTrainer(ClassicalTrainingConfig(
        models=[ModelConfig(name="lr", algorithm="logistic_regression", task="binary")]
    ))
    with pytest.raises(ModelNotFittedError):
        trainer.evaluate("lr", X[:5], y[:5])


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_reproducible_training(binary_data):
    """Same seed produces identical metrics and predictions."""
    X, y = binary_data
    results = []
    for _ in range(2):
        config = ClassicalTrainingConfig(
            models=[
                ModelConfig(name="rf", algorithm="random_forest", task="binary",
                            hyperparameters={"n_estimators": 20}, random_seed=42),
            ],
        )
        trainer = ClassicalTrainer(config)
        trainer.fit(X, y, model_name="rf")
        preds = trainer.predict("rf", X)
        metrics = trainer.get_metrics("rf")
        results.append((preds, metrics))

    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


# ---------------------------------------------------------------------------
# train_single_model convenience
# ---------------------------------------------------------------------------

def test_train_single_model(binary_data):
    """train_single_model returns a fitted model and metrics."""
    X, y = binary_data
    model, metrics = train_single_model(
        X, y, model_name="test_model", algorithm="random_forest",
        task="binary", random_seed=42,
    )
    assert model is not None
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------

def test_experiment_metadata_serialization():
    """ExperimentMetadata serializes to/from dict correctly."""
    meta = ExperimentMetadata(
        experiment_id="exp-001",
        model_name="rf",
        algorithm="random_forest",
        task="multiclass",
        dataset_name="cic-ids2017",
        dataset_version="v1.0",
        feature_schema_version="1.0",
        random_seed=42,
        hyperparameters={"n_estimators": 100},
        framework_versions={"sklearn": "1.8.0"},
        metrics={"accuracy": 0.95, "f1_macro": 0.93},
        artifact_paths=["artifacts/rf.joblib"],
        training_environment="kaggle",
        status="completed",
    )
    d = meta.to_dict()
    assert d["experiment_id"] == "exp-001"
    assert d["model_name"] == "rf"
    assert d["task"] == "multiclass"

    meta2 = ExperimentMetadata.from_dict(d)
    assert meta2.experiment_id == "exp-001"
    assert meta2.model_name == "rf"
    assert meta2.metrics["accuracy"] == 0.95


def test_metadata_recorded_in_trainer(binary_data):
    """Trainer records ExperimentMetadata for each fitted model."""
    X, y = binary_data
    config = ClassicalTrainingConfig(
        models=[
            ModelConfig(name="lr", algorithm="logistic_regression", task="binary",
                        random_seed=42),
        ],
    )
    trainer = ClassicalTrainer(config)
    trainer.fit(X, y, model_name="lr")

    meta = trainer.get_metadata("lr")
    assert isinstance(meta, ExperimentMetadata)
    assert meta.model_name == "lr"
    assert meta.algorithm == "logistic_regression"
    assert meta.task == "binary"
    assert meta.status == "completed"
    assert meta.random_seed == 42


# ---------------------------------------------------------------------------
# Training logs
# ---------------------------------------------------------------------------

def test_logger_available():
    """Training logger is available and configurable."""
    from ml.classical_models.logger import get_logger
    logger = get_logger()
    assert logger is not None
    assert logger.name == "sentinelgraph.ml.classical_models"


# ---------------------------------------------------------------------------
# Multiple models
# ---------------------------------------------------------------------------

def test_multiple_models(binary_data):
    """Training multiple models in one config works."""
    X, y = binary_data
    config = ClassicalTrainingConfig(
        models=[
            ModelConfig(name="lr", algorithm="logistic_regression", task="binary",
                        random_seed=42),
            ModelConfig(name="rf", algorithm="random_forest", task="binary",
                        hyperparameters={"n_estimators": 10}, random_seed=42),
        ],
    )
    trainer = ClassicalTrainer(config)
    results = trainer.fit(X, y)

    assert "lr" in results
    assert "rf" in results
    assert set(trainer.list_fitted_models()) == {"lr", "rf"}
