"""Tests for src.train — model building, predictions, and quality thresholds."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from src.train import FEATURE_NAMES, build_model, compute_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = Path("models/model.pkl")
TEST_DATA_PATH = Path("data/processed/test.csv")
TARGET_COL = "target"
ACCURACY_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load test split from data/processed/test.csv.

    Skips all dependent tests if the file does not exist (pipeline not run yet).
    """
    if not TEST_DATA_PATH.exists():
        pytest.skip(f"Test data not found at {TEST_DATA_PATH}. Run `dvc repro` first.")
    df = pd.read_csv(TEST_DATA_PATH)
    X_test = df.drop(columns=[TARGET_COL])
    y_test = df[TARGET_COL]
    return X_test, y_test


@pytest.fixture(scope="session")
def trained_model():
    """Load the trained model from models/model.pkl.

    Skips all dependent tests if the file does not exist (pipeline not run yet).
    """
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}. Run `dvc repro` first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def iris_sample() -> tuple[pd.DataFrame, pd.Series]:
    """Small in-memory Iris-like dataset for fast unit tests (no I/O needed)."""
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
    y = pd.Series(iris.target, name=TARGET_COL)
    return X, y


# ---------------------------------------------------------------------------
# Tests: model accuracy threshold
# ---------------------------------------------------------------------------


def test_model_accuracy_threshold(trained_model, test_data):
    """Trained model must achieve accuracy > 0.8 on the held-out test set."""
    X_test, y_test = test_data
    y_pred = trained_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > ACCURACY_THRESHOLD, (
        f"Accuracy {accuracy:.4f} is below the required threshold of {ACCURACY_THRESHOLD}."
    )


def test_model_accuracy_threshold_rf(iris_sample):
    """RandomForest trained on full Iris must exceed accuracy threshold."""
    X, y = iris_sample
    model = build_model("random_forest", {"n_estimators": 50, "random_state": 42})
    model.fit(X, y)
    accuracy = accuracy_score(y, model.predict(X))
    assert accuracy > ACCURACY_THRESHOLD


def test_model_accuracy_threshold_xgb(iris_sample):
    """XGBoost trained on full Iris must exceed accuracy threshold."""
    X, y = iris_sample
    model = build_model("xgboost", {"n_estimators": 50, "max_depth": 3, "random_state": 42})
    model.fit(X, y)
    accuracy = accuracy_score(y, model.predict(X))
    assert accuracy > ACCURACY_THRESHOLD


# ---------------------------------------------------------------------------
# Tests: output shape
# ---------------------------------------------------------------------------


def test_model_output_shape(trained_model, test_data):
    """Predictions array must have one entry per test sample."""
    X_test, y_test = test_data
    y_pred = trained_model.predict(X_test)
    assert len(y_pred) == len(y_test), (
        f"Expected {len(y_test)} predictions, got {len(y_pred)}."
    )


def test_model_output_shape_rf(iris_sample):
    """RF predictions shape matches input row count."""
    X, y = iris_sample
    model = build_model("random_forest", {"n_estimators": 10, "random_state": 0})
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


def test_model_output_shape_xgb(iris_sample):
    """XGBoost predictions shape matches input row count."""
    X, y = iris_sample
    model = build_model("xgboost", {"n_estimators": 10, "max_depth": 3, "random_state": 0})
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


def test_model_output_values_are_valid_classes(trained_model, test_data):
    """All predicted class labels must belong to the known set {0, 1, 2}."""
    X_test, _ = test_data
    y_pred = trained_model.predict(X_test)
    assert set(np.unique(y_pred)).issubset({0, 1, 2}), (
        f"Unexpected predicted classes: {set(np.unique(y_pred))}"
    )


# ---------------------------------------------------------------------------
# Tests: required features
# ---------------------------------------------------------------------------


def test_required_features_present_in_test_data(test_data):
    """Test CSV must contain all four expected Iris feature columns."""
    X_test, _ = test_data
    missing = [col for col in FEATURE_NAMES if col not in X_test.columns]
    assert not missing, f"Missing required feature columns: {missing}"


def test_required_features_order(test_data):
    """Feature columns must appear in the canonical order expected by the model."""
    X_test, _ = test_data
    actual = list(X_test.columns)
    assert actual == FEATURE_NAMES, (
        f"Column order mismatch.\n  Expected: {FEATURE_NAMES}\n  Got:      {actual}"
    )


def test_required_features_no_nulls(test_data):
    """Test set must contain no missing values in feature columns."""
    X_test, _ = test_data
    null_counts = X_test[FEATURE_NAMES].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    assert not cols_with_nulls, f"Null values found in features: {cols_with_nulls}"


def test_model_rejects_wrong_features(iris_sample):
    """Model must raise an error when given columns with wrong names."""
    X, y = iris_sample
    model = build_model("random_forest", {"n_estimators": 5, "random_state": 0})
    model.fit(X, y)
    X_wrong = X.rename(columns={"sepal_length": "wrong_col"})
    with pytest.raises((ValueError, KeyError)):
        model.predict(X_wrong)


# ---------------------------------------------------------------------------
# Tests: build_model
# ---------------------------------------------------------------------------


def test_build_random_forest_returns_correct_type():
    model = build_model("random_forest", {"n_estimators": 10, "random_state": 0})
    assert isinstance(model, RandomForestClassifier)


def test_build_xgboost_returns_correct_type():
    model = build_model("xgboost", {"n_estimators": 10, "max_depth": 3, "random_state": 0})
    assert isinstance(model, XGBClassifier)


def test_build_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_model("logistic_regression", {})


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect_predictions():
    y = pd.Series([0, 1, 2, 0, 1, 2])
    metrics = compute_metrics(y, y)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_compute_metrics_keys():
    y = pd.Series([0, 1, 2])
    metrics = compute_metrics(y, y)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}


def test_compute_metrics_values_in_range():
    y_true = pd.Series([0, 1, 2, 0, 1, 2])
    y_pred = pd.Series([0, 1, 1, 0, 2, 2])
    metrics = compute_metrics(y_true, y_pred)
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"Metric '{name}' out of range: {value}"
