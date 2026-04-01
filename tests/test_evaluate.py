"""Unit tests for src.evaluate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import compute_metrics


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_required_keys():
    """All four expected metric keys must be present in the result."""
    y = pd.Series([0, 1, 2, 0, 1, 2])
    metrics = compute_metrics(y, y)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}


def test_compute_metrics_perfect_multiclass():
    """Perfect predictions must yield all metrics equal to 1.0."""
    y = pd.Series([0, 1, 2, 0, 1, 2])
    metrics = compute_metrics(y, y)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_compute_metrics_binary():
    """Metrics must be computed correctly for a binary classification."""
    y_true = pd.Series([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = compute_metrics(y_true, y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_compute_metrics_values_in_valid_range():
    """All metric values must lie in [0.0, 1.0]."""
    y_true = pd.Series([0, 1, 2, 0, 1, 2])
    y_pred = pd.Series([0, 1, 1, 0, 2, 2])
    metrics = compute_metrics(y_true, y_pred)
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"Metric '{name}' out of range: {value}"


def test_compute_metrics_imperfect_predictions():
    """Accuracy must be strictly less than 1.0 when predictions are wrong."""
    y_true = pd.Series([0, 1, 2, 0, 1, 2])
    y_pred = pd.Series([1, 0, 2, 0, 2, 1])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] < 1.0


def test_compute_metrics_returns_floats():
    """All metric values must be Python floats."""
    y = pd.Series([0, 1, 2])
    metrics = compute_metrics(y, y)
    for name, value in metrics.items():
        assert isinstance(value, float), f"Metric '{name}' is not a float: {type(value)}"
