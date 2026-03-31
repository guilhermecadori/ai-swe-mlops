"""Unit tests for src.evaluate."""

import numpy as np
import pandas as pd

from src.evaluate import compute_metrics


def test_compute_metrics_binary():
    y_true = pd.Series([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = compute_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_compute_metrics_with_proba():
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.8, 0.15, 0.9])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] == 1.0


def test_perfect_prediction_multiclass():
    y_true = pd.Series([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 1])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
