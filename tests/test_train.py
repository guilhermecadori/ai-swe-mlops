"""Unit tests for src.train."""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.train import build_model


def _fit_predict(model_name: str, params: dict):
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    y = pd.Series([0, 1, 0, 1, 0])
    model = build_model(model_name, params)
    model.fit(X, y)
    return model.predict(X)


def test_build_random_forest():
    model = build_model("random_forest", {"n_estimators": 10, "random_state": 0})
    assert isinstance(model, RandomForestClassifier)


def test_build_logistic_regression():
    model = build_model("logistic_regression", {"random_state": 0, "max_iter": 200})
    assert isinstance(model, LogisticRegression)


def test_build_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("xgboost_turbo", {})


def test_random_forest_predicts():
    preds = _fit_predict("random_forest", {"n_estimators": 5, "random_state": 0})
    assert len(preds) == 5
