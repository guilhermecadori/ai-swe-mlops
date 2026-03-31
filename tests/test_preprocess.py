"""Unit tests for src.preprocess."""

import pandas as pd
import pytest

from src.preprocess import clean, encode_target, scale_features, split_features_target


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 2.0, 4.0, 5.0],
            "feature_b": [10.0, 20.0, 20.0, 40.0, 50.0],
            "label": ["cat", "dog", "dog", "cat", "bird"],
        }
    )


def test_clean_removes_duplicates(sample_df):
    result = clean(sample_df)
    assert len(result) == 4  # one duplicate row removed


def test_encode_target(sample_df):
    df, le = encode_target(sample_df, "label")
    assert df["label"].dtype in (int, "int64", "int32")
    assert set(le.classes_) == {"bird", "cat", "dog"}


def test_split_features_target(sample_df):
    X, y = split_features_target(sample_df, "label")
    assert "label" not in X.columns
    assert y.name == "label"
    assert len(X) == len(y)


def test_scale_features():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    X_test = pd.DataFrame({"a": [4.0], "b": [40.0]})
    X_tr_s, X_te_s, scaler = scale_features(X_train, X_test)
    assert abs(X_tr_s["a"].mean()) < 1e-10
    assert abs(X_tr_s["b"].mean()) < 1e-10
