"""Unit tests for src.preprocess."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.preprocess import FEATURE_NAMES, TARGET_COL, normalize, split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def iris_df() -> pd.DataFrame:
    """Minimal Iris-like DataFrame for fast, filesystem-free tests."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "sepal_length": rng.uniform(4.3, 7.9, n),
            "sepal_width":  rng.uniform(2.0, 4.4, n),
            "petal_length": rng.uniform(1.0, 6.9, n),
            "petal_width":  rng.uniform(0.1, 2.5, n),
            "target":       np.tile([0, 1, 2], n // 3),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Tests: split
# ---------------------------------------------------------------------------


def test_split_row_counts(iris_df):
    """Train + test rows must equal total input rows."""
    train_df, test_df = split(iris_df, test_size=0.2, random_state=42)
    assert len(train_df) + len(test_df) == len(iris_df)


def test_split_test_size_ratio(iris_df):
    """Test split must be approximately 20% of the data."""
    train_df, test_df = split(iris_df, test_size=0.2, random_state=42)
    ratio = len(test_df) / len(iris_df)
    assert abs(ratio - 0.2) < 0.05


def test_split_no_overlap(iris_df):
    """No row must appear in both train and test splits."""
    train_df, test_df = split(iris_df, test_size=0.2, random_state=42)
    combined = pd.merge(train_df, test_df, how="inner")
    assert len(combined) == 0, "Duplicate rows found between train and test splits"


def test_split_preserves_columns(iris_df):
    """Both splits must retain all original columns."""
    train_df, test_df = split(iris_df, test_size=0.2, random_state=42)
    assert list(train_df.columns) == list(iris_df.columns)
    assert list(test_df.columns) == list(iris_df.columns)


def test_split_stratification(iris_df):
    """Each class must appear in both train and test splits."""
    train_df, test_df = split(iris_df, test_size=0.2, random_state=42)
    assert set(train_df[TARGET_COL].unique()) == {0, 1, 2}
    assert set(test_df[TARGET_COL].unique()) == {0, 1, 2}


def test_split_reproducible(iris_df):
    """Same random_state must produce identical splits."""
    train1, test1 = split(iris_df, test_size=0.2, random_state=0)
    train2, test2 = split(iris_df, test_size=0.2, random_state=0)
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


# ---------------------------------------------------------------------------
# Tests: normalize
# ---------------------------------------------------------------------------


def test_normalize_train_mean_near_zero(iris_df):
    """After StandardScaler, feature means on train split must be ~0."""
    train_df, test_df = split(iris_df, test_size=0.2, random_state=42)
    train_s, _ = normalize(train_df, test_df, FEATURE_NAMES)
    means = train_s[FEATURE_NAMES].mean()
    assert (means.abs() < 1e-10).all(), f"Means not near zero: {means.to_dict()}"


def test_normalize_train_std_near_one(iris_df):
    """After StandardScaler, feature std on train split must be ~1."""
    train_df, test_df = split(iris_df, test_size=0.2, random_state=42)
    train_s, _ = normalize(train_df, test_df, FEATURE_NAMES)
    stds = train_s[FEATURE_NAMES].std()
    assert (abs(stds - 1.0) < 0.1).all(), f"Stds not near 1: {stds.to_dict()}"


def test_normalize_preserves_target(iris_df):
    """Normalisation must not alter the target column."""
    train_df, test_df = split(iris_df, test_size=0.2, random_state=42)
    train_s, test_s = normalize(train_df, test_df, FEATURE_NAMES)
    pd.testing.assert_series_equal(
        train_s[TARGET_COL].reset_index(drop=True),
        train_df[TARGET_COL].reset_index(drop=True),
    )


def test_normalize_no_data_leakage(iris_df):
    """Scaler must be fitted only on train; test mean should differ from 0."""
    train_df, test_df = split(iris_df, test_size=0.2, random_state=42)
    _, test_s = normalize(train_df, test_df, FEATURE_NAMES)
    # Test set is transformed with train stats, so its mean won't be 0
    test_means = test_s[FEATURE_NAMES].mean().abs()
    assert (test_means > 0).any(), "Test means are 0 — possible data leakage"


# ---------------------------------------------------------------------------
# Tests: load_iris_dataframe
# ---------------------------------------------------------------------------


def test_load_iris_raises_if_missing(tmp_path):
    """FileNotFoundError must be raised when raw CSV does not exist."""
    from src.preprocess import load_iris_dataframe
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="Run validate.py first"):
        load_iris_dataframe(raw_file=missing)


def test_load_iris_returns_correct_columns(tmp_path):
    """Loaded DataFrame must contain all feature columns and target."""
    from src.preprocess import load_iris_dataframe
    # Write a minimal valid CSV
    df = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2, 0]],
        columns=FEATURE_NAMES + [TARGET_COL],
    )
    path = tmp_path / "iris.csv"
    df.to_csv(path, index=False)
    result = load_iris_dataframe(raw_file=path)
    assert list(result.columns) == FEATURE_NAMES + [TARGET_COL]
