"""Load Iris dataset, apply optional normalisation, and persist train/test splits."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PARAMS_FILE = Path("params.yaml")
RAW_FILE = Path("data/raw/iris.csv")
PROCESSED_DIR = Path("data/processed")
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COL = "target"


def load_params(path: Path = PARAMS_FILE) -> dict:
    """Read pipeline parameters from params.yaml.

    Args:
        path: Path to the YAML parameters file.

    Returns:
        Parsed parameters dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def load_iris_dataframe(raw_file: Path = RAW_FILE) -> pd.DataFrame:
    """Load the validated Iris dataset from data/raw/iris.csv.

    Args:
        raw_file: Path to the CSV produced by validate.py.

    Returns:
        DataFrame with feature columns and integer target column.

    Raises:
        FileNotFoundError: If the raw file does not exist.
    """
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw data not found: {raw_file}. Run validate.py first."
        )
    df = pd.read_csv(raw_file)
    logger.info("Loaded Iris dataset: %d samples, %d features", len(df), len(FEATURE_NAMES))
    return df


def split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split preserving the target distribution.

    Args:
        df: Full dataset including the target column.
        test_size: Fraction of samples reserved for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )
    logger.info(
        "Split -> train: %d rows | test: %d rows (test_size=%.0f%%)",
        len(train_df),
        len(test_df),
        test_size * 100,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def normalize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler on train features and apply to both splits.

    The scaler is fitted only on training data to prevent data leakage.

    Args:
        train_df: Training split.
        test_df:  Test split.
        feature_cols: Column names to scale.

    Returns:
        Tuple of (scaled_train_df, scaled_test_df).
    """
    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols].to_numpy())
    test_df[feature_cols] = scaler.transform(test_df[feature_cols].to_numpy())

    logger.info(
        "Features normalised (StandardScaler) — mean: %s",
        np.round(scaler.mean_, 3),
    )
    return train_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist train and test DataFrames to CSV.

    Args:
        train_df: Training split.
        test_df:  Test split.
        output_dir: Directory where CSVs are written.

    Returns:
        Tuple of (train_path, test_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Saved train split -> %s", train_path)
    logger.info("Saved test split  -> %s", test_path)
    return train_path, test_path


def run(
    raw_file: Path = RAW_FILE,
    output_dir: Path = PROCESSED_DIR,
    params_file: Path = PARAMS_FILE,
    normalize_features: bool = False,
) -> tuple[Path, Path]:
    """End-to-end preprocessing pipeline.

    Args:
        raw_file: Path to the validated CSV produced by validate.py.
        output_dir: Destination directory for processed CSVs.
        params_file: Path to params.yaml.
        normalize_features: Whether to apply StandardScaler to features.

    Returns:
        Tuple of (train_path, test_path).
    """
    params = load_params(params_file)
    test_size: float = params["data"]["test_size"]
    random_state: int = params["data"]["random_state"]

    df = load_iris_dataframe(raw_file)
    train_df, test_df = split(df, test_size=test_size, random_state=random_state)

    if normalize_features:
        train_df, test_df = normalize(train_df, test_df, FEATURE_NAMES)

    return save_splits(train_df, test_df, output_dir)


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Preprocess Iris dataset for DVC pipeline.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory for processed CSV files (default: data/processed)",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=PARAMS_FILE,
        help="Path to params.yaml (default: params.yaml)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Apply StandardScaler normalisation to features",
    )
    args = parser.parse_args()

    run(
        output_dir=args.output_dir,
        params_file=args.params,
        normalize_features=args.normalize,
    )


if __name__ == "__main__":
    main()
