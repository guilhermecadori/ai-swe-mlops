"""Validate the Iris dataset schema, types, nulls, and feature ranges using Pandera."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
from sklearn.datasets import load_iris

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_FILE = RAW_DIR / "iris.csv"
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COL = "target"

# ---------------------------------------------------------------------------
# Pandera schema
# ---------------------------------------------------------------------------
# Ranges are slightly wider than the Iris dataset min/max to accommodate
# realistic new data: sepal_length 4.3–7.9, sepal_width 2.0–4.4,
# petal_length 1.0–6.9, petal_width 0.1–2.5.
# ---------------------------------------------------------------------------

IRIS_SCHEMA = DataFrameSchema(
    columns={
        "sepal_length": Column(
            dtype=float,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(4.0),
                Check.less_than_or_equal_to(8.5),
            ],
            description="Sepal length in cm. Expected range: [4.0, 8.5].",
        ),
        "sepal_width": Column(
            dtype=float,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(1.5),
                Check.less_than_or_equal_to(5.0),
            ],
            description="Sepal width in cm. Expected range: [1.5, 5.0].",
        ),
        "petal_length": Column(
            dtype=float,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(0.5),
                Check.less_than_or_equal_to(7.5),
            ],
            description="Petal length in cm. Expected range: [0.5, 7.5].",
        ),
        "petal_width": Column(
            dtype=float,
            nullable=False,
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(3.0),
            ],
            description="Petal width in cm. Expected range: [0.0, 3.0].",
        ),
        "target": Column(
            dtype=int,
            nullable=False,
            checks=Check.isin([0, 1, 2]),
            description="Class label: 0=setosa, 1=versicolor, 2=virginica.",
        ),
    },
    strict=True,        # reject any column not declared above
    coerce=False,       # do not silently cast — types must already be correct
    name="IrisSchema",
)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_iris_dataframe() -> pd.DataFrame:
    """Load the Iris dataset from sklearn as a typed DataFrame.

    Returns:
        DataFrame with four float feature columns and one int target column.
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
    df[TARGET_COL] = iris.target.astype("int64")
    logger.info("Loaded Iris dataset: %d rows, %d columns", len(df), len(df.columns))
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Run Pandera schema validation and return the validated DataFrame.

    Validates:
    - No null values in any column
    - Correct dtypes (float for features, int for target)
    - Feature value ranges within expected bounds
    - No unexpected extra columns (strict mode)

    Args:
        df: Raw Iris DataFrame to validate.

    Returns:
        The validated DataFrame (unchanged if all checks pass).

    Raises:
        pandera.errors.SchemaError: If any validation check fails.
    """
    logger.info("Running schema validation...")
    validated = IRIS_SCHEMA.validate(df, lazy=True)
    logger.info("Validation passed: %d rows, %d columns", len(validated), len(validated.columns))
    return validated


def save_raw(df: pd.DataFrame, path: Path = RAW_FILE) -> Path:
    """Persist the validated dataset to data/raw/iris.csv.

    Args:
        df: Validated DataFrame to save.
        path: Destination file path.

    Returns:
        Path to the saved CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Raw validated data saved to %s", path)
    return path


def run(output_path: Path = RAW_FILE) -> Path:
    """End-to-end validate pipeline stage.

    Loads Iris, validates schema, and writes data/raw/iris.csv.

    Args:
        output_path: Destination for the validated CSV.

    Returns:
        Path to the written CSV.
    """
    df = load_iris_dataframe()
    validated_df = validate(df)
    return save_raw(validated_df, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point — exits with code 1 on validation failure."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Validate Iris dataset schema before preprocessing."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_FILE,
        help="Destination for validated CSV (default: data/raw/iris.csv)",
    )
    args = parser.parse_args()

    try:
        path = run(output_path=args.output)
        logger.info("Stage complete. Output: %s", path)
    except pa.errors.SchemaErrors as exc:  # type: ignore[attr-defined]
        logger.error("Validation FAILED:\n%s", exc.failure_cases.to_string())
        sys.exit(1)


if __name__ == "__main__":
    main()
