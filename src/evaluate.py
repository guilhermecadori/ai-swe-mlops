"""Load a trained model, evaluate on test data, and persist metrics and plots."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "Classification-MLOps"
MODEL_PATH = Path("models/model.pkl")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
METRICS_FILE = REPORTS_DIR / "metrics.json"
CM_FILE = REPORTS_DIR / "confusion_matrix.png"
TARGET_COL = "target"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def load_model(path: Path = MODEL_PATH):
    """Load a pickled scikit-learn model from disk.

    Args:
        path: Path to the .pkl model file.

    Returns:
        Deserialised model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", path)
    return model


def load_test_data(processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Load test split from data/processed/test.csv.

    Args:
        processed_dir: Directory containing test.csv.

    Returns:
        Tuple of (X_test, y_test).

    Raises:
        FileNotFoundError: If test.csv does not exist.
    """
    test_path = processed_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}. Run preprocess.py first.")
    df = pd.read_csv(test_path)
    X_test = df.drop(columns=[TARGET_COL])
    y_test = df[TARGET_COL]
    logger.info("Test data loaded: %d samples, %d features", len(X_test), X_test.shape[1])
    return X_test, y_test


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 (weighted average).

    Args:
        y_true: Ground-truth labels.
        y_pred: Model predictions.

    Returns:
        Dictionary mapping metric name to its value.
    """
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 6),
        "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 6),
        "f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 6),
    }


def save_metrics(metrics: dict[str, float], path: Path = METRICS_FILE) -> None:
    """Serialise metrics dict to a JSON file.

    Args:
        metrics: Dictionary of metric name → value.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", path)


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    class_names: list[str],
    path: Path = CM_FILE,
) -> None:
    """Generate and save a confusion matrix PNG.

    Args:
        y_true: Ground-truth labels.
        y_pred: Model predictions.
        class_names: Display labels for each class.
        path: Destination file path for the PNG.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix — Iris (test set)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", path)


def log_to_mlflow(
    metrics: dict[str, float],
    cm_path: Path,
    metrics_path: Path,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """Log metrics and artifacts to MLflow under the evaluation experiment.

    Args:
        metrics: Dictionary of metric name → value.
        cm_path: Path to the confusion matrix PNG artifact.
        metrics_path: Path to the metrics JSON artifact.
        experiment_name: MLflow experiment name.

    Returns:
        The MLflow run ID.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="evaluate") as run:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")
        run_id = run.info.run_id

    logger.info("MLflow evaluation run logged: %s", run_id)
    return run_id


def run(
    model_path: Path = MODEL_PATH,
    processed_dir: Path = PROCESSED_DIR,
    reports_dir: Path = REPORTS_DIR,
    experiment_name: str = EXPERIMENT_NAME,
) -> dict[str, float]:
    """End-to-end evaluation pipeline.

    Args:
        model_path: Path to the pickled model.
        processed_dir: Directory containing test.csv.
        reports_dir: Directory where outputs are written.
        experiment_name: MLflow experiment name.

    Returns:
        Dictionary of computed metrics.
    """
    model = load_model(model_path)
    X_test, y_test = load_test_data(processed_dir)

    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    for name, value in metrics.items():
        logger.info("  %-12s %.6f", name, value)

    metrics_path = reports_dir / "metrics.json"
    cm_path = reports_dir / "confusion_matrix.png"

    save_metrics(metrics, metrics_path)
    save_confusion_matrix(y_test, y_pred, CLASS_NAMES, cm_path)
    log_to_mlflow(metrics, cm_path, metrics_path, experiment_name)

    return metrics


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate trained model and log results.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to model.pkl (default: models/model.pkl)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory with test.csv (default: data/processed)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Output directory for metrics and plots (default: reports)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=EXPERIMENT_NAME,
        help=f"MLflow experiment name (default: {EXPERIMENT_NAME})",
    )
    args = parser.parse_args()

    run(
        model_path=args.model,
        processed_dir=args.data_dir,
        reports_dir=args.reports_dir,
        experiment_name=args.experiment,
    )


if __name__ == "__main__":
    main()
