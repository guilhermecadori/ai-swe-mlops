"""Model training with MLflow experiment tracking and Model Registry."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
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
REGISTERED_MODEL_NAME = "OpClassifier"
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
TARGET_COL = "target"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def load_train_data(
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load training split produced by preprocess.py.

    Args:
        processed_dir: Directory containing train.csv.

    Returns:
        Tuple of (X_train, y_train).

    Raises:
        FileNotFoundError: If train.csv does not exist.
    """
    train_path = processed_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}. Run preprocess.py first.")
    df = pd.read_csv(train_path)
    X_train = df.drop(columns=[TARGET_COL])
    y_train = df[TARGET_COL]
    logger.info("Training data loaded: %d samples, %d features", len(X_train), X_train.shape[1])
    return X_train, y_train


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Compute weighted accuracy, precision, recall, and F1.

    Args:
        y_true: Ground-truth labels.
        y_pred: Model predictions.

    Returns:
        Dictionary mapping metric name to value.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Generate and save a confusion matrix PNG artifact.

    Args:
        y_true: Ground-truth labels.
        y_pred: Model predictions.
        class_names: Display labels for each class.
        output_path: Destination file path.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix — Iris (train set)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)


def save_model_locally(model: RandomForestClassifier, path: Path) -> None:
    """Persist model as a pickle file for DVC tracking.

    Args:
        model: Trained scikit-learn model.
        path: Destination .pkl file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved locally to %s", path)


def train(
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42,
) -> None:
    """End-to-end training pipeline with MLflow tracking.

    Args:
        n_estimators: Number of trees in the random forest.
        max_depth: Maximum depth of each tree (None = unlimited).
        random_state: Seed for reproducibility.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # --- Data ---
    X_train, y_train = load_train_data()
    logger.info("Dataset: %d training samples", len(X_train))

    # --- MLflow experiment ---
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s", run_id)

        # --- Model ---
        params: dict = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # --- Log parameters ---
        mlflow.log_params(params)

        # --- Metrics (on train set) ---
        y_pred = model.predict(X_train)
        metrics = compute_metrics(y_train, y_pred)
        mlflow.log_metrics(metrics)
        for name, value in metrics.items():
            logger.info("  %-12s %.4f", name, value)

        # --- Log model with signature ---
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        logger.info("Model registered in MLflow as '%s'", REGISTERED_MODEL_NAME)

        # --- Confusion matrix artifact ---
        cm_path = Path("reports") / "confusion_matrix_train.png"
        save_confusion_matrix(y_train, y_pred, CLASS_NAMES, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # --- Local pickle for DVC ---
        local_model_path = MODELS_DIR / "model.pkl"
        save_model_locally(model, local_model_path)
        mlflow.log_artifact(str(local_model_path), artifact_path="local_model")

        logger.info("Run complete. ID: %s", run_id)


def main() -> None:
    """Parse CLI arguments and launch training."""
    parser = argparse.ArgumentParser(
        description="Train a RandomForest on Iris with MLflow tracking."
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the forest (default: 100)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth; omit for unlimited (default: None)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
