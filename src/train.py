"""Model training with MLflow experiment tracking and Model Registry."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Union

import hydra
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from mlflow.models.signature import infer_signature
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "Classification-MLOps"
REGISTERED_MODEL_NAME = "OpClassifier"
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
TARGET_COL = "target"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

Model = Union[RandomForestClassifier, XGBClassifier]


def load_train_data(processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.Series]:
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


def build_model(model_type: str, params: dict) -> Model:
    """Instantiate the requested model with given parameters.

    Args:
        model_type: Either "random_forest" or "xgboost".
        params: Hyperparameter dict forwarded to the model constructor.

    Returns:
        Untrained model instance.

    Raises:
        ValueError: If model_type is not recognised.
    """
    if model_type == "random_forest":
        rf_params = {k: v for k, v in params.items() if k in ("n_estimators", "max_depth", "random_state")}
        return RandomForestClassifier(**rf_params)
    if model_type == "xgboost":
        xgb_params = {k: v for k, v in params.items() if k in (
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "random_state",
        )}
        return XGBClassifier(
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
            **xgb_params,
        )
    raise ValueError(f"Unknown model_type '{model_type}'. Choose 'random_forest' or 'xgboost'.")


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
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


def save_model_locally(model: Model, path: Path) -> None:
    """Persist model as a pickle file for DVC tracking.

    Args:
        model: Trained model instance.
        path: Destination .pkl file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved locally to %s", path)


def log_model_to_mlflow(model: Model, model_type: str, X_train: pd.DataFrame) -> None:
    """Log model to MLflow using the appropriate flavor.

    Args:
        model: Trained model instance.
        model_type: Either "random_forest" or "xgboost".
        X_train: Training features used to infer the model signature.
    """
    signature = infer_signature(X_train, model.predict(X_train))
    if model_type == "xgboost":
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )
    else:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )


def train(
    model_type: str,
    n_estimators: int,
    max_depth: int | None,
    random_state: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
) -> None:
    """End-to-end training pipeline with MLflow tracking.

    Args:
        model_type: Model architecture — "random_forest" or "xgboost".
        n_estimators: Number of trees / boosting rounds.
        max_depth: Maximum tree depth (None = unlimited for RF).
        random_state: Seed for reproducibility.
        learning_rate: Step size shrinkage (XGBoost only).
        subsample: Row subsampling ratio per tree (XGBoost only).
        colsample_bytree: Feature subsampling ratio per tree (XGBoost only).
    """
    X_train, y_train = load_train_data()
    logger.info("Dataset: %d training samples | model_type: %s", len(X_train), model_type)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_type) as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: %s", run_id)

        all_params: dict = {
            "model_type": model_type,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
        model = build_model(model_type, all_params)
        model.fit(X_train, y_train)

        mlflow.log_params(all_params)

        y_pred = model.predict(X_train)
        metrics = compute_metrics(y_train, y_pred)
        mlflow.log_metrics(metrics)
        for name, value in metrics.items():
            logger.info("  %-12s %.4f", name, value)

        log_model_to_mlflow(model, model_type, X_train)
        logger.info("Model registered in MLflow as '%s'", REGISTERED_MODEL_NAME)

        cm_path = Path("reports") / "confusion_matrix_train.png"
        save_confusion_matrix(y_train, y_pred, CLASS_NAMES, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        local_model_path = MODELS_DIR / "model.pkl"
        save_model_locally(model, local_model_path)
        mlflow.log_artifact(str(local_model_path), artifact_path="local_model")

        logger.info("Run complete. ID: %s", run_id)


# ---------------------------------------------------------------------------
# Hydra entry point
# config_path=".." resolves to the project root relative to src/train.py,
# so Hydra reads params.yaml — the same file DVC tracks.
# ---------------------------------------------------------------------------

@hydra.main(config_path="..", config_name="params", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point — reads params.yaml and accepts CLI overrides.

    DVC usage (pipeline):
        python src/train.py hydra.run.dir=. hydra.output_subdir=null

    Manual overrides (Hydra syntax, no dashes):
        python src/train.py model.n_estimators=150 model.max_depth=10
        python src/train.py model.model_type=xgboost xgboost.learning_rate=0.05

    Active config is logged to INFO for full reproducibility.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger.info("Hydra config:\n%s", OmegaConf.to_yaml(cfg))

    model_type: str = cfg.model.model_type

    # Route hyperparameters to the correct section
    if model_type == "xgboost":
        xgb = cfg.xgboost
        train(
            model_type=model_type,
            n_estimators=xgb.n_estimators,
            max_depth=xgb.max_depth,
            random_state=xgb.random_state,
            learning_rate=xgb.learning_rate,
            subsample=xgb.subsample,
            colsample_bytree=xgb.colsample_bytree,
        )
    else:
        m = cfg.model
        train(
            model_type=model_type,
            n_estimators=m.n_estimators,
            max_depth=m.max_depth,
            random_state=m.random_state,
            learning_rate=1.0,    # unused for RF
            subsample=1.0,
            colsample_bytree=1.0,
        )


if __name__ == "__main__":
    main()
