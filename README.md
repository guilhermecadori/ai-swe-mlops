# ai-swe-mlops-project

MLOps pipeline for classification tasks. Covers data versioning with DVC, experiment tracking with MLflow, and a reproducible training/evaluation workflow.

## Project Structure

```
ai-swe-mlops-project/
├── configs/            # Experiment and pipeline configuration (YAML)
├── data/
│   ├── raw/            # Immutable source data (tracked by DVC)
│   └── processed/      # Feature-engineered data ready for training
├── models/             # Serialized model artifacts
├── notebooks/          # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── preprocess.py   # Data loading and feature engineering
│   ├── train.py        # Model training with MLflow logging
│   └── evaluate.py     # Model evaluation and metrics reporting
└── tests/              # Unit and integration tests
```

## Quickstart

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Pull data (requires DVC remote configured)
dvc pull

# 3. Run the training pipeline
python -m src.train --config configs/train_config.yaml

# 4. Evaluate
python -m src.evaluate --config configs/train_config.yaml

# 5. Start MLflow UI
mlflow ui
```

## Running Tests

```bash
pytest
```

## Configuration

All hyperparameters and pipeline settings live in `configs/train_config.yaml`. Override any value via CLI args passed to `src.train` or `src.evaluate`.

## Experiment Tracking

MLflow is configured locally by default (`mlruns/`). Set `MLFLOW_TRACKING_URI` to point to a remote server.

## Data Versioning

Raw data is tracked with DVC. After adding new data:

```bash
dvc add data/raw/<file>
git add data/raw/<file>.dvc .gitignore
git commit -m "add new raw data"
dvc push
```
