"""Send prediction requests to a locally running MLflow model server."""

from __future__ import annotations

import argparse
import json
import logging
import sys

import requests

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://127.0.0.1"
DEFAULT_PORT = 5001
CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}

# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

# Single sample: well-known setosa
SAMPLE_SINGLE = {
    "dataframe_split": {
        "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "data": [[5.1, 3.5, 1.4, 0.2]],
    }
}

# Batch of three samples (one per class)
SAMPLE_BATCH = {
    "dataframe_split": {
        "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "data": [
            [5.1, 3.5, 1.4, 0.2],   # setosa
            [6.0, 2.9, 4.5, 1.5],   # versicolor
            [6.9, 3.1, 5.4, 2.1],   # virginica
        ],
    }
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def predict(
    payload: dict,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> list[int]:
    """Send a prediction request to the MLflow REST API.

    The server must be running:
        mlflow models serve -m "models:/OpClassifier/6" --port 5001 --no-conda

    Args:
        payload: Request body in MLflow ``dataframe_split`` format.
        host: Server host (default: http://127.0.0.1).
        port: Server port (default: 5001).

    Returns:
        List of predicted class indices.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status code.
        requests.ConnectionError: If the server is not reachable.
    """
    url = f"{host}:{port}/invocations"
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
    response.raise_for_status()

    predictions: list[int] = response.json()["predictions"]
    return predictions


def format_results(predictions: list[int], payload: dict) -> str:
    """Format predictions alongside the input samples for display.

    Args:
        predictions: List of predicted class indices.
        payload: Original request payload.

    Returns:
        Human-readable result string.
    """
    columns = payload["dataframe_split"]["columns"]
    rows = payload["dataframe_split"]["data"]

    lines = [f"{'Input':<50} {'Class':>5}  Label"]
    lines.append("-" * 70)
    for row, pred in zip(rows, predictions):
        features = ", ".join(f"{c}={v}" for c, v in zip(columns, row))
        label = CLASS_NAMES.get(pred, "unknown")
        lines.append(f"{features:<50} {pred:>5}  {label}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Send one or more sample requests and print results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Run inference against a locally served MLflow model."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Server host (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Server port (default: {DEFAULT_PORT})")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "custom"],
        default="batch",
        help="Use a preset sample or provide --data (default: batch)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=(
            'Custom input as JSON dataframe_split, e.g.: '
            '\'{"dataframe_split": {"columns": [...], "data": [[...]]}}\''
        ),
    )
    args = parser.parse_args()

    if args.mode == "single":
        payload = SAMPLE_SINGLE
    elif args.mode == "batch":
        payload = SAMPLE_BATCH
    else:
        if not args.data:
            logger.error("--data is required when --mode=custom")
            sys.exit(1)
        payload = json.loads(args.data)

    try:
        predictions = predict(payload, host=args.host, port=args.port)
        print(format_results(predictions, payload))
    except requests.ConnectionError:
        logger.error(
            "Could not connect to %s:%d. Is the MLflow server running?\n"
            "Start it with:\n"
            "  mlflow models serve -m \"models:/OpClassifier/6\" --port %d --no-conda",
            args.host, args.port, args.port,
        )
        sys.exit(1)
    except requests.HTTPError as exc:
        logger.error("Server returned an error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
