"""
Module for saving and loading trained machine learning models and their evaluation artifacts.

This module provides lightweight I/O utilities to persist trained pipelines and
store performance metrics in JSON format for reproducibility and CI/CD workflows.
"""

from pathlib import Path

import joblib
import json


def save_model(model, model_path: Path):
    """
    Save a trained scikit-learn model pipeline to disk.

    Serializes the fitted model or pipeline using joblib and stores it under the specified directory.
    Automatically creates directories if they do not exist.

    Args:
        model (sklearn.base.BaseEstimator):
            The trained scikit-learn model or pipeline to save.
        model_path (Path):
            Directory path (relative or absolute) where the model will be stored.

    Returns:
        Path: Full path to the saved model file (e.g., 'artifacts/wine-classifier.joblib').
    """
    model_path.mkdir(parents=True, exist_ok=True)

    model_path = model_path / "wine-classifier.joblib"
    joblib.dump(model, model_path)
    return model_path


def save_artifacts(metrics, artifacts_dir: Path):
    """
    Save model evaluation metrics and metadata as a JSON file.

    Args:
        metrics (dict):
            Dictionary containing evaluation results such as accuracy, F1 score,
            confusion matrix, and classification report.
        artifacts_dir (Path):
            Directory path where the metrics JSON file will be stored.

    Returns:
        Path: Full path to the saved metrics JSON file.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = artifacts_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    return metrics_path


def load_model(model_path: Path):
    """
    Load a serialized scikit-learn model pipeline from disk.

    Args:
        model_path (Path):
            Full path to the '.joblib' model file.

    Returns:
        sklearn.base.BaseEstimator:
            The deserialized scikit-learn model or pipeline.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
    """
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} does not exist")
    else:
        return joblib.load(model_path)
