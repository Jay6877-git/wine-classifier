from pathlib import Path
import pytest
import json

from src.wine_classifier.load_wine_data import load_wine_data
from src.wine_classifier.train_best_model import train_model
from src.wine_classifier.model_io import load_model, save_model, save_artifacts


def test_model_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Test end-to-end model I/O workflow for persistence and reproducibility.

    This test validates the following behaviors:
        1. The model and evaluation artifacts can be saved successfully.
        2. The saved files exist on disk and contain valid data.
        3. The saved JSON metrics file has the expected structure and types.
        4. The saved model file can later be reloaded (implicitly verified by joblib).

    It uses pytest's `tmp_path` fixture to create an isolated temporary directory,
    ensuring no pollution of the project repository.

    Args:
        tmp_path (Path): Temporary directory provided by pytest for test isolation.
        monkeypatch (pytest.MonkeyPatch): Utility for temporarily modifying the environment
                                          (used here to change working directory).

    Raises:
        AssertionError: If model or artifact files are missing, invalid, or improperly structured.
    """
    # --- Arrange: prepare clean working environment ---
    # Switch current directory to the temporary test folder
    monkeypatch.chdir(tmp_path)

    # Load sample dataset
    X, y, _, _ = load_wine_data()

    # Train a model using the baseline training pipeline
    result = train_model(X, y)
    model = result["model"]

    # --- Act: save model and artifacts ---
    # Save trained model under a "models" subfolder
    model_path = save_model(model, tmp_path / "models")

    # Save evaluation metrics as JSON (excluding model object itself)
    artifacts = {k: v for k, v in result.items() if k != "model"}

    artifacts["confusion_matrix"] = artifacts["confusion_matrix"].tolist()
    metrics_path = save_artifacts(artifacts, tmp_path / "artifacts")

    # --- Assert: verify files exist and contain valid data ---
    # Check that both the model and metrics were written to disk
    assert model_path.exists(), "Model file was not saved successfully."
    assert metrics_path.exists(), "Metrics file was not saved successfully."

    # Read and validate the JSON structure
    saved = json.loads(metrics_path.read_text())
    # Check that required keys are present
    assert (
        "accuracy" in saved and "report" in saved and "confusion_matrix" in saved
    ), "Missing one or more expected keys in metrics.json."
    assert isinstance(
        saved["accuracy"], (int, float)
    ), "Accuracy should be a numeric value."
    assert isinstance(
        saved["report"], dict
    ), "Classification report should be a dictionary."

    # --- (Optional) Load and verify model reloadability ---
    # Ensure model reloads successfully from disk without errors
    reloaded_model = load_model(model_path)
    assert hasattr(
        reloaded_model, "predict"
    ), "Reloaded model does not implement the predict() method."
