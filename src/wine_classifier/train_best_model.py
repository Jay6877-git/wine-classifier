import logging
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.wine_classifier.load_wine_data import load_wine_data
from src.wine_classifier.model_io import save_model, save_artifacts


def train_model(X, y):
    """
    Train and evaluate the final Logistic Regression model for the Wine Classification project.

    This function executes a complete supervised learning workflow:
        1. Splits the dataset into training and test sets (80/20 split).
        2. Builds a reproducible pipeline using StandardScaler and LogisticRegression.
        3. Fits the model on training data and evaluates performance on the test set.
        4. Computes and returns key evaluation metrics.

    Args:
        X (pd.DataFrame | np.ndarray):
            Feature matrix of shape (n_samples, n_features).
        y (pd.Series | np.ndarray):
            Target vector of shape (n_samples,).

    Returns:
        dict:
            {
                "model": sklearn.Pipeline,
                "accuracy": float,
                "report": dict (classification report as JSON-compatible structure),
                "confusion_matrix": np.ndarray
            }

    Notes:
        - The trained model can be persisted using `model_io.save_model`.
        - The confusion matrix is a 3×3 array corresponding to the three Wine classes.
        - Random state is fixed for reproducibility (42).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nPer-class report:\n", classification_report(y_test, y_pred, digits=4))

    return {
        "model": pipeline,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": matrix,
    }


def main() -> None:
    # Load preprocessed Wine dataset (features and labels)
    X, y, _, _ = load_wine_data()

    # Train model and evaluate on test split
    result = train_model(X, y)
    trained_model = result["model"]

    # Separate model object from metrics for saving
    artifacts = {k: v for k, v in result.items() if k != "model"}
    # Convert NumPy array to list for JSON serialization
    artifacts["confusion_matrix"] = artifacts["confusion_matrix"].tolist()

    # Define project root and save directories
    root = Path(__file__).resolve().parents[2]
    model_path = root / "models"
    artifact_path = root / "artifacts"

    # Persist model and metrics
    model_path = save_model(trained_model, model_path)
    artifact_path = save_artifacts(artifacts, artifact_path)

    # Output confirmation with absolute paths
    logging.info(f"Model saved to : {model_path}")
    logging.info(f"Artifact saved to : {artifact_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
