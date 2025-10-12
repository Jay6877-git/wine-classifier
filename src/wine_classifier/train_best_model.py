from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_model(X, y):
    """
    Train and evaluate the final Logistic Regression model for the Wine Classification project.

    This function executes a complete supervised learning workflow:
        1. Splits the dataset into training and test sets (80/20).
        2. Builds a reproducible pipeline using StandardScaler and LogisticRegression.
        3. Fits the model on the training data and evaluates on the test set.
        4. Computes key evaluation metrics (accuracy, classification report, and confusion matrix).

    Args:
        X (pd.DataFrame or np.ndarray):
            Feature matrix of shape (n_samples, n_features).
        y (pd.Series or np.ndarray):
            Target vector of shape (n_samples,).

    Returns:
        dict:
            A dictionary containing:
                - 'model': Trained sklearn Pipeline object.
                - 'accuracy': float, overall accuracy on the test set.
                - 'report': str, detailed per-class performance report.
                - 'confusion_matrix': np.ndarray, confusion matrix (n_classes × n_classes).

    Notes:
        - The trained model is intended for persistence (saving/loading via `model_io.py`).
        - Random state is fixed for reproducibility (42).
        - This function replaces the earlier `model_baseline` training function.
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
