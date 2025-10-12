from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

"""
    Evaluate a trained classification model on test data.

    Computes standard evaluation metrics for classification models:
    accuracy, weighted F1-score, confusion matrix, and classification report.

    Args:
        model (sklearn.base.BaseEstimator):
            A fitted scikit-learn model or pipeline.
        X_test (array-like or pd.DataFrame):
            Test feature matrix of shape (n_samples, n_features).
        y_test (array-like or pd.Series):
            True target labels of shape (n_samples,).

    Returns:
        dict:
            Dictionary containing:
                - 'accuracy': float, overall test accuracy.
                - 'f1': float, weighted F1-score (accounts for class imbalance).
                - 'confusion_matrix': np.ndarray, test confusion matrix.
                - 'report': str, detailed per-class metrics summary.

    Notes:
        - Designed for use with classification tasks only.
        - Weighted F1 is used for better comparison across imbalanced datasets.
    """


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }
