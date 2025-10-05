from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_baseline_model(X, y):
    """
        Trains and evaluates a baseline Logistic Regression model on the Wine dataset.

        This function performs a complete minimal ML workflow:
            1. Splits the dataset into training and test sets (80/20 split).
            2. Builds a pipeline with StandardScaler and LogisticRegression.
            3. Fits the pipeline on training data and evaluates on test data.
            4. Computes accuracy, classification report, and confusion matrix.

        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (pd.Series or np.ndarray): Target labels of shape (n_samples,).

        Returns:
            dict: A dictionary containing:
                - 'model': the trained pipeline object
                - 'accuracy': float, overall accuracy on the test set
                - 'report': str, detailed classification report
                - 'confusion_matrix': np.ndarray, confusion matrix of shape (n_classes, n_classes)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
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

