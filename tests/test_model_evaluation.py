import numpy as np

from src.wine_classifier.model_evaluation import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.wine_classifier.load_wine_data import load_wine_data


def test_model_evaluation():
    X, y, feature_names, target_names = load_wine_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("classifier", LogisticRegression())]
    )
    pipe.fit(X_train, y_train)

    result = evaluate_model(pipe, X_test, y_test)

    for keys in ("accuracy", "f1", "confusion_matrix", "report"):
        assert keys in result

    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["f1"] <= 1

    cm = result["confusion_matrix"]
    assert isinstance(cm, (np.ndarray, list))
    cm = np.array(cm)
    assert cm.shape == (3, 3)

    assert isinstance(result["report"], str)
    assert len(result["report"]) > 0
