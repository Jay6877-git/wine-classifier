from unittest import result

from src.wine_classifier.load_wine_data import load_wine_data
from src.wine_classifier.model_baseline import train_baseline_model

import numpy as np

def test_baseline_end_to_end():
    # load data (frames or arrays both fine)
    X, y,_,_  = load_wine_data()

    # train & evaluate
    result = train_baseline_model(X, y)

    # basic contract checks
    for key in ("model", "accuracy", "report", "confusion_matrix"):
        assert key in result

    # accuracy floor so test is stable but meaningful
    assert result["accuracy"] > 0.85

    # confusion matrix shape should match 3 classes
    cm = result["confusion_matrix"]
    assert isinstance(cm, (np.ndarray, list))
    cm = np.array(cm)
    assert cm.shape == (3, 3)