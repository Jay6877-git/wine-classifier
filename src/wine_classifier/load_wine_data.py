from sklearn.datasets import load_wine
import pandas as pd
from typing import Tuple


def load_wine_data(as_frame: bool = True) -> Tuple[pd.DataFrame, pd.Series, list, list]:
    """
    Loads the sklearn wine dataset.

    Returns:
        X: features (DataFrame if as_frame=True)
        y: target labels (Series)
        feature_names: list of column names
        target_names: list of class names
    """
    wine = load_wine(as_frame=as_frame)
    X = wine.data if as_frame else pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target if as_frame else pd.Series(wine.target, name="target")
    return X, y, wine.feature_names, list(wine.target_names)
