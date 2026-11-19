"""Split data by time and measure model performance."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Oldest rows -> train, newest rows -> test. No shuffling.

    Shuffling would let the model train on days AFTER the ones it is tested
    on, which is cheating for time-based data.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    df = df.sort_values("date")
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut], df.iloc[cut:]


def compute_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE and R² (the three metrics named in the README)."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_feature_importance(model, feature_names):
    """Return a sorted Series of importances, or None if the model has none."""
    if not hasattr(model, "feature_importances_"):
        return None
    importance = pd.Series(model.feature_importances_, index=feature_names)
    return importance.sort_values(ascending=False)
