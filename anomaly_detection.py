"""Simple statistical anomaly detection (z-score per district and month)."""

import numpy as np
import pandas as pd


def flag_anomalies(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Add z-score columns and an `is_anomaly` flag.

    A day is "unusual" if its rainfall is more than `threshold` standard
    deviations away from the average for the same district and calendar month.
    """
    out = df.copy()
    out["month"] = out["date"].dt.month

    group = out.groupby(["district", "month"])["rainfall_mm"]
    out["month_mean"] = group.transform("mean")
    out["month_std"] = group.transform("std")

    # Avoid dividing by zero in months with no variation.
    safe_std = out["month_std"].replace(0, np.nan)
    out["z_score"] = ((out["rainfall_mm"] - out["month_mean"]) / safe_std).fillna(0)

    out["is_anomaly"] = out["z_score"].abs() > threshold
    return out
