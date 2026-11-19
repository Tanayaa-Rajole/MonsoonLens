"""Turn raw daily rainfall into features a machine learning model can use."""

import pandas as pd

TARGET = "rainfall_mm"
FEATURE_COLUMNS = ["rain_lag_1", "rain_lag_2", "rain_roll_7", "month", "is_monsoon"]


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling-average and calendar features per district.

    Every feature only uses information from BEFORE the day being predicted,
    so the model never "peeks" at the answer (this is called data leakage).
    """
    out = df.sort_values(["district", "date"]).copy()
    by_district = out.groupby("district")["rainfall_mm"]

    out["rain_lag_1"] = by_district.shift(1)  # yesterday
    out["rain_lag_2"] = by_district.shift(2)  # day before yesterday

    # 7-day average of the PREVIOUS 7 days (shift first, then roll).
    out["rain_roll_7"] = by_district.transform(lambda s: s.shift(1).rolling(7).mean())

    out["month"] = out["date"].dt.month
    out["is_monsoon"] = out["month"].between(6, 9).astype(int)  # June-September

    # The first rows of each district have no history, so drop them.
    return out.dropna().reset_index(drop=True)


def features_for_next_day(recent_rainfall, target_date) -> pd.DataFrame:
    """Build ONE feature row for a future day from the latest observations.

    `recent_rainfall` is a list of the most recent daily values, oldest first,
    ending on the day before `target_date`. It needs at least 7 values.

    This mirrors add_basic_features exactly, so training and prediction use
    the same definition of every feature.
    """
    values = list(recent_rainfall)
    if len(values) < 7:
        raise ValueError("Need at least 7 days of recent rainfall.")
    target_date = pd.Timestamp(target_date)
    month = target_date.month
    row = {
        "rain_lag_1": values[-1],
        "rain_lag_2": values[-2],
        "rain_roll_7": sum(values[-7:]) / 7,
        "month": month,
        "is_monsoon": int(6 <= month <= 9),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
