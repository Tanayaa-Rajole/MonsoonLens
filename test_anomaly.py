import pandas as pd

from src.anomaly_detection import flag_anomalies


def test_extreme_day_is_flagged():
    rain = [5.0] * 30 + [500.0]  # one huge outlier
    df = pd.DataFrame(
        {
            "date": pd.date_range("2022-07-01", periods=31),
            "district": "Pune",
            "rainfall_mm": rain,
        }
    )
    out = flag_anomalies(df, threshold=3.0)
    assert out["is_anomaly"].sum() == 1
    assert out.loc[out["is_anomaly"], "rainfall_mm"].iloc[0] == 500.0


def test_constant_rainfall_has_no_anomalies():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2022-07-01", periods=10),
            "district": "Pune",
            "rainfall_mm": [3.0] * 10,
        }
    )
    assert not flag_anomalies(df)["is_anomaly"].any()
