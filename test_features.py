import pandas as pd

from src.feature_engineering import add_basic_features


def make_df():
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-06-01", periods=10),
            "district": "Pune",
            "rainfall_mm": [0, 5, 10, 0, 2, 4, 6, 8, 1, 3],
        }
    )


def test_lag_1_is_yesterdays_rainfall():
    out = add_basic_features(make_df())
    first = out.iloc[0]  # first row that has 7 days of history
    assert first["date"] == pd.Timestamp("2022-06-08")
    assert first["rain_lag_1"] == 6  # rainfall on 7 June


def test_rolling_mean_excludes_today():
    out = add_basic_features(make_df())
    first = out.iloc[0]  # 8 June, previous 7 days = 1-7 June
    assert first["rain_roll_7"] == (0 + 5 + 10 + 0 + 2 + 4 + 6) / 7


def test_monsoon_flag():
    out = add_basic_features(make_df())
    assert out["is_monsoon"].all()  # all June
