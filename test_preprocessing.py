import pandas as pd

from src.preprocessing import clean_rainfall


def raw(rows):
    return pd.DataFrame(rows, columns=["date", "district", "rainfall_mm"])


def test_removes_bad_dates_and_negative_values():
    df = raw([
        ["2022-06-01", "Pune", 5.0],
        ["not a date", "Pune", 3.0],
        ["2022-06-02", "Pune", -9.0],
        ["2022-06-03", "Pune", 2.0],
    ])
    clean, report = clean_rainfall(df)
    assert report["bad_dates"] == 1
    assert report["negative_rainfall"] == 1
    assert (clean["rainfall_mm"].dropna() >= 0).all()


def test_duplicates_are_dropped():
    df = raw([["2022-06-01", "Pune", 5.0], ["2022-06-01", "Pune", 7.0]])
    clean, report = clean_rainfall(df)
    assert report["duplicates"] == 1
    assert len(clean) == 1


def test_missing_days_become_explicit_nan_rows():
    df = raw([["2022-06-01", "Pune", 1.0], ["2022-06-04", "Pune", 2.0]])
    clean, report = clean_rainfall(df)
    assert len(clean) == 4  # 1, 2, 3, 4 June
    assert report["still_missing"] == 2


def test_text_rainfall_becomes_missing():
    df = raw([["2022-06-01", "Pune", "NA"], ["2022-06-02", "Pune", "4.5"]])
    clean, report = clean_rainfall(df)
    assert report["missing_rainfall"] == 1
    assert clean["rainfall_mm"].iloc[1] == 4.5
