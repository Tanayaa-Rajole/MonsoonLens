"""Clean raw rainfall data so later steps can trust it."""

import pandas as pd


def clean_rainfall(df: pd.DataFrame):
    """Validate and clean a raw rainfall table.

    Returns (clean_df, report). `report` is a dict counting what was removed,
    so problems with a dataset are visible instead of silently hidden.
    """
    report = {"rows_in": len(df)}
    out = df.copy()

    # 1. Dates: convert text to real dates; unreadable ones become NaT ("not a time").
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    report["bad_dates"] = int(out["date"].isna().sum())
    out = out.dropna(subset=["date"])

    # 2. Rainfall: convert to numbers; text like "NA" becomes NaN.
    out["rainfall_mm"] = pd.to_numeric(out["rainfall_mm"], errors="coerce")
    report["missing_rainfall"] = int(out["rainfall_mm"].isna().sum())

    # 3. Rainfall can never be negative, so those values are invalid.
    negative = out["rainfall_mm"] < 0
    report["negative_rainfall"] = int(negative.sum())
    out.loc[negative, "rainfall_mm"] = float("nan")

    # 4. Duplicates: keep one row per district per day.
    before = len(out)
    out = out.drop_duplicates(subset=["district", "date"], keep="first")
    report["duplicates"] = before - len(out)

    # 5. Make every district's calendar complete (one row for every day).
    #    Missing days become NaN rows so gaps are explicit and lag features
    #    never silently skip across a gap.
    out = _fill_calendar(out)
    report["rows_out"] = len(out)
    report["still_missing"] = int(out["rainfall_mm"].isna().sum())

    return out.reset_index(drop=True), report


def _fill_calendar(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for district, part in df.groupby("district"):
        part = part.set_index("date").sort_index()
        full_range = pd.date_range(part.index.min(), part.index.max(), freq="D")
        part = part.reindex(full_range)
        part["district"] = district
        part.index.name = "date"
        frames.append(part.reset_index())
    return pd.concat(frames, ignore_index=True)[["date", "district", "rainfall_mm"]]
