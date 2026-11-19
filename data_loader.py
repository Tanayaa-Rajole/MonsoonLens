"""Load rainfall data from a CSV file, or fall back to synthetic sample data."""

from pathlib import Path

import pandas as pd

from src.sample_data import generate_sample_data

REQUIRED_COLUMNS = ["date", "district", "rainfall_mm"]


def load_rainfall(path=None) -> pd.DataFrame:
    """Return a raw rainfall table with columns: date, district, rainfall_mm.

    If `path` is None we use the synthetic sample data. Otherwise we read the
    CSV at `path`. The CSV must have the three required columns.
    """
    if path is None:
        return generate_sample_data()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found: {list(df.columns)}. Rename your columns to match."
        )
    return df[REQUIRED_COLUMNS]
