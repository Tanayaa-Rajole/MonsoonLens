"""Synthetic sample rainfall data for Day-1 development.

IMPORTANT: this is NOT real IMD data. It is randomly generated so the
dashboard has something to show before the real datasets are acquired.
It will be replaced by src/data_loader.py once real data is available.
"""

import numpy as np
import pandas as pd

# Each district gets a "wetness" (how rainy overall) and the month its rain peaks.
DISTRICTS = {
    "Pune": {"wetness": 0.8, "peak_month": 7.5},
    "Mumbai": {"wetness": 1.6, "peak_month": 7.0},
    "Nagpur": {"wetness": 0.9, "peak_month": 7.5},
    "Jaipur": {"wetness": 0.5, "peak_month": 8.0},
    "Chennai": {"wetness": 1.0, "peak_month": 10.5},
}


def generate_sample_data(start="2019-01-01", end="2023-12-31", seed=42):
    """Return a DataFrame with columns: date, district, rainfall_mm."""
    rng = np.random.default_rng(seed)  # fixed seed -> same data every run
    dates = pd.date_range(start, end, freq="D")
    month = dates.month.to_numpy()
    frames = []

    for name, params in DISTRICTS.items():
        # Seasonality: a bell curve that is highest at the district's peak month.
        season = np.exp(-((month - params["peak_month"]) ** 2) / (2 * 1.3**2))

        # Step 1: does it rain today? (more likely in the wet season)
        rain_prob = 0.03 + 0.7 * season * params["wetness"] / 1.6
        rains = rng.random(len(dates)) < rain_prob

        # Step 2: if it rains, how much? (gamma = lots of small, few big days)
        amount = rng.gamma(shape=0.8, scale=12 * params["wetness"] * (0.3 + season))

        # Step 3: rare extreme days (about 0.3% of days) so anomalies exist.
        spike = np.where(rng.random(len(dates)) < 0.003, 4.0, 1.0)

        rainfall = np.where(rains, amount * spike, 0.0).round(1)
        frames.append(
            pd.DataFrame({"date": dates, "district": name, "rainfall_mm": rainfall})
        )

    return pd.concat(frames, ignore_index=True)
