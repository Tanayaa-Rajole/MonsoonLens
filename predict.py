"""Load a saved model and estimate rainfall for the next day."""

from pathlib import Path

import joblib

from src.feature_engineering import features_for_next_day

MODELS_DIR = Path("models")


def model_exists(district: str, models_dir=MODELS_DIR) -> bool:
    return (Path(models_dir) / f"{district}.joblib").exists()


def load_model(district: str, models_dir=MODELS_DIR) -> dict:
    path = Path(models_dir) / f"{district}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model for {district}. Run: python -m src.train"
        )
    return joblib.load(path)


def predict_next_day(bundle: dict, recent_rainfall, target_date) -> float:
    """Estimate rainfall (mm) for `target_date` from the previous days' values."""
    row = features_for_next_day(recent_rainfall, target_date)
    prediction = float(bundle["model"].predict(row[bundle["features"]])[0])
    return max(prediction, 0.0)  # rainfall can't be negative
