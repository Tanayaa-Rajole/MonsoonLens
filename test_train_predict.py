import pandas as pd
import pytest

from src.feature_engineering import FEATURE_COLUMNS, add_basic_features, features_for_next_day
from src.predict import load_model, model_exists, predict_next_day
from src.sample_data import generate_sample_data
from src.train import train_all


def test_next_day_features_match_training_features():
    """Train/serve consistency: both paths must compute identical features."""
    df = generate_sample_data("2022-01-01", "2022-03-31")
    df = df[df["district"] == "Pune"]
    feats = add_basic_features(df)
    row = feats.iloc[20]
    history = df[df["date"] < row["date"]]["rainfall_mm"].tolist()
    rebuilt = features_for_next_day(history, row["date"]).iloc[0]
    for col in FEATURE_COLUMNS:
        assert rebuilt[col] == pytest.approx(row[col])


def test_needs_seven_days():
    with pytest.raises(ValueError):
        features_for_next_day([1, 2, 3], "2022-06-01")


def test_train_save_load_predict(tmp_path):
    summary = train_all(None, tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert model_exists("Pune", tmp_path)

    bundle = load_model("Pune", tmp_path)
    assert set(bundle["metrics"]) == {"Mean baseline", "Linear Regression", "Random Forest"}
    # chronological: test period starts after train period ends
    assert bundle["train_period"][1] < bundle["test_period"][0]

    value = predict_next_day(bundle, [0, 5, 10, 2, 0, 8, 12], "2022-07-10")
    assert value >= 0
    assert "Pune" in summary["districts"]


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.train"):
        load_model("Atlantis", tmp_path)
