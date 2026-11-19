"""Train and compare rainfall models, then save the best one per district.

Run from the project root:
    python -m src.train                      # uses synthetic sample data
    python -m src.train --data data/raw/rain.csv
"""

import argparse
import json
from pathlib import Path

import joblib
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.data_loader import load_rainfall
from src.evaluation import chronological_split, compute_metrics, get_feature_importance
from src.feature_engineering import FEATURE_COLUMNS, TARGET, add_basic_features
from src.preprocessing import clean_rainfall

MODELS_DIR = Path("models")


def build_models() -> dict:
    """The candidates from the README. A fresh set is made for each district."""
    return {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
    }


def train_district(features_df, district: str, test_fraction: float = 0.2) -> dict:
    """Train every candidate on one district and return a result bundle."""
    data = features_df[features_df["district"] == district]
    train, test = chronological_split(data, test_fraction)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET]

    fitted, metrics = {}, {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        # Rainfall can't be negative, so clip predictions at 0.
        predictions = model.predict(X_test).clip(min=0)
        metrics[name] = compute_metrics(y_test, predictions)
        fitted[name] = model

    # Choose the best learned model by lowest RMSE (baseline is for comparison only).
    contenders = [n for n in metrics if n != "Mean baseline"]
    best_name = min(contenders, key=lambda n: metrics[n]["rmse"])

    importance = get_feature_importance(fitted["Random Forest"], FEATURE_COLUMNS)
    return {
        "district": district,
        "best_model_name": best_name,
        "model": fitted[best_name],
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "feature_importance": importance.to_dict() if importance is not None else {},
        "train_period": [str(train["date"].min().date()), str(train["date"].max().date())],
        "test_period": [str(test["date"].min().date()), str(test["date"].max().date())],
    }


def train_all(data_path=None, models_dir=MODELS_DIR) -> dict:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    raw = load_rainfall(data_path)
    clean, report = clean_rainfall(raw)
    features = add_basic_features(clean)

    summary = {"data_source": str(data_path) if data_path else "synthetic sample data",
               "cleaning_report": report, "districts": {}}

    for district in sorted(features["district"].unique()):
        bundle = train_district(features, district)
        joblib.dump(bundle, models_dir / f"{district}.joblib")
        # The saved JSON must not contain the model object itself.
        summary["districts"][district] = {k: v for k, v in bundle.items() if k != "model"}

    (models_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MonsoonLens models.")
    parser.add_argument("--data", default=None, help="CSV with date, district, rainfall_mm")
    parser.add_argument("--out", default=str(MODELS_DIR), help="folder for saved models")
    args = parser.parse_args()

    result = train_all(args.data, args.out)
    print(f"Data source: {result['data_source']}")
    for name, info in result["districts"].items():
        print(f"\n{name} (best: {info['best_model_name']})")
        for model_name, m in info["metrics"].items():
            print(f"  {model_name:<18} MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R2={m['r2']:.3f}")
