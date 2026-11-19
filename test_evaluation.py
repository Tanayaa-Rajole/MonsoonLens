import pandas as pd
import pytest

from src.evaluation import chronological_split, compute_metrics


def test_split_is_chronological_with_no_overlap():
    df = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=10)}).sample(
        frac=1, random_state=0  # shuffled on purpose
    )
    train, test = chronological_split(df, 0.2)
    assert len(train) == 8 and len(test) == 2
    assert train["date"].max() < test["date"].min()


def test_bad_fraction_rejected():
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame({"date": []}), 1.5)


def test_metrics_known_values():
    m = compute_metrics([0, 10], [2, 6])  # errors: 2 and 4
    assert m["mae"] == pytest.approx(3.0)
    assert m["rmse"] == pytest.approx(((4 + 16) / 2) ** 0.5)


def test_perfect_prediction_gives_r2_of_one():
    assert compute_metrics([1, 2, 3], [1, 2, 3])["r2"] == pytest.approx(1.0)
