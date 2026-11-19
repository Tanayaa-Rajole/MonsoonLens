from src.sample_data import DISTRICTS, generate_sample_data


def test_has_expected_columns_and_districts():
    df = generate_sample_data("2020-01-01", "2020-12-31")
    assert list(df.columns) == ["date", "district", "rainfall_mm"]
    assert set(df["district"]) == set(DISTRICTS)


def test_no_negative_or_missing_rainfall():
    df = generate_sample_data("2020-01-01", "2020-12-31")
    assert df["rainfall_mm"].notna().all()
    assert (df["rainfall_mm"] >= 0).all()


def test_same_seed_gives_same_data():
    a = generate_sample_data("2020-01-01", "2020-03-31", seed=1)
    b = generate_sample_data("2020-01-01", "2020-03-31", seed=1)
    assert a.equals(b)
