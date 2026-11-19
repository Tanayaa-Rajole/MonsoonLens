import pytest

from src.data_loader import load_rainfall


def test_default_is_sample_data():
    df = load_rainfall()
    assert {"date", "district", "rainfall_mm"} <= set(df.columns)


def test_reads_valid_csv(tmp_path):
    f = tmp_path / "rain.csv"
    f.write_text("date,district,rainfall_mm\n2022-06-01,Pune,5\n")
    assert len(load_rainfall(f)) == 1


def test_missing_columns_gives_clear_error(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("day,place,rain\n2022-06-01,Pune,5\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_rainfall(f)


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rainfall(tmp_path / "nope.csv")
