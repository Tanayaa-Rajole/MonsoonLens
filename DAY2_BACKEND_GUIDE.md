# MonsoonLens Day 2: Backend Guide

Day 1 gave you the face (Streamlit pages). Day 2 builds the **brain**: loading, cleaning, training, evaluating and predicting. As before, each block has **what / how / why**.

Contents:
1. What changed today
2. The data journey (one picture)
3. `src/data_loader.py`
4. `src/preprocessing.py`
5. `src/feature_engineering.py` (updated)
6. `src/evaluation.py`
7. `src/train.py`
8. `src/predict.py`
9. `app/pages/4_Prediction.py` (updated)
10. New tests
11. Reading your first results honestly
12. Git plan for Day 2 and next steps

---

## 1. What changed today

| New / changed | Role |
|---|---|
| `src/data_loader.py` (new) | Reads a CSV or falls back to sample data |
| `src/preprocessing.py` (new) | Cleans data and reports what it fixed |
| `src/evaluation.py` (new) | Time-based split, MAE / RMSE / R², feature importance |
| `src/train.py` (new) | Trains 3 models per district, keeps the best, saves it |
| `src/predict.py` (new) | Loads a saved model and estimates one day |
| `src/feature_engineering.py` | Added `FEATURE_COLUMNS`, `TARGET`, `features_for_next_day` |
| `app/pages/4_Prediction.py` | Now shows results and a next-day estimate |
| 16 new tests | 24 tests in total |

Run the whole backend with one command:

```bash
python -m src.train
```

It prints a results table and writes models to `models/` (ignored by Git because trained files are large and re-creatable).

---

## 2. The data journey

```
CSV or sample data
      |  data_loader.load_rainfall
      v
raw table (date, district, rainfall_mm)
      |  preprocessing.clean_rainfall
      v
clean table + report of what was fixed
      |  feature_engineering.add_basic_features
      v
feature table (lags, rolling mean, month, monsoon flag)
      |  evaluation.chronological_split   (old -> train, new -> test)
      v
train.train_district: fit 3 models, score on test set
      |  joblib.dump
      v
models/Pune.joblib  +  models/metrics.json
      |  predict.load_model + predict_next_day
      v
one estimate in mm
```

Each stage has one job and hands a table to the next. That is why each lives in its own file.

---

## 3. `src/data_loader.py`

**What:** One function, `load_rainfall(path=None)`.

```python
if path is None:
    return generate_sample_data()
```
- *Why:* Keeps the app working today with no real data. When you download IMD data, you only pass a path; nothing else changes.

```python
if not path.exists():
    raise FileNotFoundError(f"Data file not found: {path}")
```
- *What:* Fails early with a plain message.
- *Why:* A typo in a path should say "file not found: ..." instead of a confusing pandas traceback.

```python
missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(...)
```
- *What:* A list comprehension that collects any required column the file lacks.
- *Why:* Real datasets rarely use your exact column names. The error lists what it found so you know what to rename. This is called a **data contract**: the rest of the code is allowed to assume these three columns exist.

```python
return df[REQUIRED_COLUMNS]
```
- Keeps only the columns we need, in a fixed order.

---

## 4. `src/preprocessing.py`

**What:** `clean_rainfall(df)` returns `(clean_df, report)`.

**Why a report?** Silent cleaning hides problems. If 30% of rows were bad, you want to know. The report is a dictionary of counts.

```python
out["date"] = pd.to_datetime(out["date"], errors="coerce")
report["bad_dates"] = int(out["date"].isna().sum())
out = out.dropna(subset=["date"])
```
- *What:* Converts text to real dates. `errors="coerce"` turns unreadable ones into `NaT` (the date version of NaN) instead of crashing. We count them, then drop them.
- *Why:* A row with no valid date cannot be placed in a time series.

```python
out["rainfall_mm"] = pd.to_numeric(out["rainfall_mm"], errors="coerce")
```
- *What:* Text like `"NA"` or `"-"` becomes NaN.
- *Why:* Many meteorological files use text placeholders for missing values.

```python
negative = out["rainfall_mm"] < 0
out.loc[negative, "rainfall_mm"] = float("nan")
```
- *What:* Negative values are impossible, so we mark them missing. `out.loc[mask, column] = value` is the safe pandas way to change selected cells.
- *Why not delete the row?* Deleting would remove a calendar day and break the lag features (see below). A NaN keeps the day's slot but marks it unknown.

```python
out = out.drop_duplicates(subset=["district", "date"], keep="first")
```
- *Why:* Two rows for the same district and day would double-count rainfall.

**`_fill_calendar` (the important one):**

```python
full_range = pd.date_range(part.index.min(), part.index.max(), freq="D")
part = part.reindex(full_range)
```
- *What:* For each district, build every day between its first and last date, and `reindex` so absent days appear as NaN rows.
- *Why:* `shift(1)` means "the previous *row*", not "the previous *day*". If 3 June is missing and the data jumps from 2 June to 4 June, `shift(1)` would treat 2 June as "yesterday" for 4 June, which is wrong. A complete calendar makes row and day the same thing. The NaN rows are then removed safely by `dropna()` in feature engineering, along with any row whose window touches them.

---

## 5. `src/feature_engineering.py` (updated)

```python
TARGET = "rainfall_mm"
FEATURE_COLUMNS = ["rain_lag_1", "rain_lag_2", "rain_roll_7", "month", "is_monsoon"]
```
- *Why:* One shared definition of "inputs" and "output". Training and prediction import the same list, so they can never drift apart.

```python
out["is_monsoon"] = out["month"].between(6, 9).astype(int)
```
- Changed from True/False to 1/0. *Why:* scikit-learn models want numbers; this removes any ambiguity.

```python
def features_for_next_day(recent_rainfall, target_date):
```
- *What:* Builds one feature row for a future day from the last 7 observed values.
- *Why it exists:* At prediction time there's no table, just "here is the last week; what about tomorrow?". Computing features by hand in the app would risk a **train/serve mismatch** (training uses one definition, prediction another), which silently ruins accuracy. A test (`test_next_day_features_match_training_features`) proves both paths give identical numbers.

---

## 6. `src/evaluation.py`

```python
def chronological_split(df, test_fraction=0.2):
    df = df.sort_values("date")
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut], df.iloc[cut:]
```
- *What:* Oldest 80% for training, newest 20% for testing. `iloc[:cut]` means "rows before position cut".
- *Why not `train_test_split`?* That shuffles. For time data, shuffling lets the model learn from days *after* the test days, like studying tomorrow's newspaper before predicting tomorrow. Your README promised this chronological split.

```python
"mae":  mean_absolute_error(...),
"rmse": np.sqrt(mean_squared_error(...)),
"r2":   r2_score(...)
```
- **MAE:** the average miss in mm. Easy to explain.
- **RMSE:** squares errors before averaging, so one big miss hurts a lot. Good for rainfall because large misses are what matter.
- **R²:** 1 is perfect, 0 means "no better than always guessing the average", negative means worse than that.
- `float(...)` converts NumPy numbers to plain Python numbers so they can be saved as JSON.

```python
def get_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return None
```
- *What:* Random Forests expose `feature_importances_`, which says how much each input helped. Other models don't, so we return `None`.
- *Why:* This is your README's "Explainable ML" section starting to exist. Pandas `Series(..., index=names)` pairs numbers with names so the chart is readable.

---

## 7. `src/train.py`

```python
def build_models():
    return {"Mean baseline": DummyRegressor(strategy="mean"),
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(...)}
```
- *Why a baseline?* It always predicts the training average. Any real model must beat it, otherwise it has learned nothing. Without it, a number like MAE = 2.1 means nothing.
- *Why a function, not a global?* Models remember what they were trained on. A fresh set per district prevents one district's training from leaking into the next.
- Random Forest settings: `n_estimators=200` (200 trees), `max_depth=10` and `min_samples_leaf=5` (limit complexity so it doesn't memorise training days), `random_state=42` (repeatable), `n_jobs=-1` (use all CPU cores).

```python
for name, model in build_models().items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test).clip(min=0)
    metrics[name] = compute_metrics(y_test, predictions)
```
- *What:* The core ML loop: **fit** on the past, **predict** the held-out future, **score**. `.clip(min=0)` stops impossible negative rainfall (linear regression can output them).
- *Why a loop?* All models share the same interface (`fit` / `predict`), so adding XGBoost later is one more dictionary entry.

```python
best_name = min(contenders, key=lambda n: metrics[n]["rmse"])
```
- *What:* Picks the model with the lowest RMSE, excluding the baseline.
- *`key=lambda ...`* tells `min` what to compare by.

```python
joblib.dump(bundle, models_dir / f"{district}.joblib")
```
- *What:* Saves a **bundle** (model + feature names + metrics + date ranges) to disk.
- *Why a bundle?* The model alone is not enough: prediction needs to know which columns, in which order. Saving them together prevents mismatches.

```python
summary["districts"][district] = {k: v for k, v in bundle.items() if k != "model"}
(models_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
```
- *What:* A dictionary comprehension that copies everything except the model object, since JSON can't store a model. The JSON file is a human-readable results log.

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(...)
```
- *What:* Lets you run `python -m src.train --data my.csv`. The `if __name__` guard means this part runs only when executed as a script, not when tests import the file.
- `-m src.train` runs it as part of the package so `from src...` imports work.

---

## 8. `src/predict.py`

- `model_exists` / `load_model`: check for and load `models/<district>.joblib`. A missing model raises a message telling you the exact command to run, which is far kinder than a raw file error.
- `predict_next_day(bundle, recent_rainfall, target_date)`:
  1. builds one feature row with `features_for_next_day`,
  2. selects columns in the saved order (`row[bundle["features"]]`),
  3. predicts, then `max(prediction, 0.0)` to avoid negatives.

*Why this is separate from `train.py`:* the dashboard only needs prediction, not training code or heavy imports.

---

## 9. `app/pages/4_Prediction.py` (updated)

- `if not model_exists(district): st.info(...)`: the page degrades gracefully with instructions instead of crashing.
- A results table: `pd.DataFrame(bundle["metrics"]).T` flips the dictionary so models are rows and metrics are columns.
- `st.bar_chart(imp, horizontal=True)`: feature importance chart.
- 7 `st.number_input` boxes created in a loop, then `st.button("Estimate")`. When clicked, it calls `predict_next_day`. The result is labelled "Educational estimate from sample data only. Not a forecast."
- The Day 1 feature table moved into an expander to keep the page clean.

---

## 10. New tests (24 total)

| File | Protects against |
|---|---|
| `test_data_loader.py` | Wrong paths, wrong column names |
| `test_preprocessing.py` | Bad dates, negatives, duplicates, gaps, text like "NA" |
| `test_evaluation.py` | Shuffled/overlapping splits; metric formulas checked against hand-calculated values |
| `test_train_predict.py` | Train/serve feature mismatch; full train, save, load, predict round trip; clear error for missing model |

`tmp_path` in tests is a throwaway folder pytest creates, so tests never touch your real `models/` folder.

---

## 11. Reading your first results honestly

When you run `python -m src.train` on the sample data, you'll see R² values of roughly 0 to 0.15, and Random Forest sometimes *worse* than Linear Regression. **That is correct, not a bug.**

Why: the sample generator makes each day's rain independent of yesterday's, apart from the season. So the only learnable signal is "it rains more in July", and lag features add almost nothing. A real dataset has wet spells (rainy days cluster), so lags should help there.

Lessons to take away:
- **Never report these numbers as results.** They describe synthetic data. That's why the README roadmap boxes for training and evaluating Random Forest stay **unticked** until you run this on real IMD/CHIRPS data. The code is ready; the evidence isn't yet.
- Daily rainfall is hard to predict. Low R² is normal even on real data, so judge models by *improvement over the baseline* (MAE and RMSE), not by R² alone.
- Always compare against the baseline. It's what turns a number into a conclusion.

---

## 12. Git plan for Day 2 and next steps

Commit each step once it works (and `pytest` passes):

| Commit message | Files |
|---|---|
| `feat: add CSV data loader with validation` | `src/data_loader.py`, `tests/test_data_loader.py` |
| `feat: add preprocessing pipeline with cleaning report` | `src/preprocessing.py`, `tests/test_preprocessing.py` |
| `feat: add chronological split and regression metrics` | `src/evaluation.py`, `tests/test_evaluation.py` |
| `feat: add model training with baseline comparison` | `src/train.py`, `src/feature_engineering.py`, `requirements.txt` |
| `feat: add next-day prediction and wire into dashboard` | `src/predict.py`, `app/pages/4_Prediction.py`, `tests/test_train_predict.py` |
| `docs: add Day 2 backend guide` | this file |

Your README roadmap: "Implement data preprocessing pipeline" is done in code, but wait to tick the model items until real-data results exist.

**Day 3 suggestions:**
1. Get one real dataset (IMD gridded data or a CHIRPS extract for a district), convert it to `date,district,rainfall_mm`, and run `python -m src.train --data data/raw/yourfile.csv`.
2. Open `notebooks/01_data_exploration.ipynb`: plot the real data and check the cleaning report.
3. Compare real-data results with the baseline, then record them in `reports/model_results.md`.
4. Try adding a 3rd lag or a 30-day rolling total and see whether RMSE improves. Each experiment is a good commit.
