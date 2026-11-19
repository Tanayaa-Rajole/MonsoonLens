# MonsoonLens Day 1: Learning Guide

This guide explains **what** each piece of code does, **how** it works, and **why** it exists. Read it next to the code, one file at a time.

Contents:
1. The big picture
2. How to run it
3. `src/`: the logic (no screens)
4. `app/`: the screens (Streamlit)
5. `tests/`: how we know it works
6. Repo files (`requirements.txt`, `.gitignore`, etc.)
7. A healthy Git plan for your first week
8. Honest limitations and what to build next

---

## 1. The big picture

The project is split into two layers:

```
src/   -> the "brain": generates data, builds features, finds anomalies
app/   -> the "face": Streamlit pages that call the brain and show charts
tests/ -> the "safety net": small checks that the brain behaves correctly
```

**Why split them?** If the logic lives inside the screens, you can't test it and can't reuse it. When you later add a real model in `src/train.py`, the dashboard just imports it. This is the same structure as your README's repository layout.

**Why synthetic data on Day 1?** You haven't downloaded IMD data yet. Instead of waiting, `src/sample_data.py` generates fake-but-realistic rainfall so you can build and see the whole dashboard today. Every page shows a yellow banner saying the data is synthetic, so nobody mistakes it for real measurements. Later you replace one function and every page keeps working.

---

## 2. How to run it

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
streamlit run app/Home.py      # opens http://localhost:8501
pytest                         # runs the tests
```

Run commands from the project root (the `MonsoonLens/` folder), because `pytest` and the imports expect that.

---

## 3. `src/`: the logic

### 3.1 `src/sample_data.py`

**What:** Creates a table with three columns: `date`, `district`, `rainfall_mm`, for five districts over 2019-2023.

**Block by block:**

```python
DISTRICTS = {"Pune": {"wetness": 0.8, "peak_month": 7.5}, ...}
```
- *What:* A dictionary of settings per district.
- *Why:* Different places get different amounts of rain at different times. Mumbai is very wet and peaks in July; Chennai peaks around October-November (northeast monsoon); Jaipur is dry. Keeping settings in one dictionary means adding a district is one line, not a code change.

```python
rng = np.random.default_rng(seed)
```
- *What:* A random number generator with a fixed **seed** (42).
- *Why:* A seed makes "random" numbers repeatable. Without it, every refresh gives different data, charts jump around, and tests become unreliable. (A test checks exactly this.)

```python
dates = pd.date_range(start, end, freq="D")
month = dates.month.to_numpy()
```
- *What:* One entry per day, plus an array of each day's month number.
- *Why:* Rainfall depends heavily on the month, so we need the month for every day.

```python
season = np.exp(-((month - params["peak_month"]) ** 2) / (2 * 1.3**2))
```
- *What:* A **bell curve** (Gaussian). It equals 1 at the peak month and falls toward 0 as you move away.
- *Why:* This is a simple way to fake seasonality: lots of rain near the peak, almost none far from it.

```python
rain_prob = 0.03 + 0.7 * season * params["wetness"] / 1.6
rains = rng.random(len(dates)) < rain_prob
```
- *What:* Step 1 of the two-step rain model. Each day gets a probability of rain (a small 3% baseline plus a seasonal part). We draw a random number in [0, 1); if it is below the probability, it rains.
- *Why:* Real rainfall has many zero days. Modelling "does it rain?" separately from "how much?" reproduces that.

```python
amount = rng.gamma(shape=0.8, scale=12 * params["wetness"] * (0.3 + season))
```
- *What:* Step 2. If it rains, draw the amount from a **gamma distribution**.
- *Why:* Gamma gives many small values and occasional big ones, which is how daily rainfall behaves. A normal (bell) distribution would wrongly produce negative rainfall.

```python
spike = np.where(rng.random(len(dates)) < 0.003, 4.0, 1.0)
rainfall = np.where(rains, amount * spike, 0.0).round(1)
```
- *What:* About 0.3% of days get their rainfall multiplied by 4. Then `np.where` keeps the amount on rainy days and sets 0 on dry days.
- *Why:* We want a few extreme days so the anomaly page has something to find.

```python
return pd.concat(frames, ignore_index=True)
```
- *What:* Stacks the five district tables into one long table.
- *Why:* "Long format" (one row per day per district) is what pandas and Plotly like best for grouping and colouring by district.

> Note: the numbers are plausible in shape, not calibrated to real climate. For example, annual totals come out lower than real figures for some districts. That is fine for a UI demo, and it is why the banner exists.

---

### 3.2 `src/anomaly_detection.py`

**What:** Marks days whose rainfall is unusually far from normal for that district and month.

**Idea (z-score):** `z = (value - average) / standard deviation`. A z-score of 3 means "3 standard deviations above this month's average", which is rare.

```python
out = df.copy()
```
- *Why:* `.copy()` stops us from accidentally changing the caller's original table. A very common pandas bug source.

```python
group = out.groupby(["district", "month"])["rainfall_mm"]
out["month_mean"] = group.transform("mean")
out["month_std"]  = group.transform("std")
```
- *What:* Computes the average and spread **per district and per month**, then puts that value on every row.
- *Why:* 40 mm in July in Mumbai is normal; 40 mm in March in Jaipur is extraordinary. Comparing each day to its own district-month is what makes the result meaningful (your README says "deviate from seasonal and regional patterns").
- `transform` (instead of `agg`) keeps the table the same length so we can do row-by-row maths.

```python
safe_std = out["month_std"].replace(0, np.nan)
out["z_score"] = (...) / safe_std).fillna(0)
```
- *What:* If a group has zero spread, dividing would crash or give infinity. We swap 0 for "not a number", divide, then set those results to 0.
- *Why:* Defensive coding for edge cases (like a month that was always dry).

```python
out["is_anomaly"] = out["z_score"].abs() > threshold
```
- *What:* True/False flag. `.abs()` catches both unusually high and unusually low values.
- *Why:* The threshold is a slider on the page so you can see how sensitivity changes the result.

> **Limitation to learn from:** rainfall is skewed (many zeros, a few huge days), but z-scores assume a roughly bell-shaped spread. So this method mostly flags *high* days and over-flags in dry months. Good first version; later you could try IQR or percentile-based methods and compare. That is a great README-worthy experiment.

---

### 3.3 `src/feature_engineering.py`

**What:** Converts the raw table into **features** (input columns) for a future model that predicts tomorrow's rain.

```python
out = df.sort_values(["district", "date"]).copy()
by_district = out.groupby("district")["rainfall_mm"]
```
- *Why sort:* Lags only make sense if rows are in time order.
- *Why group:* Without grouping, the first day of Mumbai would get "yesterday" from the last day of Pune. Grouping keeps each district's history separate.

```python
out["rain_lag_1"] = by_district.shift(1)
out["rain_lag_2"] = by_district.shift(2)
```
- *What:* `shift(1)` slides values down one row, so each row now holds the previous day's rainfall.
- *Why:* Yesterday's rain is one of the best clues for today's rain (wet spells cluster).

```python
out["rain_roll_7"] = by_district.transform(lambda s: s.shift(1).rolling(7).mean())
```
- *What:* The average rainfall of the previous 7 days.
- *Why `shift(1)` first:* A plain 7-day rolling average would include **today's** rainfall, which is the answer we are trying to predict. That is **data leakage**: the model would look amazing in testing and fail in real use. Shifting first guarantees only past information is used. The test `test_rolling_mean_excludes_today` protects this.

```python
out["month"] = out["date"].dt.month
out["is_monsoon"] = out["month"].between(6, 9)
```
- *What:* Calendar features. June to September is the Indian southwest monsoon season.
- *Why:* Models can't read dates; they need numbers. A simple monsoon flag gives them the season directly.

```python
return out.dropna().reset_index(drop=True)
```
- *What:* The first rows of each district have no history (their lags are empty), so we drop them.
- *Why:* Most models can't handle missing values.

---

## 4. `app/`: the screens

Streamlit rule of thumb: **the script runs top to bottom every time the user touches a widget.** Widgets (sliders, dropdowns) return their current value, and the code below uses it. That's the whole mental model.

Streamlit also auto-builds the sidebar menu from `app/pages/`. Files are listed in order of their number prefix (`1_`, `2_`...), and the name becomes the label.

### 4.1 `app/common.py`: shared helpers

```python
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```
- *What:* Finds the project root and adds it to Python's import search path.
- *Why:* Streamlit only adds the `app/` folder to the path, so `from src.sample_data import ...` would fail without this. It makes the app work no matter where you launch it from.

```python
def setup_page(title, icon="🌧️"):
    st.set_page_config(...)
    st.title(...)
```
- *What:* Sets the browser tab title, icon and wide layout, then draws the page heading.
- *Why:* All five pages need this. Writing it once avoids five copies. (`set_page_config` must be the first Streamlit call on a page.)

```python
@st.cache_data
def load_data():
    return generate_sample_data()
```
- *What:* `@st.cache_data` remembers the result after the first call.
- *Why:* Since the script reruns on every click, without caching you would regenerate (or re-read) the dataset every time. With real CSVs this matters a lot.

```python
def show_sample_banner(): st.warning(...)
```
- *Why:* Honesty about data. When you switch to real data, delete this one function's body and every banner disappears.

### 4.2 `app/Home.py`: landing page

- `setup_page(...)` and `st.subheader(...)`: heading and tagline.
- `df = load_data()` then `st.columns(3)` and `col.metric(...)`: three headline numbers (districts, days, rows). **Why:** metrics are the quickest way to show "the data is loaded and is this big".
- `st.markdown("""...""")`: a bulleted list of pages. **Why:** a visitor (or recruiter) should understand the app in 10 seconds.
- `st.progress(0.15, ...)`: a progress bar for project status. **Why:** a visible roadmap signal; update the number as you finish roadmap items.
- `st.caption(...)`: the disclaimer from your README's Limitations section. **Why:** an app showing rainfall should say it's not for disaster warnings.

### 4.3 `app/pages/1_Rainfall_Explorer.py`

```python
district = st.sidebar.selectbox("District", sorted(df["district"].unique()))
year_range = st.sidebar.slider("Years", min, max, (min, max))
```
- *What:* A dropdown and a two-handle slider in the sidebar. The slider returns a pair like `(2019, 2023)`.
- *Why:* The user chooses what to look at; the rest of the page reacts.

```python
mask = (df["district"] == district) & (df["date"].dt.year >= year_range[0]) & (...)
data = df[mask]
```
- *What:* A **boolean mask**: a True/False column. `df[mask]` keeps only the True rows. `&` means "and" (use `&`, not `and`, in pandas).
- *Why:* This is the core pandas filtering pattern; you'll use it constantly.

```python
c1.metric("Total rainfall", f"{data['rainfall_mm'].sum():,.0f} mm")
c2.metric("Rainy days", int((data["rainfall_mm"] > 0).sum()))
```
- *What:* Sum, and a count of days with rain above zero (`True` counts as 1 when summed).
- *Why:* Totals and rainy-day counts are the two most common rainfall summaries.

```python
monthly = data.set_index("date")["rainfall_mm"].resample("MS").sum().reset_index()
fig = px.bar(monthly, x="date", y="rainfall_mm", ...)
```
- *What:* `resample("MS")` groups daily data into months (MS = month start) and sums them. `px.bar` draws a bar chart.
- *Why:* Daily data over 5 years is ~1,800 noisy points; monthly totals show the monsoon rhythm clearly.

```python
by_month = data.assign(month=...).groupby("month")["rainfall_mm"].mean().reset_index()
fig2 = px.line(...)
```
- *What:* Average daily rainfall for each calendar month, across all selected years.
- *Why:* This is the **seasonal pattern** chart: the clearest way to see when the district's wet season is.

`st.plotly_chart(fig, width="stretch")` draws the chart at full page width. `st.expander` hides the raw table until clicked, keeping the page tidy.

### 4.4 `app/pages/2_Anomaly_Detection.py`

- Imports `flag_anomalies` from `src`. **Why:** the page only *displays*; the maths lives in `src` where it is tested.
- `threshold = st.sidebar.slider(..., 1.0, 5.0, 3.0, 0.5)`: min, max, default, step. **Why:** lets you feel the trade-off. Low threshold means many flags; high means only extremes.
- `px.scatter(..., color="is_anomaly", color_discrete_map={False: grey, True: red})`: every day is a dot; flagged days are red. **Why:** you see *when* anomalies happen and how far they stick out.
- `flagged.sort_values("z_score", ascending=False)[[...]].head(10)`: the 10 most extreme days as a table. **Why:** a chart shows the pattern; a table gives exact dates and numbers.
- Final `st.caption`: restates "statistical deviation, not an emergency" from your README.

### 4.5 `app/pages/3_Regional_Comparison.py`

```python
chosen = st.sidebar.multiselect("Districts to compare", all_districts, default=all_districts[:3])
if not chosen:
    st.info("Pick at least one district in the sidebar.")
    st.stop()
```
- *What:* Multi-select box. If the user clears it, we show a message and `st.stop()` ends the page.
- *Why:* Without this, empty selections would crash later charts. Handling the empty case is what makes an app feel solid.

- Yearly totals: `groupby(["year", "district"])["rainfall_mm"].sum()` then a grouped bar (`barmode="group"`). **Why:** compare wetness between districts and between years.
- Monthly average lines, one per district (`color="district"`). **Why:** shows that Chennai peaks later than Pune or Mumbai; different *timing*, not only different amounts.
- `.agg(total_mm="sum", avg_daily_mm="mean", max_daily_mm="max")`: named aggregations make a clean summary table in one step.

### 4.6 `app/pages/4_Prediction.py`

- **No prediction is made, on purpose.** No model exists yet, and inventing numbers would be misleading. The page says so in an `st.info` box.
- It calls `add_basic_features(...)` and shows `features.tail(15)`. **Why:** you can *see* the lag/rolling columns that the model will use, which makes the next step (training) much easier to understand.
- The markdown table defines each column; the "Planned models" list mirrors the README.
- When you train your first model, this page is where its output goes.

---

## 5. `tests/`: how we know it works

A test is a small function that runs your code on a tiny input where you already know the answer. If someone (including future you) breaks the code, `pytest` turns red.

| File | What it checks | Why it matters |
|---|---|---|
| `test_sample_data.py` | Correct columns and districts; no negative or missing rainfall; same seed gives identical data | Bad data silently ruins everything downstream |
| `test_features.py` | `rain_lag_1` is really yesterday; rolling mean **excludes today**; monsoon flag works | Guards against data leakage |
| `test_anomaly.py` | One huge day among normal days is flagged; constant rainfall flags nothing | Checks both the obvious case and an edge case |

Reading one example:

```python
rain = [5.0] * 30 + [500.0]       # 30 normal days and 1 huge one
out = flag_anomalies(df, threshold=3.0)
assert out["is_anomaly"].sum() == 1
```
`assert` means "this must be true, otherwise fail". Here: exactly one day should be flagged. Tests this small are quick to write and extremely valuable.

---

## 6. Repo files

| File | What and why |
|---|---|
| `requirements.txt` | Lists the libraries (streamlit, pandas, numpy, plotly, pytest). Anyone can recreate your environment with `pip install -r requirements.txt`. Streamlit is pinned to 1.50+ because the code uses `width="stretch"`. |
| `.gitignore` | Tells Git what **not** to upload: virtual environment, caches, big data files, models, secrets. Keeps the repo small and safe. `.gitkeep` files are empty placeholders so Git still creates the otherwise-empty `data/raw`, `data/processed` and `models` folders. |
| `LICENSE` | MIT licence, as your README states. Lets others use your code while keeping your credit. |
| `data/README.md` | Says what goes in `data/` and that real data isn't committed. |
| `reports/model_results.md` | Placeholder. Your README promises results only after training, so it says "none yet". |
| `src/__init__.py` | An empty file that marks `src` as a Python package, so `from src.x import y` works. |
| `README.md` | Your own README, unchanged. |

---

## 7. A healthy Git plan for your first week

The contribution graph rewards **real, steady work**, and the best way to get a green graph is to commit small pieces as you actually finish them. Please don't backdate commits or make empty ones; recruiters and reviewers can tell, and it teaches you nothing. The upside is that this project is naturally full of small steps, so you can honestly commit almost daily.

One-time setup (replace the URL with your repo):

```bash
git init
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/MonsoonLens.git
```

Suggested commits, one per sitting, each pushed with `git push`:

| Day | Commit message | What you add |
|---|---|---|
| 1 | `docs: add project README` | `README.md`, `LICENSE`, `.gitignore` |
| 1 | `chore: add folder structure and requirements` | folders, `.gitkeep`, `requirements.txt`, `data/README.md` |
| 2 | `feat: add synthetic sample data generator` | `src/sample_data.py` and its test |
| 2 | `feat: add Streamlit home page` | `app/common.py`, `app/Home.py` |
| 3 | `feat: add rainfall explorer page` | `1_Rainfall_Explorer.py` |
| 4 | `feat: add z-score anomaly detection` | `src/anomaly_detection.py`, test, page 2 |
| 5 | `feat: add regional comparison page` | page 3 |
| 6 | `feat: add feature engineering and prediction preview` | `src/feature_engineering.py`, test, page 4 |
| 7 | `docs: add learning guide, tick roadmap items` | this file and README checkboxes |

Commands for each commit:

```bash
git add <the files for that step>
git commit -m "feat: add synthetic sample data generator"
git push -u origin main      # -u only needed the first time
```

Tips: commit when one small thing works; write messages as "what changed"; run `pytest` before committing; tick the matching box in the README roadmap when a step is truly done (for example, "Develop rainfall-related features" after Day 6).

---

## 8. Honest limitations and what to build next

- The data is synthetic; charts show the *shape* of the app, not real climate.
- Z-score anomaly detection is a rough first method (see 3.2).
- The Prediction page makes no predictions yet.

**Next steps (matching your roadmap):**
1. Download a real IMD or CHIRPS file for one district and write `src/data_loader.py` so `load_data()` reads it instead of generating fake data.
2. Write `src/preprocessing.py` (missing values, date validation) with tests.
3. In `notebooks/03_model_experiments.ipynb`, split data **by time** (train on earlier years, test on the latest), then compare the mean baseline, Linear Regression and Random Forest with MAE, RMSE and R².
4. Move the working code to `src/train.py`, save the model with Joblib, and show its results on the Prediction page.
5. Add feature importance (`model.feature_importances_`) for the explainability section.

Learn-by-doing exercise for tomorrow: change `threshold` defaults, add a sixth district to `DISTRICTS`, and watch every page update without touching any app code. That is the payoff of the `src/` and `app/` split.
