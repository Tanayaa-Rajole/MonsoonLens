"""Prediction page: model results and a next-day estimate (needs a trained model)."""

import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from common import setup_page, show_sample_banner, load_data
from src.feature_engineering import add_basic_features
from src.predict import load_model, model_exists, predict_next_day

setup_page("Prediction", "🔮")
show_sample_banner()

df = load_data()
district = st.sidebar.selectbox("District", sorted(df["district"].unique()))

# ---------------------------------------------------------------- model section
if not model_exists(district):
    st.info(
        "No trained model found. In a terminal, from the project folder, run "
        "`python -m src.train`, then refresh this page."
    )
else:
    bundle = load_model(district)

    st.markdown("### Model comparison (test set)")
    metrics = pd.DataFrame(bundle["metrics"]).T.round(3)
    metrics.columns = ["MAE (mm)", "RMSE (mm)", "R²"]
    st.dataframe(metrics, width="stretch")
    st.caption(
        f"Trained on {bundle['train_period'][0]} to {bundle['train_period'][1]}; "
        f"tested on {bundle['test_period'][0]} to {bundle['test_period'][1]}. "
        "Lower MAE/RMSE is better; R² near 0 means no better than guessing the average."
    )

    if bundle["feature_importance"]:
        st.markdown("### Which features mattered? (Random Forest)")
        imp = pd.Series(bundle["feature_importance"]).sort_values()
        st.bar_chart(imp, horizontal=True)

    st.markdown(f"### Try it: estimate rainfall for one day ({bundle['best_model_name']})")
    st.write("Enter rainfall (mm) for the last 7 days, oldest first.")
    cols = st.columns(7)
    last_week = [
        cols[i].number_input(f"Day {i + 1}", min_value=0.0, value=0.0, step=1.0, key=f"d{i}")
        for i in range(7)
    ]
    target = st.date_input("Day to estimate", date.today() + timedelta(days=1))
    if st.button("Estimate"):
        value = predict_next_day(bundle, last_week, target)
        st.metric("Estimated rainfall", f"{value:.1f} mm")
        st.caption("Educational estimate from sample data only. Not a forecast.")

# ------------------------------------------------------------ feature preview
with st.expander("Feature table used for training"):
    features = add_basic_features(df[df["district"] == district])
    st.dataframe(features.tail(15), width="stretch")
    st.markdown(
        """
| Column | Meaning |
|---|---|
| `rainfall_mm` | **Target**: the value we predict |
| `rain_lag_1`, `rain_lag_2` | Rainfall 1 and 2 days earlier |
| `rain_roll_7` | Average of the previous 7 days |
| `month`, `is_monsoon` | Calendar and season indicators |
"""
    )
