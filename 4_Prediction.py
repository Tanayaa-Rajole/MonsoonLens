"""Prediction page (Day 1: feature preview only, no trained model yet)."""

import streamlit as st
from common import setup_page, show_sample_banner, load_data
from src.feature_engineering import add_basic_features

setup_page("Prediction", "🔮")
show_sample_banner()

st.info(
    "No model is trained yet, so this page does **not** make predictions. "
    "Below is the feature table the models will learn from."
)

df = load_data()
district = st.sidebar.selectbox("District", sorted(df["district"].unique()))

features = add_basic_features(df[df["district"] == district])

st.markdown("### Features built from past rainfall")
st.dataframe(features.tail(15), width="stretch")

st.markdown(
    """
| Column | Meaning |
|---|---|
| `rainfall_mm` | **Target** – the value we want to predict |
| `rain_lag_1`, `rain_lag_2` | Rainfall 1 and 2 days earlier |
| `rain_roll_7` | Average of the previous 7 days |
| `month`, `is_monsoon` | Calendar and season indicators |
"""
)

st.markdown("### Planned models")
st.markdown(
    "1. Mean baseline  \n2. Linear Regression  \n3. Random Forest  \n"
    "4. XGBoost (optional)"
)
st.caption("Results will be added to this page after training and evaluation.")
