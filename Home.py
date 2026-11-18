"""MonsoonLens home page. Run with: streamlit run app/Home.py"""

import streamlit as st
from common import setup_page, show_sample_banner, load_data

setup_page("MonsoonLens", "🌧️")
st.subheader("Turning rainfall data into meaningful intelligence")
show_sample_banner()

st.write(
    "MonsoonLens explores historical rainfall patterns across Indian districts, "
    "flags unusual rainfall, and (soon) estimates next-day rainfall with machine learning."
)

# --- Quick numbers about the loaded data ---
df = load_data()
col1, col2, col3 = st.columns(3)
col1.metric("Districts", df["district"].nunique())
col2.metric("Days of data", df["date"].nunique())
col3.metric("Total records", f"{len(df):,}")

# --- Feature overview ---
st.markdown("### What you can do")
st.markdown(
    """
- 📈 **Rainfall Explorer** – monthly totals and seasonal patterns
- 🚨 **Anomaly Detection** – spot unusually heavy rainfall days
- 🗺️ **Regional Comparison** – compare districts side by side
- 🔮 **Prediction** – feature preview now, model coming soon
"""
)
st.info("👈 Use the sidebar to open a page.")

# --- Roadmap status ---
st.markdown("### Project status")
st.progress(0.15, text="Day 1: dashboard skeleton ready")
st.caption(
    "Educational and analytical use only – not for operational weather "
    "forecasting or disaster warnings."
)
