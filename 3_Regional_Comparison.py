"""Compare rainfall between districts."""

import plotly.express as px
import streamlit as st
from common import setup_page, show_sample_banner, load_data

setup_page("Regional Comparison", "🗺️")
show_sample_banner()

df = load_data()
all_districts = sorted(df["district"].unique())

chosen = st.sidebar.multiselect("Districts to compare", all_districts, default=all_districts[:3])

if not chosen:
    st.info("Pick at least one district in the sidebar.")
    st.stop()  # stop running the page here

data = df[df["district"].isin(chosen)]

# --- Chart 1: total rainfall per year ---
yearly = (
    data.assign(year=data["date"].dt.year)
    .groupby(["year", "district"])["rainfall_mm"]
    .sum()
    .reset_index()
)
fig = px.bar(yearly, x="year", y="rainfall_mm", color="district", barmode="group",
             title="Total yearly rainfall")
fig.update_layout(xaxis_title="Year", yaxis_title="Rainfall (mm)")
st.plotly_chart(fig, width="stretch")

# --- Chart 2: seasonal shape of each district ---
monthly = (
    data.assign(month=data["date"].dt.month)
    .groupby(["month", "district"])["rainfall_mm"]
    .mean()
    .reset_index()
)
fig2 = px.line(monthly, x="month", y="rainfall_mm", color="district", markers=True,
               title="Average daily rainfall by month")
fig2.update_layout(xaxis_title="Month (1 = Jan)", yaxis_title="Avg daily rainfall (mm)")
st.plotly_chart(fig2, width="stretch")

# --- Summary table ---
summary = data.groupby("district")["rainfall_mm"].agg(
    total_mm="sum", avg_daily_mm="mean", max_daily_mm="max"
).round(1)
st.dataframe(summary, width="stretch")
