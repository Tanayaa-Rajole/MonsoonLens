"""Explore historical rainfall for one district."""

import plotly.express as px
import streamlit as st
from common import setup_page, show_sample_banner, load_data

setup_page("Rainfall Explorer", "📈")
show_sample_banner()

df = load_data()

# --- Sidebar controls ---
district = st.sidebar.selectbox("District", sorted(df["district"].unique()))
years = sorted(df["date"].dt.year.unique())
year_range = st.sidebar.slider("Years", min(years), max(years), (min(years), max(years)))

# --- Filter the data based on the controls ---
mask = (
    (df["district"] == district)
    & (df["date"].dt.year >= year_range[0])
    & (df["date"].dt.year <= year_range[1])
)
data = df[mask]

# --- Headline numbers ---
c1, c2, c3 = st.columns(3)
c1.metric("Total rainfall", f"{data['rainfall_mm'].sum():,.0f} mm")
c2.metric("Rainy days", int((data["rainfall_mm"] > 0).sum()))
c3.metric("Heaviest day", f"{data['rainfall_mm'].max():.1f} mm")

# --- Chart 1: monthly totals over time ---
monthly = data.set_index("date")["rainfall_mm"].resample("MS").sum().reset_index()
fig = px.bar(monthly, x="date", y="rainfall_mm", title=f"Monthly rainfall – {district}")
fig.update_layout(xaxis_title="Month", yaxis_title="Rainfall (mm)")
st.plotly_chart(fig, width="stretch")

# --- Chart 2: average rainfall by calendar month (seasonality) ---
by_month = (
    data.assign(month=data["date"].dt.month)
    .groupby("month")["rainfall_mm"]
    .mean()
    .reset_index()
)
fig2 = px.line(by_month, x="month", y="rainfall_mm", markers=True,
               title="Average daily rainfall by month (seasonal pattern)")
fig2.update_layout(xaxis_title="Month (1 = Jan)", yaxis_title="Avg daily rainfall (mm)")
st.plotly_chart(fig2, width="stretch")

with st.expander("Show raw data"):
    st.dataframe(data, width="stretch")
