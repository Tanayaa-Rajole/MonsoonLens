"""Highlight unusually heavy rainfall days using a z-score."""

import plotly.express as px
import streamlit as st
from common import setup_page, show_sample_banner, load_data
from src.anomaly_detection import flag_anomalies

setup_page("Anomaly Detection", "🚨")
show_sample_banner()

df = load_data()

district = st.sidebar.selectbox("District", sorted(df["district"].unique()))
threshold = st.sidebar.slider("Sensitivity (z-score threshold)", 1.0, 5.0, 3.0, 0.5)
st.sidebar.caption("Lower = more days flagged. Higher = only the most extreme.")

result = flag_anomalies(df[df["district"] == district], threshold)
flagged = result[result["is_anomaly"]]

c1, c2 = st.columns(2)
c1.metric("Days flagged", len(flagged))
c2.metric("Share of all days", f"{len(flagged) / len(result):.1%}")

fig = px.scatter(
    result, x="date", y="rainfall_mm", color="is_anomaly",
    color_discrete_map={False: "#9aa5b1", True: "#d62728"},
    title=f"Daily rainfall – {district} (red = unusual)",
)
fig.update_layout(xaxis_title="Date", yaxis_title="Rainfall (mm)")
st.plotly_chart(fig, width="stretch")

st.markdown("### Most extreme days")
top = flagged.sort_values("z_score", ascending=False)[
    ["date", "rainfall_mm", "month_mean", "z_score"]
].head(10)
st.dataframe(top, width="stretch")

st.caption(
    "A flagged day is a statistical deviation, not necessarily a meteorological "
    "emergency."
)
