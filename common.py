"""Helpers shared by every Streamlit page."""

import sys
from pathlib import Path

# Make the project root importable so pages can do `from src... import ...`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.sample_data import generate_sample_data  # noqa: E402


def setup_page(title: str, icon: str = "🌧️") -> None:
    """Set the browser tab title/icon and the page heading."""
    st.set_page_config(page_title=f"MonsoonLens | {title}", page_icon=icon, layout="wide")
    st.title(f"{icon} {title}")


@st.cache_data  # run once, then reuse the result (keeps the app fast)
def load_data():
    return generate_sample_data()


def show_sample_banner() -> None:
    st.warning(
        "Showing **synthetic sample data** for development. "
        "Real IMD data has not been added yet."
    )
