import streamlit as st
from pathlib import Path
import base64

st.set_page_config(
    page_title="SVG Icon Test",
    layout="wide"
)

BASE = Path(__file__).resolve().parent
ICON_DIR = BASE / "assets" / "icons"


def svg_to_data_uri(filename):
    path = ICON_DIR / filename

    if not path.exists():
        return None

    svg = path.read_text(encoding="utf-8")

    encoded = base64.b64encode(
        svg.encode("utf-8")
    ).decode("utf-8")

    return f"data:image/svg+xml;base64,{encoded}"


icons = [
    ("building-factory-2.svg", "Factory"),
    ("home.svg", "Home"),
    ("tool.svg", "Predict Output"),
    ("chart-dots-2.svg", "Model Insights"),
    ("database-search.svg", "Data Exploration"),
    ("info-circle.svg", "About"),
    ("chevron-left.svg", "Collapse"),
    ("settings-2.svg", "Settings"),
]


st.title("SVG Icon Experiment 2")

st.write("Icon folder:")

st.code(str(ICON_DIR))

st.divider()


for filename, label in icons:

    uri = svg_to_data_uri(filename)

    st.subheader(label)

    if uri is None:

        st.error(f"❌ {filename} not found")

    else:

        # IMPORTANT:
        # Everything is deliberately on ONE LINE.
        html = (
            f'<div style="display:flex;align-items:center;'
            f'gap:20px;margin:15px 0;">'
            f'<img src="{uri}" width="40" height="40">'
            f'<span style="font-size:20px;">{filename}</span>'
            f'</div>'
        )

        st.markdown(
            html,
            unsafe_allow_html=True
        )

    st.divider()