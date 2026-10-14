import streamlit as st

import pandas as pd

import numpy as np

import joblib

import json

import base64

from pathlib import Path
from urllib.parse import quote





# ============================================================

# PAGE CONFIG

# ============================================================



st.set_page_config(

    page_title="Manufacturing Output Predictor",

    page_icon="🏭",

    layout="wide",

    initial_sidebar_state="expanded",

)





# ============================================================

# PATHS

# ============================================================



BASE = Path(__file__).resolve().parent



MODEL_PATH = BASE / "models" / "manufacturing_output_prediction_model.pkl"

METRICS_PATH = BASE / "models" / "model_metrics.csv"

CONFIG_PATH = BASE / "models" / "feature_config.json"

DATA_PATH = BASE / "data" / "manufacturing_dataset_1000_samples.csv"

HERO_PATH = BASE / "assets" / "factory_background.jpg"



ICON_DIR = BASE / "assets" / "icons"





# ============================================================

# LOAD MODEL / DATA

# ============================================================



@st.cache_resource

def get_model():

    return joblib.load(MODEL_PATH)





@st.cache_data

def get_data():

    return pd.read_csv(DATA_PATH)





@st.cache_data

def get_metrics():

    return pd.read_csv(METRICS_PATH)





@st.cache_data

def get_config():

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:

        return json.load(f)





model = get_model()

df = get_data()

metrics = get_metrics()

config = get_config()



TARGET = config.get("target", "Parts_Per_Hour")

FEATURES = config.get("features_used", [])





# ============================================================

# HELPERS

# ============================================================



def val(column, fallback=0.0):

    if column not in df.columns:

        return fallback



    x = pd.to_numeric(

        df[column],

        errors="coerce"

    ).median()



    return fallback if pd.isna(x) else float(x)





def cats(column, fallback):

    if column not in df.columns:

        return fallback



    x = (

        df[column]

        .dropna()

        .astype(str)

        .unique()

        .tolist()

    )



    return sorted(x) if x else fallback





def get_metric(metric_name, fallback=0.0):



    if metrics.empty:

        return fallback



    possible = {

        "MSE": ["MSE", "mse"],

        "RMSE": ["RMSE", "rmse"],

        "MAE": ["MAE", "mae"],

        "R2": ["R²", "R2", "r2", "R_Squared"],

    }.get(metric_name, [metric_name])



    row = metrics



    if "Model" in metrics.columns:



        refined = metrics[

            metrics["Model"]

            .astype(str)

            .str.contains(

                "Refined",

                case=False,

                na=False

            )

        ]



        if not refined.empty:

            row = refined



    for col in possible:



        if col in row.columns:



            try:

                return float(row.iloc[-1][col])

            except Exception:

                pass



    return fallback





MSE = get_metric("MSE")

RMSE = get_metric("RMSE")

MAE = get_metric("MAE")

R2 = get_metric("R2")





def add_time_features(frame):



    x = frame.copy()



    if "Timestamp" in x.columns:



        ts = pd.to_datetime(

            x["Timestamp"],

            errors="coerce"

        )



        hour = ts.dt.hour.fillna(12)



        x["Hour_Sin"] = np.sin(

            2 * np.pi * hour / 24

        )



        x["Hour_Cos"] = np.cos(

            2 * np.pi * hour / 24

        )



    return x





def html_one_line(value):

    return " ".join(str(value).split())





def custom_html(value):

    st.markdown(

        html_one_line(value),

        unsafe_allow_html=True

    )





# ============================================================

# SVG ICON HELPERS

# ============================================================



def icon_svg(name):



    path = ICON_DIR / f"{name}.svg"



    if path.exists():



        return path.read_text(

            encoding="utf-8"

        )



    return ""





def icon_data_uri(name, color=None):



    raw = icon_svg(name)



    if not raw:

        return ""



    if color:

        raw = raw.replace(

            "currentColor",

            color

        )



    encoded = base64.b64encode(

        raw.encode("utf-8")

    ).decode("ascii")



    return (

        "data:image/svg+xml;base64,"

        + encoded

    )





# ============================================================

# MAIN CSS

# ============================================================



st.markdown(

    """

<style>



@import url(

    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'

);





/* ============================================================

   GLOBAL

   ============================================================ */



:root {

    --navy: #062441;

    --navy2: #0a3155;

    --blue: #1769e8;

    --green: #12a66a;

    --purple: #7445e8;

    --orange: #ef9515;

    --ink: #0b2b4b;

    --muted: #617792;

    --border: #dce6f0;

    --bg: #f5f8fc;

}



html,

body,

[class*="css"] {

    font-family: 'Inter', sans-serif;

}



.stApp {

    background: var(--bg);

    color: var(--ink);

}



header[data-testid="stHeader"] {

    background: transparent;

}



.block-container {

    max-width: 1390px;

    padding-top: 1rem;

    padding-bottom: 2rem;

}





/* ============================================================

   SIDEBAR

   ============================================================ */



[data-testid="stSidebar"] {

    background:

        linear-gradient(

            180deg,

            #061f39 0%,

            #062d4e 100%

        );

    border-right: none;

}



[data-testid="stSidebar"] > div:first-child {

    padding-top: 1rem;

}





/* BRAND */



.sidebar-brand {

    padding: 22px 18px 24px;

    color: #ffffff;

}



.sidebar-brand-icon {

    width: 45px;

    height: 45px;

    margin-bottom: 15px;

    color: #ffffff;

}



.sidebar-brand-icon svg {

    width: 45px !important;

    height: 45px !important;

}



.sidebar-brand-title {

    font-size: 16px;

    line-height: 1.35;

    font-weight: 800;

}





/* NAVIGATION TITLE */



.sidebar-nav-title {

    color: #a8bfd4;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1.2px;

    margin: 6px 18px 8px;

}





/* ============================================================
   CUSTOM SVG NAVIGATION
   ============================================================ */

.sidebar-nav {
    display: flex;
    flex-direction: column;
    gap: 3px;
    width: 100%;
}

.sidebar-nav-item {
    display: flex;
    align-items: center;
    width: calc(100% - 16px);
    min-height: 46px;
    box-sizing: border-box;
    margin: 0 8px;
    padding: 8px 12px;
    border-radius: 8px;
    text-decoration: none !important;
    color: #ffffff !important;
    font-size: 12px;
    font-weight: 600;
    line-height: 1;
    transition: background 0.15s ease, box-shadow 0.15s ease;
}

.sidebar-nav-item:hover {
    background: rgba(255,255,255,.07);
}

.sidebar-nav-item.active {
    background: #1976ed;
    box-shadow: 0 7px 17px rgba(25,118,237,.25);
}

.sidebar-nav-item img {
    width: 20px !important;
    height: 20px !important;
    flex: 0 0 20px;
    display: block !important;
    margin: 0 12px 0 2px !important;
    object-fit: contain;
}

.sidebar-nav-item span {
    color: #ffffff !important;
    white-space: nowrap;
}

.sidebar-nav-item:focus-visible {
    outline: 2px solid rgba(255,255,255,.75);
    outline-offset: 1px;
}





/* ============================================================
   COLLAPSED SIDEBAR REOPEN BUTTON
   ============================================================ */

div[data-testid="collapsedControl"],
[data-testid="collapsedControl"] {
    position: fixed !important;
    top: 16px !important;
    left: 12px !important;
    right: auto !important;
    bottom: auto !important;
    z-index: 999999 !important;
}

div[data-testid="collapsedControl"] button,
[data-testid="collapsedControl"] button,
div[data-testid="collapsedControl"] button[data-testid="baseButton-headerNoPadding"] {
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    min-height: 40px !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 9px !important;
    background: #1769e8 !important;
    background-color: #1769e8 !important;
    border: 1px solid #1769e8 !important;
    color: #ffffff !important;
    box-shadow: 0 5px 16px rgba(23,105,232,.30) !important;
}

div[data-testid="collapsedControl"] button:hover,
[data-testid="collapsedControl"] button:hover {
    background: #0f5fd4 !important;
    background-color: #0f5fd4 !important;
    border-color: #0f5fd4 !important;
}

div[data-testid="collapsedControl"] button svg,
[data-testid="collapsedControl"] button svg {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
}

div[data-testid="collapsedControl"] button > div,
[data-testid="collapsedControl"] button > div {
    display: none !important;
}

div[data-testid="collapsedControl"] button::before,
[data-testid="collapsedControl"] button::before {
    content: ">" !important;
    display: block !important;
    color: #ffffff !important;
    font-size: 24px !important;
    font-weight: 800 !important;
    line-height: 1 !important;
    font-family: Arial, sans-serif !important;
}

div[data-testid="collapsedControl"] button::after,
[data-testid="collapsedControl"] button::after {
    content: none !important;
}

/* Streamlit header sidebar-toggle fallbacks */

button[aria-label="Open sidebar"],
button[aria-label="Open sidebar"][data-testid="baseButton-headerNoPadding"],
div[data-testid="collapsedControl"] button[aria-label="Open sidebar"] {
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    min-height: 40px !important;
    padding: 0 !important;
    border-radius: 9px !important;
    background: #1769e8 !important;
    background-color: #1769e8 !important;
    border: 1px solid #1769e8 !important;
    color: #ffffff !important;
    box-shadow: 0 5px 16px rgba(23,105,232,.30) !important;
}

button[aria-label="Open sidebar"] svg,
button[aria-label="Open sidebar"] path {
    display: none !important;
    visibility: hidden !important;
}

button[aria-label="Open sidebar"]::before {
    content: ">" !important;
    display: block !important;
    color: #ffffff !important;
    font-size: 24px !important;
    font-weight: 800 !important;
    line-height: 1 !important;
    font-family: Arial, sans-serif !important;
}


/* ============================================================

   SIDEBAR FOOTER

   ============================================================ */



.sidebar-footer {



    color: #93aac0;



    font-size: 9px;



    line-height: 1.55;



    padding: 12px 18px;

}



.sidebar-footer-icon {



    width: 18px;



    height: 18px;



    display: inline-block;



    vertical-align: middle;



    color: #93aac0;



    margin-right: 7px;

}



.sidebar-footer-icon svg {



    width: 18px !important;



    height: 18px !important;

}



.sidebar-collapse-icon {



    width: 18px;



    height: 18px;



    display: inline-block;



    vertical-align: middle;



    color: #93aac0;

}



.sidebar-collapse-icon svg {



    width: 18px !important;



    height: 18px !important;

}





/* ============================================================

   PAGE TITLES

   ============================================================ */



.page-title {



    color: var(--ink);



    font-size: 30px;



    font-weight: 800;



    letter-spacing: -0.5px;



    margin: 8px 0 5px;

}



.page-subtitle {



    color: var(--muted);



    font-size: 12px;



    margin-bottom: 20px;

}





/* ============================================================

   HERO

   ============================================================ */



.hero-wrap {



    position: relative;



    width: 100%;



    min-height: 400px;



    border-radius: 14px;



    overflow: hidden;



    box-shadow:

        0 12px 30px

        rgba(20, 55, 90, 0.14);



    border: 1px solid #d8e2ec;



    background: #082f52;

}



.hero-img {



    position: absolute;



    inset: 0;



    display: block;



    width: 100%;



    height: 100%;



    object-fit: cover;



    object-position: 58% center;



    z-index: 0;

}



.hero-overlay {



    position: absolute;



    inset: 0;



    z-index: 1;



    background:

        linear-gradient(

            90deg,

            rgba(3,25,47,.88) 0%,

            rgba(3,25,47,.78) 22%,

            rgba(3,30,54,.58) 42%,

            rgba(3,35,62,.34) 64%,

            rgba(3,35,62,.18) 82%,

            rgba(3,35,62,.12) 100%

        ),

        linear-gradient(

            180deg,

            rgba(4,31,56,.10),

            rgba(4,31,56,.30)

        );

}



.hero-content {



    position: relative;



    z-index: 2;



    min-height: 400px;



    display: flex;



    flex-direction: column;



    justify-content: center;



    padding: 44px 50px;



    max-width: 690px;



    color: #ffffff;

}



.hero-badge {



    display: inline-flex;



    align-items: center;



    gap: 7px;



    width: max-content;



    background:

        rgba(255,255,255,.12);



    border:

        1px solid

        rgba(255,255,255,.22);



    padding: 7px 12px;



    border-radius: 999px;



    font-size: 10px;



    font-weight: 600;



    margin-bottom: 18px;



    backdrop-filter: blur(5px);

}



.hero-title {



    color: #ffffff;



    font-size: 40px;



    line-height: 1.08;



    font-weight: 800;



    letter-spacing: -0.8px;



    margin-bottom: 16px;

}



.hero-subtitle {



    color: #dce9f6;



    font-size: 15px;



    line-height: 1.65;



    max-width: 520px;

}





/* ============================================================

   FEATURE CARDS

   ============================================================ */



.feature-card {



    background: #ffffff;



    border: 1px solid var(--border);



    border-radius: 11px;



    padding: 16px;



    min-height: 122px;



    box-shadow:

        0 4px 15px

        rgba(20,55,90,.05);

}



.feature-icon {



    width: 37px;



    height: 37px;



    border-radius: 9px;



    display: flex;



    align-items: center;



    justify-content: center;



    margin-bottom: 10px;

}



.blue {



    background: #e7f0ff;



    color: #1769e8;

}



.green {



    background: #e7f8ef;



    color: #0e9f61;

}



.purple {



    background: #f0eaff;



    color: #7445e8;

}



.orange {



    background: #fff1d8;



    color: #ed900d;

}



.feature-title {



    font-size: 11px;



    font-weight: 800;



    color: var(--ink);



    margin-bottom: 4px;

}



.feature-text {



    font-size: 9px;



    color: var(--muted);



    line-height: 1.5;

}





/* ============================================================

   SECTION TITLES

   ============================================================ */



.section-title {



    font-size: 17px;



    font-weight: 800;



    color: var(--ink);



    margin: 24px 0 11px;

}





/* ============================================================

   QUICK START

   ============================================================ */



.quick-card {



    position: relative;



    background: #0b3459;



    border: 1px solid #17476d;



    border-radius: 10px;



    padding: 15px;



    min-height: 116px;



    color: #ffffff;

}



.quick-number {



    width: 28px;



    height: 28px;



    border-radius: 50%;



    background: #237bf0;



    display: flex;



    align-items: center;



    justify-content: center;



    font-size: 11px;



    font-weight: 800;



    margin-bottom: 11px;

}



.quick-title {



    font-size: 10px;



    font-weight: 800;



    margin-bottom: 5px;

}



.quick-text {



    color: #c1d3e4;



    font-size: 9px;



    line-height: 1.5;

}



.quick-arrow {



    position: absolute;



    right: -17px;



    top: 42px;



    color: #2e86f7;



    font-size: 22px;



    z-index: 2;

}





/* ============================================================

   QUOTE

   ============================================================ */



.quote {



    color: #6c8095;



    text-align: center;



    font-size: 10px;



    font-style: italic;



    margin: 14px 0;

}





/* ============================================================

   MODEL BADGE

   ============================================================ */



.model-badge {



    background: #ffffff;



    border: 1px solid #dce6f0;



    border-radius: 10px;



    padding: 9px 13px;



    display: inline-block;



    float: right;



    margin-top: -63px;



    color: #173653;



    font-size: 9px;



    line-height: 1.55;

}





/* ============================================================

   STREAMLIT BORDERED CARDS

   ============================================================ */



div[data-testid="stVerticalBlockBorderWrapper"] {



    background: #ffffff;



    border: 1px solid #dce6f0;



    border-radius: 12px;



    box-shadow:

        0 4px 14px

        rgba(20,55,90,.04);



    padding: 6px 12px 6px;

}





/* ============================================================

   FORM SECTIONS

   ============================================================ */



.form-section-title {



    display: flex;



    align-items: center;



    gap: 8px;



    font-size: 13px;



    font-weight: 800;



    margin-bottom: 12px;

}



.form-section-title.blue-t {



    color: #1769e8;

}



.form-section-title.green-t {



    color: #0e9f61;

}



.form-section-title.purple-t {



    color: #7445e8;

}



.form-section-title.orange-t {



    color: #ed900d;

}





/* ============================================================

   FORM INPUTS

   ============================================================ */



label[data-testid="stWidgetLabel"] p {



    color: #405a73 !important;



    font-size: 9px !important;



    font-weight: 600 !important;

}



.stNumberInput input,

.stTextInput input,

[data-baseweb="select"] > div {



    background: #f8fbfe !important;



    border: 1px solid #d6e0ea !important;



    border-radius: 7px !important;



    color: #173653 !important;



    font-size: 12px !important;

}



.stNumberInput button {



    background: #eef3f8 !important;



    color: #173653 !important;



    border: none !important;

}





/* ============================================================

   PREDICT BUTTON

   ============================================================ */



/* Streamlit form submit button */

div[data-testid="stFormSubmitButton"] button,
button[kind="secondaryFormSubmit"],
button[data-testid="stFormSubmitButton"] {
    width: 100% !important;
    background: #2378ed !important;
    background-color: #2378ed !important;
    color: #ffffff !important;
    border: 1px solid #2378ed !important;
    border-radius: 9px !important;
    min-height: 48px !important;
    font-size: 13px !important;
    font-weight: 800 !important;
    box-shadow: 0 8px 18px rgba(35,120,237,.23) !important;
}

div[data-testid="stFormSubmitButton"] button p,
div[data-testid="stFormSubmitButton"] button span,
button[kind="secondaryFormSubmit"] p,
button[kind="secondaryFormSubmit"] span {
    color: #ffffff !important;
    font-weight: 800 !important;
}

div[data-testid="stFormSubmitButton"] button:hover,
button[kind="secondaryFormSubmit"]:hover,
button[data-testid="stFormSubmitButton"]:hover {
    background: #1769dc !important;
    background-color: #1769dc !important;
    border-color: #1769dc !important;
}

.predict-button button {



    background: #2378ed !important;



    color: #ffffff !important;



    border: none !important;



    border-radius: 8px !important;



    min-height: 47px !important;



    font-size: 13px !important;



    font-weight: 800 !important;



    box-shadow:

        0 8px 18px

        rgba(35,120,237,.23);

}



.predict-button button:hover {



    background: #1769dc !important;

}





/* ============================================================

   METRIC CARDS

   ============================================================ */



.metric-card {



    background: #ffffff;



    border: 1px solid var(--border);



    border-radius: 11px;



    padding: 15px 17px;



    min-height: 94px;



    box-shadow:

        0 4px 14px

        rgba(20,55,90,.04);

}



.metric-label {



    color: #71839a;



    font-size: 8px;



    font-weight: 800;



    text-transform: uppercase;



    letter-spacing: .7px;

}



.metric-value {



    font-size: 25px;



    font-weight: 800;



    margin-top: 8px;

}





/* ============================================================

   RESULT

   ============================================================ */



.result-card {



    background:

        linear-gradient(

            135deg,

            #edfff6,

            #f9fffc

        );



    border: 1px solid #ccebdc;



    border-radius: 13px;



    padding: 25px;

}



.result-label {



    color: #60768d;



    font-size: 10px;



    font-weight: 700;

}



.result-number {



    color: #0b9560;



    font-size: 46px;



    font-weight: 800;



    line-height: 1;



    margin-top: 5px;

}



.result-unit {



    color: #0b9560;



    font-size: 15px;



    font-weight: 700;

}





/* ============================================================

   SMALL CARDS

   ============================================================ */



.small-card {



    background: #ffffff;



    border: 1px solid var(--border);



    border-radius: 10px;



    padding: 15px;



    min-height: 100px;

}



.small-card-title {



    color: #6b7f94;



    font-size: 9px;



    font-weight: 700;

}



.small-card-value {



    color: var(--ink);



    font-size: 21px;



    font-weight: 800;



    margin-top: 7px;

}





/* ============================================================

   ABOUT

   ============================================================ */



.about-card {



    background: #ffffff;



    border: 1px solid var(--border);



    border-radius: 10px;



    padding: 15px;



    min-height: 110px;

}



.about-title {



    font-size: 10px;



    font-weight: 800;



    color: var(--ink);

}



.about-value {



    font-size: 15px;



    font-weight: 800;



    color: var(--ink);



    margin-top: 7px;

}



.about-text {



    font-size: 9px;



    color: var(--muted);



    line-height: 1.5;



    margin-top: 4px;

}





/* ============================================================

   FOOTER

   ============================================================ */



.footer {



    color: #8495a7;



    font-size: 8px;



    text-align: center;



    margin-top: 24px;

}





/* ============================================================

   REMOVE EXTRA SPACING

   ============================================================ */



div[data-testid="stVerticalBlock"]

> div:has(> div.stMarkdown) {



    gap: 0;

}





/* ============================================================

   RESPONSIVE

   ============================================================ */



@media (max-width: 900px) {



    .block-container {

        padding-left: 0.8rem !important;

        padding-right: 0.8rem !important;

    }



    .page-title {

        font-size: 28px !important;

        line-height: 1.15 !important;

    }



    .page-subtitle {

        font-size: 11px !important;

    }



    .hero-wrap,

    .hero-content {



        min-height: 320px;

    }



    .hero-title {



        font-size: 30px;

    }



    .hero-content {



        padding: 30px;

    }



    .sidebar-brand {

        padding-left: 14px !important;

        padding-right: 14px !important;

    }



    .sidebar-nav-item {

        min-height: 42px !important;

        font-size: 11px !important;

    }



    .sidebar-nav-item img {

        width: 19px !important;

        height: 19px !important;

        flex-basis: 19px !important;

    }



    .model-badge {



        float: none;



        margin:

            0 0 15px;

    }

}



</style>

""".replace(

        "ICON_HOME",

        icon_data_uri("home", "#ffffff")

    ).replace(

        "ICON_TOOL",

        icon_data_uri("tool", "#ffffff")

    ).replace(

        "ICON_CHART",

        icon_data_uri("chart-dots-2", "#ffffff")

    ).replace(

        "ICON_DATABASE",

        icon_data_uri("database-search", "#ffffff")

    ).replace(

        "ICON_INFO",

        icon_data_uri("info-circle", "#ffffff")

    ),

    unsafe_allow_html=True

)





# ============================================================

# SMALL INTERNAL SVG ICONS

# ============================================================



def svg(name, size=20):



    paths = {



        "home":

        f"""

        <svg width="{size}" height="{size}"

             viewBox="0 0 24 24"

             fill="currentColor">

            <path d="M3 10.8 12 3l9 7.8V21a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1v-10.2Z"/>

        </svg>

        """,



        "chart":

        f"""

        <svg width="{size}" height="{size}"

             viewBox="0 0 24 24"

             fill="none"

             stroke="currentColor"

             stroke-width="2"

             stroke-linecap="round"

             stroke-linejoin="round">

            <path d="M4 19V5M4 19h17"/>

            <path d="m7 15 4-4 3 2 5-7"/>

        </svg>

        """,



        "database":

        f"""

        <svg width="{size}" height="{size}"

             viewBox="0 0 24 24"

             fill="none"

             stroke="currentColor"

             stroke-width="2"

             stroke-linecap="round"

             stroke-linejoin="round">

            <ellipse cx="12" cy="5" rx="8" ry="3"/>

            <path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>

            <path d="M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7"/>

        </svg>

        """,



        "user":

        f"""

        <svg width="{size}" height="{size}"

             viewBox="0 0 24 24"

             fill="none"

             stroke="currentColor"

             stroke-width="2"

             stroke-linecap="round"

             stroke-linejoin="round">

            <circle cx="12" cy="7" r="4"/>

            <path d="M4 21c.8-4 3.4-6 8-6s7.2 2 8 6"/>

        </svg>

        """,



        "calendar":

        f"""

        <svg width="{size}" height="{size}"

             viewBox="0 0 24 24"

             fill="none"

             stroke="currentColor"

             stroke-width="2"

             stroke-linecap="round"

             stroke-linejoin="round">

            <rect x="3" y="4" width="18" height="17" rx="2"/>

            <path d="M16 2v4M8 2v4M3 10h18"/>

        </svg>

        """,



        "gear":

        f"""

        <svg width="{size}" height="{size}"

             viewBox="0 0 24 24"

             fill="none"

             stroke="currentColor"

             stroke-width="2"

             stroke-linecap="round"

             stroke-linejoin="round">

            <circle cx="12" cy="12" r="3"/>

            <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-1.8 1.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.5V20h-2.6v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.9.3l-.1.1-1.8-1.8.1-.1A1.7 1.7 0 0 0 8 15a1.7 1.7 0 0 0-1.5-1H6v-2.6h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1L9 6.6l.1.1a1.7 1.7 0 0 0 1.9.3 1.7 1.7 0 0 0 1-1.5V5h2.6v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1 1.8 1.8-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.5 1h.1V14h-.1a1.7 1.7 0 0 0-1.1 1Z"/>

        </svg>

        """

    }



    return paths.get(

        name,

        paths["gear"]

    )





# ============================================================
# SIDEBAR
# ============================================================

PAGES = [
    "Home",
    "Predict Output",
    "Model Insights",
    "Data Exploration",
    "About",
]

if "page" not in st.session_state:
    st.session_state.page = "Home"

query_page = st.query_params.get("page")
if query_page in PAGES:
    st.session_state.page = query_page

page = st.session_state.page

with st.sidebar:

    # --------------------------------------------------------
    # LOAD REAL DOWNLOADED SVGs
    # --------------------------------------------------------
    factory_icon = icon_svg("building-factory-2")
    settings_icon = icon_svg("settings-2")
    chevron_icon = icon_svg("chevron-left")

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------
    custom_html(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">
                {factory_icon}
            </div>
            <div class="sidebar-brand-title">
                Manufacturing<br>
                Output Predictor
            </div>
        </div>
        """
    )

    custom_html(
        '<div class="sidebar-nav-title">NAVIGATION</div>'
    )

    # --------------------------------------------------------
    # CUSTOM SVG NAVIGATION
    # --------------------------------------------------------
    nav_items = [
        ("Home", "home"),
        ("Predict Output", "tool"),
        ("Model Insights", "chart-dots-2"),
        ("Data Exploration", "database-search"),
        ("About", "info-circle"),
    ]

    nav_html = ['<div class="sidebar-nav">']

    for label, icon_name in nav_items:
        active_class = " active" if page == label else ""
        href = "?page=" + quote(label, safe="")
        icon_uri = icon_data_uri(icon_name, "#ffffff")

        nav_html.append(
            f'<a class="sidebar-nav-item{active_class}" '
            f'href="{href}" aria-label="{label}">'
            f'<img src="{icon_uri}" width="20" height="20" alt="">'
            f'<span>{label}</span>'
            f'</a>'
        )

    nav_html.append('</div>')
    custom_html("".join(nav_html))

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------
    custom_html(
        f"""
        <div class="sidebar-footer">
            <span class="sidebar-footer-icon">
                {settings_icon}
            </span>
            Made with Streamlit
            <br>
            &nbsp;&nbsp;&nbsp;&nbsp;v1.0
        </div>
        <div style="height:4px"></div>
        <div style="padding:0 18px; color:#93aac0; text-align:right;">
            <span class="sidebar-collapse-icon">
                {chevron_icon}
            </span>
        </div>
        """
    )


# ============================================================


# HOME

# ============================================================



if page == "Home":



    # --------------------------------------------------------

    # HERO

    # --------------------------------------------------------



    if HERO_PATH.exists():



        b64 = base64.b64encode(

            HERO_PATH.read_bytes()

        ).decode("utf-8")



        custom_html(

            f"""

            <div class="hero-wrap">



                <img

                    class="hero-img"

                    src="data:image/jpeg;base64,{b64}"

                >



                <div class="hero-overlay"></div>



                <div class="hero-content">



                    <div class="hero-badge">

                        🏭 &nbsp;

                        Manufacturing Intelligence

                    </div>



                    <div class="hero-title">

                        Manufacturing Equipment<br>

                        Output Prediction

                    </div>



                    <div class="hero-subtitle">

                        Predict production output using

                        machine parameters with

                        Machine Learning

                    </div>



                </div>



            </div>

            """

        )



    else:



        custom_html(

            """

            <div class="hero-wrap">



                <div class="hero-overlay"></div>



                <div class="hero-content">



                    <div class="hero-title">

                        Manufacturing Equipment<br>

                        Output Prediction

                    </div>



                    <div class="hero-subtitle">

                        Predict production output using

                        machine parameters with

                        Machine Learning

                    </div>



                </div>



            </div>

            """

        )





    # --------------------------------------------------------

    # WHAT YOU CAN DO

    # --------------------------------------------------------



    st.markdown(

        '<div class="section-title">'

        'What You Can Do'

        '</div>',

        unsafe_allow_html=True

    )





    cols = st.columns(4)



    cards = [



        (

            "blue",

            "chart",

            "Accurate Predictions",

            "ML-powered output estimation"

        ),



        (

            "green",

            "gear",

            "Optimize Production",

            "Improve efficiency and reduce downtime"

        ),



        (

            "purple",

            "database",

            "Data-Driven Decisions",

            "Leverage machine data for better planning"

        ),



        (

            "orange",

            "gear",

            "Easy to Use",

            "Simple and intuitive interface"

        ),

    ]





    for col, (

        color,

        ico,

        title,

        desc

    ) in zip(cols, cards):



        with col:



            custom_html(

                f"""

                <div class="feature-card">



                    <div class="feature-icon {color}">

                        {svg(ico, 20)}

                    </div>



                    <div class="feature-title">

                        {title}

                    </div>



                    <div class="feature-text">

                        {desc}

                    </div>



                </div>

                """

            )





    # --------------------------------------------------------

    # QUICK START

    # --------------------------------------------------------



    st.markdown(

        '<div class="section-title">'

        'Quick Start'

        '</div>',

        unsafe_allow_html=True

    )





    cols = st.columns(3)



    quick = [



        (

            "1",

            "Enter Machine Parameters",

            "Provide machine, material and environment details."

        ),



        (

            "2",

            "Get Prediction",

            "Our model will estimate parts per hour."

        ),



        (

            "3",

            "Optimize",

            "Use insights to improve production efficiency."

        ),

    ]





    for i, (

        col,

        (

            num,

            title,

            desc

        )

    ) in enumerate(

        zip(cols, quick)

    ):



        with col:



            arrow = (

                '<div class="quick-arrow">→</div>'

                if i < 2

                else ""

            )



            custom_html(

                f"""

                <div class="quick-card">



                    <div class="quick-number">

                        {num}

                    </div>



                    <div class="quick-title">

                        {title}

                    </div>



                    <div class="quick-text">

                        {desc}

                    </div>



                    {arrow}



                </div>

                """

            )





    custom_html(

        '<div class="quote">'

        '“Data today, better production tomorrow.”'

        '</div>'

    )





# ============================================================

# PREDICT OUTPUT

# ============================================================



elif page == "Predict Output":



    st.markdown(

        '<div class="page-title">'

        'Predict Manufacturing Output'

        '</div>',

        unsafe_allow_html=True

    )



    st.markdown(

        '<div class="page-subtitle">'

        'Enter the machine parameters below to predict '

        'the number of parts produced per hour.'

        '</div>',

        unsafe_allow_html=True

    )





    # --------------------------------------------------------

    # MODEL BADGE

    # --------------------------------------------------------



    custom_html(

        f"""

        <div class="model-badge">



            <b>⚙ Linear Regression Model</b>

            <br>



            Trained on {len(df):,} samples



        </div>

        """

    )





    shifts = cats(

        "Shift",

        ["Day", "Evening", "Night"]

    )



    machine_types = cats(

        "Machine_Type",

        ["Type_A", "Type_B", "Type_C"]

    )



    grades = cats(

        "Material_Grade",

        ["Economy", "Standard", "Premium"]

    )



    days = cats(

        "Day_of_Week",

        [

            "Monday",

            "Tuesday",

            "Wednesday",

            "Thursday",

            "Friday",

            "Saturday",

            "Sunday"

        ]

    )





    # --------------------------------------------------------

    # FORM

    # --------------------------------------------------------



    with st.form("prediction_form"):



        # ====================================================

        # MACHINE PARAMETERS

        # ====================================================



        with st.container(border=True):



            custom_html(

                f"""

                <div class="form-section-title blue-t">

                    {svg("gear", 20)}

                    Machine Parameters

                </div>

                """

            )



            a, b = st.columns(2)





            with a:



                injection_temperature = st.number_input(

                    "Injection Temperature (°C)",

                    value=val(

                        "Injection_Temperature"

                    ),

                    step=0.1,

                    format="%.1f"

                )



                cycle_time = st.number_input(

                    "Cycle Time (s)",

                    value=val("Cycle_Time"),

                    step=0.1,

                    format="%.1f"

                )





            with b:



                injection_pressure = st.number_input(

                    "Injection Pressure (MPa)",

                    value=val(

                        "Injection_Pressure"

                    ),

                    step=0.1,

                    format="%.1f"

                )



                cooling_time = st.number_input(

                    "Cooling Time (s)",

                    value=val("Cooling_Time"),

                    step=0.1,

                    format="%.1f"

                )





        st.markdown(

            '<div style="height:6px"></div>',

            unsafe_allow_html=True

        )





        # ====================================================

        # MATERIAL & ENVIRONMENT

        # ====================================================



        with st.container(border=True):



            custom_html(

                f"""

                <div class="form-section-title green-t">

                    {svg("database", 20)}

                    Material &amp; Environment

                </div>

                """

            )



            a, b = st.columns(2)





            with a:



                material_viscosity = st.number_input(

                    "Material Viscosity",

                    value=val(

                        "Material_Viscosity"

                    ),

                    step=0.1,

                    format="%.1f"

                )



                material_grade = st.selectbox(

                    "Material Grade",

                    grades

                )





            with b:



                ambient_temperature = st.number_input(

                    "Ambient Temperature (°C)",

                    value=val(

                        "Ambient_Temperature"

                    ),

                    step=0.1,

                    format="%.1f"

                )



                shift = st.selectbox(

                    "Shift",

                    shifts

                )





        st.markdown(

            '<div style="height:6px"></div>',

            unsafe_allow_html=True

        )





        # ====================================================

        # MACHINE & OPERATOR

        # ====================================================



        with st.container(border=True):



            custom_html(

                f"""

                <div class="form-section-title purple-t">

                    {svg("user", 20)}

                    Machine &amp; Operator

                </div>

                """

            )



            a, b = st.columns(2)





            with a:



                machine_type = st.selectbox(

                    "Machine Type",

                    machine_types

                )



                machine_age = st.number_input(

                    "Machine Age (years)",

                    value=val("Machine_Age"),

                    min_value=0.0,

                    step=1.0,

                    format="%.0f"

                )





            with b:



                operator_experience = st.number_input(

                    "Operator Experience (years)",

                    value=val(

                        "Operator_Experience"

                    ),

                    min_value=0.0,

                    step=0.5,

                    format="%.1f"

                )



                maintenance_hours = st.number_input(

                    "Maintenance Hours",

                    value=val(

                        "Maintenance_Hours"

                    ),

                    min_value=0.0,

                    step=1.0,

                    format="%.1f"

                )





        st.markdown(

            '<div style="height:6px"></div>',

            unsafe_allow_html=True

        )





        # ====================================================

        # ADDITIONAL INFORMATION

        # ====================================================



        with st.container(border=True):



            custom_html(

                f"""

                <div class="form-section-title orange-t">

                    {svg("calendar", 20)}

                    Additional Information

                </div>

                """

            )



            a, b = st.columns(2)





            with a:



                day_of_week = st.selectbox(

                    "Day of Week",

                    days

                )



                machine_utilization = st.number_input(

                    "Machine Utilization",

                    value=val(

                        "Machine_Utilization"

                    ),

                    min_value=0.0,

                    step=0.01,

                    format="%.2f"

                )





            with b:



                efficiency_score = st.number_input(

                    "Efficiency Score",

                    value=val(

                        "Efficiency_Score"

                    ),

                    min_value=0.0,

                    step=0.01,

                    format="%.2f"

                )



                hour = st.slider(

                    "Production Hour",

                    0,

                    23,

                    12

                )





        st.markdown(

            '<div style="height:8px"></div>',

            unsafe_allow_html=True

        )





        st.markdown(

            '<div style="height:4px"></div>',

            unsafe_allow_html=True

        )



        submitted = st.form_submit_button(

            "▶  Predict Output",

            use_container_width=True

        )





    # ========================================================

    # PREDICTION

    # ========================================================



    if submitted:



        hsin = np.sin(

            2 * np.pi * hour / 24

        )



        hcos = np.cos(

            2 * np.pi * hour / 24

        )





        row = pd.DataFrame(

            [{

                "Injection_Temperature":

                    injection_temperature,



                "Injection_Pressure":

                    injection_pressure,



                "Cycle_Time":

                    cycle_time,



                "Cooling_Time":

                    cooling_time,



                "Material_Viscosity":

                    material_viscosity,



                "Ambient_Temperature":

                    ambient_temperature,



                "Operator_Experience":

                    operator_experience,



                "Maintenance_Hours":

                    maintenance_hours,



                "Shift":

                    shift,



                "Machine_Type":

                    machine_type,



                "Material_Grade":

                    material_grade,



                "Day_of_Week":

                    day_of_week,



                "Machine_Utilization":

                    machine_utilization,



                "Hour_Sin":

                    hsin,



                "Hour_Cos":

                    hcos,

            }]

        )





        try:



            prediction = float(

                model.predict(

                    row[FEATURES]

                )[0]

            )





            st.session_state.prediction = prediction



            st.session_state.prediction_inputs = row



            st.session_state.prediction_display = {



                "Machine Age":

                    machine_age,



                "Efficiency Score":

                    efficiency_score,



                "Hour":

                    hour,

            }





            st.markdown(

                '<div style="height:8px"></div>',

                unsafe_allow_html=True

            )





            # ------------------------------------------------

            # RESULT

            # ------------------------------------------------



            custom_html(

                f"""

                <div class="result-card">



                    <div style="

                        color:#0b9560;

                        font-size:14px;

                        font-weight:800;

                    ">

                        ✓ Prediction Result

                    </div>



                    <div class="result-label"

                         style="margin-top:4px;">

                        Based on the provided machine parameters

                    </div>



                    <div style="margin-top:20px;">



                        <div class="result-label">

                            Predicted Output

                        </div>



                        <div class="result-number">

                            {prediction:.1f}

                        </div>



                        <div class="result-unit">

                            Parts per Hour

                        </div>



                    </div>



                </div>

                """

            )





            st.markdown(

                '<div style="height:8px"></div>',

                unsafe_allow_html=True

            )





            # ------------------------------------------------

            # RESULT CARDS

            # ------------------------------------------------



            c1, c2, c3 = st.columns(3)





            with c1:



                custom_html(

                    f"""

                    <div class="small-card">



                        <div class="small-card-title">

                            Estimated Daily Output

                        </div>



                        <div class="small-card-value">

                            {prediction * 8:,.0f}

                        </div>



                        <div class="small-card-title">

                            parts · 8-hour shift

                        </div>



                    </div>

                    """

                )





            with c2:



                custom_html(

                    f"""

                    <div class="small-card">



                        <div class="small-card-title">

                            Machine Utilization

                        </div>



                        <div class="small-card-value">

                            {machine_utilization * 100:.1f}%

                        </div>



                        <div class="small-card-title">

                            selected setting

                        </div>



                    </div>

                    """

                )





            with c3:



                avg = float(

                    df[TARGET].mean()

                )



                status = (

                    "Good"

                    if prediction >= avg

                    else "Below Average"

                )



                status_color = (

                    "#0b9560"

                    if status == "Good"

                    else "#df8a00"

                )





                custom_html(

                    f"""

                    <div class="small-card">



                        <div class="small-card-title">

                            Production Status

                        </div>



                        <div

                            class="small-card-value"

                            style="color:{status_color};"

                        >

                            {status}

                        </div>



                        <div class="small-card-title">

                            compared with dataset average

                        </div>



                    </div>

                    """

                )





        except Exception as e:



            st.error(

                "Prediction could not be generated."

            )



            st.exception(e)





# ============================================================

# MODEL INSIGHTS

# ============================================================



elif page == "Model Insights":



    st.markdown(

        '<div class="page-title">'

        'Model Performance'

        '</div>',

        unsafe_allow_html=True

    )



    st.markdown(

        '<div class="page-subtitle">'

        'Evaluation metrics for the trained '

        'Linear Regression model.'

        '</div>',

        unsafe_allow_html=True

    )





    # --------------------------------------------------------

    # METRICS

    # --------------------------------------------------------



    cols = st.columns(4)



    metric_cards = [



        (

            "R² Score",

            f"{R2:.3f}",

            "#0b9b62"

        ),



        (

            "Mean Absolute Error",

            f"{MAE:.2f}",

            "#1769e8"

        ),



        (

            "Root Mean Squared Error",

            f"{RMSE:.2f}",

            "#7445e8"

        ),



        (

            "Mean Squared Error",

            f"{MSE:.1f}",

            "#ed900d"

        ),

    ]





    for col, (

        name,

        value,

        color

    ) in zip(cols, metric_cards):



        with col:



            custom_html(

                f"""

                <div class="metric-card">



                    <div class="metric-label">

                        {name}

                    </div>



                    <div

                        class="metric-value"

                        style="color:{color};"

                    >

                        {value}

                    </div>



                </div>

                """

            )





    # --------------------------------------------------------

    # ACTUAL VS PREDICTED

    # --------------------------------------------------------



    st.markdown(

        '<div class="section-title">'

        'Actual vs Predicted'

        '</div>',

        unsafe_allow_html=True

    )





    try:



        model_input = add_time_features(

            df

        )



        model_input = model_input[

            FEATURES

        ].copy()



        predictions = model.predict(

            model_input

        )





        plot_df = pd.DataFrame({



            "Actual":

                df[TARGET].values,



            "Predicted":

                predictions

        })





        st.scatter_chart(

            plot_df,

            x="Actual",

            y="Predicted",

            height=350

        )





    except Exception as e:



        st.error(

            "The actual-vs-predicted chart "

            "could not be generated."

        )



        st.code(str(e))





    # --------------------------------------------------------

    # FEATURE INFLUENCE

    # --------------------------------------------------------



    st.markdown(

        '<div class="section-title">'

        'Feature Influence'

        '</div>',

        unsafe_allow_html=True

    )





    try:



        prep = None

        estimator = None





        if hasattr(model, "named_steps"):



            for step_name, step in (

                model.named_steps.items()

            ):



                if hasattr(

                    step,

                    "get_feature_names_out"

                ):



                    prep = step



                if hasattr(

                    step,

                    "coef_"

                ):



                    estimator = step





        if (

            prep is not None

            and estimator is not None

        ):



            names = (

                prep

                .get_feature_names_out()

            )



            coefs = np.asarray(

                estimator.coef_

            ).ravel()





            coef_df = pd.DataFrame({



                "Feature":

                    names,



                "Coefficient":

                    coefs,

            })





            coef_df[

                "Absolute Coefficient"

            ] = (

                coef_df["Coefficient"]

                .abs()

            )





            coef_df = coef_df.sort_values(

                "Absolute Coefficient",

                ascending=False

            )





            chart = (

                coef_df

                .head(10)

                .set_index("Feature")

                [["Absolute Coefficient"]]

            )





            st.bar_chart(

                chart,

                height=350

            )





            st.dataframe(

                coef_df

                .head(20)

                .round(4),

                use_container_width=True,

                hide_index=True

            )





        else:



            st.info(

                "Coefficient details are unavailable "

                "for this saved pipeline."

            )





    except Exception:



        st.info(

            "Coefficient details are unavailable "

            "for this saved pipeline."

        )





    # --------------------------------------------------------

    # MODEL COMPARISON

    # --------------------------------------------------------



    st.markdown(

        '<div class="section-title">'

        'Model Comparison'

        '</div>',

        unsafe_allow_html=True

    )





    st.dataframe(

        metrics,

        use_container_width=True,

        hide_index=True

    )





# ============================================================

# DATA EXPLORATION

# ============================================================



elif page == "Data Exploration":



    st.markdown(

        '<div class="page-title">'

        'Data Exploration'

        '</div>',

        unsafe_allow_html=True

    )



    st.markdown(

        '<div class="page-subtitle">'

        'Explore the manufacturing dataset used '

        'for model development.'

        '</div>',

        unsafe_allow_html=True

    )





    # --------------------------------------------------------

    # DATASET STATISTICS

    # --------------------------------------------------------



    cols = st.columns(4)



    stats = [



        (

            "Records",

            f"{len(df):,}"

        ),



        (

            "Features",

            f"{len(df.columns):,}"

        ),



        (

            "Average Output",

            f"{df[TARGET].mean():.2f}"

        ),



        (

            "Missing Values",

            f"{int(df.isna().sum().sum()):,}"

        ),

    ]





    for col, (

        label,

        value

    ) in zip(cols, stats):



        with col:



            custom_html(

                f"""

                <div class="metric-card">



                    <div class="metric-label">

                        {label}

                    </div>



                    <div class="metric-value">

                        {value}

                    </div>



                </div>

                """

            )





    # --------------------------------------------------------

    # OUTPUT DISTRIBUTION

    # --------------------------------------------------------



    st.markdown(

        '<div class="section-title">'

        'Parts Per Hour Distribution'

        '</div>',

        unsafe_allow_html=True

    )





    output_values = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    ).dropna()

    if len(output_values) > 0:

        counts, edges = np.histogram(
            output_values,
            bins=12
        )

        centers = (
            edges[:-1] + edges[1:]
        ) / 2

        distribution_df = pd.DataFrame(
            {"Frequency": counts},
            index=np.round(centers, 1)
        )

        distribution_df.index.name = "Parts Per Hour"

        st.bar_chart(
            distribution_df,
            height=330
        )

    else:

        st.info(
            "No valid Parts Per Hour values are available for the distribution chart."
        )





    # --------------------------------------------------------

    # FEATURE EXPLORER

    # --------------------------------------------------------



    st.markdown(

        '<div class="section-title">'

        'Feature Explorer'

        '</div>',

        unsafe_allow_html=True

    )





    numerical = (

        df

        .select_dtypes(include=np.number)

        .columns

        .tolist()

    )





    selected = st.selectbox(

        "Select numerical feature",

        numerical

    )





    st.line_chart(

        df[[selected]]

        .reset_index(drop=True),

        height=280

    )





    # --------------------------------------------------------

    # DATASET PREVIEW

    # --------------------------------------------------------



    st.markdown(

        '<div class="section-title">'

        'Dataset Preview'

        '</div>',

        unsafe_allow_html=True

    )





    st.dataframe(

        df.head(20),

        use_container_width=True,

        hide_index=True

    )





# ============================================================

# ABOUT

# ============================================================



elif page == "About":



    st.markdown(

        '<div class="page-title">'

        'About This Project'

        '</div>',

        unsafe_allow_html=True

    )



    st.markdown(

        '<div class="page-subtitle">'

        'Manufacturing Equipment Output Prediction'

        '</div>',

        unsafe_allow_html=True

    )





    # --------------------------------------------------------

    # IMAGE

    # --------------------------------------------------------



    if HERO_PATH.exists():



        st.image(

            str(HERO_PATH),

            use_container_width=True

        )





    # --------------------------------------------------------

    # DESCRIPTION

    # --------------------------------------------------------



    custom_html(

        """

        <div style="

            background:#ffffff;

            border:1px solid #dce6f0;

            border-radius:10px;

            padding:14px 16px;

            margin-top:12px;

            color:#173653;

            font-size:10px;

            line-height:1.6;

        ">



            This project uses machine learning to

            predict the number of parts produced per

            hour based on machine, material, operator

            and production parameters.



            The model supports data-driven manufacturing

            decisions and helps estimate production

            output from operating conditions.



        </div>

        """

    )





    # --------------------------------------------------------

    # PROJECT CARDS

    # --------------------------------------------------------



    cols = st.columns(4)



    cards = [



        (

            "Dataset",

            f"{len(df):,} records",

            "manufacturing records"

        ),



        (

            "Model",

            "Linear Regression",

            "refined pipeline"

        ),



        (

            "Technologies",

            "Python, Scikit-learn",

            "Streamlit"

        ),



        (

            "Objective",

            "Predict parts/hour",

            "optimize production"

        ),

    ]





    for col, (

        title,

        value,

        desc

    ) in zip(cols, cards):



        with col:



            custom_html(

                f"""

                <div class="about-card">



                    <div class="about-title">

                        {title}

                    </div>



                    <div class="about-value">

                        {value}

                    </div>



                    <div class="about-text">

                        {desc}

                    </div>



                </div>

                """

            )





    custom_html(

        '<div class="quote">'

        '“Turning data into higher productivity.”'

        '</div>'

    )





# ============================================================

# GLOBAL FOOTER

# ============================================================



custom_html(

    '<div class="footer">'

    'Manufacturing Equipment Output Prediction '

    '· Data Science Capstone Project'

    '</div>'

)