
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from pathlib import Path
from html import escape

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - handled in app runtime
    CatBoostClassifier = None

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="MEPS Risk Dashboard",
    page_icon="MEPS",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True
dark_mode = st.session_state["dark_mode"]

theme = {
    "paper": "#0b0b0d" if dark_mode else "#efede7",
    "paper_2": "#121216" if dark_mode else "#f8f6f2",
    "panel": "rgba(19,19,23,0.82)" if dark_mode else "rgba(255,255,255,0.74)",
    "panel_strong": "rgba(28,28,33,0.94)" if dark_mode else "rgba(255,255,255,0.96)",
    "stroke": "rgba(255,255,255,0.10)" if dark_mode else "rgba(16,16,16,0.12)",
    "stroke_strong": "rgba(255,255,255,0.22)" if dark_mode else "rgba(16,16,16,0.24)",
    "text": "#f3f1ea" if dark_mode else "#101010",
    "muted": "#b9b1a7" if dark_mode else "#6c675f",
    "muted_soft": "rgba(243,241,234,0.72)" if dark_mode else "#7d776d",
    "accent": "#f2c400",
    "shadow": "0 18px 42px rgba(0, 0, 0, 0.32)" if dark_mode else "0 18px 40px rgba(0, 0, 0, 0.08)",
    "sidebar_bg": "linear-gradient(180deg, rgba(16,16,19,0.96) 0%, rgba(12,12,15,0.98) 100%)" if dark_mode else "linear-gradient(180deg, rgba(255,255,255,0.78) 0%, rgba(248,246,242,0.96) 100%)",
    "app_overlay": "rgba(0,0,0,0.08)" if dark_mode else "rgba(255,255,255,0.42)",
    "hero_overlay_1": "rgba(255,255,255,0.04)" if dark_mode else "rgba(255,255,255,0.62)",
    "hero_overlay_2": "rgba(255,255,255,0.02)" if dark_mode else "rgba(255,255,255,0.36)",
    "hero_note_bg": "#050506" if dark_mode else "#0b0b0c",
    "hero_note_text": "#f5f1e9",
    "hero_note_muted": "rgba(243,239,231,0.72)" if dark_mode else "rgba(255,255,255,0.74)",
    "hero_note_outline": "rgba(255,255,255,0.10)",
    "hero_note_label": "rgba(243,239,231,0.68)",
    "dot_color": "rgba(255,255,255,0.10)" if dark_mode else "rgba(16,16,16,0.08)",
    "indicator": "#f3f1ea" if dark_mode else "#111111",
    "button_bg": "#17171b" if dark_mode else "#ffffff",
    "button_text": "#f3f1ea" if dark_mode else "#111111",
    "button_border": "rgba(255,255,255,0.18)" if dark_mode else "rgba(16,16,16,0.18)",
    "button_hover_bg": "#202026" if dark_mode else "#f8f6f2",
    "button_hover_text": "#f3f1ea" if dark_mode else "#111111",
    "tab_active_bg": "#f2f0ea" if dark_mode else "#111111",
    "tab_active_text": "#111111" if dark_mode else "#f7f5ef",
    "chip_bg": "#111111" if dark_mode else "#111111",
    "chip_text": "#f7f5ef",
    "control_bg": "#111214" if dark_mode else "#e7e3dc",
    "control_text": "#f3f1ea" if dark_mode else "#141414",
    "control_icon": "#f3f1ea" if dark_mode else "#141414",
    "toggle_track": "#4b4b54" if dark_mode else "#d3d0c8",
    "toggle_knob": "#faf8f1" if dark_mode else "#ffffff",
    "toggle_border": "#6f6f78" if dark_mode else "#c7c3bb",
    "slider_track": "#3c3c43" if dark_mode else "#d8d3cb",
    "slider_thumb_border": "#0f0f10" if dark_mode else "#ffffff",
    "table_bg": "#f7f5ef",
    "table_text": "#151515",
    "table_border": "#d7d1c8",
}

# -------------------------
# Styling (clean, no icon leakage)
# -------------------------
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600;700&family=Space+Mono:wght@400;700&family=DotGothic16&display=swap');

:root {
  --paper: %(paper)s;
  --paper-2: %(paper_2)s;
  --panel: %(panel)s;
  --panel-strong: %(panel_strong)s;
  --stroke: %(stroke)s;
  --stroke-strong: %(stroke_strong)s;
  --text: %(text)s;
  --muted: %(muted)s;
  --muted-soft: %(muted_soft)s;
  --accent: %(accent)s;
  --shadow: %(shadow)s;
  --app-overlay: %(app_overlay)s;
  --hero-overlay-1: %(hero_overlay_1)s;
  --hero-overlay-2: %(hero_overlay_2)s;
  --hero-note-bg: %(hero_note_bg)s;
  --hero-note-text: %(hero_note_text)s;
  --hero-note-muted: %(hero_note_muted)s;
  --hero-note-outline: %(hero_note_outline)s;
  --hero-note-label: %(hero_note_label)s;
  --indicator: %(indicator)s;
  --sidebar-bg: %(sidebar_bg)s;
  --button-bg: %(button_bg)s;
  --button-text: %(button_text)s;
  --button-border: %(button_border)s;
  --button-hover-bg: %(button_hover_bg)s;
  --button-hover-text: %(button_hover_text)s;
  --tab-active-bg: %(tab_active_bg)s;
  --tab-active-text: %(tab_active_text)s;
  --chip-bg: %(chip_bg)s;
  --chip-text: %(chip_text)s;
  --control-bg: %(control_bg)s;
  --control-text: %(control_text)s;
  --control-icon: %(control_icon)s;
  --toggle-track: %(toggle_track)s;
  --toggle-knob: %(toggle_knob)s;
  --toggle-border: %(toggle_border)s;
  --slider-track: %(slider_track)s;
  --slider-thumb-border: %(slider_thumb_border)s;
  --table-bg: %(table_bg)s;
  --table-text: %(table_text)s;
  --table-border: %(table_border)s;
}

html, body, [class*="css"], [class*="st-"] {
  font-family: 'Inter', sans-serif;
}

header { visibility: hidden; height: 0; }
[data-testid="stToolbar"] { display: none; }

h1, h2, h3, h4 {
  font-family: 'Cormorant Garamond', serif;
  color: var(--text);
  letter-spacing: -0.02em;
  font-weight: 600;
}

.stApp {
  background:
    radial-gradient(circle at 1px 1px, %(dot_color)s 1px, transparent 0),
    linear-gradient(180deg, var(--app-overlay), var(--app-overlay)),
    var(--paper);
  background-size: 18px 18px, auto, auto;
  color: var(--text);
}

[data-testid="stSidebar"] {
  background: var(--sidebar-bg);
  border-right: 1px solid var(--stroke);
  box-shadow: 10px 0 32px rgba(0, 0, 0, 0.04);
}

/* Remove Streamlit default collapse/expander icons that render as text */
button[aria-label="Open sidebar"],
button[aria-label="Close sidebar"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
  display: none !important;
}

.sidebar-card {
  background: var(--panel);
  border: 1px solid var(--stroke);
  border-radius: 10px;
  padding: 16px 16px 14px 16px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
  position: relative;
}

.sidebar-card::before {
  content: "";
  position: absolute;
  left: 16px;
  top: 0;
  width: 46px;
  height: 3px;
  background: var(--accent);
}

.sidebar-card h3 {
  margin: 0;
  font-size: 1.5rem;
}

.sidebar-kicker {
  font-family: 'DotGothic16', monospace;
  font-size: 0.76rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 10px;
}

.sidebar-card, .sidebar-card * {
  color: var(--text) !important;
}

.stForm {
  background: var(--panel);
  border: 1px solid var(--stroke);
  border-radius: 10px;
  padding: 14px 14px 8px 14px;
  box-shadow: var(--shadow);
}

.control-label-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin: 0 0 8px 0;
}

.control-label {
  color: var(--text);
  font-family: 'DotGothic16', monospace;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.toggle-inline-label {
  color: var(--text);
  font-size: 0.96rem;
  font-weight: 500;
  line-height: 1.2;
  padding-top: 6px;
}

.control-label-row .info-dot {
  width: 22px;
  height: 22px;
  font-size: 0.78rem;
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] .stToggle *,
[data-testid="stSidebar"] .stSlider *,
[data-testid="stSidebar"] .stSelectbox *,
[data-testid="stSidebar"] .stMultiSelect * {
  color: var(--text) !important;
}

[data-testid="stWidgetLabel"] p,
.stToggle p,
.stSelectbox p,
.stMultiSelect p,
.stSlider p,
.stMarkdown p,
.stMarkdown li,
.stMarkdown span {
  color: var(--text);
}

div[data-baseweb="select"] *,
div[data-baseweb="tag"] * {
  color: var(--text) !important;
}

[data-testid="stSlider"] div,
[data-testid="stSlider"] span,
[data-testid="stSlider"] p {
  color: var(--text) !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="popover"] > div {
  border-radius: 8px !important;
}

div[data-baseweb="select"] > div {
  background: var(--control-bg) !important;
  border: 1px solid var(--stroke-strong) !important;
  min-height: 3.1rem !important;
  padding: 0 10px !important;
  box-shadow: none !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div span,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div input,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div span,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div div {
  color: var(--control-text) !important;
  -webkit-text-fill-color: var(--control-text) !important;
}

div[data-baseweb="select"] svg {
  fill: var(--control-icon) !important;
  color: var(--control-icon) !important;
}

div.stButton > button,
div.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button,
button[kind="primary"],
button[kind="secondary"] {
  border-radius: 8px !important;
  border: 1px solid var(--button-border) !important;
  background: var(--button-bg) !important;
  color: var(--button-text) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.03em;
  padding: 0.65rem 1.1rem !important;
  box-shadow: none !important;
}

div.stButton > button *,
div.stDownloadButton > button *,
div[data-testid="stFormSubmitButton"] > button *,
button[kind="primary"] *,
button[kind="secondary"] * {
  color: var(--button-text) !important;
  fill: var(--button-text) !important;
  -webkit-text-fill-color: var(--button-text) !important;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover {
  background: var(--button-hover-bg) !important;
  color: var(--button-hover-text) !important;
  border-color: var(--accent) !important;
}

div.stButton > button:hover *,
div.stDownloadButton > button:hover *,
div[data-testid="stFormSubmitButton"] > button:hover *,
button[kind="primary"]:hover *,
button[kind="secondary"]:hover * {
  color: var(--button-hover-text) !important;
  fill: var(--button-hover-text) !important;
  -webkit-text-fill-color: var(--button-hover-text) !important;
}

div[data-baseweb="select"] span[data-baseweb="tag"],
span[data-baseweb="tag"] {
  background: var(--chip-bg) !important;
  color: var(--chip-text) !important;
  border: 1px solid transparent !important;
  border-radius: 6px !important;
  font-family: 'Space Mono', monospace !important;
  padding: 6px 10px !important;
}

div[data-baseweb="select"] span[data-baseweb="tag"] *,
div[data-baseweb="select"] span[data-baseweb="tag"] span,
div[data-baseweb="select"] span[data-baseweb="tag"] div,
div[data-baseweb="select"] span[data-baseweb="tag"] svg,
div[data-baseweb="select"] span[data-baseweb="tag"] path,
span[data-baseweb="tag"] svg,
span[data-baseweb="tag"] path,
span[data-baseweb="tag"] *,
[data-baseweb="tag"] span,
[data-baseweb="tag"] div {
  color: var(--chip-text) !important;
  fill: var(--chip-text) !important;
  -webkit-text-fill-color: var(--chip-text) !important;
}

.hero-shell {
  background:
    linear-gradient(180deg, var(--hero-overlay-1), var(--hero-overlay-2)),
    var(--paper-2);
  border: 1px solid var(--stroke);
  border-radius: 10px;
  padding: 18px 24px;
  box-shadow: var(--shadow);
}

.hero-grid {
  display: grid;
  grid-template-columns: 1.75fr 0.9fr;
  gap: 18px;
  align-items: start;
}

.mono-tag {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: 'DotGothic16', monospace;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}

.mono-tag::before {
  content: "";
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--indicator);
}

.hero-shell h1 {
  font-size: clamp(2.3rem, 3vw, 3.5rem);
  line-height: 0.95;
  margin: 14px 0 8px 0;
  max-width: 14ch;
}

.hero-copy {
  font-size: 0.98rem;
  line-height: 1.6;
  color: var(--muted);
  max-width: 34rem;
}

.pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 16px;
}

.pill {
  display: inline-flex;
  align-items: center;
  padding: 10px 16px;
  border-radius: 8px;
  border: 1px solid var(--stroke-strong);
  background: var(--panel);
  font-family: 'Space Mono', monospace;
  color: var(--text);
  font-size: 0.88rem;
}

.hero-note {
  position: relative;
  overflow: hidden;
  background: var(--hero-note-bg);
  color: var(--hero-note-text);
  border-radius: 10px;
  padding: 18px 18px 14px 18px;
  min-height: 170px;
}

.hero-note::after {
  content: "";
  position: absolute;
  inset: 14px;
  border: 1px solid var(--hero-note-outline);
  border-radius: 8px;
  pointer-events: none;
}

.hero-note .mono-tag {
  color: var(--hero-note-label);
}

.hero-note .mono-tag::before {
  background: var(--accent);
}

.hero-note-title {
  color: var(--hero-note-text) !important;
  font-family: 'Inter', sans-serif;
  font-size: 0.98rem;
  line-height: 1.3;
  font-weight: 700;
  margin: 10px 0 8px 0;
  max-width: none;
}

.hero-note-copy {
  color: var(--hero-note-muted) !important;
  line-height: 1.5;
  margin-bottom: 0;
  font-size: 0.92rem;
}

.kpi-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-top: 18px;
}

.kpi-card {
  background: var(--panel-strong);
  border: 1px solid var(--stroke);
  border-radius: 8px;
  padding: 18px 20px;
  min-height: 110px;
  box-shadow: var(--shadow);
}

.kpi-title {
  color: var(--muted);
  font-family: 'DotGothic16', monospace;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.kpi-value {
  font-size: 28px;
  font-weight: 700;
  color: var(--text);
  margin-top: 6px;
}

.kpi-sub {
  color: var(--muted);
  font-size: 12px;
  margin-top: 6px;
}

.ribbon {
  margin-top: 18px;
  padding: 8px 14px;
  border-radius: 8px;
  color: #111;
  font-weight: 600;
  text-align: center;
  background: var(--accent);
  border: 1px solid var(--stroke-strong);
  font-family: 'Space Mono', monospace;
  letter-spacing: 0.04em;
  font-size: 0.92rem;
}

.section {
  margin-top: 22px;
  background: var(--panel);
  border: 1px solid var(--stroke);
  border-radius: 10px;
  padding: 18px 20px;
  box-shadow: var(--shadow);
}

.section-head {
  margin-top: 20px;
  display: grid;
  gap: 6px;
}

.title-row {
  display: inline-flex;
  align-items: center;
  gap: 10px;
}

.info-dot {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  border: 1px solid var(--stroke-strong);
  background: var(--panel);
  color: var(--muted-soft);
  font-family: 'Inter', sans-serif;
  font-size: 0.84rem;
  font-weight: 700;
  cursor: help;
  flex: 0 0 auto;
}

.section-kicker {
  font-family: 'DotGothic16', monospace;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}

.section-title {
  margin: 0;
  font-size: clamp(1.65rem, 1.8vw, 2.2rem);
  line-height: 0.95;
}

.section-copy {
  color: var(--muted);
  max-width: 58rem;
  line-height: 1.6;
}

.studio-note {
  margin-top: 14px;
  border: 1px solid var(--stroke);
  background: var(--panel);
  border-radius: 8px;
  padding: 16px 18px;
  box-shadow: var(--shadow);
}

.studio-note ul {
  margin: 10px 0 0 18px;
  color: var(--muted);
  line-height: 1.7;
}

.studio-note strong {
  color: var(--text);
}

.insight-list {
  margin: 0;
  padding-left: 18px;
  color: var(--muted);
  line-height: 1.8;
}

.insight-list strong {
  color: var(--text);
}

.small-muted { color: var(--muted); font-size: 12px; }

.static-control {
  margin: 0 0 8px 0;
  min-height: 3.1rem;
  display: flex;
  align-items: center;
  border: 1px solid var(--stroke-strong);
  border-radius: 8px;
  padding: 0 14px;
  background: var(--paper-2);
  color: var(--text);
  font-size: 0.98rem;
}

.static-control .static-label {
  color: var(--muted);
  margin-right: 8px;
}

.help-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  margin-top: 10px;
}

.help-card {
  background: var(--panel);
  border: 1px solid var(--stroke);
  border-radius: 8px;
  padding: 14px 16px;
}

.help-card h4 {
  margin: 0 0 8px 0;
  font-size: 1.15rem;
}

.help-card ul {
  margin: 0;
  padding-left: 18px;
  line-height: 1.7;
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--table-border);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: var(--shadow);
  background: var(--table-bg) !important;
}

div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--stroke);
  border-radius: 8px;
  padding: 12px 14px;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"],
div[data-testid="stMetricDelta"],
div[data-testid="stMetricLabel"] *,
div[data-testid="stMetricValue"] *,
div[data-testid="stMetricDelta"] *,
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] p,
div[data-testid="stMetric"] span {
  color: var(--text) !important;
}

div[data-testid="stMetricLabel"] *,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
  color: var(--muted-soft) !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 10px;
}

.stTabs [data-baseweb="tab"] {
  border-radius: 8px;
  border: 1px solid var(--stroke);
  background: var(--panel);
  padding: 10px 16px;
  color: var(--muted) !important;
}

.stTabs [data-baseweb="tab-list"] {
  margin-top: 16px;
}

.stTabs [aria-selected="true"] {
  background: var(--tab-active-bg) !important;
  color: var(--tab-active-text) !important;
  border-color: var(--tab-active-bg) !important;
}

.stTabs [aria-selected="true"] * {
  color: var(--tab-active-text) !important;
}

[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] td {
  color: var(--table-text) !important;
  background: var(--table-bg) !important;
}

[data-testid="stToggle"] {
  margin: 0 !important;
  display: flex !important;
  align-items: center !important;
  min-height: 30px !important;
}

[data-testid="stToggle"] label {
  gap: 0 !important;
  display: flex !important;
  align-items: center !important;
  min-height: 30px !important;
}

[data-testid="stToggle"] div[role="switch"],
[data-testid="stToggle"] [data-baseweb="switch"] {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: flex-start !important;
  width: 50px !important;
  min-width: 50px !important;
  height: 28px !important;
  padding: 2px !important;
  border-radius: 999px !important;
  background: var(--toggle-track) !important;
  border: 1px solid var(--toggle-border) !important;
  box-shadow: none !important;
}

[data-testid="stToggle"] div[role="switch"][aria-checked="true"],
[data-testid="stToggle"] [data-baseweb="switch"][aria-checked="true"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}

[data-testid="stToggle"] div[role="switch"] > div,
[data-testid="stToggle"] [data-baseweb="switch"] > div {
  background: var(--toggle-knob) !important;
  width: 22px !important;
  height: 22px !important;
  border-radius: 999px !important;
  box-shadow: none !important;
}

[data-testid="stSlider"] div[data-baseweb="slider"] {
  padding: 8px 0 2px 0;
}

[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {
  background: var(--slider-track) !important;
  height: 6px !important;
  border-radius: 999px !important;
}

[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:nth-child(2) {
  background: var(--accent) !important;
  height: 6px !important;
  border-radius: 999px !important;
}

[data-testid="stSlider"] div[role="slider"] {
  background: var(--accent) !important;
  border: 2px solid var(--slider-thumb-border) !important;
  width: 18px !important;
  height: 18px !important;
  box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.18) !important;
}

[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"],
[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] {
  color: var(--muted-soft) !important;
}

[data-testid="stSidebar"] .stButton > button {
  width: 100%;
}

@media (max-width: 1100px) {
  .hero-grid { grid-template-columns: 1fr; }
  .kpi-row { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 700px) {
  .hero-shell { padding: 18px 16px; }
  .kpi-row { grid-template-columns: 1fr; }
  .help-grid { grid-template-columns: 1fr; }
}
</style>
"""
for key, value in theme.items():
    css = css.replace(f"%({key})s", value)
st.markdown(css, unsafe_allow_html=True)

# -------------------------
# Data loading
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH_EVENTS = BASE_DIR / "data" / "processed" / "meps_group6_analysis_ready_events.parquet"
DATA_PATH_BASE = BASE_DIR / "data" / "processed" / "meps_group6_analysis_ready.parquet"
DATA_PATH = DATA_PATH_EVENTS if DATA_PATH_EVENTS.exists() else DATA_PATH_BASE
NEG_CODES = [-1, -7, -8, -9]
RANDOM_STATE = 42

plt.rcParams.update(
    {
        "figure.facecolor": theme["paper_2"],
        "axes.facecolor": theme["paper_2"],
        "axes.edgecolor": "#5e594f" if dark_mode else "#b3ada4",
        "axes.labelcolor": theme["text"],
        "xtick.color": theme["text"],
        "ytick.color": theme["text"],
        "text.color": theme["text"],
        "axes.titleweight": "semibold",
        "axes.titlesize": 14,
        "font.family": "DejaVu Sans",
        "grid.color": "#403c37" if dark_mode else "#d7d1c8",
    }
)

curve_line_color = "#fff7d6" if dark_mode else "#111111"
curve_fill_color = "#f2c400" if dark_mode else "#f2c400"
curve_marker_color = "#ffd84d" if dark_mode else "#c89b00"
diag_line_color = "#b9b1a7" if dark_mode else "#8a847c"

@st.cache_data
def load_data(uploaded_bytes=None):
    if uploaded_bytes is not None:
        df = pd.read_parquet(io.BytesIO(uploaded_bytes))
        return df
    if not DATA_PATH.exists():
        st.error(f"Missing file: {DATA_PATH}. Run the notebook to create it.")
        st.stop()
    df = pd.read_parquet(DATA_PATH)
    return df


def select_features(df, allowed_prefixes, exclude_cols, include_suffixes=("_y1",)):
    cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in allowed_prefixes) or any(c.endswith(s) for s in include_suffixes)
    ]
    cols = [c for c in cols if "Y2" not in c]
    cols = [c for c in cols if c not in exclude_cols]
    if "AGE_MAX" in df.columns and "AGE_MAX" not in cols:
        cols.append("AGE_MAX")
    return cols


def build_dataset(df):
    df = df.copy()
    df = df.replace(NEG_CODES, np.nan)
    target_raw = "IPDISY2"
    df[target_raw] = df[target_raw].replace(NEG_CODES, np.nan)
    df = df.dropna(subset=[target_raw]).copy()

    def make_target_class(value):
        if pd.isna(value):
            return np.nan
        if value == 0:
            return 0
        if value == 1:
            return 1
        return 2

    df["Y_IPDISY2_CLASS"] = df[target_raw].apply(make_target_class).astype(int)

    allowed = (
        "AGE", "SEX", "RACE", "HISP",
        "EDUC", "POVCAT", "INSUR",
        "ASTH", "DIAB", "ARTH", "HYPER", "CHRON",
        "RTHLTH", "MNHLTH",
        "ERDISY1", "IPDISY1",
        "TOTEXPY1", "RXEXPY1"
    )
    exclude = {
        "DUPERSID", "SOURCE_PANEL", "CANCERY1", "CANCERY2",
        "IPDISY2", "Y_IPDISY2_CLASS"
    }

    X = df[select_features(df, allowed, exclude)].copy()
    missing_rate = X.isna().mean()
    high_missing = missing_rate[missing_rate > 0.5].index.tolist()
    if high_missing:
        X = X.drop(columns=high_missing)

    if "AGE_MAX" in X.columns:
        age_cols_to_drop = [c for c in X.columns if c.startswith("AGE") and c != "AGE_MAX"]
        X = X.drop(columns=age_cols_to_drop)

    y = df["Y_IPDISY2_CLASS"].astype(int)
    panels = df["PANEL"].astype(int)
    meta_cols = [c for c in ["DUPERSID", "PANEL", "AGE_MAX", "SEX", "INSURCY1", "POVCATY1"] if c in df.columns]
    meta = df[meta_cols].copy()
    return X, y, panels, X.columns.tolist(), meta


def identify_categorical_columns(X):
    cost_count_cols = [c for c in X.columns if c.endswith("_cost_y1") or c.endswith("_count_y1")]
    numeric_force = set(cost_count_cols + ["AGE_MAX", "TOTEXPY1", "RXEXPY1", "IPDISY1", "ERDISY1"])
    cat_cols = []
    for c in X.columns:
        if c in numeric_force:
            continue
        if X[c].dtype == "object":
            cat_cols.append(c)
        elif pd.api.types.is_integer_dtype(X[c]) and X[c].nunique(dropna=True) <= 50:
            cat_cols.append(c)
        elif pd.api.types.is_float_dtype(X[c]):
            vals = X[c].dropna()
            if len(vals) > 0 and (vals % 1 == 0).all() and vals.nunique() <= 50:
                cat_cols.append(c)
    return cat_cols


def prepare_baseline_frames(X_train, X_test, cat_cols):
    X_train_base = X_train.copy()
    X_test_base = X_test.copy()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_train_base[num_cols] = num_imputer.fit_transform(X_train_base[num_cols])
        X_test_base[num_cols] = num_imputer.transform(X_test_base[num_cols])

    if cat_cols:
        for col in cat_cols:
            X_train_base[col] = X_train_base[col].astype("string").fillna("NA")
            X_test_base[col] = X_test_base[col].astype("string").fillna("NA")
        X_train_base = pd.get_dummies(X_train_base, columns=cat_cols, dummy_na=False)
        X_test_base = pd.get_dummies(X_test_base, columns=cat_cols, dummy_na=False)
        X_train_base, X_test_base = X_train_base.align(X_test_base, join="left", axis=1, fill_value=0)

    return X_train_base, X_test_base


def prepare_catboost_frames(X_train, X_test, cat_cols):
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_train_cb[num_cols] = num_imputer.fit_transform(X_train_cb[num_cols])
        X_test_cb[num_cols] = num_imputer.transform(X_test_cb[num_cols])

    for col in cat_cols:
        X_train_cb[col] = X_train_cb[col].astype("string").fillna("NA")
        X_test_cb[col] = X_test_cb[col].astype("string").fillna("NA")

    cat_idx = [X_train_cb.columns.get_loc(c) for c in cat_cols]
    return X_train_cb, X_test_cb, cat_idx


def compute_lift_table(y_true_binary, scores):
    score_df = pd.DataFrame({"y_true": y_true_binary, "score": scores})
    score_df["decile"] = pd.qcut(score_df["score"], 10, labels=False, duplicates="drop")
    lift_table = (
        score_df.groupby("decile")
        .agg(n=("y_true", "size"), positives=("y_true", "sum"))
        .reset_index()
    )
    lift_table["rate"] = lift_table["positives"] / lift_table["n"]
    base_rate = score_df["y_true"].mean()
    lift_table["lift"] = lift_table["rate"] / base_rate if base_rate > 0 else 0.0
    lift_table = lift_table.sort_values("decile", ascending=False).reset_index(drop=True)
    return lift_table, base_rate


@st.cache_data(show_spinner=False)
def train_and_eval(X, y, panels, meta, model_names, pred_pct):
    unique_panels = sorted(int(p) for p in panels.dropna().unique())
    holdout_panel = max(unique_panels)
    train_idx = panels.isin([p for p in unique_panels if p != holdout_panel])
    test_idx = panels.isin([holdout_panel])

    X_train = X.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()
    y_train = y.loc[train_idx].copy()
    y_test = y.loc[test_idx].copy()
    meta_test = meta.loc[test_idx].reset_index(drop=True).copy()

    cat_cols = identify_categorical_columns(X_train)
    X_train_base, X_test_base = prepare_baseline_frames(X_train, X_test, cat_cols)
    X_train_cb, X_test_cb, cat_idx = prepare_catboost_frames(X_train, X_test, cat_cols)

    class_counts = y_train.value_counts().sort_index()
    class_weights = (class_counts.sum() / class_counts).values.tolist()
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    models = {}
    if "CatBoost" in model_names and CatBoostClassifier is not None:
        models["CatBoost"] = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_seed=RANDOM_STATE,
            verbose=False,
            class_weights=class_weights,
            thread_count=-1,
        )
    if "GB" in model_names:
        models["GB"] = GradientBoostingClassifier(random_state=RANDOM_STATE)
    if "HGB" in model_names:
        models["HGB"] = HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_depth=6)
    if "RF" in model_names:
        models["RF"] = RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )

    results = []
    preds = {}
    probas = {}
    diagnostics = {}

    for name, model in models.items():
        if name == "CatBoost":
            model.fit(X_train_cb, y_train, cat_features=cat_idx)
            y_pred = np.asarray(model.predict(X_test_cb)).astype(int).flatten()
            y_proba = model.predict_proba(X_test_cb)
        else:
            model.fit(X_train_base, y_train, sample_weight=sample_weight)
            y_pred = model.predict(X_test_base)
            y_proba = model.predict_proba(X_test_base)

        preds[name] = y_pred
        probas[name] = y_proba

        y_test_bin = (y_test == 2).astype(int)
        recall_class2 = recall_score(y_test_bin, (y_pred == 2).astype(int), zero_division=0)
        pr_auc_class2 = average_precision_score(y_test_bin, y_proba[:, 2])
        lift_table, base_rate = compute_lift_table(y_test_bin, y_proba[:, 2])
        top_decile_lift = float(lift_table.loc[0, "lift"]) if not lift_table.empty else np.nan
        frac_pos, mean_pred = calibration_curve(y_test_bin, y_proba[:, 2], n_bins=10)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_bin, y_proba[:, 2])

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "Macro-F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall (2+)": recall_class2,
            "PR-AUC (2+)": pr_auc_class2,
            "Top-Decile Lift": top_decile_lift,
        })

        diagnostics[name] = {
            "precision_curve": precision_curve,
            "recall_curve": recall_curve,
            "frac_pos": frac_pos,
            "mean_pred": mean_pred,
            "lift_table": lift_table,
            "base_rate": base_rate,
        }

    results_df = pd.DataFrame(results).sort_values("PR-AUC (2+)", ascending=False).reset_index(drop=True)
    pred_df, score_cols, flag_cols = build_prediction_table(meta_test, probas, pred_pct)
    return results_df, preds, probas, y_test, diagnostics, pred_df, score_cols, flag_cols


def build_prediction_table(meta, probas_dict, pred_pct):
    model_names = list(probas_dict.keys())
    pred_df = meta.copy().reset_index(drop=True)

    for name in model_names:
        scores = probas_dict[name][:, 2]
        cutoff = np.quantile(scores, 1 - pred_pct / 100)
        pred_df[f"{name}_score"] = scores
        pred_df[f"{name}_flag"] = (scores >= cutoff).astype(int)

    score_cols = [f"{name}_score" for name in model_names]
    flag_cols = [f"{name}_flag" for name in model_names]
    pred_df["avg_score"] = pred_df[score_cols].mean(axis=1)
    pred_df["consensus_count"] = pred_df[flag_cols].sum(axis=1)
    pred_df["consensus_all"] = pred_df["consensus_count"] == len(model_names)
    return pred_df, score_cols, flag_cols


def info_icon_html(help_text):
    return f'<span class="info-dot" title="{escape(help_text)}">i</span>'


def render_control_label(label, help_text=None):
    info_html = info_icon_html(help_text) if help_text else ""
    st.markdown(
        f"""
<div class="control-label-row">
  <span class="control-label">{label}</span>
  {info_html}
</div>
""",
        unsafe_allow_html=True,
    )


def render_toggle_row(label, value, key, help_text, container=st):
    return container.toggle(label, value=value, key=key, help=help_text)


def render_section_header(title, kicker, copy_text, help_text=None):
    info_html = ""
    if help_text:
        info_html = info_icon_html(help_text)
    st.markdown(
        f"""
<div class="section-head">
  <div class="section-kicker">{kicker}</div>
  <div class="title-row">
    <h2 class="section-title">{title}</h2>
    {info_html}
  </div>
  <div class="section-copy">{copy_text}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# -------------------------
# Sidebar Controls
# -------------------------
with st.sidebar:
    light_toggle = render_toggle_row(
        "Light mode",
        value=not dark_mode,
        key="light_mode_toggle",
        help_text="Switches the interface to a light presentation mode.",
        container=st,
    )
    if light_toggle == dark_mode:
        st.session_state["dark_mode"] = not light_toggle
        st.rerun()

    st.markdown(
        """
<div class='sidebar-card'>
  <div class='sidebar-kicker'>Control Panel</div>
  <h3>Analysis Controls</h3>
  <div class='small-muted'>Panel-validated inpatient-risk review.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    available_models = ["CatBoost", "HGB", "GB", "RF"] if CatBoostClassifier is not None else ["HGB", "GB", "RF"]
    default_models = [m for m in ["CatBoost"] if m in available_models] or ["HGB"]

    with st.form("controls"):
        render_control_label(
            "Target",
            "Fixed target for the final dashboard: Year-2 inpatient admissions grouped as 0, 1, and 2+."
        )
        st.markdown(
            "<div class='static-control'><span class='static-label'>Year-2 admissions:</span><strong>0 / 1 / 2+</strong></div>",
            unsafe_allow_html=True,
        )

        render_control_label(
            "Models",
            "Compare panel-validated baselines against the final CatBoost model."
        )
        model_choices = st.multiselect(
            "Models",
            available_models,
            default=default_models,
            label_visibility="collapsed",
        )

        render_control_label(
            "Predicted Cohort Size (%)",
            "Top share of the holdout cohort retained for ranked export, based on the class-2 probability."
        )
        pred_pct = st.slider(
            "Predicted Cohort Size (%)",
            1,
            20,
            5,
            1,
            label_visibility="collapsed",
        )
        render_control_label(
            "Max Rows To Display",
            "Maximum number of holdout records shown in the cohort tables."
        )
        max_rows = st.slider(
            "Max Rows To Display",
            50,
            500,
            200,
            50,
            label_visibility="collapsed",
        )

        run_btn = st.form_submit_button("Run analysis")

    render_control_label(
        "Data Source",
        "Optional: upload a new analysis-ready parquet file with the same schema. The dashboard will treat the most recent panel in that file as the holdout panel.",
    )
    uploaded_file = st.file_uploader(
        "Upload analysis-ready parquet",
        type=["parquet"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        st.markdown("<div class='small-muted'>Using uploaded analysis-ready parquet.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='small-muted'>Using the project analysis-ready MEPS parquet.</div>", unsafe_allow_html=True)

# -------------------------
# Main Layout
# -------------------------
uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
with st.spinner("Loading data..."):
    df = load_data(uploaded_bytes)

X, y, panels, feature_cols, meta = build_dataset(df)
panel_values = sorted(int(p) for p in panels.dropna().unique())
if len(panel_values) < 2:
    st.error("At least two panels are required to run panel-based validation.")
    st.stop()
holdout_panel = max(panel_values)
train_panels = [p for p in panel_values if p != holdout_panel]
holdout_mask = panels.isin([holdout_panel])
holdout_n = int(holdout_mask.sum())
holdout_rate = float((y.loc[holdout_mask] == 2).mean())
panel_label = f"Group 6 / Panels {min(panel_values)}-{max(panel_values)} / 2018-2023"

st.markdown(
    f"""
<div class="hero-shell">
  <div class="hero-grid">
    <div>
      <div class="mono-tag">{panel_label}</div>
      <h1>MEPS Inpatient Risk</h1>
      <p class="hero-copy">
        Panel-validated multiclass prediction of Year-2 inpatient admissions from Year-1 MEPS features.
      </p>
    </div>
    <div class="hero-note">
      <div class="mono-tag">Overview</div>
      <div class="hero-note-title">CatBoost, panel split, calibration, lift.</div>
      <p class="hero-note-copy">
        Target: 0 / 1 / 2+ admissions. Holdout: panel {holdout_panel}. Output: model comparison, high-risk ranking, and export-ready cohort lists.
      </p>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

show_help = render_toggle_row(
    "Show dashboard help",
    value=False,
    key="show_dashboard_help",
    help_text="Shows definitions and workflow notes for first-time users.",
)

if show_help:
    render_section_header(
        "Dashboard help",
        "Guide",
        "Key terms and workflow notes for first-time use.",
        "Explains the target, validation design, and how to interpret the main ranking metrics.",
    )
    st.markdown(
        f"""
<div class="help-grid">
  <div class="help-card">
    <h4>What this dashboard does</h4>
    <ul>
      <li>Uses <strong>Year-1 features</strong> to predict <strong>Year-2 inpatient admissions</strong>.</li>
      <li>Groups the outcome into <strong>0</strong>, <strong>1</strong>, and <strong>2+</strong> admissions.</li>
      <li>Evaluates models by training on earlier panels and testing on the most recent panel.</li>
    </ul>
  </div>
  <div class="help-card">
    <h4>Key terms</h4>
    <ul>
      <li><strong>Holdout panel</strong>: the final unseen panel used only for testing. Here: <strong>{holdout_panel}</strong>.</li>
      <li><strong>Class 2</strong>: members with <strong>2 or more</strong> Year-2 inpatient admissions.</li>
      <li><strong>Lift</strong>: how much more concentrated true class-2 cases are in a top-ranked group versus random selection.</li>
    </ul>
  </div>
  <div class="help-card">
    <h4>How to read the metrics</h4>
    <ul>
      <li><strong>Balanced accuracy</strong> prevents the majority class from overstating model quality.</li>
      <li><strong>Macro-F1</strong> reflects overall balance across all three classes.</li>
      <li><strong>PR-AUC (2+)</strong> focuses on ranking quality for the rare high-risk class.</li>
    </ul>
  </div>
  <div class="help-card">
    <h4>Scaling the workflow</h4>
    <ul>
      <li>You can upload a new analysis-ready parquet with later MEPS panels.</li>
      <li>The dashboard will automatically treat the <strong>latest panel as holdout</strong>.</li>
      <li>This keeps the same Year-1 to Year-2 prediction design while extending to new years.</li>
    </ul>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

ribbon_text = f"Target: Year-2 inpatient admissions (0 / 1 / 2+) | Holdout class-2 rate: {holdout_rate:.1%}"
st.markdown(f"<div class='ribbon'>{ribbon_text}</div>", unsafe_allow_html=True)

st.markdown(
    f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-title">Model Cohort</div>
    <div class="kpi-value">{len(y):,}</div>
    <div class="kpi-sub">Rows available for modeling</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Holdout Panel</div>
    <div class="kpi-value">{holdout_n:,}</div>
    <div class="kpi-sub">Panel {holdout_panel} rows</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Predictors</div>
    <div class="kpi-value">{len(feature_cols)}</div>
    <div class="kpi-sub">Year-1 features</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Validation</div>
    <div class="kpi-value">{min(train_panels)}-{max(train_panels)} -> {holdout_panel}</div>
    <div class="kpi-sub">Panel-based holdout</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Run analysis on submit
if run_btn:
    if not model_choices:
        st.warning("Please select at least one model.")
    else:
        with st.spinner("Training panel-validated models..."):
            results_df, preds, probas, y_test, diagnostics, pred_df, score_cols, flag_cols = train_and_eval(
                X, y, panels, meta, model_choices, pred_pct
            )
            st.session_state["results_df"] = results_df
            st.session_state["preds"] = preds
            st.session_state["probas"] = probas
            st.session_state["y_test"] = y_test
            st.session_state["diagnostics"] = diagnostics
            st.session_state["pred_df"] = pred_df
            st.session_state["score_cols"] = score_cols
            st.session_state["flag_cols"] = flag_cols

# Predicted cohort section (collapsible via toggle)
show_pred = render_toggle_row(
    "Show holdout high-risk cohort",
    value=True,
    key="show_predicted_high_risk_cohort",
    help_text="Displays the panel 27 cohort ranked by class-2 probability.",
)
if show_pred:
    render_section_header(
        "Holdout High-Risk Cohort",
        "Panel 27 scoring",
        "Consensus and ranked views derived from class-2 probabilities on the holdout panel.",
        "Consensus keeps only holdout cases flagged by every selected model. Ranked keeps the highest average class-2 scores.",
    )
    if "pred_df" in st.session_state:
        pred_df = st.session_state["pred_df"]
        consensus_df = pred_df[pred_df["consensus_all"]].sort_values("avg_score", ascending=False)
        top_df = pred_df.sort_values("avg_score", ascending=False).head(max_rows)

        meta_cols = [c for c in ["DUPERSID", "PANEL", "AGE_MAX", "SEX", "INSURCY1", "POVCATY1"] if c in pred_df.columns]
        display_cols = meta_cols + ["avg_score", "consensus_count"]

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Consensus cases", f"{len(consensus_df):,}")
        metric_col2.metric("Average cohort score", f"{pred_df['avg_score'].mean():.3f}")
        metric_col3.metric("Selected cohort size", f"Top {pred_pct}%")

        st.markdown(
            f"""
<div class="studio-note">
  <div class="section-kicker">Consensus Rule</div>
  Individuals must fall within the top {pred_pct}% of class-2 risk for <strong>every selected model</strong>. The ranked tab retains broader coverage for review and export.
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        tab_consensus, tab_ranked = st.tabs(["Consensus cohort", "Ranked cohort"])

        with tab_consensus:
            if len(consensus_df) == 0:
                st.markdown(
                    "<div class='studio-note'>No consensus cases at the current threshold. Increase the cohort size to broaden coverage.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.dataframe(consensus_df[display_cols].head(max_rows), width="stretch", hide_index=True)
                csv_consensus = consensus_df[display_cols].to_csv(index=False).encode("utf-8")
                st.download_button("Download consensus CSV", csv_consensus, "consensus_high_risk.csv", "text/csv")

        with tab_ranked:
            st.dataframe(top_df[display_cols], width="stretch", hide_index=True)
            csv_top = top_df[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button("Download ranked CSV", csv_top, "top_high_risk.csv", "text/csv")
    else:
        st.markdown(
            "<div class='studio-note'>Run analysis to generate panel-validated holdout cohort rankings.</div>",
            unsafe_allow_html=True,
        )

# Display results if available
if "results_df" in st.session_state:
    results_df = st.session_state["results_df"]
    preds = st.session_state["preds"]
    probas = st.session_state["probas"]
    y_test = st.session_state["y_test"]
    diagnostics = st.session_state["diagnostics"]

    render_section_header(
        "Model Comparison",
        "Panel 27 evaluation",
        "Compare multiclass baselines against the final panel-validated CatBoost model using class-2 performance as the main ranking signal.",
        "Balanced accuracy, macro-F1, class-2 recall, and class-2 PR-AUC are the primary metrics for this dashboard.",
    )
    display_results = results_df.copy()
    metric_cols = [c for c in display_results.columns if c != "Model"]
    display_results[metric_cols] = display_results[metric_cols].applymap(lambda x: round(float(x), 3))
    st.dataframe(display_results, width="stretch", hide_index=True)

    render_control_label(
        "Model To Visualize",
        "Choose the model to inspect through confusion, PR, calibration, and lift diagnostics."
    )
    model_to_show = st.selectbox(
        "Model To Visualize",
        results_df["Model"].tolist(),
        label_visibility="collapsed",
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        render_section_header(
            "Confusion Matrix",
            "Classification",
            f"Three-class holdout predictions for {model_to_show}.",
            "Rows correspond to observed classes and columns correspond to predicted classes on panel 27.",
        )
        cm = confusion_matrix(y_test, preds[model_to_show], labels=[0, 1, 2])
        fig_cm, ax_cm = plt.subplots(figsize=(5.2, 4.2))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1", "2+"])
        disp.plot(cmap="Greys", values_format="d", ax=ax_cm, colorbar=False)
        ax_cm.grid(False)
        st.pyplot(fig_cm)

    with chart_col2:
        render_section_header(
            "PR Curve (Class 2)",
            "Rare-event fit",
            "Precision-recall for the 2+ admissions class on panel 27.",
            "This curve is the most relevant discrimination view for the rare high-risk class.",
        )
        precision = diagnostics[model_to_show]["precision_curve"]
        recall = diagnostics[model_to_show]["recall_curve"]
        fig_pr, ax_pr = plt.subplots(figsize=(5.2, 4.2))
        ax_pr.plot(recall, precision, color=curve_line_color, linewidth=2.2)
        ax_pr.fill_between(recall, precision, color=curve_fill_color, alpha=0.32)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("Class 2 PR Curve")
        ax_pr.grid(alpha=0.35)
        st.pyplot(fig_pr)

    chart_col3, chart_col4 = st.columns([1.15, 0.85])
    with chart_col3:
        render_section_header(
            "Calibration (Class 2)",
            "Probability check",
            "Observed versus predicted probabilities for the 2+ admissions class.",
            "Curves below the diagonal indicate over-confident probabilities.",
        )
        frac_pos = diagnostics[model_to_show]["frac_pos"]
        mean_pred = diagnostics[model_to_show]["mean_pred"]
        fig_cal, ax_cal = plt.subplots(figsize=(5.6, 4.2))
        ax_cal.plot(
            mean_pred,
            frac_pos,
            marker="o",
            color=curve_line_color,
            markerfacecolor=curve_marker_color,
            markeredgecolor=curve_line_color,
            markersize=7,
            linewidth=2.2,
        )
        ax_cal.plot([0, 1], [0, 1], linestyle="--", color=diag_line_color)
        ax_cal.set_xlabel("Mean predicted probability")
        ax_cal.set_ylabel("Fraction of positives")
        ax_cal.set_title("Class 2 Calibration")
        ax_cal.grid(alpha=0.35)
        st.pyplot(fig_cal)

    with chart_col4:
        render_section_header(
            "Lift by Decile",
            "Ranking utility",
            "Observed class-2 concentration across holdout risk deciles.",
            "Higher lift in the top decile indicates stronger ranking utility for operational targeting.",
        )
        lift_table = diagnostics[model_to_show]["lift_table"]
        fig_lift, ax_lift = plt.subplots(figsize=(4.5, 4.2))
        ax_lift.bar(
            lift_table["decile"].astype(int),
            lift_table["lift"],
            color=curve_fill_color,
            edgecolor=curve_line_color,
            linewidth=1.2,
            alpha=0.85,
        )
        ax_lift.set_xlabel("Decile")
        ax_lift.set_ylabel("Lift")
        ax_lift.set_title("Class 2 Lift")
        ax_lift.invert_xaxis()
        ax_lift.grid(alpha=0.25, axis="y")
        st.pyplot(fig_lift)
        top_lift = float(lift_table.iloc[0]["lift"]) if not lift_table.empty else np.nan
        st.markdown(
            f"<div class='studio-note'><div class='section-kicker'>Selected model</div>Top-decile lift: <strong>{top_lift:.2f}x</strong></div>",
            unsafe_allow_html=True,
        )

else:
    st.markdown(
        "<div class='studio-note'>Run analysis to generate the panel-validated model comparison and holdout cohort output.</div>",
        unsafe_allow_html=True,
    )
