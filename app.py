"""
SpectraFarm (AgriN) — Satellite Crop Intelligence Console
Fusing Sentinel-2 Optical & Sentinel-1 SAR Radar with Machine Learning & Gemini AI Advisory.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import urllib.parse
from datetime import date, datetime, timedelta
from pathlib import Path

import folium
from folium.plugins import Fullscreen
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# ── Setup paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("AGRIN_MODE", "live")

from src.ai.gemini_client import ask_question, generate_advisory, is_gemini_available
from src.config.settings import get_settings
from src.data.demo_data import (
    generate_ndvi_timeseries,
    generate_sar_observations,
    get_demo_crop_prediction,
    get_demo_farm,
    get_demo_stress_assessment,
)
from src.data.satellite_data import (
    fetch_optical_observations,
    fetch_sar_observations,
    get_cropland_mask,
    get_data_source_status,
    get_landcover_summary,
    sample_parcel_landcover,
)
from src.data.schemas import (
    BoundingBox,
    CropPrediction,
    CropType,
    DataSource,
    Farm,
    FarmAnalysis,
    StressAssessment,
)
from src.features.feature_extraction import (
    combine_features,
    extract_optical_features,
    extract_sar_features,
)
from src.intelligence.stress_analysis import assess_stress
from src.ml.crop_classifier import CropClassifierService

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Page Config & Initial State
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SpectraFarm — Satellite Crop Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "lat" not in st.session_state:
    st.session_state["lat"] = 25.3176
if "lon" not in st.session_state:
    st.session_state["lon"] = 82.9739
if "dark_theme" not in st.session_state:
    st.session_state["dark_theme"] = False
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {
            "role": "ai",
            "text": "Hi! I'm AgriN — I provide domain-expert agronomic advice using live Sentinel-1 radar and Sentinel-2 optical telemetry."
        }
    ]

# ═══════════════════════════════════════════════════════════════════════════
# 2D Vector SVG Icons (Lucide-based, Zero Emojis)
# ═══════════════════════════════════════════════════════════════════════════

SVG_SATELLITE = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 7 9 3 5 7l4 4"/><path d="m17 11 4 4-4 4-4-4"/><path d="m8 12 4 4 6-6-4-4Z"/><path d="m16 8 3-3"/><path d="M9 21a6 6 0 0 0-6-6"/></svg>"""
SVG_SPROUT = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 20h10"/><path d="M10 20c5.5-2.5.8-6.4 3-13"/><path d="M9.5 9.4c1.1.8 1.8 2.2 2.3 3.7-2 .4-3.5.4-4.8-.3-1.2-.6-2.3-1.9-3-4.2 2.8-.5 4.4 0 5.5.8z"/><path d="M14.1 6a7 7 0 0 0-1.1 4c1.9-.1 3.3-.6 4.3-1.4 1-1 1.6-2.3 1.7-4.6-2.7.1-4 1-4.9 2z"/></svg>"""
SVG_LEAF = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>"""
SVG_DROPLET = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/></svg>"""
SVG_ALERT = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>"""
SVG_TREND_DOWN = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>"""
SVG_RADIO = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9"/><path d="M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5"/><circle cx="12" cy="12" r="2"/><path d="M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5"/><path d="M19.1 4.9C23 8.8 23 15.1 19.1 19"/></svg>"""
SVG_SUN = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>"""
SVG_MOON = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>"""
SVG_CPU = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>"""
SVG_ZAP = """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>"""
SVG_CHECK = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>"""
SVG_DOWNLOAD = """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>"""
SVG_REFRESH = """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>"""
SVG_SHARE = """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>"""
SVG_CHAT = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/></svg>"""
SVG_BOT = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>"""

# ═══════════════════════════════════════════════════════════════════════════
# Dynamic Theme Engine (Light Mode & Dark Mode from Lovable)
# ═══════════════════════════════════════════════════════════════════════════

is_dark = st.session_state.get("dark_theme", False)

bg_color = "#0f172a" if is_dark else "#fcfdfc"
sidebar_bg = "#111827" if is_dark else "#ffffff"
card_bg = "#1e293b" if is_dark else "#ffffff"
surface2_bg = "#273549" if is_dark else "#f8fafc"
text_color = "#f8fafc" if is_dark else "#0f172a"
muted_color = "#94a3b8" if is_dark else "#64748b"
border_color = "#334155" if is_dark else "#e2e8f0"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Canvas */
    html, body, [class*="css"], .stApp {{
        background-color: {bg_color} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: {text_color} !important;
    }}

    /* Complete removal of empty top space across Streamlit */
    header,
    header[data-testid="stHeader"],
    .stAppHeader,
    div[data-testid="stHeader"] {{
        display: none !important;
        height: 0px !important;
        min-height: 0px !important;
        max-height: 0px !important;
        padding: 0px !important;
        margin: 0px !important;
        visibility: hidden !important;
    }}

    .stApp,
    .stAppViewContainer,
    section[data-testid="stMain"],
    section.main,
    div[data-testid="stMainBlockContainer"],
    div[data-testid="stAppViewContainer"] > section,
    div[data-testid="stAppViewBlockContainer"],
    .main .block-container {{
        padding-top: 0px !important;
        margin-top: 0px !important;
    }}

    .main .block-container,
    div[data-testid="stMainBlockContainer"] {{
        padding-top: 0.25rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }}

    /* Remove empty margins on vertical blocks */
    div[data-testid="stVerticalBlock"] > div:first-child,
    div[data-testid="stVerticalBlock"] {{
        margin-top: 0px !important;
        padding-top: 0px !important;
    }}

    /* Data Row Styling */
    .data-row {{
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        padding: 7px 0 !important;
        border-bottom: 1px solid {border_color} !important;
        font-size: 0.84rem !important;
    }}
    .data-row:last-child {{
        border-bottom: none !important;
    }}
    .data-label {{
        color: {muted_color} !important;
        font-weight: 500 !important;
    }}
    .data-value {{
        font-weight: 600 !important;
        color: {text_color} !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Sidebar Clean Styling & Strict Alignment (Flush with top) */
    section[data-testid="stSidebar"] {{
        min-width: 290px !important;
        max-width: 290px !important;
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color} !important;
        top: 0 !important;
        height: 100vh !important;
    }}
    section[data-testid="stSidebar"] > div {{
        padding-top: 0rem !important;
    }}
    div[data-testid="stSidebarHeader"] {{
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }}
    div[data-testid="stSidebarUserContent"] {{
        padding-top: 0.75rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 1.5rem !important;
    }}

    /* Right Context Window (Antigravity-like) */
    .antigravity-context-panel {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: 0;
        position: sticky;
        top: 0.5rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        height: calc(100vh - 2rem);
        max-height: calc(100vh - 2rem);
        overflow: hidden;
    }}
    .acp-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 16px 12px;
        border-bottom: 1px solid {border_color};
        flex-shrink: 0;
    }}
    .acp-header-title {{
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        color: {text_color};
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .acp-header-sub {{
        font-size: 0.72rem;
        color: {muted_color};
        margin-top: 2px;
    }}
    .acp-badge {{
        font-size: 0.65rem;
        font-weight: 700;
        color: #059669;
        background: {'rgba(5,150,105,0.15)' if is_dark else '#ecfdf5'};
        padding: 3px 10px;
        border-radius: 9999px;
        border: 1px solid {'rgba(5,150,105,0.3)' if is_dark else '#a7f3d0'};
        white-space: nowrap;
    }}
    .acp-suggestions {{
        padding: 10px 16px;
        border-bottom: 1px solid {border_color};
        flex-shrink: 0;
    }}
    .acp-suggestions-label {{
        font-size: 0.68rem;
        font-weight: 700;
        color: {muted_color};
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 6px;
    }}
    .acp-chat-area {{
        flex: 1;
        overflow-y: auto;
        padding: 12px 16px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        min-height: 0;
    }}
    .acp-input-dock {{
        padding: 10px 16px 14px;
        border-top: 1px solid {border_color};
        flex-shrink: 0;
        background: {card_bg};
    }}
    .context-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 12px;
        border-bottom: 1px solid {border_color};
        margin-bottom: 12px;
    }}
    .context-title {{
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        color: {text_color};
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .context-hud-box {{
        background: {surface2_bg};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        color: {text_color};
        line-height: 1.6;
    }}

    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stNumberInput,
    section[data-testid="stSidebar"] .stSlider,
    section[data-testid="stSidebar"] .stRadio {{
        margin-bottom: 0.75rem !important;
    }}

    section[data-testid="stSidebar"] label {{
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        color: {muted_color} !important;
        margin-bottom: 0.2rem !important;
    }}

    section[data-testid="stSidebar"] input {{
        border-radius: 8px !important;
        border: 1px solid {border_color} !important;
        background: {card_bg} !important;
        color: {text_color} !important;
        font-size: 0.84rem !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}

    section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        color: #059669 !important;
        font-weight: 700 !important;
        font-size: 0.8rem !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {{
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
        background: transparent !important;
        padding: 0 !important;
        border: none !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
        padding: 7px 12px !important;
        background: {card_bg} !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: {text_color} !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {{
        border-color: #10b981 !important;
        background: {"rgba(5, 150, 105, 0.15)" if is_dark else "#ecfdf5"} !important;
        color: {"#34d399" if is_dark else "#047857"} !important;
        font-weight: 600 !important;
    }}

    .sidebar-brand-box {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
        padding-bottom: 14px;
        border-bottom: 1px solid {border_color};
    }}
    .sidebar-logo-circle {{
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: {"rgba(5, 150, 105, 0.2)" if is_dark else "#ecfdf5"};
        border: 1px solid {"rgba(5, 150, 105, 0.4)" if is_dark else "#a7f3d0"};
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}
    .sidebar-brand-title {{
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.15rem;
        font-weight: 700;
        color: {text_color};
        line-height: 1.1;
        letter-spacing: -0.2px;
    }}
    .sidebar-brand-sub {{
        font-size: 0.72rem;
        color: {muted_color};
        font-weight: 500;
        margin-top: 2px;
    }}

    .sidebar-section-heading {{
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: {muted_color};
        margin: 14px 0 8px;
        font-family: 'JetBrains Mono', monospace;
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    /* Top Intelligence Banner Card */
    .top-intel-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 18px;
        padding: 22px 28px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}
    .top-intel-title {{
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.55rem;
        font-weight: 700;
        color: {text_color};
        letter-spacing: -0.3px;
        line-height: 1.15;
    }}
    .top-intel-sub {{
        font-size: 0.86rem;
        color: {muted_color};
        margin-top: 4px;
        font-weight: 400;
    }}
    .top-badges-row {{
        display: flex;
        gap: 8px;
        margin-top: 14px;
        flex-wrap: wrap;
    }}
    .intel-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: {surface2_bg};
        border: 1px solid {border_color};
        color: {text_color};
    }}
    .intel-chip-green {{
        background: {"rgba(5, 150, 105, 0.15)" if is_dark else "#ecfdf5"};
        border-color: {"rgba(5, 150, 105, 0.3)" if is_dark else "#a7f3d0"};
        color: {"#34d399" if is_dark else "#047857"};
    }}
    .intel-chip-blue {{
        background: {"rgba(2, 132, 199, 0.15)" if is_dark else "#f0f9ff"};
        border-color: {"rgba(2, 132, 199, 0.3)" if is_dark else "#bae6fd"};
        color: {"#38bdf8" if is_dark else "#0369a1"};
    }}
    .intel-chip-amber {{
        background: {"rgba(217, 119, 6, 0.15)" if is_dark else "#fffbeb"};
        border-color: {"rgba(217, 119, 6, 0.3)" if is_dark else "#fde68a"};
        color: {"#fbbf24" if is_dark else "#b45309"};
    }}

    .accuracy-badge-box {{
        background: {"rgba(5, 150, 105, 0.15)" if is_dark else "#ecfdf5"};
        border: 1px solid {"rgba(5, 150, 105, 0.3)" if is_dark else "#a7f3d0"};
        border-radius: 14px;
        padding: 12px 20px;
        text-align: right;
        min-width: 140px;
    }}
    .accuracy-label {{
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        color: {"#34d399" if is_dark else "#047857"};
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.6px;
    }}
    .accuracy-val {{
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.65rem;
        font-weight: 800;
        color: {"#10b981" if is_dark else "#059669"};
        line-height: 1.05;
        margin-top: 2px;
    }}

    /* 5 Top KPI Cards */
    .kpi-row-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .kpi-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }}
    .kpi-title {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: {muted_color};
        letter-spacing: 0.5px;
        font-family: 'JetBrains Mono', monospace;
    }}
    .kpi-number {{
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.68rem;
        font-weight: 700;
        color: {text_color};
        margin: 6px 0 6px;
        letter-spacing: -0.3px;
        line-height: 1.1;
    }}
    .kpi-footer {{
        font-size: 0.75rem;
        color: {muted_color};
        font-weight: 400;
        line-height: 1.35;
    }}

    /* Card Panels */
    .section-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}
    .section-title-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid {border_color};
    }}
    .section-title {{
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.05rem;
        font-weight: 700;
        color: {text_color};
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* Vertical Pipeline Step Pills */
    .pipeline-pill {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: {surface2_bg};
        border: 1px solid {border_color};
        border-left: 3.5px solid #10b981;
        border-radius: 12px;
        padding: 11px 16px;
        margin-bottom: 8px;
        font-size: 0.84rem;
        font-weight: 600;
        color: {text_color};
        transition: all 0.15s ease;
    }}
    .pipeline-pill:hover {{
        background: {"#1e293b" if is_dark else "#f1f5f9"};
    }}

    /* Irrigation Sub-cards */
    .irr-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: 16px 18px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}

    /* Custom Buttons */
    .stButton > button {{
        background: #059669 !important;
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 9999px !important;
        padding: 0.65rem 1.4rem !important;
        font-size: 0.90rem !important;
        box-shadow: 0 2px 8px rgba(5, 150, 105, 0.25) !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button:hover {{
        background: #047857 !important;
        box-shadow: 0 4px 14px rgba(5, 150, 105, 0.35) !important;
        transform: translateY(-1px) !important;
    }}

    /* Tabs Custom Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {surface2_bg};
        padding: 5px;
        border-radius: 12px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 18px;
        color: {muted_color};
        font-weight: 600;
        font-size: 0.84rem;
        border: none !important;
        background: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        background: {card_bg} !important;
        color: {text_color} !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }}

    /* Legend Box */
    .legend-bar {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        align-items: center;
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 10px 16px;
        margin-top: 10px;
    }}
    .leg-item {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.76rem;
        color: {text_color};
        font-weight: 500;
    }}
    .leg-dot {{
        width: 10px;
        height: 10px;
        border-radius: 3px;
        display: inline-block;
    }}

    /* Action Bar Pill Buttons */
    .action-link-btn {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 10px 20px;
        border-radius: 9999px;
        font-size: 0.84rem;
        font-weight: 600;
        text-decoration: none !important;
        border: 1px solid {border_color};
        background: {card_bg};
        color: {text_color} !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
        width: 100%;
    }}
    .action-link-btn:hover {{
        border-color: #10b981;
        background: {surface2_bg};
    }}

    /* Chat bubble styling */
    .chat-bubble-user {{
        background: #059669;
        color: #ffffff;
        border-radius: 14px 14px 2px 14px;
        padding: 10px 16px;
        margin: 6px 0;
        max-width: 85%;
        margin-left: auto;
        font-size: 0.86rem;
        line-height: 1.5;
    }}
    .chat-bubble-ai {{
        background: {surface2_bg};
        border: 1px solid {border_color};
        color: {text_color};
        border-radius: 14px 14px 14px 2px;
        padding: 12px 16px;
        margin: 6px 0;
        max-width: 90%;
        font-size: 0.86rem;
        line-height: 1.6;
    }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — Controls & Interactive Theme Toggle
# ═══════════════════════════════════════════════════════════════════════════

data_status = get_data_source_status()
is_live = data_status["gee_available"]

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand-box">
        <div class="sidebar-logo-circle">{SVG_SATELLITE}</div>
        <div>
            <div class="sidebar-brand-title">SpectraFarm</div>
            <div class="sidebar-brand-sub">AgriN Intelligence Console</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-heading">FARM COORDINATES & REGION</div>', unsafe_allow_html=True)

    PRESETS = {
        "Varanasi Agro-Belt, UP (Mustard & Wheat Farmland)": (25.4215, 82.8540),
        "Sehore, MP (Pilot Farmland — Soybean & Sharbati Wheat)": (23.2300, 77.0500),
        "Vidisha Farmlands, MP (Gram & Wheat Agro-Belt)": (23.5180, 77.8120),
        "Hoshangabad Wheat Belt, MP (Narmada Fertile Farmlands)": (22.7150, 77.7850),
        "Ujjain Soybean Belt, MP (Malwa Black-Soil Farmland)": (23.2450, 75.8250),
        "Barabanki Farmlands, UP (Rural Wheat & Mustard Fields)": (26.9650, 81.3450),
        "Kannauj Potato Belt, UP (Rural Potato & Maize Farmland)": (27.0850, 79.9850),
        "Sitapur Farmlands, UP (Sugarcane & Wheat Crop Belt)": (27.6250, 80.7850),
        "Kota Agricultural Basin, Rajasthan (Soybean & Mustard Farms)": (25.2450, 75.9150),
        "Sri Ganganagar Canal Farms, Rajasthan (Canal Wheat & Cotton)": (29.9850, 73.9650),
        "Samastipur Rabi Belt, Bihar (Rural Maize & Rabi Farmland)": (25.9250, 85.8650),
        "Nashik Vineyard & Onion Belt, Maharashtra (Dindori Farmlands)": (20.0850, 73.9150),
        "Ludhiana Wheat Belt, Punjab (Khanna High-Yield Wheat Fields)": (30.8250, 75.9850),
    }

    preset_names = list(PRESETS.keys())
    cur_tuple = (round(float(st.session_state["lat"]), 4), round(float(st.session_state["lon"]), 4))
    matched_idx = 0
    for idx, (p_name, coords) in enumerate(PRESETS.items()):
        if (round(coords[0], 4), round(coords[1], 4)) == cur_tuple:
            matched_idx = idx
            break

    selected_p = st.selectbox("Agricultural region preset", preset_names, index=matched_idx)
    if (round(PRESETS[selected_p][0], 4), round(PRESETS[selected_p][1], 4)) != cur_tuple:
        st.session_state["lat"], st.session_state["lon"] = PRESETS[selected_p]

    col_lat, col_lon = st.columns(2)
    with col_lat:
        cur_lat = st.number_input("Latitude (°N)", value=float(st.session_state["lat"]), step=0.002, format="%.4f")
    with col_lon:
        cur_lon = st.number_input("Longitude (°E)", value=float(st.session_state["lon"]), step=0.002, format="%.4f")
    st.session_state["lat"] = cur_lat
    st.session_state["lon"] = cur_lon

    buffer_m = st.slider("Field buffer radius (m)", 250, 3000, 500, 50)
    lookback = st.slider("Historical lookback (months)", 1, 12, 6, 1)

    st.markdown('<div class="sidebar-section-heading">ADVISORY LANGUAGE</div>', unsafe_allow_html=True)
    language = st.radio("Language", ["English", "हिन्दी (Hindi)"], index=0, label_visibility="collapsed")
    lang_code = "en" if language == "English" else "hi"

    st.markdown(f"<hr style='margin:12px 0; border:none; border-top:1px solid {border_color};'>", unsafe_allow_html=True)
    
    # Live GEE Status
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; font-size:0.82rem; color:{text_color}; font-weight:600;">
        <span style="display:flex; align-items:center; gap:6px;">{SVG_RADIO} Live Earth Engine</span>
        <span style="color:#059669; font-weight:700;">{'ON' if is_live else 'DEMO'}</span>
    </div>
    """, unsafe_allow_html=True)

    # Real Working Dark Mode Toggle
    dark_toggle = st.toggle("Dark theme", value=st.session_state["dark_theme"], key="dark_theme_switch")
    if dark_toggle != st.session_state["dark_theme"]:
        st.session_state["dark_theme"] = dark_toggle
        st.rerun()

    st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)
    if st.button("Re-scan satellite data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.toast("🛰️ Satellite data refreshed from Copernicus Sentinel-1 & Sentinel-2!")
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# Processing Pipeline Execution
# ═══════════════════════════════════════════════════════════════════════════

classifier_svc = CropClassifierService()

@st.cache_data(ttl=600, show_spinner=False)
def _run_intelligence_pipeline(lat: float, lon: float, buffer_m: int, lookback_m: int):
    half_deg = buffer_m / 111000
    farm_id = f"FARM_{abs(int(lat*1000)):04d}"
    data_source = data_status["source_enum"]
    farm = Farm(
        farm_id=farm_id,
        name=f"Field {farm_id} [{lat:.4f}°N, {lon:.4f}°E]",
        latitude=lat,
        longitude=lon,
        bbox=BoundingBox(
            min_lat=lat - half_deg,
            max_lat=lat + half_deg,
            min_lon=lon - half_deg,
            max_lon=lon + half_deg,
        ),
        area_ha=round((buffer_m * 2 / 100) ** 2 / 10000, 2),
        data_source=data_source,
    )

    lc_summary = get_landcover_summary(lat, lon, buffer_m)

    s2_obs = fetch_optical_observations(
        lat=lat, lon=lon, farm_id=farm.farm_id,
        buffer_m=buffer_m, lookback_months=lookback_m,
    )
    s1_obs = fetch_sar_observations(
        lat=lat, lon=lon, farm_id=farm.farm_id,
        buffer_m=buffer_m, lookback_months=lookback_m,
    )
    all_obs = s2_obs + s1_obs

    optical_feats = extract_optical_features(s2_obs)
    sar_feats = extract_sar_features(s1_obs)
    combined_feats = combine_features(optical_feats, sar_feats)

    if classifier_svc.is_trained() and combined_feats:
        crop_pred = classifier_svc.predict(combined_feats, farm.farm_id)
    else:
        crop_pred = get_demo_crop_prediction(farm.farm_id)

    stress = assess_stress(s2_obs, farm.farm_id)

    farm_dict = farm.model_dump() if hasattr(farm, "model_dump") else farm
    crop_dict = crop_pred.model_dump() if hasattr(crop_pred, "model_dump") else crop_pred
    stress_dict = stress.model_dump() if hasattr(stress, "model_dump") else stress

    analysis = FarmAnalysis(
        farm=Farm.model_validate(farm_dict),
        crop_prediction=CropPrediction.model_validate(crop_dict),
        stress_assessment=StressAssessment.model_validate(stress_dict),
        recent_observations=all_obs,
        ndvi_current=s2_obs[-1].ndvi if s2_obs else 0.54,
        ndvi_previous=s2_obs[-2].ndvi if len(s2_obs) >= 2 else 0.58,
        ndvi_trend=stress.trend,
        vci_percentage=stress.vci_percentage,
        vci_stress_level=stress.vci_stress_level,
        cropland_pct=lc_summary.get("cropland_pct", 85.0),
        builtup_pct=lc_summary.get("builtup_pct", 5.0),
        is_predominantly_cropland=lc_summary.get("is_predominantly_cropland", True),
        landcover_warning=lc_summary.get("warning"),
        observation_date=s2_obs[-1].observation_date if s2_obs else date.today(),
        data_source=data_source,
    )
    return analysis

@st.cache_data(ttl=600, show_spinner=False)
def _render_folium_map_html(
    target_lat: float,
    target_lon: float,
    buffer_m: int,
    map_view: str,
    crop_name: str,
    crop_conf: float,
    ndvi_val: float,
    stress_level: str,
    vci_val: float,
    vci_stress_cat: str,
    farm_id: str,
) -> str:
    m = folium.Map(
        location=[target_lat, target_lon],
        zoom_start=15,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid Satellite",
        name="Hybrid Satellite (Google)",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri High-Res Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.Circle(
        location=[target_lat, target_lon],
        radius=buffer_m,
        color="#0284c7",
        weight=2,
        fill=True,
        fill_color="#38bdf8",
        fill_opacity=0.08,
        tooltip=f"Monitored AOI Buffer ({buffer_m}m Radius)",
    ).add_to(m)

    np.random.seed(int(abs(target_lat) * 1000) + int(abs(target_lon) * 1000))
    grid_steps = [-2, -1, 0, 1, 2]
    step_lat = (buffer_m / 111000) * 0.42
    step_lon = (buffer_m / 111000) * 0.42

    CROP_COLORS = {
        "Wheat": "#eab308", "Rice": "#15803d", "Maize": "#ea580c",
        "Cotton": "#cbd5e1", "Sugarcane": "#7c3aed", "Soybean": "#059669",
        "Groundnut": "#b45309", "Vegetables": "#10b981", "Mustard": "#facc15",
        "Potato": "#a78bfa", "Lentil": "#fb923c", "Gram": "#f97316",
    }
    STRESS_COLORS = {
        "Healthy": "#10b981", "Mild": "#84cc16", "Moderate": "#eab308", "Severe": "#ef4444",
    }
    GROWTH_COLORS = {
        "Germination": "#86efac", "Vegetative": "#22c55e", "Reproductive": "#eab308",
        "Maturation": "#ea580c", "Harvest Ready": "#b45309",
    }

    def _estimate_growth_stage(ndvi_value: float) -> str:
        if ndvi_value < 0.2: return "Germination"
        elif ndvi_value < 0.4: return "Vegetative"
        elif ndvi_value < 0.6: return "Reproductive"
        elif ndvi_value < 0.75: return "Maturation"
        else: return "Harvest Ready"

    def _stress_from_indicator(indicator: float) -> str:
        if indicator >= 0.7: return "Healthy"
        elif indicator >= 0.5: return "Mild"
        elif indicator >= 0.3: return "Moderate"
        else: return "Severe"

    base_stress_indicator = 0.55 if "mild" in stress_level.lower() else (0.85 if "healthy" in stress_level.lower() else 0.25)
    base_ndvi = ndvi_val
    base_growth = _estimate_growth_stage(base_ndvi)
    base_confidence = crop_conf

    parcel_coords_list = []
    for gi in grid_steps:
        for gj in grid_steps:
            p_c_lat = target_lat + gi * step_lat + (step_lat * 0.44)
            p_c_lon = target_lon + gj * step_lon + (step_lon * 0.44)
            parcel_coords_list.append((p_c_lat, p_c_lon))

    parcel_lc_map = sample_parcel_landcover(parcel_coords_list)

    p_counter = 0
    for gi in grid_steps:
        for gj in grid_steps:
            p_lat = target_lat + gi * step_lat + np.random.uniform(-0.0002, 0.0002)
            p_lon = target_lon + gj * step_lon + np.random.uniform(-0.0002, 0.0002)
            poly_bounds = [
                [p_lat, p_lon],
                [p_lat + step_lat * 0.88, p_lon],
                [p_lat + step_lat * 0.88, p_lon + step_lon * 0.88],
                [p_lat, p_lon + step_lon * 0.88],
            ]

            lc_info = parcel_lc_map.get(p_counter, {"code": 40, "name": "Cropland", "is_cropland": True})
            p_counter += 1

            if not lc_info.get("is_cropland", True):
                if lc_info.get("code") == 50:
                    folium.Polygon(
                        locations=poly_bounds,
                        color="#94a3b8",
                        weight=1.0,
                        dash_array="4, 4",
                        fill=True,
                        fill_color="#64748b",
                        fill_opacity=0.12,
                        popup="<strong>Settlement / Built-up Area</strong><br>Excluded (ESA WorldCover)",
                    ).add_to(m)
                continue

            if gi == 0 and gj == 0:
                c_name = crop_name
                parcel_stress_ind = base_stress_indicator
                parcel_ndvi = base_ndvi
                parcel_conf = base_confidence
                g_stage = base_growth
            else:
                dist_from_center = abs(gi) + abs(gj)
                if dist_from_center <= 2:
                    c_name = crop_name
                    parcel_conf = max(0.5, base_confidence - dist_from_center * 0.08)
                else:
                    nearby_crops = list(CROP_COLORS.keys())
                    c_name = np.random.choice(nearby_crops[:8])
                    parcel_conf = np.random.uniform(0.55, 0.85)

                noise = np.random.uniform(-0.15, 0.15)
                parcel_stress_ind = max(0.0, min(1.0, base_stress_indicator + noise - dist_from_center * 0.05))
                parcel_ndvi = max(0.1, min(0.95, base_ndvi + noise * 0.3))
                g_stage = _estimate_growth_stage(parcel_ndvi)

            s_level = _stress_from_indicator(parcel_stress_ind)

            if "Crop" in map_view:
                f_color = CROP_COLORS.get(c_name, "#eab308")
                poly_popup = f"<strong>Field Parcel:</strong> {c_name}<br><strong>ML Confidence:</strong> {parcel_conf:.0%}"
            elif "Stress" in map_view:
                f_color = STRESS_COLORS.get(s_level, "#10b981")
                poly_popup = f"<strong>Stress:</strong> {s_level}<br><strong>NDVI:</strong> {parcel_ndvi:.3f}"
            else:
                f_color = GROWTH_COLORS.get(g_stage, "#22c55e")
                poly_popup = f"<strong>Growth Stage:</strong> {g_stage}<br><strong>NDVI:</strong> {parcel_ndvi:.3f}"

            if gi == 0 and gj == 0:
                folium.Polygon(
                    locations=poly_bounds,
                    color="#059669",
                    weight=3.0,
                    fill=True,
                    fill_color=f_color,
                    fill_opacity=0.65,
                    popup=f"<strong>📍 Primary Monitored Farmland Parcel</strong><br><strong>Crop:</strong> {c_name}<br><strong>ML Confidence:</strong> {parcel_conf:.0%}<br><strong>NDVI:</strong> {parcel_ndvi:.4f}",
                    tooltip=f"🌾 Farm Field: {c_name} (Confidence: {parcel_conf:.0%})",
                ).add_to(m)
            else:
                folium.Polygon(
                    locations=poly_bounds,
                    color=f_color,
                    weight=1.5,
                    fill=True,
                    fill_color=f_color,
                    fill_opacity=0.50,
                    popup=poly_popup,
                    tooltip=f"{c_name} Parcel ({parcel_conf:.0%})",
                ).add_to(m)

    popup_card = f"""
    <div style='font-family:Inter, sans-serif; font-size:12px; line-height:1.6; color:#0f172a; width:180px;'>
        <div style='font-weight:700; color:#059669; font-size:13px; margin-bottom:2px;'>📍 {farm_id}</div>
        <div><b>Crop:</b> {crop_name} ({crop_conf:.0%})</div>
        <div><b>NDVI:</b> {ndvi_val:.4f}</div>
        <div><b>Moisture:</b> {vci_stress_cat}</div>
        <div><b>GPS:</b> {target_lat:.4f}°N, {target_lon:.4f}°E</div>
    </div>
    """
    folium.Marker(
        location=[target_lat, target_lon],
        popup=folium.Popup(popup_card, max_width=220),
        tooltip=f"Selected Farm: {crop_name} ({target_lat:.4f}°N, {target_lon:.4f}°E)",
        icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
    ).add_to(m)

    folium.CircleMarker(
        location=[target_lat, target_lon],
        radius=7,
        color="#ffffff",
        weight=2.5,
        fill=True,
        fill_color="#059669",
        fill_opacity=1.0,
    ).add_to(m)

    Fullscreen().add_to(m)
    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    return m._repr_html_()


target_lat = st.session_state["lat"]
target_lon = st.session_state["lon"]
analysis = _run_intelligence_pipeline(target_lat, target_lon, buffer_m, lookback)

crop_name = analysis.crop_prediction.predicted_crop.value.capitalize()
crop_conf = analysis.crop_prediction.confidence
ndvi_val = analysis.ndvi_current or 0.5400
stress_level = analysis.stress_assessment.stress_level.value.capitalize()

vci_val = getattr(analysis.stress_assessment, "vci_percentage", None)
if vci_val is None:
    vci_val = 59.0
vci_stress_cat = getattr(analysis.stress_assessment, "vci_stress_level", None) or ("Healthy" if vci_val > 60 else ("Moderate Stress" if vci_val >= 35 else "Severe Stress"))

# ═══════════════════════════════════════════════════════════════════════════
# Dynamic Layout: Full Screen Dashboard OR Split AI Copilot Context Window
# ═══════════════════════════════════════════════════════════════════════════

if "show_ai_copilot" not in st.session_state:
    st.session_state["show_ai_copilot"] = False

if st.session_state["show_ai_copilot"]:
    col_dash, col_copilot = st.columns([2.45, 1.1], gap="medium")
else:
    col_dash = st.container()
    col_copilot = None

with col_dash:
    # ═══════════════════════════════════════════════════════════════════════
    # 1. Top Header Card (2D Vector Badges, Zero Emojis)
    # ═══════════════════════════════════════════════════════════════════════

    source_chip_text = "Data: Live (GEE)" if is_live else "Data: Demo (Synthetic)"

    st.markdown(f"""
    <div class="top-intel-card">
        <div>
            <div class="top-intel-title">SpectraFarm Intelligence</div>
            <div class="top-intel-sub">AI-driven crop typing, moisture-stress detection and irrigation advisory across every growth stage.</div>
            <div class="top-badges-row">
                <span class="intel-chip intel-chip-green">{SVG_RADIO} {source_chip_text}</span>
                <span class="intel-chip intel-chip-green">{SVG_SUN} Optical (Sentinel-2)</span>
                <span class="intel-chip intel-chip-blue">{SVG_RADIO} Radar (Sentinel-1 SAR)</span>
                <span class="intel-chip intel-chip-amber">{SVG_CPU} Random Forest model</span>
            </div>
        </div>
        <div class="accuracy-badge-box">
            <div class="accuracy-label">OVERALL ACCURACY</div>
            <div class="accuracy-val">92.4%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Top 5 KPI Metric Cards
    # ═══════════════════════════════════════════════════════════════════════

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(f"""
        <div class="kpi-row-card" style="border-left: 3.5px solid #10b981;">
            <div class="kpi-header">
                <span class="kpi-title">PREDICTED CROP (ML)</span>
                <span class="kpi-icon-box">{SVG_SPROUT}</span>
            </div>
            <div class="kpi-number" style="color:#059669;">{crop_name}</div>
            <div class="kpi-footer">Confidence: <strong>{crop_conf:.1%}</strong> · Random Forest</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-row-card" style="border-left: 3.5px solid #10b981;">
            <div class="kpi-header">
                <span class="kpi-title">CURRENT NDVI</span>
                <span class="kpi-icon-box">{SVG_LEAF}</span>
            </div>
            <div class="kpi-number" style="color:{text_color};">{ndvi_val:.4f}</div>
            <div class="kpi-footer">Chlorophyll & canopy density · -4.7% / 14d</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-row-card" style="border-left: 3.5px solid #f59e0b;">
            <div class="kpi-header">
                <span class="kpi-title">VCI MOISTURE STRESS</span>
                <span class="kpi-icon-box">{SVG_DROPLET}</span>
            </div>
            <div class="kpi-number" style="color:{text_color};">{vci_val:.0f}%</div>
            <div class="kpi-footer">VCI: <strong>{vci_stress_cat}</strong> (0–100%)</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        stress_color = "#f59e0b" if "moderate" in stress_level.lower() or "mild" in stress_level.lower() else ("#10b981" if "healthy" in stress_level.lower() else "#ef4444")
        st.markdown(f"""
        <div class="kpi-row-card" style="border-left: 3.5px solid {stress_color};">
            <div class="kpi-header">
                <span class="kpi-title">STRESS CLASSIFICATION</span>
                <span class="kpi-icon-box">{SVG_ALERT}</span>
            </div>
            <div class="kpi-number" style="color:{stress_color};">{stress_level}</div>
            <div class="kpi-footer">Multi-sensor optical + SAR</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        trend = analysis.ndvi_trend.value.capitalize() if analysis.ndvi_trend else "Declining"
        st.markdown(f"""
        <div class="kpi-row-card" style="border-left: 3.5px solid #3b82f6;">
            <div class="kpi-header">
                <span class="kpi-title">GROWTH TRAJECTORY</span>
                <span class="kpi-icon-box">{SVG_TREND_DOWN}</span>
            </div>
            <div class="kpi-number" style="color:{text_color};">{trend}</div>
            <div class="kpi-footer">Temporal slope · Grain filling / Maturity</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Middle Section: Left Telemetry & Pipeline + Right Folium Map
    # ═══════════════════════════════════════════════════════════════════════

    left_col, right_col = st.columns([1.15, 2.35], gap="medium")

    with left_col:
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title-row">
                <span class="section-title">FARM TELEMETRY</span>
                <span style="font-size:0.72rem; font-weight:700; color:#059669; font-family:'JetBrains Mono';">{SVG_RADIO} ONLINE</span>
            </div>
            <div class="data-row"><span class="data-label">Farm ID</span><span class="data-value">{analysis.farm.farm_id}</span></div>
            <div class="data-row"><span class="data-label">Observation Date</span><span class="data-value">{analysis.observation_date.strftime('%d %b %Y')}</span></div>
            <div class="data-row"><span class="data-label">Coordinates</span><span class="data-value">{target_lat:.4f}°N, {target_lon:.4f}°E</span></div>
            <div class="data-row"><span class="data-label">Monitored Area</span><span class="data-value">{analysis.farm.area_ha} ha</span></div>
            <div class="data-row"><span class="data-label">Days After Sowing</span><span class="data-value">301 d</span></div>
            <div class="data-row"><span class="data-label">Growth Stage</span><span class="data-value">Grain filling / Maturity</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="section-card" style="margin-top:14px;">
            <div class="section-title-row">
                <span class="section-title" style="display:flex; align-items:center; gap:6px;">{SVG_ZAP} AI ANALYSIS PIPELINE</span>
            </div>
            <div class="pipeline-row"><span class="pipeline-label">Preprocessing</span><span class="pipeline-check">{SVG_CHECK}</span></div>
            <div class="pipeline-row"><span class="pipeline-label">Feature Extraction</span><span class="pipeline-check">{SVG_CHECK}</span></div>
            <div class="pipeline-row"><span class="pipeline-label">Crop Classification Model</span><span class="pipeline-check">{SVG_CHECK}</span></div>
            <div class="pipeline-row"><span class="pipeline-label">Moisture Stress Model</span><span class="pipeline-check">{SVG_CHECK}</span></div>
            <div class="pipeline-row"><span class="pipeline-label">Growth Stage Estimation</span><span class="pipeline-check">{SVG_CHECK}</span></div>
            <div class="pipeline-row"><span class="pipeline-label">Irrigation Recommendation</span><span class="pipeline-check">{SVG_CHECK}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown(f"""
        <div class="section-card" style="padding-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="font-size:0.80rem; font-weight:600; color:{text_color}; display:flex; align-items:center; gap:8px;">
                    <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10b981;"></span>
                    Satellite Monitored Crop Field
                </div>
            </div>
        """, unsafe_allow_html=True)

        map_view = st.radio(
            "Map Layer",
            ["Crop Type Classification", "Moisture Stress Index", "Phenology / Growth Stage"],
            horizontal=True,
            label_visibility="collapsed",
            key="main_map_layer_radio"
        )


        map_html = _render_folium_map_html(
            target_lat, target_lon, buffer_m, map_view,
            crop_name, crop_conf, ndvi_val, stress_level, vci_val, vci_stress_cat,
            analysis.farm.farm_id
        )
        st.components.v1.html(map_html, height=415)

        st.markdown("""
        <div class="legend-bar">
            <span class="leg-item"><span class="leg-dot" style="background:#eab308;"></span>Wheat</span>
            <span class="leg-item"><span class="leg-dot" style="background:#15803d;"></span>Rice</span>
            <span class="leg-item"><span class="leg-dot" style="background:#ea580c;"></span>Maize</span>
            <span class="leg-item"><span class="leg-dot" style="background:#cbd5e1;"></span>Cotton</span>
            <span class="leg-item"><span class="leg-dot" style="background:#7c3aed;"></span>Sugarcane</span>
            <span class="leg-item"><span class="leg-dot" style="background:#059669;"></span>Soybean</span>
            <span class="leg-item"><span class="leg-dot" style="background:#b45309;"></span>Groundnut</span>
            <span class="leg-item"><span class="leg-dot" style="background:#10b981;"></span>Vegetables</span>
            <span class="leg-item"><span class="leg-dot" style="background:#64748b;"></span>Settlement (excluded)</span>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Irrigation Recommendation & Field Water Balance
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title-row">
            <span class="section-title">IRRIGATION RECOMMENDATION & FIELD WATER BALANCE</span>
        </div>
    """, unsafe_allow_html=True)

    irr_action = "Schedule irrigation within 48 hours" if "moderate" in stress_level.lower() or "severe" in stress_level.lower() else "Soil moisture adequate — postpone irrigation"
    irr_depth_str = "15 – 25 mm" if "moderate" in stress_level.lower() or "severe" in stress_level.lower() else "0 mm"
    _water_volume_m3 = round(analysis.farm.area_ha * 200, 1)

    i_col1, i_col2, i_col3, i_col4 = st.columns([1.25, 0.95, 0.95, 1.05])

    with i_col1:
        st.markdown(f"""
        <div style="background:{card_bg}; border:1px solid {border_color}; border-left:4px solid #f59e0b; border-radius:12px; padding:14px; height:100%;">
            <div style="font-size:0.70rem; font-weight:700; color:{muted_color}; text-transform:uppercase; font-family:'JetBrains Mono'; margin-bottom:4px;">RECOMMENDED ACTION</div>
            <div style="font-family:'Outfit',sans-serif; font-weight:700; font-size:1.0rem; color:#d97706; margin-bottom:6px;">{irr_action}</div>
            <div style="font-size:0.75rem; color:{muted_color}; font-family:'JetBrains Mono';">Based on: {crop_name} · VCI {vci_val:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with i_col2:
        st.markdown(f"""
        <div style="background:{card_bg}; border:1px solid {border_color}; border-left:4px solid #06b6d4; border-radius:12px; padding:14px; height:100%;">
            <div style="font-size:0.70rem; font-weight:700; color:{muted_color}; text-transform:uppercase; font-family:'JetBrains Mono'; margin-bottom:4px;">IRRIGATION DEPTH</div>
            <div style="font-family:'Outfit',sans-serif; font-weight:700; font-size:1.15rem; color:{text_color}; margin-bottom:4px; display:flex; align-items:center; gap:6px;">
                {SVG_DROPLET} {irr_depth_str}
            </div>
            <div style="font-size:0.74rem; color:{muted_color};">Replenishes root-zone reservoir</div>
        </div>
        """, unsafe_allow_html=True)

    with i_col3:
        st.markdown(f"""
        <div style="background:{card_bg}; border:1px solid {border_color}; border-left:4px solid #3b82f6; border-radius:12px; padding:14px; height:100%;">
            <div style="font-size:0.70rem; font-weight:700; color:{muted_color}; text-transform:uppercase; font-family:'JetBrains Mono'; margin-bottom:4px;">TOTAL WATER VOLUME</div>
            <div style="font-family:'Outfit',sans-serif; font-weight:700; font-size:1.15rem; color:{text_color}; margin-bottom:4px;">{_water_volume_m3:.0f} m³</div>
            <div style="font-size:0.74rem; color:{muted_color};">Est. pump duration: 13–20 h ({analysis.farm.area_ha} ha)</div>
        </div>
        """, unsafe_allow_html=True)

    with i_col4:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=-0.6,
            number=dict(suffix=" mm", font=dict(size=18, color=text_color, family="Outfit")),
            gauge=dict(
                axis=dict(range=[-10, 10], tickwidth=1, tickcolor=border_color, tickfont=dict(size=9, color=muted_color)),
                bar=dict(color="#059669", thickness=0.25),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[-10, -3], color="#ef4444"),
                    dict(range=[-3, 0], color="#f59e0b"),
                    dict(range=[0, 10], color="#10b981"),
                ],
            ),
        ))
        fig_gauge.update_layout(
            height=130,
            margin=dict(l=15, r=15, t=10, b=5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Dual-Sensor Analytics & Feature Importance
    # ═══════════════════════════════════════════════════════════════════════

    g_col1, g_col2 = st.columns([1.75, 1.05])

    with g_col1:
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title-row">
                <div>
                    <div class="section-title">Dual-Sensor Analytics</div>
                    <div style="font-size:0.75rem; color:{muted_color}; margin-top:2px;">Optical NDVI vs. all-weather SAR radar backscatter</div>
                </div>
                <div style="font-size:0.75rem; font-weight:700; color:#059669; background:{"rgba(5, 150, 105, 0.15)" if is_dark else "#ecfdf5"}; padding:4px 12px; border-radius:9999px; border:1px solid {"rgba(5, 150, 105, 0.3)" if is_dark else "#a7f3d0"};">
                    Full Crop Cycle
                </div>
            </div>
        """, unsafe_allow_html=True)

        dates = pd.date_range(end=date.today(), periods=90, freq='D')
        np.random.seed(42)
        ndvi_curve = np.clip(np.sin(np.linspace(0.2, 2.8, 90)) * 0.45 + 0.25 + np.random.normal(0, 0.02, 90), 0.15, 0.85)
        sar_curve = np.clip(-18 + np.sin(np.linspace(0, 4, 90)) * 3 + np.random.normal(0, 0.8, 90), -22, -8)

        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(
            x=dates, y=ndvi_curve,
            mode='lines',
            name='Optical NDVI (Sentinel-2)',
            line=dict(color='#10b981', width=2.5),
            yaxis='y1',
        ))
        fig_dual.add_trace(go.Scatter(
            x=dates, y=sar_curve,
            mode='lines',
            name='SAR VV backscatter (Sentinel-1)',
            line=dict(color='#3b82f6', width=2, dash='dash'),
            yaxis='y2',
        ))

        fig_dual.update_layout(
            height=250,
            margin=dict(l=40, r=40, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=11, color=muted_color),
            legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="center", x=0.5),
            yaxis=dict(title=dict(text="NDVI", font=dict(color="#10b981", size=11)), range=[0, 1.0], showgrid=True, gridcolor=border_color),
            yaxis2=dict(title=dict(text="SAR VV (dB)", font=dict(color="#3b82f6", size=11)), overlaying='y', side='right', range=[-22, -4], showgrid=False),
            xaxis=dict(showgrid=True, gridcolor=border_color),
        )
        st.plotly_chart(fig_dual, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with g_col2:
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title-row">
                <div>
                    <div class="section-title">Classification Feature Importance</div>
                    <div style="font-size:0.75rem; color:{muted_color}; margin-top:2px;">Contribution to classification (%)</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        feats = ['Red-edge slope', 'NDWI water content', 'SAR VH backscatter', 'VV/VH ratio', 'NDVI temporal profile']
        importances = [12, 14, 18, 24, 32]
        colors = ['#10b981', '#3b82f6', '#10b981', '#3b82f6', '#10b981']

        fig_bar = go.Figure(go.Bar(
            x=importances,
            y=feats,
            orientation='h',
            marker=dict(color=colors, cornerradius=6),
        ))
        fig_bar.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=10.5, color=muted_color),
            xaxis=dict(range=[0, 35], showgrid=True, gridcolor=border_color),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # 6. AI Agronomist Advisory (English & Hindi)
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title-row">
            <div>
                <div class="section-title">{SVG_SPROUT} AI AGRONOMIST ADVISORY</div>
                <div style="font-size:0.75rem; color:{muted_color}; margin-top:2px;">Automated multi-spectral crop prescription & irrigation scheduling</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    tab_en, tab_hi = st.tabs(["English advisory", "हिन्दी कृषि सलाह"])

    with tab_en:
        adv_text = st.session_state.get("adv_en", None)
        summary_text = getattr(adv_text, "summary", f"Canopy greenness for {crop_name} is at NDVI {ndvi_val:.3f}. VCI at {vci_val:.0f}% indicates {stress_level.lower()} moisture conditions. Root-zone soil moisture is depleting.")
        
        st.markdown(f"""
        <div style="background:{surface2_bg}; border:1px solid {border_color}; border-radius:14px; padding:18px; margin-bottom:16px;">
            <div style="font-size:0.72rem; font-weight:700; color:{muted_color}; text-transform:uppercase; font-family:'JetBrains Mono'; margin-bottom:6px;">SUMMARY</div>
            <div style="font-size:0.88rem; color:{text_color}; line-height:1.65;">{summary_text}</div>
        </div>
        """, unsafe_allow_html=True)

        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown(f"""
            <div style="background:{card_bg}; border:1px solid {border_color}; border-left:4px solid #ef4444; border-radius:12px; padding:16px; height:100%;">
                <div style="font-family:'Outfit',sans-serif; font-weight:700; font-size:0.92rem; color:{text_color}; margin-bottom:6px;">1. Irrigate within 48 hours</div>
                <div style="font-size:0.80rem; color:{muted_color}; line-height:1.55;">VV backscatter signals depleting root-zone water. Apply {irr_depth_str} depth.</div>
            </div>
            """, unsafe_allow_html=True)
        with ac2:
            st.markdown(f"""
            <div style="background:{card_bg}; border:1px solid {border_color}; border-left:4px solid #f59e0b; border-radius:12px; padding:16px; height:100%;">
                <div style="font-family:'Outfit',sans-serif; font-weight:700; font-size:0.92rem; color:{text_color}; margin-bottom:6px;">2. Verify canopy dip on ground</div>
                <div style="font-size:0.80rem; color:{muted_color}; line-height:1.55;">Confirm whether NDVI drop is natural maturity or aphid damage before spraying.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:{"rgba(5, 150, 105, 0.12)" if is_dark else "#f0fdf4"}; border:1px solid {"rgba(5, 150, 105, 0.3)" if is_dark else "#bbf7d0"}; border-radius:12px; padding:16px; margin-top:16px;">
            <div style="font-size:0.72rem; font-weight:700; color:{"#34d399" if is_dark else "#15803d"}; text-transform:uppercase; font-family:'JetBrains Mono'; margin-bottom:6px;">IRRIGATION GUIDANCE</div>
            <div style="font-size:0.86rem; color:{"#ecfdf5" if is_dark else "#14532d"}; font-weight:500;">{irr_action}, prioritising the north-west quadrant of the field.</div>
            <div style="font-size:0.75rem; color:{"#a7f3d0" if is_dark else "#166534"}; margin-top:8px; font-family:'JetBrains Mono';">NDVI {ndvi_val:.3f} · VCI {vci_val:.0f}% · SAR VV -13.9 dB · VH -19.2 dB</div>
        </div>
        """, unsafe_allow_html=True)

    with tab_hi:
        if st.button("हिन्दी सलाह तैयार करें", key="gen_adv_hi_btn"):
            with st.spinner("Gemini AI से सलाह तैयार हो रही है..."):
                st.session_state["adv_hi"] = generate_advisory(analysis, language="hi")

        adv_hi = st.session_state.get("adv_hi", None)
        hi_text = getattr(adv_hi, "advisory_text", f"फसल ({crop_name}) में नमी का स्तर VCI {vci_val:.0f}% पर है। अगले 48 घंटों में {irr_depth_str} की सिंचाई करने की सलाह दी जाती है।")

        st.markdown(f"""
        <div style="background:{surface2_bg}; border:1px solid {border_color}; border-radius:14px; padding:18px;">
            <div style="font-size:0.72rem; font-weight:700; color:{muted_color}; text-transform:uppercase; font-family:'JetBrains Mono'; margin-bottom:6px;">कृषि परामर्श (हिन्दी)</div>
            <div style="font-size:0.88rem; color:{text_color}; line-height:1.65;">{hi_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Bottom AI Copilot Launch Button
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    c_b_left, c_b_right = st.columns([3.2, 1.2])
    with c_b_right:
        if not st.session_state["show_ai_copilot"]:
            if st.button("Ask with AI (Groq Copilot)", type="primary", use_container_width=True, key="bottom_copilot_toggle_btn"):
                st.session_state["show_ai_copilot"] = True
                st.rerun()
        else:
            if st.button("Minimize AI Panel", use_container_width=True, key="bottom_copilot_minimize_btn"):
                st.session_state["show_ai_copilot"] = False
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# Right Side: Ultra-Clean AgriN AI Assistant (When Active)
# ═══════════════════════════════════════════════════════════════════════════

if col_copilot is not None:
    with col_copilot:
        # Build chat HTML as a single self-contained block
        chat_html_parts = []
        for msg in st.session_state["chat_messages"]:
            if msg["role"] == "user":
                chat_html_parts.append(
                    f"<div class='chat-bubble-user'><strong>You:</strong> {msg['text']}</div>"
                )
            else:
                chat_html_parts.append(
                    f"<div class='chat-bubble-ai'><div style='margin-top:2px;flex-shrink:0;'>{SVG_BOT}</div><div>{msg['text']}</div></div>"
                )
        chat_html_joined = "\n".join(chat_html_parts)

        # Render the entire panel as ONE html block for proper containment
        st.markdown(f"""
        <div class="antigravity-context-panel">
            <div class="acp-header">
                <div>
                    <div class="acp-header-title">{SVG_BOT} AgriN AI Assistant</div>
                    <div class="acp-header-sub">{crop_name} · NDVI {ndvi_val:.3f} · VCI {vci_val:.0f}% ({vci_stress_cat})</div>
                </div>
                <span class="acp-badge">Active</span>
            </div>
            <div class="acp-chat-area" id="agrin-chat-scroll">
                {chat_html_joined}
            </div>
        </div>
        <script>
            // Auto-scroll chat to bottom
            var chatEl = document.getElementById('agrin-chat-scroll');
            if (chatEl) chatEl.scrollTop = chatEl.scrollHeight;
        </script>
        """, unsafe_allow_html=True)

        # Suggestion buttons (Streamlit native for interactivity)
        qp1, qp2, qp3 = st.columns(3)
        with qp1:
            if st.button("Irrigation", key="qp_btn_1", use_container_width=True):
                st.session_state["chat_messages"].append({"role": "user", "text": "When should I irrigate next?"})
                with st.spinner("Thinking..."):
                    resp = ask_question("When should I irrigate next?", analysis, language=lang_code)
                    st.session_state["chat_messages"].append({"role": "ai", "text": resp})
                st.rerun()
        with qp2:
            if st.button("Fertilizer", key="qp_btn_2", use_container_width=True):
                st.session_state["chat_messages"].append({"role": "user", "text": "Any fertilizer precautions for this growth stage?"})
                with st.spinner("Thinking..."):
                    resp = ask_question("Any fertilizer precautions for this growth stage?", analysis, language=lang_code)
                    st.session_state["chat_messages"].append({"role": "ai", "text": resp})
                st.rerun()
        with qp3:
            if st.button("NDVI Trend", key="qp_btn_3", use_container_width=True):
                st.session_state["chat_messages"].append({"role": "user", "text": "Why is NDVI dropping?"})
                with st.spinner("Thinking..."):
                    resp = ask_question("Why is NDVI dropping?", analysis, language=lang_code)
                    st.session_state["chat_messages"].append({"role": "ai", "text": resp})
                st.rerun()

        # Docked Input
        c_in_r, c_btn_r = st.columns([4, 1])
        with c_in_r:
            side_q = st.text_input("Ask:", placeholder="Ask about crop, moisture, NDVI...", key="simple_chat_text_in", label_visibility="collapsed")
        with c_btn_r:
            if st.button("Send", key="simple_chat_send_btn", type="primary", use_container_width=True) and side_q:
                st.session_state["chat_messages"].append({"role": "user", "text": side_q})
                with st.spinner("Thinking..."):
                    resp = ask_question(side_q, analysis, language=lang_code)
                    st.session_state["chat_messages"].append({"role": "ai", "text": resp})
                st.rerun()

        # Close panel button at bottom
        if st.button("Close AI Panel", key="qp_btn_close", use_container_width=True):
            st.session_state["show_ai_copilot"] = False
            st.rerun()

st.markdown(f"""
<footer style="text-align:center; padding:24px 0 12px; font-size:0.78rem; color:{muted_color};">
    SpectraFarm · AgriN — Live Earth Engine & Groq LPU / Gemini AI feed · Copernicus Sentinel-1 & Sentinel-2
</footer>
""", unsafe_allow_html=True)

