"""
AgriN — Apple HIG Inspired Agricultural Intelligence Dashboard
Crafted with Apple Human Interface Guidelines: Precision, Clarity, Depth, and Elegance.
"""

import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.gemini_client import ask_question, generate_advisory, is_gemini_available
from src.config.settings import get_settings
from src.data.demo_data import (
    generate_demo_crop_map_data,
    generate_demo_stress_map_data,
    get_demo_farm,
)
from src.data.schemas import DataSource, HealthTrend, StressLevel
from src.intelligence.farm_analyzer import FarmAnalyzer

# ═══════════════════════════════════════════════════════════════════════════
# Page Configuration
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AgriN — Smart Crop Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# Apple Human Interface Guidelines (HIG) Styling
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Typography & Layout ── */
    html, body, [class*="css"], .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", "Helvetica Neue", sans-serif !important;
        background-color: #F5F5F7 !important;
        color: #1D1D1F !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Sidebar (Apple macOS Sidebar Aesthetic) ── */
    section[data-testid="stSidebar"] {
        background-color: rgba(245, 245, 247, 0.92) !important;
        backdrop-filter: blur(25px) saturate(180%);
        -webkit-backdrop-filter: blur(25px) saturate(180%);
        border-right: 1px solid rgba(0, 0, 0, 0.08) !important;
    }
    
    section[data-testid="stSidebar"] hr {
        border-color: rgba(0, 0, 0, 0.06) !important;
        margin: 1.2rem 0 !important;
    }

    /* ── Apple Glass Card System ── */
    .apple-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(20px) saturate(160%);
        -webkit-backdrop-filter: blur(20px) saturate(160%);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.03), 0 2px 6px -1px rgba(0, 0, 0, 0.02);
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .apple-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px -4px rgba(0, 0, 0, 0.07), 0 4px 12px -2px rgba(0, 0, 0, 0.03);
    }

    /* ── Header Banner (Apple Keynote Style) ── */
    .apple-hero {
        background: linear-gradient(135deg, #FFFFFF 0%, #F9F9FB 100%);
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 24px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px -2px rgba(0, 0, 0, 0.04);
        position: relative;
        overflow: hidden;
    }
    .apple-hero h1 {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        color: #1D1D1F;
        margin: 0;
        line-height: 1.15;
    }
    .apple-hero .subtitle {
        color: #86868B;
        font-size: 1rem;
        font-weight: 400;
        margin-top: 0.4rem;
        letter-spacing: -0.01em;
    }
    .apple-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-demo {
        background: rgba(255, 149, 0, 0.12);
        color: #D97706;
        border: 1px solid rgba(255, 149, 0, 0.25);
    }
    .badge-live {
        background: rgba(52, 199, 89, 0.12);
        color: #248A3D;
        border: 1px solid rgba(52, 199, 89, 0.25);
    }
    .badge-region {
        background: rgba(0, 122, 255, 0.1);
        color: #0071E3;
        border: 1px solid rgba(0, 122, 255, 0.2);
    }

    /* ── Apple Health Widget Style Metrics ── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-widget {
        background: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 20px;
        padding: 1.3rem 1.1rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s;
    }
    .metric-widget:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }
    .widget-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.6rem;
    }
    .widget-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #86868B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .widget-icon {
        width: 32px;
        height: 32px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    .widget-value {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #1D1D1F;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }
    .widget-caption {
        font-size: 0.75rem;
        color: #86868B;
        font-weight: 500;
    }

    /* ── Apple Color Accents ── */
    .bg-green-soft { background: rgba(52, 199, 89, 0.12); color: #34C759; }
    .bg-orange-soft { background: rgba(255, 149, 0, 0.12); color: #FF9500; }
    .bg-blue-soft { background: rgba(0, 122, 255, 0.12); color: #007AFF; }
    .bg-red-soft { background: rgba(255, 59, 48, 0.12); color: #FF3B30; }
    .bg-purple-soft { background: rgba(175, 82, 222, 0.12); color: #AF52DE; }

    .color-green { color: #34C759; }
    .color-orange { color: #FF9500; }
    .color-blue { color: #007AFF; }
    .color-red { color: #FF3B30; }

    /* ── Section Title ── */
    .apple-section-title {
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #1D1D1F;
        margin: 1.8rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Apple Intelligence Advisory Card ── */
    .intelligence-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(245, 247, 255, 0.95) 100%);
        border: 1px solid rgba(0, 122, 255, 0.15);
        border-radius: 24px;
        padding: 1.8rem 2rem;
        box-shadow: 0 4px 24px rgba(0, 122, 255, 0.04);
        position: relative;
    }
    .intelligence-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1rem;
    }
    .intelligence-pill {
        background: linear-gradient(135deg, #007AFF, #5856D6);
        color: #FFFFFF;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        text-transform: uppercase;
    }
    .intelligence-content {
        color: #1D1D1F;
        font-size: 0.95rem;
        line-height: 1.7;
        font-weight: 400;
        white-space: pre-line;
    }

    /* ── Q&A Card ── */
    .qa-box {
        background: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 20px;
        padding: 1.5rem 1.8rem;
        margin-top: 1rem;
        box-shadow: 0 2px 14px rgba(0, 0, 0, 0.03);
        color: #1D1D1F;
        font-size: 0.93rem;
        line-height: 1.7;
    }

    /* ── Apple Segmented Controls / Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #E5E5EA;
        padding: 4px;
        border-radius: 14px;
        gap: 4px;
        border: none;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #636366 !important;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.5rem 1.2rem;
        border: none !important;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #1D1D1F !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    }

    /* ── Quick Question Pills ── */
    .stButton > button {
        background: #FFFFFF !important;
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        color: #1D1D1F !important;
        border-radius: 999px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.4rem 1rem !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button:hover {
        border-color: #007AFF !important;
        color: #007AFF !important;
        background: rgba(0, 122, 255, 0.04) !important;
        transform: scale(1.02);
    }

    /* ── Input Styling ── */
    .stTextInput input {
        background: #FFFFFF !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        border-radius: 14px !important;
        color: #1D1D1F !important;
        padding: 0.8rem 1.1rem !important;
        font-size: 0.92rem !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02) inset !important;
    }
    .stTextInput input:focus {
        border-color: #007AFF !important;
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.15) !important;
    }
    
    /* Map Frame Smoothing */
    iframe {
        border-radius: 18px !important;
        border: 1px solid rgba(0, 0, 0, 0.06) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# State Management
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_analyzer():
    return FarmAnalyzer()


def get_analysis():
    if "analysis" not in st.session_state:
        analyzer = get_analyzer()
        st.session_state.analysis = analyzer.analyze()
    return st.session_state.analysis


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar (macOS Style Navigation & Settings)
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.75rem; padding:0.5rem 0 1rem 0;">
        <div style="font-size:1.8rem; background:rgba(52,199,89,0.15); width:44px; height:44px; border-radius:12px; display:flex; align-items:center; justify-content:center;">🌾</div>
        <div>
            <div style="font-size:1.25rem; font-weight:800; color:#1D1D1F; letter-spacing:-0.03em;">AgriN</div>
            <div style="font-size:0.75rem; color:#86868B; font-weight:500;">Crop Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    settings = get_settings()
    lang_options = {l["code"]: l["name"] for l in settings.languages}
    selected_lang = st.selectbox(
        "Language / भाषा",
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        key="language",
    )

    st.markdown("---")

    farm = get_demo_farm()
    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid rgba(0,0,0,0.06); border-radius:16px; padding:1rem; box-shadow:0 1px 4px rgba(0,0,0,0.02);">
        <div style="font-size:0.7rem; font-weight:700; color:#86868B; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.4rem;">Pilot Field</div>
        <div style="font-size:0.95rem; font-weight:700; color:#1D1D1F;">{farm.name}</div>
        <div style="font-size:0.8rem; color:#86868B; margin-top:0.4rem; line-height:1.6;">
            📍 {farm.latitude}°N, {farm.longitude}°E<br>
            📐 {farm.area_ha} Hectares<br>
            🌱 Season: {farm.season.capitalize() if farm.season else 'Rabi'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    gee_ok = not settings.is_demo
    gem_ok = is_gemini_available()
    st.markdown(f"""
    <div style="font-size:0.7rem; font-weight:700; color:#86868B; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">Pipeline Telemetry</div>
    <div style="display:flex; flex-direction:column; gap:0.4rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.82rem; color:#1D1D1F; padding:0.4rem 0.6rem; background:#FFFFFF; border-radius:8px; border:1px solid rgba(0,0,0,0.04);">
            <span>Core Mode</span>
            <span style="font-weight:600; color:{'#34C759' if not settings.is_demo else '#D97706'};">{settings.mode.upper()}</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.82rem; color:#1D1D1F; padding:0.4rem 0.6rem; background:#FFFFFF; border-radius:8px; border:1px solid rgba(0,0,0,0.04);">
            <span>Sentinel GEE</span>
            <span style="font-weight:600; color:{'#34C759' if gee_ok else '#86868B'};">{'Active' if gee_ok else 'Simulated'}</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.82rem; color:#1D1D1F; padding:0.4rem 0.6rem; background:#FFFFFF; border-radius:8px; border:1px solid rgba(0,0,0,0.04);">
            <span>Gemini AI</span>
            <span style="font-weight:600; color:{'#34C759' if gem_ok else '#86868B'};">{'Connected' if gem_ok else 'Demo Model'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Refresh Telemetry", use_container_width=True):
        st.session_state.pop("analysis", None)
        st.session_state.pop("advisory", None)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Main Canvas
# ═══════════════════════════════════════════════════════════════════════════

analysis = get_analysis()

# Hero Banner
st.markdown(f"""
<div class="apple-hero">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
            <h1>AgriN Crop Intelligence</h1>
            <div class="subtitle">Autonomous satellite synthesis, ML classification & AI decision support</div>
        </div>
        <div style="display:flex; gap:0.5rem;">
            <span class="apple-badge badge-region">📍 Ludhiana Pilot</span>
            <span class="apple-badge {'badge-live' if analysis.data_source == DataSource.LIVE else 'badge-demo'}">
                {'● LIVE DATA' if analysis.data_source == DataSource.LIVE else '🔶 DEMO DATASET'}
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Health & Telemetry Metrics (Apple Health Style Widgets) ──

crop_name = analysis.crop_prediction.predicted_crop.value.capitalize() if analysis.crop_prediction else "—"
crop_conf = f"{analysis.crop_prediction.confidence:.0%}" if analysis.crop_prediction else "—"
stress_level = analysis.stress_assessment.stress_level.value if analysis.stress_assessment else "moderate"
trend_val = analysis.ndvi_trend.value if analysis.ndvi_trend else "stable"
ndvi_val = f"{analysis.ndvi_current:.3f}" if analysis.ndvi_current else "0.520"
obs_date = str(analysis.observation_date) if analysis.observation_date else "Today"

# Stress styling
stress_theme = {
    "healthy": ("bg-green-soft", "color-green", "🟢", "Healthy"),
    "mild": ("bg-orange-soft", "color-orange", "🟡", "Mild Stress"),
    "moderate": ("bg-orange-soft", "color-orange", "🟠", "Moderate Stress"),
    "severe": ("bg-red-soft", "color-red", "🔴", "Severe Stress"),
}.get(stress_level, ("bg-green-soft", "color-green", "🟢", "Healthy"))

trend_theme = {
    "improving": ("bg-green-soft", "color-green", "↗", "Improving"),
    "stable": ("bg-blue-soft", "color-blue", "→", "Stable"),
    "declining": ("bg-red-soft", "color-red", "↘", "Declining"),
}.get(trend_val, ("bg-blue-soft", "color-blue", "→", "Stable"))

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-widget">
        <div class="widget-header">
            <span class="widget-label">Detected Crop</span>
            <div class="widget-icon bg-orange-soft">🌾</div>
        </div>
        <div class="widget-value">{crop_name}</div>
        <div class="widget-caption">Confidence {crop_conf}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-widget">
        <div class="widget-header">
            <span class="widget-label">Crop Status</span>
            <div class="widget-icon {stress_theme[0]}">{stress_theme[2]}</div>
        </div>
        <div class="widget-value {stress_theme[1]}">{stress_theme[3].split()[0]}</div>
        <div class="widget-caption">{stress_theme[3]}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-widget">
        <div class="widget-header">
            <span class="widget-label">Current NDVI</span>
            <div class="widget-icon bg-green-soft">🍃</div>
        </div>
        <div class="widget-value color-green">{ndvi_val}</div>
        <div class="widget-caption">Vegetation Index</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-widget">
        <div class="widget-header">
            <span class="widget-label">Health Trend</span>
            <div class="widget-icon {trend_theme[0]}">{trend_theme[2]}</div>
        </div>
        <div class="widget-value {trend_theme[1]}">{trend_theme[3]}</div>
        <div class="widget-caption">Recent Revisit Slope</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-widget">
        <div class="widget-header">
            <span class="widget-label">Observation</span>
            <div class="widget-icon bg-blue-soft">🛰️</div>
        </div>
        <div class="widget-value" style="font-size:1.15rem;">{obs_date}</div>
        <div class="widget-caption">Sentinel-2 Pass</div>
    </div>
    """, unsafe_allow_html=True)


# ── High-Fidelity Maps & Scientific Trajectory ──

st.markdown('<div class="apple-section-title">Geospatial Intelligence & Spectral Dynamics</div>', unsafe_allow_html=True)

tab_crop, tab_stress, tab_ndvi = st.tabs([
    "  🗺️ Crop Classification Raster  ",
    "  💧 Moisture Stress Assessment  ",
    "  📈 NDVI & NDWI Temporal Trajectory  ",
])

# High-resolution Apple styled crop map
with tab_crop:
    crop_map_data = generate_demo_crop_map_data(
        center_lat=analysis.farm.latitude,
        center_lon=analysis.farm.longitude,
        grid_size=24,
    )

    m = folium.Map(
        location=[analysis.farm.latitude, analysis.farm.longitude],
        zoom_start=15,
        tiles="CartoDB Positron",
        attr="AgriN / Copernicus",
    )

    CROP_PALETTE = {
        "wheat": "#D97706",   # Apple Warm Amber
        "rice": "#16A34A",    # Apple Deep Green
        "other": "#2563EB",   # Apple Vivid Blue
    }

    cell_size = 0.00045
    for cell in crop_map_data:
        color = CROP_PALETTE.get(cell["crop"], "#64748B")
        folium.Rectangle(
            bounds=[
                [cell["lat"] - cell_size, cell["lon"] - cell_size],
                [cell["lat"] + cell_size, cell["lon"] + cell_size],
            ],
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.78,
            weight=0.5,
            popup=folium.Popup(
                f"<div style='font-family:-apple-system,sans-serif; font-size:12px;'>"
                f"<b>Crop:</b> {cell['crop'].capitalize()}<br>"
                f"<b>Confidence:</b> {cell['confidence']:.0%}</div>",
                max_width=180,
            ),
        ).add_to(m)

    # Farm AOI boundary with Apple rounded aesthetic
    bbox = analysis.farm.bbox
    folium.Rectangle(
        bounds=[[bbox.min_lat, bbox.min_lon], [bbox.max_lat, bbox.max_lon]],
        color="#007AFF",
        fill=False,
        weight=2.5,
        dash_array="6, 6",
        popup="Farm AOI Boundary",
    ).add_to(m)

    legend_crop = """
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;
         background:rgba(255,255,255,0.92);padding:10px 16px;border-radius:14px;
         border:1px solid rgba(0,0,0,0.08);box-shadow:0 4px 16px rgba(0,0,0,0.08);
         font-family:-apple-system,BlinkMacSystemFont,sans-serif;font-size:12px;color:#1D1D1F;backdrop-filter:blur(16px);">
        <div style="font-weight:700;margin-bottom:4px;">Crop Layer</div>
        <div style="display:flex;gap:12px;align-items:center;">
            <span><span style="color:#D97706;font-size:14px;">■</span> Wheat</span>
            <span><span style="color:#16A34A;font-size:14px;">■</span> Rice</span>
            <span><span style="color:#2563EB;font-size:14px;">■</span> Other</span>
        </div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_crop))

    st_folium(m, width=None, height=480, key="apple_crop_map")

# High-resolution Apple styled stress map
with tab_stress:
    stress_map_data = generate_demo_stress_map_data(
        center_lat=analysis.farm.latitude,
        center_lon=analysis.farm.longitude,
        grid_size=24,
    )

    m2 = folium.Map(
        location=[analysis.farm.latitude, analysis.farm.longitude],
        zoom_start=15,
        tiles="CartoDB Positron",
        attr="AgriN / Copernicus",
    )

    STRESS_PALETTE = {
        "healthy": "#16A34A",
        "mild": "#EAB308",
        "moderate": "#F97316",
        "severe": "#DC2626",
    }

    for cell in stress_map_data:
        color = STRESS_PALETTE.get(cell["stress_level"], "#64748B")
        folium.Rectangle(
            bounds=[
                [cell["lat"] - cell_size, cell["lon"] - cell_size],
                [cell["lat"] + cell_size, cell["lon"] + cell_size],
            ],
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=0.5,
            popup=folium.Popup(
                f"<div style='font-family:-apple-system,sans-serif; font-size:12px;'>"
                f"<b>Stress:</b> {cell['stress_level'].capitalize()}<br>"
                f"<b>Indicator:</b> {cell['indicator']:.2f}</div>",
                max_width=180,
            ),
        ).add_to(m2)

    folium.Rectangle(
        bounds=[[bbox.min_lat, bbox.min_lon], [bbox.max_lat, bbox.max_lon]],
        color="#007AFF",
        fill=False,
        weight=2.5,
        dash_array="6, 6",
    ).add_to(m2)

    legend_stress = """
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;
         background:rgba(255,255,255,0.92);padding:10px 16px;border-radius:14px;
         border:1px solid rgba(0,0,0,0.08);box-shadow:0 4px 16px rgba(0,0,0,0.08);
         font-family:-apple-system,BlinkMacSystemFont,sans-serif;font-size:12px;color:#1D1D1F;backdrop-filter:blur(16px);">
        <div style="font-weight:700;margin-bottom:4px;">Moisture / Crop Stress</div>
        <div style="display:flex;gap:12px;align-items:center;">
            <span><span style="color:#16A34A;font-size:14px;">■</span> Healthy</span>
            <span><span style="color:#EAB308;font-size:14px;">■</span> Mild</span>
            <span><span style="color:#F97316;font-size:14px;">■</span> Moderate</span>
            <span><span style="color:#DC2626;font-size:14px;">■</span> Severe</span>
        </div>
    </div>"""
    m2.get_root().html.add_child(folium.Element(legend_stress))

    st_folium(m2, width=None, height=480, key="apple_stress_map")

# Apple Health style chart
with tab_ndvi:
    s2_obs = [
        o for o in analysis.recent_observations
        if o.satellite == "Sentinel-2" and o.ndvi is not None
    ]

    if s2_obs:
        df = pd.DataFrame([
            {"Date": o.observation_date, "NDVI": o.ndvi, "NDWI": o.ndwi}
            for o in sorted(s2_obs, key=lambda x: x.observation_date)
        ])

        fig = go.Figure()

        # Apple Health style smooth green curve
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["NDVI"], name="NDVI (Vegetation Index)",
            mode="lines+markers",
            line=dict(color="#34C759", width=3.5, shape="spline"),
            marker=dict(size=7, color="#FFFFFF", line=dict(width=2.5, color="#34C759")),
            fill="tozeroy",
            fillcolor="rgba(52, 199, 89, 0.08)",
        ))

        if df["NDWI"].notna().any():
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df["NDWI"], name="NDWI (Water Index)",
                mode="lines+markers",
                line=dict(color="#007AFF", width=2, dash="dot", shape="spline"),
                marker=dict(size=5, color="#FFFFFF", line=dict(width=2, color="#007AFF")),
            ))

        thresholds = get_settings().ndvi_thresholds
        fig.add_hline(y=thresholds["dense_vegetation_min"], line_dash="dash",
                      line_color="rgba(52,199,89,0.35)", annotation_text="Dense Canopy Baseline (0.50)",
                      annotation_font_color="#34C759", annotation_font_size=11)
        fig.add_hline(y=thresholds["bare_soil_max"], line_dash="dash",
                      line_color="rgba(255,59,48,0.35)", annotation_text="Bare Soil Threshold (0.15)",
                      annotation_font_color="#FF3B30", annotation_font_size=11)

        fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0.7)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif", color="#636366"),
            xaxis=dict(
                title="Observation Timeline",
                gridcolor="rgba(0,0,0,0.04)",
                linecolor="rgba(0,0,0,0.08)",
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title="Spectral Index",
                range=[-0.15, 1.0],
                gridcolor="rgba(0,0,0,0.04)",
                linecolor="rgba(0,0,0,0.08)",
                tickfont=dict(size=11),
            ),
            height=440,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
                font=dict(size=12),
            ),
            margin=dict(l=45, r=25, t=20, b=45),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No optical observations available.")


# ── Feature Importance (Random Forest Explainability) ──

if analysis.crop_prediction and analysis.crop_prediction.feature_importance:
    with st.expander("Model Interpretability & Feature Attribution"):
        imp = analysis.crop_prediction.feature_importance
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        names = [x[0].replace("_", " ").title() for x in sorted_imp]
        values = [x[1] for x in sorted_imp]

        fig_imp = go.Figure(go.Bar(
            x=values, y=names, orientation="h",
            marker=dict(
                color=values,
                colorscale=[[0, "#E5E5EA"], [1, "#007AFF"]],
                line=dict(width=0),
            ),
        ))
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif", color="#636366"),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(title="Gini Feature Importance", gridcolor="rgba(0,0,0,0.04)"),
            height=300,
            margin=dict(l=140, r=20, t=10, b=35),
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ── Apple Intelligence Advisory ──

st.markdown('<div class="apple-section-title">✨ Apple Intelligence & Crop Advisory</div>', unsafe_allow_html=True)

lang = st.session_state.get("language", "en")

if "advisory" not in st.session_state or st.session_state.get("advisory_lang") != lang:
    advisory = generate_advisory(analysis, language=lang)
    st.session_state.advisory = advisory
    st.session_state.advisory_lang = lang

advisory = st.session_state.advisory

st.markdown(f"""
<div class="intelligence-card">
    <div class="intelligence-header">
        <span class="intelligence-pill">✨ AgriN Advisory</span>
        <span style="font-size:0.8rem; color:#86868B; font-weight:500;">
            {'Synthesized via Google Gemini 1.5' if advisory.data_source == DataSource.LIVE else 'Demo Synthesis Mode'}
        </span>
    </div>
    <div class="intelligence-content">{advisory.advisory_text}</div>
</div>
""", unsafe_allow_html=True)


# ── Ask AgriN (Apple Siri / Assistant Style) ──

st.markdown('<div class="apple-section-title">💬 Ask AgriN</div>', unsafe_allow_html=True)

if lang == "hi":
    examples = ["मेरी फसल की स्थिति कैसी है?", "NDVI का क्या मतलब है?", "क्या मुझे सिंचाई करनी चाहिए?"]
else:
    examples = ["What is the health status of my crop?", "Explain my current NDVI index", "Should I irrigate my wheat field?"]

cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    with cols[i]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.user_question = ex

user_question = st.text_input(
    "Ask a question regarding your farm...",
    value=st.session_state.get("user_question", ""),
    key="qa_input",
    placeholder="e.g. Is my crop showing signs of moisture stress?" if lang == "en" else "उदा: क्या मेरी फसल में नमी का तनाव है?",
)

if user_question:
    with st.spinner("Analyzing farm context..."):
        response = ask_question(user_question, analysis, language=lang)
    st.markdown(f"""
    <div class="qa-box">
        <div style="font-size:0.75rem; font-weight:700; color:#007AFF; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.4rem;">
            Response
        </div>
        {response}
    </div>
    """, unsafe_allow_html=True)


# ── Footer ──

st.markdown("""
<div style="text-align:center; padding:2rem 0; color:#86868B; font-size:0.8rem;">
    <b>AgriN</b> &nbsp;•&nbsp; Designed with Apple Human Interface Principles &nbsp;•&nbsp; v0.1.0<br>
    <span style="font-size:0.75rem; color:#A1A1A6;">Satellite indicators represent decision-support estimates. Consult local agricultural authorities for treatment verification.</span>
</div>
""", unsafe_allow_html=True)
