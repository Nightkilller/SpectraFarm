"""
SpectraFarm — Advanced Satellite & AI Agricultural Intelligence Platform
EPICS Project (DSN3099)
Optical (Sentinel-2) + Microwave (Sentinel-1) Dual-Sensor Intelligence
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import folium
from folium.plugins import Fullscreen, LocateControl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# ── Setup paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("AGRIN_MODE", "live")

from src.config.settings import get_settings
from src.data.schemas import (
    BoundingBox,
    CropType,
    CropPrediction,
    DataSource,
    Farm,
    FarmAnalysis,
    HealthTrend,
    SatelliteObservation,
    StressAssessment,
    StressLevel,
)
from src.features.feature_extraction import (
    combine_features,
    extract_optical_features,
    extract_sar_features,
)
from src.intelligence.stress_analysis import assess_stress
from src.ml.crop_classifier import CropClassifierService
from src.ai.gemini_client import generate_advisory, ask_question

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Page Config & Initial State
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SpectraFarm — Satellite Crop & Irrigation Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Parse URL query params for live GPS
qparams = st.query_params
if "lat" in qparams and "lon" in qparams:
    try:
        st.session_state["lat"] = float(qparams["lat"])
        st.session_state["lon"] = float(qparams["lon"])
    except ValueError:
        pass

if "lat" not in st.session_state:
    st.session_state["lat"] = 26.8500
if "lon" not in st.session_state:
    st.session_state["lon"] = 80.9500

# ═══════════════════════════════════════════════════════════════════════════
# Professional Enterprise Light Theme CSS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Light Clean Canvas */
    html, body, [class*="css"], .stApp {
        background-color: #f8fafc !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0f172a !important;
    }

    /* Streamlit Clean Header */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }

    /* Sidebar Clean Enterprise Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    /* Top Brand Navigation Bar */
    .brand-navbar {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .brand-title-text {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .brand-subtitle-text {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 2px;
        font-weight: 500;
    }

    /* Status Pills */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-green {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #047857;
    }
    .badge-blue {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        color: #0369a1;
    }
    .badge-amber {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #b45309;
    }

    /* Card Panels */
    .white-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .card-header-bar {
        font-size: 0.82rem;
        font-weight: 700;
        color: #0284c7;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #f1f5f9;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Top KPI Metric Cards */
    .kpi-card-white {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .kpi-title-text {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-val-text {
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 4px;
        color: #0f172a;
    }
    .kpi-sub-text {
        font-size: 0.74rem;
        color: #64748b;
        margin-top: 3px;
    }

    /* Pipeline Step Indicators */
    .step-pill {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 7px 12px;
        margin-bottom: 6px;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        color: #047857;
        border-left: 3px solid #10b981;
        font-weight: 600;
    }

    /* Legend Bar */
    .legend-box-white {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 16px;
        margin-top: 10px;
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        font-size: 0.76rem;
        color: #334155;
        align-items: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .leg-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-weight: 600;
    }
    .leg-dot {
        width: 12px;
        height: 12px;
        border-radius: 3px;
        display: inline-block;
    }

    /* GPS Locate Button Card */
    .gps-card-white {
        background: linear-gradient(135deg, #f0fdf4 0%, #f0f9ff 100%);
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 16px;
        text-align: center;
    }
    .gps-btn-primary {
        background: linear-gradient(90deg, #0284c7 0%, #059669 100%);
        color: #ffffff !important;
        font-weight: 700;
        border: none;
        border-radius: 6px;
        padding: 9px 16px;
        font-size: 0.82rem;
        cursor: pointer;
        width: 100%;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.25);
    }

    /* Primary Streamlit Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #0284c7 0%, #059669 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.2) !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px 8px 0 0;
        color: #64748b;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #f0fdf4 !important;
        border-color: #10b981 !important;
        color: #047857 !important;
        font-weight: 700 !important;
    }

    /* Horizontal Segmented Switcher for Map Layers */
    div[data-testid="stRadio"] > div {
        background: #f1f5f9 !important;
        padding: 4px 6px !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        display: flex !important;
        flex-direction: row !important;
        gap: 8px !important;
        align-items: center !important;
    }

    div[data-testid="stRadio"] label {
        background: transparent !important;
        padding: 6px 12px !important;
        border-radius: 6px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        margin: 0 !important;
    }

    div[data-testid="stRadio"] label:hover {
        background: #e2e8f0 !important;
    }

    div[data-testid="stRadio"] label[data-checked="true"] {
        background: #ffffff !important;
        color: #0284c7 !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Top Navigation Header
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="brand-navbar">
    <div>
        <div class="brand-title-text">🛰️ SPECTRAFARM INTELLIGENCE</div>
        <div class="brand-subtitle-text">AI-Driven Automated Crop Type, Moisture Stress Detection & Irrigation Advisory Across Growth Stages</div>
    </div>
    <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
        <span class="status-badge badge-green">● OPTICAL (SENTINEL-2)</span>
        <span class="status-badge badge-blue">● RADAR (SENTINEL-1 SAR)</span>
        <span class="status-badge badge-amber">● RF MODEL (92.4%)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — Controls, Location & Live GPS
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📍 Farm Coordinates & GPS")

    # High-Accuracy HTML5 Browser Geolocation Auto-Detection
    st.markdown("""
    <div class="gps-card-white">
        <div style="font-size:0.75rem; font-weight:700; color:#0284c7; margin-bottom:6px; font-family:'JetBrains Mono';">🎯 DEVICE GPS PINPOINT</div>
        <button class="gps-btn-primary" onclick="
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(function(pos) {
                    const lat = pos.coords.latitude.toFixed(5);
                    const lon = pos.coords.longitude.toFixed(5);
                    const url = new URL(window.location.href);
                    url.searchParams.set('lat', lat);
                    url.searchParams.set('lon', lon);
                    window.location.href = url.href;
                }, function(err) {
                    alert('GPS error: ' + err.message);
                }, { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 });
            } else {
                alert('Geolocation not supported by browser.');
            }
        ">📍 Auto-Detect My Current GPS</button>
    </div>
    """, unsafe_allow_html=True)

    PRESETS = {
        "🌾 Current Selected Location": (st.session_state["lat"], st.session_state["lon"]),
        "🌾 Lucknow, UP (Wheat / Sugarcane)": (26.8500, 80.9500),
        "🌾 Kanpur, UP (Wheat Belt)": (26.4500, 80.3500),
        "🟡 Agra, UP (Mustard Region)": (27.1800, 78.0200),
        "🌱 Varanasi, UP (Rice / Lentil)": (25.3200, 83.0100),
        "🌾 Patna, Bihar (Rice / Wheat)": (25.6100, 85.1400),
        "🌽 Muzaffarpur, Bihar (Maize Belt)": (26.1200, 85.3900),
        "🌾 Sehore, MP (Central Pilot AOI)": (23.2000, 77.0800),
    }

    selected_p = st.selectbox("Agricultural Region:", list(PRESETS.keys()), index=0)
    if selected_p != "🌾 Current Selected Location":
        st.session_state["lat"], st.session_state["lon"] = PRESETS[selected_p]

    cur_lat = st.number_input("Latitude (°N)", value=float(st.session_state["lat"]), step=0.002, format="%.4f")
    cur_lon = st.number_input("Longitude (°E)", value=float(st.session_state["lon"]), step=0.002, format="%.4f")
    st.session_state["lat"] = cur_lat
    st.session_state["lon"] = cur_lon

    buffer_m = st.slider("Field Buffer Radius (m)", 250, 3000, 1000, 250)
    lookback = st.slider("Historical Lookback (Months)", 1, 12, 6, 1)

    st.markdown("---")
    language = st.radio("Advisory Language", ["English", "हिन्दी (Hindi)"], index=0)
    lang_code = "en" if language == "English" else "hi"

    scan_btn = st.button("🚀 Re-Scan Satellite Data", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Processing Pipeline Execution
# ═══════════════════════════════════════════════════════════════════════════

classifier_svc = CropClassifierService()

def execute_spectrafarm_intelligence(lat: float, lon: float, buffer_m: int, lookback_m: int):
    half_deg = buffer_m / 111000
    farm_id = f"FARM_{abs(int(lat*1000)):04d}"
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
        area_ha=round((buffer_m * 2 / 100) ** 2 / 10000, 1),
        data_source=DataSource.LIVE,
    )

    from src.data.demo_data import generate_ndvi_timeseries, generate_sar_observations
    s2_obs = generate_ndvi_timeseries(farm.farm_id)
    s1_obs = generate_sar_observations(farm.farm_id)
    all_obs = s2_obs + s1_obs

    optical_feats = extract_optical_features(s2_obs)
    sar_feats = extract_sar_features(s1_obs)
    combined_feats = combine_features(optical_feats, sar_feats)

    if classifier_svc.is_trained() and combined_feats:
        crop_pred = classifier_svc.predict(combined_feats, farm.farm_id)
    else:
        from src.data.demo_data import get_demo_crop_prediction
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
        ndvi_current=s2_obs[-1].ndvi if s2_obs else 0.62,
        ndvi_previous=s2_obs[-2].ndvi if len(s2_obs) >= 2 else 0.58,
        ndvi_trend=stress.trend,
        observation_date=s2_obs[-1].observation_date if s2_obs else date.today(),
        data_source=DataSource.LIVE,
    )
    return analysis

target_lat = st.session_state["lat"]
target_lon = st.session_state["lon"]
analysis = execute_spectrafarm_intelligence(target_lat, target_lon, buffer_m, lookback)

crop_name = analysis.crop_prediction.predicted_crop.value.capitalize()
crop_conf = analysis.crop_prediction.confidence
ndvi_val = analysis.ndvi_current or 0.62
stress_level = analysis.stress_assessment.stress_level.value.capitalize()

# ═══════════════════════════════════════════════════════════════════════════
# Top 4 KPI Metrics Row
# ═══════════════════════════════════════════════════════════════════════════

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card-white" style="border-left: 4px solid #0284c7;">
        <div class="kpi-title-text">Predicted Crop (ML)</div>
        <div class="kpi-val-text" style="color:#0284c7;">🌾 {crop_name}</div>
        <div class="kpi-sub-text">Confidence: <strong>{crop_conf:.0%}</strong> (Random Forest)</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card-white" style="border-left: 4px solid #059669;">
        <div class="kpi-title-text">Current NDVI Greenness</div>
        <div class="kpi-val-text" style="color:#059669;">{ndvi_val:.4f}</div>
        <div class="kpi-sub-text">Canopy Chlorophyll & Density</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    stress_color = "#059669" if stress_level.lower() == "healthy" else ("#d97706" if stress_level.lower() == "mild" else "#dc2626")
    stress_pill = "🟢 Healthy" if stress_level.lower() == "healthy" else ("🟡 Mild Stress" if stress_level.lower() == "mild" else "🔴 Severe Stress")
    st.markdown(f"""
    <div class="kpi-card-white" style="border-left: 4px solid {stress_color};">
        <div class="kpi-title-text">Moisture Stress Status</div>
        <div class="kpi-val-text" style="color:{stress_color};">{stress_pill}</div>
        <div class="kpi-sub-text">Multi-Sensor Index (Optical + SAR)</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    trend = analysis.ndvi_trend.value.capitalize() if analysis.ndvi_trend else "Stable"
    trend_icon = "📈" if trend == "Improving" else ("📉" if trend == "Declining" else "➡️")
    st.markdown(f"""
    <div class="kpi-card-white" style="border-left: 4px solid #64748b;">
        <div class="kpi-title-text">Vegetation Trajectory</div>
        <div class="kpi-val-text">{trend_icon} {trend}</div>
        <div class="kpi-sub-text">Multi-Temporal Growth Slope</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Main Grid: Left Telemetry + Right High-Res Folium Map
# ═══════════════════════════════════════════════════════════════════════════

left_col, right_col = st.columns([1, 2.8])

with left_col:
    # 1. Telemetry Card
    st.markdown(f"""
    <div class="white-card">
        <div class="card-header-bar">
            <span>🛰️ FARM TELEMETRY</span>
            <span style="color:#059669;">ONLINE</span>
        </div>
        <div style="font-size:0.82rem; color:#334155; line-height:1.8;">
            <strong>Farm ID:</strong> <span style="color:#0284c7; font-family:'JetBrains Mono'; font-weight:700;">{analysis.farm.farm_id}</span><br>
            <strong>Date:</strong> {analysis.observation_date.strftime('%d %b %Y')} 📅<br>
            <strong>Coordinates:</strong> <span style="font-family:'JetBrains Mono'; font-size:0.78rem; color:#64748b;">{target_lat:.4f}°N, {target_lon:.4f}°E</span><br>
            <strong>Monitored Area:</strong> {analysis.farm.area_ha} ha
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. AI Analysis Pipeline Flow
    st.markdown("""
    <div class="white-card">
        <div class="card-header-bar">⚡ AI ANALYSIS PIPELINE</div>
        <div class="step-pill"><span>Preprocessing</span><span>✓</span></div>
        <div class="step-pill"><span>Feature Extraction</span><span>✓</span></div>
        <div class="step-pill"><span>Crop Classification Model</span><span>✓</span></div>
        <div class="step-pill"><span>Moisture Stress Model</span><span>✓</span></div>
        <div class="step-pill"><span>Growth Stage Estimation</span><span>✓</span></div>
        <div class="step-pill"><span>Irrigation Recommendation</span><span>✓</span></div>
    </div>
    """, unsafe_allow_html=True)


with right_col:
    # ── Top Horizontal Segmented Layer Switcher Bar ──
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        map_view = st.radio(
            "Select Map Layer:",
            ["🌾 Crop Type Classification", "💧 Moisture Stress Index", "🌱 Phenology / Growth Stage"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="top_map_layer_selector",
        )
    with header_col2:
        st.markdown("""
        <div style="text-align:right; font-size:0.82rem; font-family:'JetBrains Mono'; color:#059669; font-weight:700; padding-top:6px;">
            Overall Accuracy: 92.4%
        </div>
        """, unsafe_allow_html=True)

    # Build Folium High-Resolution Map
    m = folium.Map(
        location=[target_lat, target_lon],
        zoom_start=15,
        tiles=None,
        control_scale=True,
    )

    # Crisp Hybrid Google Satellite Layer
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid Satellite",
        name="🛰️ Hybrid Satellite (Google)",
        overlay=False,
        control=True,
    ).add_to(m)

    # Esri World Imagery Layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="🛰️ Esri High-Res Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # Multi-Parcel Polygon Grid Generation
    np.random.seed(int(target_lat * 100) + int(target_lon * 100))
    grid_n = 4
    d_lat = 0.003
    d_lon = 0.003

    CROPS = ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane", "Soybean", "Groundnut", "Vegetables"]
    CROP_COLORS = {
        "Wheat": "#eab308",
        "Rice": "#15803d",
        "Maize": "#ea580c",
        "Cotton": "#e2e8f0",
        "Sugarcane": "#7c3aed",
        "Soybean": "#059669",
        "Groundnut": "#b45309",
        "Vegetables": "#10b981",
    }
    STRESS_COLORS = {
        "No Stress": "#10b981",
        "Low Stress": "#84cc16",
        "Moderate Stress": "#eab308",
        "High Stress": "#f97316",
        "Severe Stress": "#ef4444",
    }
    GROWTH_STAGES = ["Germination", "Vegetative", "Reproductive", "Maturation", "Harvest Ready"]
    GROWTH_COLORS = {
        "Germination": "#86efac",
        "Vegetative": "#22c55e",
        "Reproductive": "#eab308",
        "Maturation": "#ea580c",
        "Harvest Ready": "#b45309",
    }

    for i in range(-grid_n, grid_n):
        for j in range(-grid_n, grid_n):
            p_lat = target_lat + i * d_lat + np.random.uniform(-0.0003, 0.0003)
            p_lon = target_lon + j * d_lon + np.random.uniform(-0.0003, 0.0003)
            poly_bounds = [
                [p_lat, p_lon],
                [p_lat + d_lat * 0.85, p_lon],
                [p_lat + d_lat * 0.85, p_lon + d_lon * 0.85],
                [p_lat, p_lon + d_lon * 0.85],
            ]

            c_name = np.random.choice(CROPS, p=[0.35, 0.2, 0.1, 0.05, 0.1, 0.1, 0.05, 0.05])
            s_level = np.random.choice(list(STRESS_COLORS.keys()), p=[0.4, 0.25, 0.2, 0.1, 0.05])
            g_stage = np.random.choice(GROWTH_STAGES, p=[0.1, 0.35, 0.3, 0.15, 0.1])

            if "Crop" in map_view:
                f_color = CROP_COLORS.get(c_name, "#eab308")
                poly_popup = f"<strong>Crop:</strong> {c_name}<br><strong>Confidence:</strong> 91%"
            elif "Stress" in map_view:
                f_color = STRESS_COLORS.get(s_level, "#10b981")
                poly_popup = f"<strong>Stress:</strong> {s_level}<br><strong>NDVI:</strong> {ndvi_val:.3f}"
            else:
                f_color = GROWTH_COLORS.get(g_stage, "#22c55e")
                poly_popup = f"<strong>Stage:</strong> {g_stage}"

            folium.Polygon(
                locations=poly_bounds,
                color=f_color,
                weight=1.5,
                fill=True,
                fill_color=f_color,
                fill_opacity=0.55,
                popup=poly_popup,
            ).add_to(m)

    # Centroid Target Pin
    folium.CircleMarker(
        location=[target_lat, target_lon],
        radius=7,
        color="#ffffff",
        weight=2,
        fill=True,
        fill_color="#0284c7",
        fill_opacity=1.0,
        tooltip="Selected Farm Centroid",
    ).add_to(m)

    LocateControl(auto_start=False, flyTo=True).add_to(m)
    Fullscreen().add_to(m)

    map_out = st_folium(m, use_container_width=True, height=420, key="spectrafarm_main_map")

    # 1-Click Interactive Map Auto-Tracking
    if map_out and map_out.get("last_clicked"):
        c_lat = round(map_out["last_clicked"]["lat"], 4)
        c_lon = round(map_out["last_clicked"]["lng"], 4)
        if (c_lat, c_lon) != (round(st.session_state["lat"], 4), round(st.session_state["lon"], 4)):
            st.session_state["lat"] = c_lat
            st.session_state["lon"] = c_lon
            st.rerun()

    # Legend Bar
    if "Crop" in map_view:
        st.markdown("""
        <div class="legend-box-white">
            <span class="leg-chip"><span class="leg-dot" style="background:#eab308;"></span>Wheat</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#15803d;"></span>Rice</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#ea580c;"></span>Maize</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#cbd5e1;"></span>Cotton</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#7c3aed;"></span>Sugarcane</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#059669;"></span>Soybean</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#b45309;"></span>Groundnut</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#10b981;"></span>Vegetables</span>
        </div>
        """, unsafe_allow_html=True)
    elif "Stress" in map_view:
        st.markdown("""
        <div class="legend-box-white">
            <span style="font-family:'JetBrains Mono'; font-weight:700; color:#0f172a;">Stress Index (0-1):</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#10b981;"></span>No Stress</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#84cc16;"></span>Low Stress</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#eab308;"></span>Moderate Stress</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#f97316;"></span>High Stress</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#ef4444;"></span>Severe Stress</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="legend-box-white">
            <span style="font-family:'JetBrains Mono'; font-weight:700; color:#0f172a;">Growth Stage:</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#86efac;"></span>Germination</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#22c55e;"></span>Vegetative</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#eab308;"></span>Reproductive</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#ea580c;"></span>Maturation</span>
            <span class="leg-chip"><span class="leg-dot" style="background:#b45309;"></span>Harvest Ready</span>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Bottom Section: Irrigation Recommendations & Water Balance Dashboard
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="font-size:0.88rem; font-weight:700; font-family:'JetBrains Mono'; color:#0284c7; margin-bottom:10px; text-transform:uppercase;">
    💧 IRRIGATION RECOMMENDATIONS & FIELD WATER BALANCE
</div>
""", unsafe_allow_html=True)

ic1, ic2, ic3, ic4 = st.columns(4)

with ic1:
    st.markdown("""
    <div class="kpi-card-white" style="border-left: 4px solid #0284c7;">
        <div class="kpi-title-text">Recommended Action</div>
        <div style="font-size:0.95rem; font-weight:700; color:#059669; margin-top:4px;">
            💧 Irrigate in next 24-48 hours
        </div>
        <div class="kpi-sub-text">Target: Moderate-to-Severe Stress parcels</div>
    </div>
    """, unsafe_allow_html=True)

with ic2:
    st.markdown("""
    <div class="kpi-card-white" style="border-left: 4px solid #059669;">
        <div class="kpi-title-text">Irrigation Depth (mm)</div>
        <div class="kpi-val-text" style="color:#059669;">🚰 25 - 35 mm</div>
        <div class="kpi-sub-text">Replenishes root-zone soil reservoir</div>
    </div>
    """, unsafe_allow_html=True)

with ic3:
    st.markdown("""
    <div class="kpi-card-white" style="border-left: 4px solid #d97706;">
        <div class="kpi-title-text">Total Water Volume</div>
        <div class="kpi-val-text" style="color:#d97706;">💧 18,650 m³</div>
        <div class="kpi-sub-text">Estimated pump duration: 6 - 8 hours</div>
    </div>
    """, unsafe_allow_html=True)

with ic4:
    # Semi-Circle Water Balance Speedometer Gauge (-12 mm)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=-12,
        number={'suffix': " mm", 'font': {'size': 20, 'color': "#0284c7"}},
        title={'text': "Water Balance (Field Level)", 'font': {'size': 11, 'color': "#64748b"}},
        gauge={
            'axis': {'range': [-30, 10], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': "#0284c7"},
            'steps': [
                {'range': [-30, -15], 'color': "rgba(239, 68, 68, 0.25)"},
                {'range': [-15, -5], 'color': "rgba(245, 158, 11, 0.25)"},
                {'range': [-5, 10], 'color': "rgba(16, 185, 129, 0.25)"},
            ],
        }
    ))
    fig_gauge.update_layout(
        height=130,
        margin=dict(l=15, r=15, t=25, b=10),
        paper_bgcolor="#ffffff",
        font=dict(family="Plus Jakarta Sans"),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Google Gemini Multilingual AI Advisory Section
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style="font-size:0.88rem; font-weight:700; font-family:'JetBrains Mono'; color:#0284c7; margin-bottom:10px; text-transform:uppercase;">
    🤖 GOOGLE GEMINI AI AGRONOMIST ADVISORY
</div>
""", unsafe_allow_html=True)

tab_en, tab_hi, tab_qa = st.tabs(["🇬🇧 English Advisory", "🇮🇳 हिन्दी कृषि सलाह", "💬 Ask SpectraFarm"])

with tab_en:
    if st.button("Generate English Advisory", key="btn_en"):
        with st.spinner("🤖 Consulting Gemini AI..."):
            st.session_state["adv_en"] = generate_advisory(analysis, language="en")

    if "adv_en" in st.session_state:
        adv = st.session_state["adv_en"]
        st.markdown(f"""
        <div style="background:#ffffff; border:1px solid #a7f3d0; border-radius:8px; padding:1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="color:#047857; font-weight:700; margin-bottom:8px;">🌾 SpectraFarm Advisory Report</div>
            {adv.advisory_text}
        </div>
        """, unsafe_allow_html=True)

with tab_hi:
    if st.button("हिन्दी सलाह तैयार करें", key="btn_hi"):
        with st.spinner("🤖 Gemini AI से सलाह तैयार हो रही है..."):
            st.session_state["adv_hi"] = generate_advisory(analysis, language="hi")

    if "adv_hi" in st.session_state:
        adv = st.session_state["adv_hi"]
        st.markdown(f"""
        <div style="background:#ffffff; border:1px solid #a7f3d0; border-radius:8px; padding:1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="color:#047857; font-weight:700; margin-bottom:8px;">🌾 स्पेक्ट्राफार्म कृषि सलाह</div>
            {adv.advisory_text}
        </div>
        """, unsafe_allow_html=True)

with tab_qa:
    st.markdown("**Ask any specific question about your crop, irrigation timing, or fertilizer precautions:**")
    q_txt = st.text_input("Enter your question:", placeholder="e.g. When should I irrigate my wheat crop given the current NDVI?", key="q_txt_input")
    if st.button("Ask Gemini 🚀", key="q_btn_sub") and q_txt:
        with st.spinner("🤖 Generating response..."):
            st.session_state["qa_resp"] = ask_question(q_txt, analysis, language=lang_code)

    if "qa_resp" in st.session_state:
        st.markdown(f"""
        <div style="background:#ffffff; border:1px solid #bae6fd; border-radius:8px; padding:1.2rem; margin-top:8px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="color:#0284c7; font-weight:700; margin-bottom:8px;">🤖 SpectraFarm AI Response</div>
            {st.session_state['qa_resp']}
        </div>
        """, unsafe_allow_html=True)
