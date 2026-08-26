"""
AgriN — Interactive Web Dashboard (Streamlit)

A farmer-friendly web interface where users enter any location in India
and receive real-time satellite-based crop intelligence:

  1. Multi-temporal Sentinel-2 NDVI & Sentinel-1 SAR analysis
  2. ML-based crop classification (Random Forest)
  3. Vegetation stress assessment
  4. Google Gemini AI advisory (English / Hindi)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Setup paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("AGRIN_MODE", "live")

from src.config.settings import get_settings
from src.data.schemas import (
    BoundingBox,
    CropType,
    DataSource,
    Farm,
    FarmAnalysis,
    HealthTrend,
    SatelliteObservation,
    StressLevel,
)
from src.data.demo_data import (
    generate_ndvi_timeseries,
    generate_sar_observations,
    get_demo_crop_prediction,
    get_demo_farm,
    get_demo_stress_assessment,
    get_demo_farm_analysis,
)
from src.features.feature_extraction import (
    combine_features,
    extract_optical_features,
    extract_sar_features,
)
from src.intelligence.stress_analysis import assess_stress
from src.ai.gemini_client import generate_advisory, ask_question, is_gemini_available

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AgriN — Smart Crop Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS for premium look
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0d4d2c 0%, #1a7a4a 40%, #2d9b5e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(13, 77, 44, 0.3);
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }

    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8faf9 100%);
        border: 1px solid #e0e8e3;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }

    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #6b7c72;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a3d2a;
        line-height: 1.2;
    }

    .metric-delta {
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 0.2rem;
    }

    .status-healthy { color: #16a34a; }
    .status-mild { color: #eab308; }
    .status-moderate { color: #f97316; }
    .status-severe { color: #dc2626; }

    .advisory-card {
        background: linear-gradient(145deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1px solid #bbf7d0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .advisory-card h3 {
        color: #166534;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .sidebar-section {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a3d1f 0%, #0d5a30 100%);
    }

    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown h3,
    div[data-testid="stSidebar"] .stMarkdown h2 {
        color: #d1e7db !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🌾 AgriN — Smart Crop Intelligence</h1>
    <p>AI-powered satellite analysis for Indian agriculture · Sentinel-2 + Sentinel-1 + Google Gemini</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — Location Input
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📍 Select Location")

    PRESET_LOCATIONS = {
        "Lucknow, UP (Wheat/Sugarcane)": (26.85, 80.95),
        "Kanpur, UP (Wheat)": (26.45, 80.35),
        "Agra, UP (Mustard)": (27.18, 78.02),
        "Varanasi, UP (Rice/Lentil)": (25.32, 83.01),
        "Patna, Bihar (Rice/Wheat)": (25.61, 85.14),
        "Muzaffarpur, Bihar (Maize)": (26.12, 85.39),
        "Sehore, MP (Pilot AOI)": (23.20, 77.08),
        "Custom Coordinates": None,
    }

    location_choice = st.selectbox(
        "📌 Choose a preset location",
        list(PRESET_LOCATIONS.keys()),
        index=0,
        key="location_select",
    )

    if location_choice == "Custom Coordinates":
        lat = st.number_input("Latitude (°N)", value=26.85, min_value=8.0, max_value=37.0, step=0.01, format="%.4f")
        lon = st.number_input("Longitude (°E)", value=80.95, min_value=68.0, max_value=97.0, step=0.01, format="%.4f")
    else:
        lat, lon = PRESET_LOCATIONS[location_choice]
        st.info(f"📍 **{lat:.4f}°N, {lon:.4f}°E**")

    st.markdown("---")

    st.markdown("## ⚙️ Settings")
    buffer_m = st.slider("AOI buffer (meters)", 500, 5000, 1000, 250)
    lookback = st.slider("Lookback (months)", 1, 12, 3, 1)
    language = st.radio("🌐 Advisory language", ["English", "हिन्दी (Hindi)"], index=0)
    lang_code = "en" if language == "English" else "hi"

    st.markdown("---")

    analyze_btn = st.button("🛰️ Analyze Location", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; opacity:0.6; font-size:0.75rem; color:#a3c4b0;'>
        AgriN v0.1 · Phase 9 Dashboard<br>
        Powered by Google Earth Engine<br>
        & Google Gemini AI
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def get_stress_color(level: str) -> str:
    """Return CSS class for stress level."""
    return {
        "healthy": "status-healthy",
        "mild": "status-mild",
        "moderate": "status-moderate",
        "severe": "status-severe",
    }.get(level.lower(), "status-moderate")


def get_stress_emoji(level: str) -> str:
    return {
        "healthy": "🟢",
        "mild": "🟡",
        "moderate": "🟠",
        "severe": "🔴",
    }.get(level.lower(), "🟠")


def create_ndvi_chart(observations: list[SatelliteObservation]) -> go.Figure:
    """Create an interactive NDVI time-series chart."""
    s2_obs = sorted(
        [o for o in observations if o.satellite == "Sentinel-2" and o.ndvi is not None],
        key=lambda o: o.observation_date,
    )

    if not s2_obs:
        fig = go.Figure()
        fig.add_annotation(text="No Sentinel-2 data available", showarrow=False, font=dict(size=16))
        return fig

    dates = [o.observation_date for o in s2_obs]
    ndvi_vals = [o.ndvi for o in s2_obs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=ndvi_vals, mode="lines+markers",
        name="NDVI",
        line=dict(color="#16a34a", width=3),
        marker=dict(size=8, color="#16a34a", line=dict(width=1, color="white")),
        fill="tozeroy",
        fillcolor="rgba(22, 163, 74, 0.1)",
    ))

    # Threshold lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="#9ca3af", annotation_text="Dense Vegetation (0.5)")
    fig.add_hline(y=0.15, line_dash="dash", line_color="#ef4444", annotation_text="Bare Soil (0.15)")

    fig.update_layout(
        title=dict(text="🌿 NDVI Vegetation Index Time Series", font=dict(size=16, family="Inter")),
        xaxis_title="Date",
        yaxis_title="NDVI",
        yaxis=dict(range=[-0.1, 1.0]),
        template="plotly_white",
        height=350,
        margin=dict(l=50, r=20, t=50, b=40),
        font=dict(family="Inter"),
    )
    return fig


def create_sar_chart(observations: list[SatelliteObservation]) -> go.Figure:
    """Create an interactive SAR backscatter chart."""
    s1_obs = sorted(
        [o for o in observations if o.satellite == "Sentinel-1" and o.vv is not None],
        key=lambda o: o.observation_date,
    )

    if not s1_obs:
        fig = go.Figure()
        fig.add_annotation(text="No Sentinel-1 SAR data available", showarrow=False, font=dict(size=16))
        return fig

    dates = [o.observation_date for o in s1_obs]
    vv_vals = [o.vv for o in s1_obs]
    vh_vals = [o.vh for o in s1_obs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=vv_vals, mode="lines+markers", name="VV (dB)",
        line=dict(color="#2563eb", width=2.5),
        marker=dict(size=7, color="#2563eb"),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=vh_vals, mode="lines+markers", name="VH (dB)",
        line=dict(color="#f59e0b", width=2.5),
        marker=dict(size=7, color="#f59e0b"),
    ))

    fig.update_layout(
        title=dict(text="📡 Sentinel-1 SAR Radar Backscatter", font=dict(size=16, family="Inter")),
        xaxis_title="Date",
        yaxis_title="Backscatter (dB)",
        template="plotly_white",
        height=350,
        margin=dict(l=50, r=20, t=50, b=40),
        font=dict(family="Inter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def run_analysis(lat: float, lon: float, buffer_m: int, lookback: int) -> FarmAnalysis:
    """Run the full analysis pipeline for the given coordinates."""
    settings = get_settings()

    # Create dynamic Farm object
    half_deg = buffer_m / 111000  # rough meters to degrees
    farm = Farm(
        farm_id=f"dynamic_{lat:.4f}_{lon:.4f}",
        name=f"Field at {lat:.4f}°N, {lon:.4f}°E",
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

    # Try live Earth Engine first
    try:
        from src.geospatial.gee_client import (
            get_dynamic_aoi,
            get_sentinel1_observations,
            get_sentinel2_observations,
            init_earth_engine,
            is_gee_available,
        )

        if init_earth_engine():
            end_date = date.today()
            start_date = end_date - timedelta(days=30 * lookback)

            aoi = get_dynamic_aoi(lat, lon, buffer_m)

            s2_obs = get_sentinel2_observations(
                bbox=farm.bbox, start_date=start_date, end_date=end_date,
                farm_id=farm.farm_id, max_cloud_cover=30,
            )
            s1_obs = get_sentinel1_observations(
                bbox=farm.bbox, start_date=start_date, end_date=end_date,
                farm_id=farm.farm_id,
            )

            if s2_obs:
                # Real satellite data path
                optical_feats = extract_optical_features(s2_obs)
                sar_feats = extract_sar_features(s1_obs)
                combined = combine_features(optical_feats, sar_feats)

                # Crop classification
                try:
                    from src.ml.crop_classifier import CropClassifierService
                    classifier = CropClassifierService()
                    if classifier.is_trained() and combined:
                        crop_pred = classifier.predict(combined, farm.farm_id)
                    else:
                        crop_pred = get_demo_crop_prediction(farm.farm_id)
                except Exception:
                    crop_pred = get_demo_crop_prediction(farm.farm_id)

                # Stress
                stress = assess_stress(s2_obs, farm.farm_id)

                return FarmAnalysis(
                    farm=farm,
                    crop_prediction=crop_pred,
                    stress_assessment=stress,
                    recent_observations=s2_obs + s1_obs,
                    ndvi_current=s2_obs[-1].ndvi if s2_obs else None,
                    ndvi_previous=s2_obs[-2].ndvi if len(s2_obs) >= 2 else None,
                    ndvi_trend=stress.trend,
                    observation_date=s2_obs[-1].observation_date if s2_obs else None,
                    data_source=DataSource.LIVE,
                )
    except Exception as e:
        logger.warning(f"Live analysis failed: {e}")

    # Fallback to demo
    s2_obs = generate_ndvi_timeseries(farm.farm_id)
    s1_obs = generate_sar_observations(farm.farm_id)
    stress = assess_stress(s2_obs, farm.farm_id)
    crop_pred = get_demo_crop_prediction(farm.farm_id)

    return FarmAnalysis(
        farm=farm,
        crop_prediction=crop_pred,
        stress_assessment=stress,
        recent_observations=s2_obs + s1_obs,
        ndvi_current=stress.ndvi_current,
        ndvi_previous=stress.ndvi_previous,
        ndvi_trend=stress.trend,
        observation_date=s2_obs[-1].observation_date if s2_obs else date.today(),
        data_source=DataSource.DEMO,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main Content
# ═══════════════════════════════════════════════════════════════════════════

if analyze_btn or "analysis" not in st.session_state:
    with st.spinner("🛰️ Connecting to Google Earth Engine and analyzing satellite data..."):
        analysis = run_analysis(lat, lon, buffer_m, lookback)
        st.session_state["analysis"] = analysis
        st.session_state["lat"] = lat
        st.session_state["lon"] = lon

analysis: FarmAnalysis = st.session_state.get("analysis")

if analysis is None:
    st.info("👆 Click **Analyze Location** to start.")
    st.stop()

# ── Data Source Badge ──────────────────────────────────────────────────
source_badge = "🟢 LIVE SATELLITE DATA" if analysis.data_source == DataSource.LIVE else "🔶 DEMO DATA"
st.caption(f"Data Source: **{source_badge}** · Location: **{st.session_state.get('lat', lat):.4f}°N, {st.session_state.get('lon', lon):.4f}°E**")

# ── Top Metric Cards ──────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    crop_name = "Unknown"
    crop_conf = 0.0
    if analysis.crop_prediction:
        crop_name = analysis.crop_prediction.predicted_crop.value.capitalize()
        crop_conf = analysis.crop_prediction.confidence
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Crop</div>
        <div class="metric-value">🌾 {crop_name}</div>
        <div class="metric-delta">Confidence: {crop_conf:.0%}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    ndvi_val = analysis.ndvi_current or 0.0
    ndvi_label = "Low" if ndvi_val < 0.3 else ("Moderate" if ndvi_val < 0.5 else "High")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current NDVI</div>
        <div class="metric-value" style="color: {'#16a34a' if ndvi_val >= 0.5 else '#f97316'}">{ndvi_val:.4f}</div>
        <div class="metric-delta">Greenness: {ndvi_label}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    stress_level = "Moderate"
    if analysis.stress_assessment:
        stress_level = analysis.stress_assessment.stress_level.value.capitalize()
    emoji = get_stress_emoji(stress_level)
    css_class = get_stress_color(stress_level)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Stress Level</div>
        <div class="metric-value {css_class}">{emoji} {stress_level}</div>
        <div class="metric-delta">Satellite-derived indicator</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    trend = "Stable"
    if analysis.ndvi_trend:
        trend = analysis.ndvi_trend.value.capitalize()
    trend_emoji = {"Improving": "📈", "Stable": "➡️", "Declining": "📉"}.get(trend, "➡️")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Health Trend</div>
        <div class="metric-value">{trend_emoji} {trend}</div>
        <div class="metric-delta">Based on NDVI trajectory</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts ─────────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.plotly_chart(create_ndvi_chart(analysis.recent_observations), use_container_width=True)

with chart_col2:
    st.plotly_chart(create_sar_chart(analysis.recent_observations), use_container_width=True)

# ── Satellite Observations Table ───────────────────────────────────────
with st.expander("📋 Raw Satellite Observations", expanded=False):
    if analysis.recent_observations:
        obs_data = []
        for o in sorted(analysis.recent_observations, key=lambda x: x.observation_date, reverse=True):
            row = {
                "Date": str(o.observation_date),
                "Satellite": o.satellite,
                "NDVI": f"{o.ndvi:.4f}" if o.ndvi is not None else "—",
                "VV (dB)": f"{o.vv:.2f}" if o.vv is not None else "—",
                "VH (dB)": f"{o.vh:.2f}" if o.vh is not None else "—",
                "Cloud %": f"{o.cloud_cover:.1f}" if o.cloud_cover is not None else "—",
            }
            obs_data.append(row)
        st.dataframe(pd.DataFrame(obs_data), use_container_width=True, hide_index=True)
    else:
        st.info("No satellite observations available.")

# ── Feature Importance ─────────────────────────────────────────────────
if analysis.crop_prediction and analysis.crop_prediction.feature_importance:
    with st.expander("🔬 Feature Importance (Crop Classifier)", expanded=False):
        fi = analysis.crop_prediction.feature_importance
        fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        fi_df = pd.DataFrame(fi_sorted, columns=["Feature", "Importance"])
        st.bar_chart(fi_df.set_index("Feature"), height=300)

# ═══════════════════════════════════════════════════════════════════════════
# Gemini AI Advisory
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("## 🤖 AgriN AI Crop Advisory")

tab_en, tab_hi, tab_qa = st.tabs(["🇬🇧 English Advisory", "🇮🇳 हिन्दी Advisory", "❓ Ask AgriN"])

with tab_en:
    if st.button("Generate English Advisory", key="gen_en"):
        with st.spinner("🤖 Generating advisory with Gemini AI..."):
            advisory = generate_advisory(analysis, language="en")
            st.session_state["advisory_en"] = advisory

    if "advisory_en" in st.session_state:
        adv = st.session_state["advisory_en"]
        st.markdown(f"""
        <div class="advisory-card">
            <h3>🌾 Farm Advisory</h3>
            {adv.advisory_text}
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"Model: {adv.model_version} · Source: {adv.data_source.value.upper()}")

with tab_hi:
    if st.button("हिन्दी सलाह प्राप्त करें", key="gen_hi"):
        with st.spinner("🤖 Gemini AI से सलाह तैयार हो रही है..."):
            advisory = generate_advisory(analysis, language="hi")
            st.session_state["advisory_hi"] = advisory

    if "advisory_hi" in st.session_state:
        adv = st.session_state["advisory_hi"]
        st.markdown(f"""
        <div class="advisory-card">
            <h3>🌾 कृषि सलाह</h3>
            {adv.advisory_text}
        </div>
        """, unsafe_allow_html=True)

with tab_qa:
    st.markdown("**Ask any question about your crop and satellite data:**")
    question = st.text_input(
        "Your question",
        placeholder="e.g., Is my crop healthy? Should I irrigate?",
        key="qa_input",
    )
    if st.button("Ask AgriN 🌾", key="qa_btn") and question:
        with st.spinner("🤖 Thinking..."):
            answer = ask_question(question, analysis, language=lang_code)
            st.session_state["qa_answer"] = answer

    if "qa_answer" in st.session_state:
        st.markdown(f"""
        <div class="advisory-card">
            <h3>🌾 AgriN Response</h3>
            {st.session_state['qa_answer']}
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #9ca3af; font-size: 0.8rem; padding: 1rem 0;'>
    <strong>AgriN — AI-Powered Smart Crop Intelligence</strong><br>
    Google Earth Engine · Sentinel-2 · Sentinel-1 · Random Forest · Google Gemini AI<br>
    <em>Satellite indicators alone cannot replace in-field crop inspection. Please consult local agricultural experts for treatment decisions.</em>
</div>
""", unsafe_allow_html=True)
