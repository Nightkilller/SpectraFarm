"""
AgriN — Streamlit Dashboard

Main entry point for the AgriN dashboard.
Provides: Farm Overview, Crop Map, Stress Map, NDVI Timeline,
AI Advisory, and Ask AgriN Q&A.
"""

import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.gemini_client import ask_question, generate_advisory, is_gemini_available
from src.config.settings import get_settings
from src.data.demo_data import (
    generate_demo_crop_map_data,
    generate_demo_stress_map_data,
    get_demo_advisory,
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
# Custom CSS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #0f4c3a 0%, #1a7a5c 50%, #2ecc71 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(15, 76, 58, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* Demo banner */
    .demo-banner {
        background: linear-gradient(90deg, #ff9800, #ff5722);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border: 1px solid #e9ecef;
        border-radius: 14px;
        padding: 1.3rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #6c757d;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.3rem 0;
        color: #212529;
    }

    /* Status colors */
    .status-healthy { color: #27ae60; }
    .status-mild { color: #f1c40f; }
    .status-moderate { color: #e67e22; }
    .status-severe { color: #e74c3c; }
    .trend-improving { color: #27ae60; }
    .trend-stable { color: #3498db; }
    .trend-declining { color: #e74c3c; }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 3px solid #2ecc71;
        display: inline-block;
    }

    /* Advisory card */
    .advisory-card {
        background: linear-gradient(145deg, #f0faf4, #e8f5e9);
        border: 1px solid #c8e6c9;
        border-radius: 14px;
        padding: 1.5rem;
        line-height: 1.7;
    }

    /* Q&A section */
    .qa-response {
        background: linear-gradient(145deg, #f3f0ff, #ede7f6);
        border: 1px solid #d1c4e9;
        border-radius: 14px;
        padding: 1.5rem;
        line-height: 1.7;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f4c3a, #1a3a2a);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: #b0b0b0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Initialize State
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_analyzer():
    return FarmAnalyzer()


def get_analysis():
    """Get or compute the farm analysis (cached in session state)."""
    if "analysis" not in st.session_state:
        analyzer = get_analyzer()
        st.session_state.analysis = analyzer.analyze()
    return st.session_state.analysis


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🌾 AgriN")
    st.markdown("*Smart Crop Intelligence*")
    st.markdown("---")

    settings = get_settings()

    # Language selection
    lang_options = {l["code"]: l["name"] for l in settings.languages}
    selected_lang = st.selectbox(
        "🌐 Language",
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        key="language",
    )

    st.markdown("---")

    # Farm selection
    st.markdown("### 📍 Farm Selection")
    farm = get_demo_farm()
    st.markdown(f"**{farm.name}**")
    st.markdown(f"📌 {farm.latitude}°N, {farm.longitude}°E")
    st.markdown(f"📐 {farm.area_ha} hectares")
    st.markdown(f"🌱 Season: {farm.season.capitalize() if farm.season else 'N/A'}")

    st.markdown("---")

    # Status indicators
    st.markdown("### ⚙️ System Status")
    mode = settings.mode.upper()
    st.markdown(f"**Mode:** `{mode}`")

    gee_status = "🟢 Connected" if not settings.is_demo else "🟡 Demo"
    gemini_status = "🟢 Connected" if is_gemini_available() else "🟡 Demo"
    st.markdown(f"**Satellite:** {gee_status}")
    st.markdown(f"**Gemini AI:** {gemini_status}")

    st.markdown("---")
    if st.button("🔄 Refresh Analysis", use_container_width=True):
        st.session_state.pop("analysis", None)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# Main Content
# ═══════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>🌾 AgriN — Smart Crop Intelligence</h1>
    <p>AI-powered satellite monitoring & farmer advisory</p>
</div>
""", unsafe_allow_html=True)

# Get analysis
analysis = get_analysis()

# Demo banner
if analysis.data_source == DataSource.DEMO:
    st.markdown(
        '<div class="demo-banner">🔶 DEMO DATA — Results shown are simulated '
        'for demonstration purposes. Connect real services for live data.</div>',
        unsafe_allow_html=True,
    )

# ── Farm Overview Metrics ────────────────────────────────────────────────

st.markdown('<div class="section-header">📊 Farm Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

# Crop
crop_name = analysis.crop_prediction.predicted_crop.value.capitalize() if analysis.crop_prediction else "—"
crop_conf = f"{analysis.crop_prediction.confidence:.0%}" if analysis.crop_prediction else "—"
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Detected Crop</div>
        <div class="metric-value">🌾 {crop_name}</div>
        <div style="font-size:0.8rem;color:#888;">Confidence: {crop_conf}</div>
    </div>
    """, unsafe_allow_html=True)

# Health / Stress
stress_level = analysis.stress_assessment.stress_level.value if analysis.stress_assessment else "unknown"
stress_class = f"status-{stress_level}"
stress_emoji = {"healthy": "🟢", "mild": "🟡", "moderate": "🟠", "severe": "🔴"}.get(stress_level, "⚪")
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Crop Health</div>
        <div class="metric-value {stress_class}">{stress_emoji} {stress_level.capitalize()}</div>
        <div style="font-size:0.8rem;color:#888;">Satellite-based indicator</div>
    </div>
    """, unsafe_allow_html=True)

# NDVI
ndvi_val = f"{analysis.ndvi_current:.3f}" if analysis.ndvi_current else "—"
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current NDVI</div>
        <div class="metric-value">{ndvi_val}</div>
        <div style="font-size:0.8rem;color:#888;">Vegetation health index</div>
    </div>
    """, unsafe_allow_html=True)

# Trend
trend_val = analysis.ndvi_trend.value if analysis.ndvi_trend else "stable"
trend_class = f"trend-{trend_val}"
trend_emoji = {"improving": "📈", "stable": "➡️", "declining": "📉"}.get(trend_val, "➡️")
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Health Trend</div>
        <div class="metric-value {trend_class}">{trend_emoji} {trend_val.capitalize()}</div>
        <div style="font-size:0.8rem;color:#888;">Over recent observations</div>
    </div>
    """, unsafe_allow_html=True)

# Observation date
obs_date = str(analysis.observation_date) if analysis.observation_date else "—"
with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Last Observation</div>
        <div class="metric-value" style="font-size:1.1rem;">{obs_date}</div>
        <div style="font-size:0.8rem;color:#888;">Satellite pass date</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Maps & NDVI Chart ────────────────────────────────────────────────────

map_tab, stress_tab, ndvi_tab = st.tabs([
    "🗺️ Crop Classification Map",
    "💧 Moisture Stress Map",
    "📈 NDVI Timeline",
])

# --- Crop Map ---
with map_tab:
    crop_colors = {"wheat": "#f4a460", "rice": "#2e8b57", "other": "#b0c4de"}

    crop_map_data = generate_demo_crop_map_data(
        center_lat=analysis.farm.latitude,
        center_lon=analysis.farm.longitude,
    )

    m = folium.Map(
        location=[analysis.farm.latitude, analysis.farm.longitude],
        zoom_start=15,
        tiles="CartoDB positron",
    )

    for cell in crop_map_data:
        folium.CircleMarker(
            location=[cell["lat"], cell["lon"]],
            radius=5,
            color=crop_colors.get(cell["crop"], "#999"),
            fill=True,
            fill_color=crop_colors.get(cell["crop"], "#999"),
            fill_opacity=0.7,
            popup=f"{cell['crop'].capitalize()}: {cell['confidence']:.0%}",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
         background:white;padding:10px 14px;border-radius:8px;
         box-shadow:0 2px 8px rgba(0,0,0,0.15);font-size:13px;">
        <b>Crop Classification</b><br>
        <span style="color:#f4a460;">⬤</span> Wheat &nbsp;
        <span style="color:#2e8b57;">⬤</span> Rice &nbsp;
        <span style="color:#b0c4de;">⬤</span> Other
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=None, height=450, key="crop_map")

# --- Stress Map ---
with stress_tab:
    stress_colors = {
        "healthy": "#27ae60",
        "mild": "#f1c40f",
        "moderate": "#e67e22",
        "severe": "#e74c3c",
    }

    stress_map_data = generate_demo_stress_map_data(
        center_lat=analysis.farm.latitude,
        center_lon=analysis.farm.longitude,
    )

    m2 = folium.Map(
        location=[analysis.farm.latitude, analysis.farm.longitude],
        zoom_start=15,
        tiles="CartoDB positron",
    )

    for cell in stress_map_data:
        folium.CircleMarker(
            location=[cell["lat"], cell["lon"]],
            radius=5,
            color=stress_colors.get(cell["stress_level"], "#999"),
            fill=True,
            fill_color=stress_colors.get(cell["stress_level"], "#999"),
            fill_opacity=0.7,
            popup=f"Stress: {cell['stress_level'].capitalize()} ({cell['indicator']:.2f})",
        ).add_to(m2)

    legend_html2 = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
         background:white;padding:10px 14px;border-radius:8px;
         box-shadow:0 2px 8px rgba(0,0,0,0.15);font-size:13px;">
        <b>Stress Level</b><br>
        <span style="color:#27ae60;">⬤</span> Healthy &nbsp;
        <span style="color:#f1c40f;">⬤</span> Mild &nbsp;
        <span style="color:#e67e22;">⬤</span> Moderate &nbsp;
        <span style="color:#e74c3c;">⬤</span> Severe
    </div>
    """
    m2.get_root().html.add_child(folium.Element(legend_html2))

    st_folium(m2, width=None, height=450, key="stress_map")

# --- NDVI Timeline ---
with ndvi_tab:
    s2_obs = [
        o for o in analysis.recent_observations
        if o.satellite == "Sentinel-2" and o.ndvi is not None
    ]

    if s2_obs:
        df = pd.DataFrame([
            {
                "Date": o.observation_date,
                "NDVI": o.ndvi,
                "NDWI": o.ndwi,
            }
            for o in sorted(s2_obs, key=lambda x: x.observation_date)
        ])

        fig = go.Figure()

        # NDVI line
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["NDVI"],
            name="NDVI",
            mode="lines+markers",
            line=dict(color="#27ae60", width=3),
            marker=dict(size=8, color="#27ae60"),
            fill="tozeroy",
            fillcolor="rgba(39, 174, 96, 0.1)",
        ))

        # NDWI line
        if df["NDWI"].notna().any():
            fig.add_trace(go.Scatter(
                x=df["Date"],
                y=df["NDWI"],
                name="NDWI",
                mode="lines+markers",
                line=dict(color="#3498db", width=2, dash="dash"),
                marker=dict(size=6, color="#3498db"),
            ))

        # Threshold lines
        settings = get_settings()
        fig.add_hline(
            y=settings.ndvi_thresholds["dense_vegetation_min"],
            line_dash="dot",
            line_color="#888",
            annotation_text="Dense Veg.",
            annotation_position="right",
        )
        fig.add_hline(
            y=settings.ndvi_thresholds["bare_soil_max"],
            line_dash="dot",
            line_color="#ccc",
            annotation_text="Bare Soil",
            annotation_position="right",
        )

        fig.update_layout(
            title=dict(
                text="Vegetation Health Over Time",
                font=dict(size=18, family="Inter"),
            ),
            xaxis_title="Observation Date",
            yaxis_title="Index Value",
            yaxis=dict(range=[-0.1, 1.0]),
            template="plotly_white",
            height=420,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=50, r=30, t=60, b=50),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No NDVI observations available.")

# ── Feature Importance ───────────────────────────────────────────────────

if analysis.crop_prediction and analysis.crop_prediction.feature_importance:
    with st.expander("🔍 Model Explainability — Feature Importance", expanded=False):
        imp = analysis.crop_prediction.feature_importance
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_imp]
        values = [x[1] for x in sorted_imp]

        fig_imp = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color="#2ecc71",
        ))
        fig_imp.update_layout(
            title="Random Forest Feature Importance",
            xaxis_title="Importance",
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            height=350,
            margin=dict(l=120, r=30, t=50, b=40),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ── AI Advisory ──────────────────────────────────────────────────────────

st.markdown("---")
st.markdown('<div class="section-header">🤖 AI Farm Advisory</div>', unsafe_allow_html=True)

lang = st.session_state.get("language", "en")

# Generate advisory
if "advisory" not in st.session_state or st.session_state.get("advisory_lang") != lang:
    advisory = generate_advisory(analysis, language=lang)
    st.session_state.advisory = advisory
    st.session_state.advisory_lang = lang

advisory = st.session_state.advisory

if advisory.data_source == DataSource.DEMO:
    st.caption("🔶 Demo advisory — connect Gemini API for AI-generated responses")

st.markdown(
    f'<div class="advisory-card">{advisory.advisory_text}</div>',
    unsafe_allow_html=True,
)

# ── Ask AgriN ────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown('<div class="section-header">💬 Ask AgriN</div>', unsafe_allow_html=True)

# Example questions
if lang == "hi":
    examples = [
        "मेरी फसल की स्थिति कैसी है?",
        "NDVI का क्या मतलब है?",
        "मेरी फसल में तनाव क्यों दिख रहा है?",
        "क्या मुझे सिंचाई करनी चाहिए?",
    ]
else:
    examples = [
        "How healthy is my crop?",
        "What does NDVI mean?",
        "Why is my crop stressed?",
        "Should I irrigate my field?",
        "Is my crop improving?",
    ]

st.caption("Example questions:")
example_cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    with example_cols[i]:
        if st.button(ex, key=f"example_{i}", use_container_width=True):
            st.session_state.user_question = ex

# Text input
user_question = st.text_input(
    "Ask a question about your farm...",
    value=st.session_state.get("user_question", ""),
    key="qa_input",
    placeholder="e.g., What is happening to my crop?" if lang == "en"
    else "उदाहरण: मेरी फसल को क्या हो रहा है?",
)

if user_question:
    with st.spinner("🤔 Thinking..." if lang == "en" else "🤔 सोच रहा हूँ..."):
        response = ask_question(user_question, analysis, language=lang)

    st.markdown(
        f'<div class="qa-response">{response}</div>',
        unsafe_allow_html=True,
    )

# ── Footer ───────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    """<div style="text-align:center;color:#999;font-size:0.8rem;padding:1rem;">
    🌾 AgriN v0.1.0 — AI-Powered Smart Crop Intelligence<br>
    Satellite indicators are not a substitute for field verification.
    Consult local agricultural experts for specific decisions.
    </div>""",
    unsafe_allow_html=True,
)
