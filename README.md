# SpectraFarm — AgriN Satellite Crop Intelligence Platform

> Real-time crop monitoring and agronomic advisory powered by satellite remote sensing, machine learning, and AI.

SpectraFarm (AgriN) is an end-to-end agricultural intelligence platform that combines **Sentinel-2 optical imagery**, **Sentinel-1 SAR radar data** via **Google Earth Engine**, a **Random Forest crop classifier**, and **Groq/Gemini AI** to deliver actionable crop health insights and irrigation guidance to farmers — all through a single interactive dashboard.

---

## What It Does

1. **Identifies the crop** growing at any coordinate in India using a trained ML classifier on multi-temporal satellite features.
2. **Monitors crop health** in near-real-time via NDVI (vegetation greenness), VCI (moisture stress), and SAR radar backscatter.
3. **Detects stress** — drought, waterlogging, pest damage — by fusing optical and radar signals with agronomic thresholds.
4. **Recommends irrigation** timing, depth, and priority based on current soil moisture and canopy condition.
5. **Provides AI advisory** — an interactive AgriN AI Assistant (powered by Groq LPU / Google Gemini) answers agronomic questions grounded in live satellite telemetry.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Streamlit Dashboard (app.py)            │
│   Map · Telemetry · Charts · AI Assistant Chat      │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│          Agricultural Intelligence Layer             │
│   Combines all data into structured FarmAnalysis     │
└────┬──────────┬──────────┬──────────┬───────────────┘
     │          │          │          │
┌────▼────┐ ┌──▼────┐ ┌───▼───┐ ┌───▼──────────┐
│Satellite│ │  ML   │ │Feature│ │   AI/LLM     │
│ Service │ │Classif│ │Fusion │ │  Advisory    │
└────┬────┘ └──┬────┘ └───┬───┘ └───┬──────────┘
     │         │          │         │
┌────▼────┐ ┌──▼─────┐ ┌─▼──┐ ┌───▼──────────┐
│ Google  │ │scikit- │ │NumPy│ │ Groq LPU /   │
│ Earth   │ │ learn  │ │     │ │ Google Gemini│
│ Engine  │ │        │ │     │ │              │
└─────────┘ └────────┘ └────┘ └──────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Dashboard** | Streamlit, Plotly, Folium (Leaflet.js) |
| **Satellite Data** | Google Earth Engine (Sentinel-2 Optical + Sentinel-1 SAR) |
| **ML Classifier** | scikit-learn Random Forest (trained on AgriFieldNet public dataset) |
| **Feature Engineering** | NumPy, multi-temporal NDVI + SAR backscatter fusion |
| **AI Advisory** | Groq LPU (Compound-Mini / Qwen) + Google Gemini as fallback |
| **Data Validation** | Pydantic schemas, spatial blocking, provenance tracking |
| **Testing** | pytest (78 automated tests) |
| **Language** | Python 3.11+ |

---

## Key Features

- **Live Satellite Feed** — Real Sentinel-2 and Sentinel-1 data from Google Earth Engine, with automatic fallback to synthetic demo data when GEE is unavailable.
- **Interactive Crop Map** — Folium-based satellite map with ESA WorldCover land-use validation, 5x5 agricultural parcel grid, and Google Hybrid / Esri high-res basemaps.
- **Multi-Sensor Fusion** — Combines optical NDVI time-series with SAR radar VV/VH backscatter for robust crop identification even under cloud cover.
- **Stress Detection** — Vegetation Condition Index (VCI), SAR-derived soil moisture proxies, and configurable agronomic thresholds.
- **Irrigation Guidance** — Calculates recommended irrigation depth, pump duration, and water volume based on field size and stress level.
- **AI Copilot** — Ask questions about your crop in natural language; responses are grounded in live satellite measurements (not hallucinated).
- **Bilingual** — English and Hindi advisory support.
- **13 Pre-configured Regions** — Agricultural presets across UP, MP, Rajasthan, Bihar, Punjab, Maharashtra.

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/agriN.git
cd agriN

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your keys:

```env
GEE_PROJECT=your-gee-project-id
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
AGRIN_MODE=live
```

| Variable | Required | Purpose |
|---|---|---|
| `GEE_PROJECT` | Yes (for live data) | Google Earth Engine cloud project ID |
| `GEMINI_API_KEY` | Optional | Google Gemini AI fallback |
| `GROQ_API_KEY` | Yes (for AI chat) | Groq LPU for fast AI responses |
| `AGRIN_MODE` | Optional | `live` (default) or `demo` |

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

> **Note:** If Google Earth Engine is not configured, the app automatically falls back to synthetic demo data — no setup required for a quick demo.

---

## Project Structure

```
agriN/
├── app.py                          # Main Streamlit dashboard
├── config/
│   ├── settings.yaml               # Pilot AOI, satellite config, ML regions
│   ├── thresholds.yaml             # NDVI/VCI stress thresholds
│   └── crops.yaml                  # Crop classes & growing seasons
├── src/
│   ├── ai/
│   │   ├── gemini_client.py        # Groq/Gemini AI client & question handler
│   │   └── prompts.py              # System prompts for agronomic advisory
│   ├── config/
│   │   └── settings.py             # Central configuration loader
│   ├── data/
│   │   ├── schemas.py              # Pydantic models (Farm, FarmAnalysis, etc.)
│   │   ├── satellite_data.py       # Unified satellite data acquisition layer
│   │   ├── demo_data.py            # Synthetic demo data generator
│   │   ├── ground_truth_validator.py  # Spatial validation & provenance
│   │   └── load_ground_truth.py    # Ground truth ingestion pipeline
│   ├── features/
│   │   ├── feature_extraction.py   # Statistical feature extractors
│   │   └── fusion.py               # Optical + SAR multi-sensor fusion
│   ├── geospatial/
│   │   ├── gee_client.py           # Earth Engine initialization & queries
│   │   ├── indices.py              # Vegetation indices (NDVI, NDWI)
│   │   ├── timeseries.py           # Multi-temporal Sentinel-2 extraction
│   │   ├── sar.py                  # Multi-temporal Sentinel-1 SAR extraction
│   │   └── state_boundaries.py     # India state boundary lookups
│   ├── intelligence/
│   │   ├── farm_analyzer.py        # Combines all data into FarmAnalysis
│   │   └── stress_analysis.py      # Stress detection & VCI calculation
│   └── ml/
│       ├── crop_classifier.py      # Random Forest crop classification
│       └── train_heavy_classifier.py  # Training pipeline (AgriFieldNet)
├── tests/                          # 78 automated pytest tests
├── scripts/                        # Verification & training scripts
├── notebooks/                      # Jupyter/Colab training notebooks
├── docs/
│   ├── architecture.md             # System architecture & design
│   ├── methodology.md              # Remote sensing methodology
│   ├── assumptions.md              # Operational assumptions & limits
│   └── deployment.md               # Deployment guide
├── data/
│   └── ground_truth/               # Ground truth templates & docs
├── models/
│   └── crop_classifier/            # Trained model artifacts (gitignored)
├── .streamlit/config.toml          # Streamlit theme configuration
├── .env.example                    # Environment variable template
├── requirements.txt                # Python dependencies
└── pyproject.toml                  # Project metadata & pytest config
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 78 tests run offline without any API keys or GEE access.

---

## Data Sources

| Source | Sensor | Data Used |
|---|---|---|
| **Copernicus Sentinel-2** | Optical (10m) | NDVI, NDWI, canopy greenness |
| **Copernicus Sentinel-1** | SAR C-band (10m) | VV/VH backscatter, soil moisture proxy |
| **ESA WorldCover** | Land use (10m) | Cropland validation mask |
| **AgriFieldNet** | Ground truth | Crop labels for ML training (CC-BY-4.0) |

---

## ML Training Dataset

The crop classifier is trained on the **AgriFieldNet Competition Dataset** by Radiant Earth Foundation / IDinsight:
- **DOI:** `10.34911/rdnt.wu92p1`
- **License:** CC-BY-4.0
- **Coverage:** 7,081 labeled agricultural fields across Uttar Pradesh, Bihar, Rajasthan, and Odisha
- **Crops:** Wheat, Rice, Maize, Cotton, Sugarcane, Soybean, Groundnut, Vegetables

---

## License

MIT
