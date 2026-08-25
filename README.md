# AgriN — AI-Powered Smart Crop Intelligence

> *"What is happening to my crop, and what should I consider doing about it?"*

AgriN combines **satellite remote sensing**, **machine learning**, and **Google Gemini AI** to deliver farmer-friendly crop intelligence. It transforms complex geospatial data into actionable advisories that any farmer can understand.

---

## What It Does

| Capability | Description |
|---|---|
| 🛰️ **Satellite Monitoring** | Sentinel-2 (optical) and Sentinel-1 (SAR/radar) imagery via Google Earth Engine |
| 🌾 **Crop Classification** | Random Forest classifier — Wheat / Rice / Other |
| 📊 **NDVI Health Tracking** | Vegetation index time series with trend detection |
| 💧 **Stress Detection** | Satellite-based moisture/crop stress indicator (Healthy → Severe) |
| 🤖 **AI Advisory** | Google Gemini interprets agricultural measurements into plain-language advice |
| 💬 **Ask AgriN** | Farmers ask questions in English or Hindi, get contextual answers |
| 🌍 **Multilingual** | English + Hindi (extensible to more Indian languages) |

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- (Optional) Google Earth Engine account — for real satellite data
- (Optional) Gemini API key — for AI advisory

### Setup

```bash
# Clone and enter the project
cd agriN

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your API keys (or leave empty for demo mode)

# Run tests
pytest tests/

# Launch dashboard (Phase 1+)
# streamlit run dashboard/app.py
```

### Demo Mode

AgriN runs in **demo mode** by default — no API keys or external services needed. All demo data is clearly labeled in the UI.

Set `AGRIN_MODE=live` in `.env` when you have real credentials configured.

---

## Architecture

```
Satellite Data (Sentinel-1/2 via GEE)
    ↓
Geospatial Processing (cloud filtering, compositing)
    ↓
Feature Extraction (NDVI, NDWI, SAR features)
    ↓
Machine Learning (Random Forest crop classifier)
    ↓
Agricultural Intelligence (structured FarmAnalysis)
    ↓
Google Gemini AI (contextual explanation)
    ↓
Farmer Advisory + Q&A (Dashboard)
```

**Key principle:** ML and geospatial processing determine the measurements. Gemini explains and contextualizes them — it never fabricates scientific data.

---

## Project Structure

```
agriN/
├── config/            # YAML configuration (thresholds, crops, settings)
├── src/
│   ├── config/        # Config loader
│   ├── data/          # Pydantic schemas
│   ├── geospatial/    # GEE client, preprocessing, indices
│   ├── features/      # Feature extraction
│   ├── ml/            # Crop classifier, stress detector
│   ├── intelligence/  # Agricultural analysis layer
│   └── ai/            # Gemini client, prompts, advisory
├── dashboard/         # Streamlit UI
├── data/demo/         # Demo datasets
├── models/            # Trained model files
├── tests/             # Test suite
├── docs/              # Documentation
└── notebooks/         # Exploration notebooks
```

---

## Development Phases

| Phase | Focus | Status |
|---|---|---|
| **0** | Project setup, schemas, config | ✅ Complete |
| **1** | Dashboard skeleton with demo data | ⬜ Next |
| **2** | Real satellite data via GEE | ⬜ |
| **3** | Crop classification (Random Forest) | ⬜ |
| **4** | Stress detection | ⬜ |
| **5** | Agricultural intelligence layer | ⬜ |
| **6** | Gemini AI integration | ⬜ |
| **7** | Weather context | ⬜ |
| **8** | Multilingual (Hindi) | ⬜ |
| **9** | Disease detection (future) | ⬜ |
| **10** | Polish & documentation | ⬜ |

---

## Configuration

All thresholds and parameters are in `config/`:

- **`settings.yaml`** — App settings, pilot region, satellite config
- **`thresholds.yaml`** — NDVI, stress, SAR thresholds (configurable)
- **`crops.yaml`** — Crop classes, colors, season mapping

---

## Pilot Region

**Ludhiana, Punjab** (~30.9°N, 75.85°E) — a major wheat/rice agricultural belt with good Sentinel coverage.

---

## Important Notes

- 🔶 **Data honesty**: Demo data is always explicitly labeled. No fake measurements are presented as real.
- ⚠️ **Not a diagnosis tool**: Satellite indicators are not a substitute for field verification.
- 🔒 **Security**: API keys stored in `.env` only — never in source code.

---

## License

MIT
