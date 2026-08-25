# AgriN — AI-Powered Smart Crop Intelligence

> *"What is happening to my crop, and what should I consider doing about it?"*

AgriN combines **satellite remote sensing (Sentinel-2/Sentinel-1 via Google Earth Engine)**, **machine learning**, and **Google Gemini AI** to deliver farmer-friendly crop intelligence.

---

## Current Status: Phase 1 Complete ✅

- **Google Earth Engine Integration**: Connected and operational with Google Cloud Project `agrin-506618`.
- **Pilot Region Foundation**: **Sehore Pilot Test AOI, Madhya Pradesh, India** (`23.20°N, 77.08°E`).
- **Satellite Data Stream**: Real Sentinel-2 Surface Reflectance Harmonized (`COPERNICUS/S2_SR_HARMONIZED`) imagery verified.
- **Spectral Index Calculation**: Real-time NDVI calculation using Sentinel-2 B8 (NIR) and B4 (Red) with regional statistical aggregation (Min, Mean, Max, StdDev).

---

## Quick Start & Verification

### 1. Environment Setup

```bash
# Clone the repository
cd agriN

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` based on `.env.example`:
```env
GEE_PROJECT=agrin-506618
GEMINI_API_KEY=your_key_here
AGRIN_MODE=live
```

### 3. Run the Phase 1 Real Sentinel-2 Sehore Test

Execute the verified end-to-end satellite pipeline:

```bash
python scripts/test_sehore_sentinel2.py
```

#### Expected Output:
```text
======================================================================
AgriN — Phase 1: Real Sentinel-2 + Sehore Earth Engine Verification
======================================================================

[INFO] Initializing Google Earth Engine...
[INFO] Project: agrin-506618
[INFO] Earth Engine connection established successfully.

[INFO] Loading Pilot AOI: Sehore Pilot Test AOI
[INFO] Location: Sehore, Madhya Pradesh, India
[INFO] Center Coordinates: 23.2°N, 77.08°E (Buffer: 2000m)

[INFO] Querying Sentinel-2 Surface Reflectance (COPERNICUS/S2_SR_HARMONIZED)
[INFO] Temporal Filter: Recent observations (< 20% cloudy pixels)
[INFO] Sentinel-2 observations found: 25

--------------------------------------------------
Selected Sentinel-2 Image Metadata
--------------------------------------------------
  Image ID:          COPERNICUS/S2_SR_HARMONIZED/...
  Acquisition Date:  2026-03-11
  Cloud Percentage:  0.00%
  Spacecraft:        Sentinel-2C
  Spectral Bands:    B4 (Red, 665nm), B8 (NIR, 842nm), B2, B3, B11, B12

[INFO] Computing Normalized Difference Vegetation Index (NDVI)...
       Formula: NDVI = (B8 - B4) / (B8 + B4)
[INFO] Calculating regional NDVI statistics over Sehore AOI at 10m resolution...
--------------------------------------------------
Sehore AOI — Real Satellite NDVI Statistics
--------------------------------------------------
  Minimum NDVI:      -0.2596
  Mean NDVI:         0.2956
  Maximum NDVI:      0.9685
  Std Deviation:     0.1706
--------------------------------------------------

✅ Validation PASSED: NDVI mean is within theoretical physical range [-1.0, +1.0].
✅ Real satellite data pipeline verified from Earth Engine to AgriN.
```

---

## Test Suite

Run the full automated test suite:

```bash
pytest tests/ -v
```

---

## Project Structure

```
agriN/
├── config/
│   ├── settings.yaml        # Sehore pilot AOI, satellite collections, thresholds
│   ├── thresholds.yaml      # Configurable NDVI and stress thresholds
│   └── crops.yaml           # Crop classes & seasons
├── src/
│   ├── config/              # Central configuration loader
│   ├── data/                # Data schemas (Pydantic models)
│   ├── geospatial/
│   │   ├── gee_client.py    # Earth Engine initialization & collection filtering
│   │   └── indices.py       # Reusable spectral indices (NDVI, NDWI)
│   ├── features/            # Feature extraction modules
│   └── ai/                  # Gemini advisory integration
├── scripts/
│   └── test_sehore_sentinel2.py  # Phase 1 verification script
├── docs/
│   ├── architecture.md      # System architecture & service boundaries
│   ├── assumptions.md       # Assumptions & operational limits
│   └── methodology.md       # Scientific remote sensing & index formulations
├── tests/                   # Automated pytest suite
├── .env.example
├── requirements.txt
└── README.md
```

---

## License

MIT
