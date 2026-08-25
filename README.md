# AgriN — AI-Powered Smart Crop Intelligence

> *"What is happening to my crop, and what should I consider doing about it?"*

AgriN combines **satellite remote sensing (Sentinel-2/Sentinel-1 via Google Earth Engine)**, **machine learning**, and **Google Gemini AI** to deliver farmer-friendly crop intelligence.

---

## Current Status: Phase 2 Complete ✅

- **Google Earth Engine Integration**: Connected and operational with Google Cloud Project `agrin-506618`.
- **Pilot Region Foundation**: **Sehore Pilot Test AOI, Madhya Pradesh, India** (`23.20°N, 77.08°E`).
- **Sentinel-2 Multi-Temporal Pipeline**: Real multi-temporal NDVI trajectory extracted across chronological Sentinel-2 Surface Reflectance observations.
- **Server-Side Reduction**: Earth Engine statistical reductions (min, mean, max, stdDev) executed at native 10m pixel resolution without client raster downloads.
- **Data Integrity**: 100% real satellite data. No synthetic/demo data used in the real-mode time series.

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

---

## Multi-Temporal Sentinel-2 Verification (Phase 2)

Execute the verified multi-temporal satellite pipeline over the Sehore pilot AOI:

```bash
python scripts/test_sehore_timeseries.py
```

### Sample Output:
```text
================================================================================
AgriN — Phase 2: Multi-Temporal Sentinel-2 NDVI Time Series Pipeline
================================================================================

[INFO] Initializing Earth Engine with project: agrin-506618
[INFO] Earth Engine connection established successfully.
[INFO] Loading Pilot AOI: Sehore Pilot Test AOI (23.2°N, 77.08°E, Buffer: 2000m)
[INFO] Collection: COPERNICUS/S2_SR_HARMONIZED
[INFO] Querying and computing multi-temporal NDVI statistics over Sehore AOI...
[INFO] Successfully retrieved 23 real Sentinel-2 observations.

--------------------------------------------------------------------------------
Date         | Cloud %  | Min NDVI  | Mean NDVI  | Max NDVI  | StdDev   | Image ID            
--------------------------------------------------------------------------------
2026-03-01   |   0.00% |   -0.0911 |     0.3580 |    0.9436 |   0.2076 | 20260301T051741_20...
2026-03-06   |   0.00% |   -0.1659 |     0.3260 |    0.9544 |   0.1831 | 20260306T051649_20...
2026-03-11   |   0.00% |   -0.2596 |     0.2956 |    0.9685 |   0.1706 | 20260311T051651_20...
...
2026-06-09   |   0.59% |   -0.0657 |     0.1901 |    0.8321 |   0.1231 | 20260609T051651_20...
--------------------------------------------------------------------------------

================================================================================
PHASE 2 MULTI-TEMPORAL SENTINEL-2 NDVI SUMMARY
================================================================================
  Target Pilot AOI:          Sehore Pilot Test AOI (Sehore, MP)
  Date Range Filter:         Past 180 Days (2026-02-27 to 2026-08-26)
  Cloud Cover Threshold:     < 20%
  Total Valid Observations:  23
  First Observation Date:    2026-03-01 (Mean NDVI: 0.3580)
  Last Observation Date:     2026-06-09 (Mean NDVI: 0.1901)
  Temporal Trajectory Range: Mean NDVI spans from 0.1648 to 0.3580
  Overall Time Series Mean:  0.2315
  Data Origin:               100% Live Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED)
================================================================================
```

---

## Test Suite

Run the full automated offline test suite (57 tests passing):

```bash
pytest tests/ -v
```

---

## Project Structure

```
agriN/
├── config/
│   ├── settings.yaml        # Sehore pilot AOI, satellite collections, thresholds
│   ├── thresholds.yaml      # Guarded uncalibrated thresholds (placeholders)
│   └── crops.yaml           # Crop classes & seasons
├── src/
│   ├── config/              # Central configuration loader
│   ├── data/                # Data schemas (Pydantic models)
│   ├── geospatial/
│   │   ├── gee_client.py    # Earth Engine initialization & collection filtering
│   │   ├── indices.py       # Reusable spectral indices (NDVI, NDWI)
│   │   └── timeseries.py    # Multi-temporal NDVI extraction & reducers
│   ├── features/            # Feature extraction modules
│   └── ai/                  # Gemini advisory integration
├── scripts/
│   ├── test_sehore_sentinel2.py   # Phase 1 single-image test
│   └── test_sehore_timeseries.py  # Phase 2 multi-temporal time series test
├── docs/
│   ├── architecture.md      # System architecture & service boundaries
│   ├── assumptions.md       # Assumptions & operational limits
│   └── methodology.md       # Scientific remote sensing & index formulations
├── tests/                   # Automated pytest suite (57 tests)
├── .env.example
├── requirements.txt
└── README.md
```

---

## License

MIT
