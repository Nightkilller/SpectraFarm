# AgriN — AI-Powered Smart Crop Intelligence

> *"What is happening to my crop, and what should I consider doing about it?"*

AgriN combines **satellite remote sensing (Sentinel-2 optical + Sentinel-1 SAR via Google Earth Engine)**, **machine learning**, and **Google Gemini AI** to deliver farmer-friendly crop intelligence.

---

## Current Status: Phase 4 Complete (Optical + SAR Fusion) ✅

- **Google Earth Engine Integration**: Connected and operational with Google Cloud Project `agrin-506618`.
- **Pilot Region Foundation**: **Sehore Pilot Test AOI, Madhya Pradesh, India** (`23.20°N, 77.08°E`).
- **Sentinel-2 Optical Pipeline (Phase 1 & 2)**: Real multi-temporal NDVI trajectories (22 canonical daily observations).
- **Sentinel-1 SAR Pipeline (Phase 3)**: Real multi-temporal Synthetic Aperture Radar backscatter (`VV`, `VH`, `VV/VH` linear ratio) from `COPERNICUS/S1_GRD` (14 canonical daily observations).
- **Multi-Sensor Fusion (Phase 4)**: Temporally fused feature dataset combining optical greenness and radar backscatter across the 180-day window.
- **Statistical Feature Summary**: Full temporal feature vector (`ndvi_mean`, `ndvi_slope`, `vv_mean_db`, `vh_mean_db`, `vv_vh_ratio_mean`, etc.) generated for downstream ML readiness.
- **Data Integrity**: 100% real satellite data (`LIVE DERIVED FROM SATELLITE`). Zero synthetic or fabricated data.

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

## Multi-Sensor Optical + SAR Fusion Verification (Phase 4)

Execute the multi-sensor fusion pipeline over the Sehore pilot AOI:

```bash
python scripts/test_sehore_fusion.py
```

### Sample Output:
```text
===============================================================================================
AgriN — Phase 4: Optical + SAR Multi-Sensor Fusion Pipeline (Sehore AOI)
===============================================================================================

[INFO] Initializing Earth Engine with project: agrin-506618
[INFO] Earth Engine connection established successfully.
[INFO] Loading Pilot AOI: Sehore Pilot Test AOI (23.2°N, 77.08°E, Buffer: 2000m)

[INFO] Multi-Sensor Fusion Window: 2026-02-27 to 2026-08-26
[INFO] Optical series retrieved: 22 canonical daily observations.
[INFO] SAR series retrieved: 14 canonical daily observations.
[INFO] Performing temporal fusion (nearest observation within +/- 5 days)...
[INFO] Generated 14 fused multi-sensor observation records.

-----------------------------------------------------------------------------------------------
Target Date  | Optical Date | NDVI    | SAR Date     | VV (dB)  | VH (dB)  | VV/VH (lin) | Lag (days)
-----------------------------------------------------------------------------------------------
2026-03-05   | 2026-03-06   |  0.3260 | 2026-03-05   |   -10.43 |   -17.76 |       10.34 |          1
2026-03-17   | 2026-03-16   |  0.2588 | 2026-03-17   |   -10.45 |   -18.27 |       14.73 |          1
2026-03-29   | 2026-03-31   |  0.2172 | 2026-03-29   |   -10.33 |   -18.43 |       18.77 |          2
2026-04-10   | 2026-04-10   |  0.2386 | 2026-04-10   |   -10.47 |   -18.63 |       19.48 |          0
2026-04-22   | 2026-04-22   |  0.2298 | 2026-04-22   |   -10.14 |   -18.14 |       15.68 |          0
2026-05-04   | 2026-05-05   |  0.2112 | 2026-05-04   |    -9.98 |   -18.17 |       16.55 |          1
2026-05-16   | 2026-05-15   |  0.1952 | 2026-05-16   |   -10.04 |   -18.14 |       15.83 |          1
2026-05-28   | 2026-05-25   |  0.1648 | 2026-05-28   |    -9.76 |   -18.01 |       14.99 |          3
2026-06-09   | 2026-06-09   |  0.1901 | 2026-06-09   |    -9.66 |   -18.39 |       20.43 |          0
2026-06-21   | CLOUD GAP    |    N/A  | 2026-06-21   |    -9.49 |   -18.12 |       17.41 |         12
2026-06-28   | CLOUD GAP    |    N/A  | 2026-06-28   |    -7.18 |   -15.61 |       11.81 |         19
2026-07-10   | CLOUD GAP    |    N/A  | 2026-07-10   |    -8.59 |   -16.78 |       10.92 |         31
2026-08-03   | CLOUD GAP    |    N/A  | 2026-08-03   |    -7.10 |   -14.79 |       10.33 |         55
2026-08-15   | CLOUD GAP    |    N/A  | 2026-08-15   |    -6.75 |   -14.54 |       11.20 |         67
-----------------------------------------------------------------------------------------------

===============================================================================================
PHASE 4 TEMPORAL SUMMARY FEATURE VECTOR
===============================================================================================
  Target Pilot AOI:             Sehore Pilot Test AOI (Sehore, MP)
  Date Range:                   2026-02-27 to 2026-08-26
  Optical Observations:         22 passes (NDVI Mean: 0.2286, Range: 0.1648 to 0.3580, Trend Slope: -0.006604)
  SAR Observations:             14 passes (VV Mean: -9.31 dB, VH Mean: -17.41 dB, VV/VH Ratio: 14.89)
  Temporally Aligned Pairs:     9 multi-sensor pairs (Lag <= 3 days)
  Data Origin:                  LIVE DERIVED FROM SATELLITE (100% Earth Engine Compute)
  Scientific Status:            UNVALIDATED MULTI-SENSOR FEATURE VECTOR
===============================================================================================
```

---

## Test Suite

Run the full automated offline test suite (66 tests passing):

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
│   │   ├── indices.py       # Reusable optical indices (NDVI, NDWI)
│   │   ├── timeseries.py    # Multi-temporal Sentinel-2 NDVI extraction & deduplication
│   │   └── sar.py           # Multi-temporal Sentinel-1 SAR extraction & backscatter ratios
│   ├── features/
│   │   ├── feature_extraction.py  # Statistical feature extractors
│   │   └── fusion.py              # Optical + SAR multi-sensor fusion & temporal vector
│   └── ai/                  # Gemini advisory integration
├── scripts/
│   ├── test_sehore_sentinel2.py   # Phase 1 single-image optical test
│   ├── test_sehore_timeseries.py  # Phase 2 multi-temporal optical test
│   ├── test_sehore_sentinel1.py   # Phase 3 multi-temporal SAR radar test
│   └── test_sehore_fusion.py      # Phase 4 optical + SAR multi-sensor fusion test
├── docs/
│   ├── architecture.md      # System architecture & service boundaries
│   ├── assumptions.md       # Assumptions & operational limits
│   └── methodology.md       # Scientific remote sensing & index formulations
├── tests/                   # Automated pytest suite (66 tests)
├── .env.example
├── requirements.txt
└── README.md
```

---

## License

MIT
