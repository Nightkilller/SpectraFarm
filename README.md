# AgriN — AI-Powered Smart Crop Intelligence

> *"What is happening to my crop, and what should I consider doing about it?"*

AgriN combines **satellite remote sensing (Sentinel-2 optical + Sentinel-1 SAR via Google Earth Engine)**, **machine learning**, and **Google Gemini AI** to deliver farmer-friendly crop intelligence.

---

## Current Status: Phase 3 Complete (Sentinel-1 SAR) ✅

- **Google Earth Engine Integration**: Connected and operational with Google Cloud Project `agrin-506618`.
- **Pilot Region Foundation**: **Sehore Pilot Test AOI, Madhya Pradesh, India** (`23.20°N, 77.08°E`).
- **Sentinel-2 Optical Pipeline (Phase 1 & 2)**: Real multi-temporal NDVI trajectories (22 canonical daily observations).
- **Sentinel-1 SAR Pipeline (Phase 3)**: Real multi-temporal Synthetic Aperture Radar backscatter (`VV`, `VH`, `VV/VH` linear ratio, and `VV - VH` difference) extracted from `COPERNICUS/S1_GRD` (14 canonical daily observations).
- **All-Weather Monsoon Coverage**: Sentinel-1 C-band radar provides continuous observation continuity across June, July, and August 2026 when optical sensors are blinded by monsoon cloud cover.
- **Server-Side Reduction**: Earth Engine statistical reductions (min, mean, max, stdDev) executed at native 10m pixel resolution without client raster downloads.
- **Data Integrity**: 100% real satellite data (`LIVE DERIVED FROM SATELLITE`). No synthetic/demo data used in the real-mode time series.

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

## Sentinel-1 SAR Verification (Phase 3)

Execute the verified SAR backscatter pipeline over the Sehore pilot AOI:

```bash
python scripts/test_sehore_sentinel1.py
```

### Sample Output:
```text
==========================================================================================
AgriN — Phase 3: Real Sentinel-1 SAR Backscatter Pipeline (Sehore AOI)
==========================================================================================

[INFO] Initializing Earth Engine with project: agrin-506618
[INFO] Earth Engine connection established successfully.
[INFO] Loading Pilot AOI: Sehore Pilot Test AOI (23.2°N, 77.08°E, Buffer: 2000m)
[INFO] Collection: COPERNICUS/S1_GRD
[INFO] Temporal Filter: 2026-02-27 to 2026-08-26
[INFO] Polarizations: Dual (VV + VH) | Orbit Pass: DESCENDING | Instrument Mode: IW
[INFO] Cloud Independence: All-weather C-band active microwave radar

[INFO] Querying and reducing Sentinel-1 SAR observations over Sehore AOI...
[INFO] Retrieved 14 raw Sentinel-1 SAR observations.
[INFO] Canonical SAR series contains 14 daily observations.

------------------------------------------------------------------------------------------
Date         | Orbit      | RelOrb | VV (dB)   | VH (dB)   | VV/VH (lin) | VV-VH (dB) | Image ID       
------------------------------------------------------------------------------------------
2026-03-05   | DESCENDING | 63     |    -10.43 |    -17.76 |       10.34 |       7.34 | S1A_IW_GRDH_1S...
2026-03-17   | DESCENDING | 63     |    -10.45 |    -18.27 |       14.73 |       7.81 | S1A_IW_GRDH_1S...
2026-03-29   | DESCENDING | 63     |    -10.33 |    -18.43 |       18.77 |       8.10 | S1A_IW_GRDH_1S...
2026-04-10   | DESCENDING | 63     |    -10.47 |    -18.63 |       19.48 |       8.16 | S1A_IW_GRDH_1S...
2026-04-22   | DESCENDING | 63     |    -10.14 |    -18.14 |       15.68 |       8.00 | S1A_IW_GRDH_1S...
2026-05-04   | DESCENDING | 63     |     -9.98 |    -18.17 |       16.55 |       8.19 | S1A_IW_GRDH_1S...
2026-05-16   | DESCENDING | 63     |    -10.04 |    -18.14 |       15.83 |       8.10 | S1A_IW_GRDH_1S...
2026-05-28   | DESCENDING | 63     |     -9.76 |    -18.01 |       14.99 |       8.25 | S1A_IW_GRDH_1S...
2026-06-09   | DESCENDING | 63     |     -9.66 |    -18.39 |       20.43 |       8.73 | S1A_IW_GRDH_1S...
2026-06-21   | DESCENDING | 63     |     -9.49 |    -18.12 |       17.41 |       8.63 | S1A_IW_GRDH_1S...
2026-06-28   | DESCENDING | 63     |     -7.18 |    -15.61 |       11.81 |       8.44 | S1D_IW_GRDH_1S...
2026-07-10   | DESCENDING | 63     |     -8.59 |    -16.78 |       10.92 |       8.19 | S1D_IW_GRDH_1S...
2026-08-03   | DESCENDING | 63     |     -7.10 |    -14.79 |       10.33 |       7.69 | S1D_IW_GRDH_1S...
2026-08-15   | DESCENDING | 63     |     -6.75 |    -14.54 |       11.20 |       7.79 | S1D_IW_GRDH_1S...
------------------------------------------------------------------------------------------

==========================================================================================
PHASE 3 SENTINEL-1 SAR VERIFICATION SUMMARY
==========================================================================================
  Target Pilot AOI:             Sehore Pilot Test AOI (Sehore, MP)
  Collection Used:              COPERNICUS/S1_GRD (IW GRDH 1SDV)
  Date Range Filter:            Past 180 Days (2026-02-27 to 2026-08-26)
  Raw SAR Observations:         14
  Canonical Daily Observations: 14
  First Observation Date:       2026-03-05
  Last Observation Date:        2026-08-15
  Mean VV Backscatter:          -9.31 dB (Range: -10.47 to -6.75 dB)
  Mean VH Backscatter:          -17.41 dB (Range: -18.63 to -14.54 dB)
  Mean VV/VH Linear Ratio:      14.89 (Range: 10.33 to 20.43)
  Data Origin:                  LIVE DERIVED FROM SENTINEL-1 (100% Earth Engine Compute)
  Scientific Status:            UNVALIDATED RADAR BACKSCATTER FEATURE (No moisture inference claimed)
==========================================================================================
```

---

## Multi-Temporal Sentinel-2 Optical Verification (Phase 2)

```bash
python scripts/test_sehore_timeseries.py
```

---

## Test Suite

Run the full automated offline test suite (62 tests passing):

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
│   ├── features/            # Feature extraction modules
│   └── ai/                  # Gemini advisory integration
├── scripts/
│   ├── test_sehore_sentinel2.py   # Phase 1 single-image optical test
│   ├── test_sehore_timeseries.py  # Phase 2 multi-temporal optical test
│   └── test_sehore_sentinel1.py   # Phase 3 multi-temporal SAR radar test
├── docs/
│   ├── architecture.md      # System architecture & service boundaries
│   ├── assumptions.md       # Assumptions & operational limits
│   └── methodology.md       # Scientific remote sensing & index formulations
├── tests/                   # Automated pytest suite (62 tests)
├── .env.example
├── requirements.txt
└── README.md
```

---

## License

MIT
