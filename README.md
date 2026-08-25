# AgriN — AI-Powered Smart Crop Intelligence

> *"What is happening to my crop, and what should I consider doing about it?"*

AgriN combines **satellite remote sensing (Sentinel-2 optical + Sentinel-1 SAR via Google Earth Engine)**, **machine learning**, and **Google Gemini AI** to deliver farmer-friendly crop intelligence.

---

## Current Status: Phase 5 Complete (Ground Truth Infrastructure) ✅

- **Google Earth Engine Integration**: Connected and operational with Google Cloud Project `agrin-506618`.
- **Pilot Region Foundation**: **Sehore Pilot Test AOI, Madhya Pradesh, India** (`23.20°N, 77.08°E`).
- **Sentinel-2 Optical Pipeline (Phase 1 & 2)**: Real multi-temporal NDVI trajectories (22 canonical daily observations).
- **Sentinel-1 SAR Pipeline (Phase 3)**: Real multi-temporal Synthetic Aperture Radar backscatter (`VV`, `VH`, `VV/VH` linear ratio) from `COPERNICUS/S1_GRD` (14 canonical daily observations).
- **Multi-Sensor Fusion (Phase 4)**: Temporally fused feature dataset combining optical greenness and radar backscatter (9 `FUSED_PAIR`, 5 `SAR_STANDALONE` observations).
- **Ground Truth Ingestion Infrastructure (Phase 5)**: Cloud-ready validation pipeline with spatial bounding, duplicate coordinate detection ($<15\text{m}$ tolerance), and spatial blocking for spatial k-fold cross-validation.
- **Target Cloud Data Warehouse**: Google BigQuery (`agrin-506618.agrin_db.ground_truth_labels`) and Google Cloud Storage (`gs://agrin-ground-truth-506618/`).
- **Data Integrity**: Zero ground-truth fabrication. System status: `WAITING_FOR_DATA` (infrastructure ready for real survey datasets).

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

## Verification Pipelines

### 1. Ground Truth Validation Pipeline (Phase 5)
```bash
python scripts/validate_ground_truth.py
```

### 2. Multi-Sensor Optical + SAR Fusion Pipeline (Phase 4)
```bash
python scripts/test_sehore_fusion.py
```

### 3. Sentinel-1 SAR Backscatter Pipeline (Phase 3)
```bash
python scripts/test_sehore_sentinel1.py
```

### 4. Sentinel-2 Optical Pipeline (Phase 1 & 2)
```bash
python scripts/test_sehore_timeseries.py
```

---

## Test Suite

Run the full automated offline test suite (75 tests passing):

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
├── data/
│   ├── ground_truth/        # Ground-truth templates & documentation
│   │   ├── README.md        # Ground truth policies & BigQuery specification
│   │   └── ground_truth_template.csv  # Field survey ingestion header template
│   └── processed/sehore/    # Processed satellite datasets (NDVI, SAR, Fused)
├── src/
│   ├── config/              # Central configuration loader
│   ├── data/                # Schemas & ground-truth validation logic
│   │   ├── schemas.py       # Pydantic models (GroundTruthRecord, FusedPair, etc.)
│   │   └── ground_truth_validator.py  # Spatial checks, deduplication & blocking
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
│   ├── test_sehore_fusion.py      # Phase 4 optical + SAR multi-sensor fusion test
│   └── validate_ground_truth.py   # Phase 5 ground truth validation test
├── docs/
│   ├── architecture.md      # System architecture & service boundaries
│   ├── assumptions.md       # Assumptions & operational limits
│   └── methodology.md       # Scientific remote sensing & index formulations
├── tests/                   # Automated pytest suite (75 tests)
├── .env.example
├── requirements.txt
└── README.md
```

---

## License

MIT
