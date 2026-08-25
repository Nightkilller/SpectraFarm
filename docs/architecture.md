# AgriN — Architecture

## System Overview

AgriN is a modular agricultural intelligence system with clearly separated layers:

```
┌────────────────────────────────────────────────┐
│              Streamlit Dashboard               │
│  (Map, Charts, Advisory, Ask AgriN)            │
└──────────────────┬─────────────────────────────┘
                   │
┌──────────────────▼─────────────────────────────┐
│         Agricultural Intelligence Layer        │
│  Combines all data into structured FarmAnalysis│
└──────┬──────────┬──────────┬──────────┬────────┘
       │          │          │          │
┌──────▼───┐ ┌───▼────┐ ┌──▼───┐ ┌───▼──────┐
│ Satellite│ │   ML   │ │ AI   │ │ Weather  │
│ Service  │ │Service │ │(Gem.)│ │ (future) │
└──────┬───┘ └───┬────┘ └──┬───┘ └──────────┘
       │         │         │
┌──────▼───┐ ┌───▼────┐ ┌──▼───────────┐
│  Google  │ │ scikit │ │ Google       │
│  Earth   │ │ -learn │ │ Gemini API   │
│  Engine  │ │        │ │              │
└──────────┘ └────────┘ └──────────────┘
```

## Key Design Principles

### 1. Separation of Measurement from Interpretation

- **Backend services** (satellite, ML, indices) produce measurements
- **Gemini AI** explains and contextualizes measurements
- Gemini is never allowed to invent NDVI, crop type, stress, or other scientific values

### 2. Demo/Live Mode Separation

- `DataSource.DEMO` vs `DataSource.LIVE` tracked on every data object
- Demo data loaded from `data/demo/`, never mixed with real pipelines
- UI displays "🔶 DEMO DATA" badge when demo data is active

### 3. Configuration-Driven

- NDVI/stress thresholds in `config/thresholds.yaml`
- Crop classes in `config/crops.yaml`
- Pilot region and satellite params in `config/settings.yaml`
- Nothing hardcoded in application logic

### 4. Progressive Enhancement

Each phase adds a new service module without rewriting existing code:

| Phase | Module Added |
|---|---|
| 0 | `src/config`, `src/data/schemas` |
| 1 | `dashboard/` |
| 2 | `src/geospatial/` |
| 3 | `src/ml/`, `src/features/` |
| 4 | `src/intelligence/stress_analysis` |
| 5 | `src/intelligence/farm_analyzer` |
| 6 | `src/ai/` |
| 7 | `src/data/weather` |

## Data Flow

```
User selects field
    ↓
Farm object (validated Pydantic schema)
    ↓
Satellite service → SatelliteObservation[]
    ↓
Feature extraction → ML input
    ↓
Crop classifier → CropPrediction
    ↓
Stress detector → StressAssessment
    ↓
Farm analyzer → FarmAnalysis (structured)
    ↓
Gemini service receives FarmAnalysis
    ↓
Advisory generated → Advisory object
    ↓
Dashboard renders all outputs
```

## External Dependencies

| Service | Required Phase | Free Tier | Manual Setup |
|---|---|---|---|
| Google Earth Engine | Phase 2 | Yes (research) | GEE account + auth |
| Google Gemini API | Phase 6 | Yes (limited) | API key from AI Studio |
| Weather API | Phase 7 | TBD | TBD |
