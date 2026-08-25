# AgriN — Scientific Methodology

## 1. Remote Sensing Data Source

AgriN utilizes **Copernicus Sentinel-2 Surface Reflectance (Level-2A Harmonized)** imagery:
- **Earth Engine Collection**: `COPERNICUS/S2_SR_HARMONIZED`
- **Spatial Resolution**: 10 meters (B2, B3, B4, B8) / 20 meters (B11, B12)
- **Temporal Resolution**: ~5-day revisit cycle with the constellation (Sentinel-2A, 2B, 2C)
- **Atmospheric Correction**: Level-2A surface reflectance provides bottom-of-atmosphere (BOA) measurements, minimizing atmospheric aerosol and water vapor distortion.

---

## 2. Pilot Region & Test AOI

- **Location**: Sehore District, Madhya Pradesh, India
- **Coordinates**: Latitude 23.2000°N, Longitude 77.0800°E
- **Area of Interest (AOI)**: A dedicated 2,000-meter test buffer around the pilot center (`~12.5 km²`).
- **Agronomic Context**: Major agricultural belt in central India.

---

## 3. Spectral Index Formulation: NDVI

The Normalized Difference Vegetation Index (NDVI) assesses photosynthetic capacity and live green vegetation density:

$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}} = \frac{\text{B8} - \text{B4}}{\text{B8} + \text{B4}}$$

### Bands Used
| Band ID | Name | Central Wavelength | Native Resolution | Role in NDVI |
|---|---|---|---|---|
| **B4** | Red | 665 nm | 10 m | Chlorophyll absorption peak |
| **B8** | NIR | 842 nm | 10 m | Mesophyll cell leaf reflectance |

---

## 4. Multi-Temporal NDVI Time Series Methodology (Phase 2)

AgriN maintains a dual-tier time-series architecture:

```
COPERNICUS/S2_SR_HARMONIZED (Sehore AOI, 180-Day Window, <20% Cloud)
        ↓
Server-Side Reduction across Earth Engine Clusters
        ↓
1. Raw Granule Observation Series (23 Observations)
   - Preserves complete instrument and ingestion provenance.
   - Retains all ESA processing runs and baseline granules.
        ↓
2. Canonical Daily Agricultural Series (22 Observations)
   - Exactly ONE observation per unique calendar date.
   - Deterministic Granule Selection Rule: For dates with multiple valid granules 
     (e.g., 2026-03-11), select the granule with the latest ESA generation/processing timestamp.
   - Validated: 0 duplicate calendar dates.
```

### Why One Observation Per Calendar Date is Used for Agricultural Analytics:
1. **Model Temporal Uniformity**: Downstream agricultural models (feature extraction, temporal interpolation, cross-sensor fusion with SAR) require a single distinct state per calendar day.
2. **Preventing Artificial Step-Variance**: Retaining multiple granules for the exact same satellite pass on the same day can introduce artificial sub-pixel registration noise into gradient and slope calculations.
3. **Provenance Preservation**: The raw granule dataset is preserved side-by-side in [`data/processed/sehore/ndvi_timeseries_raw.csv`](file:///Users/adityagupta/Desktop/EPIC%20project/agriN/data/processed/sehore/ndvi_timeseries_raw.csv) so auditability is never lost.

### Scientific Rigor & Principles:
1. **Server-Side Reduction**: Statistics are computed across Earth Engine clusters via `reduceRegion`, avoiding client-side raster downloads.
2. **No Data Fabrication**: Only real, unmasked observations from valid satellite passes are retained.
3. **Strict Attribution**:
   - NDVI trajectory reflects canopy greenness dynamics over the Sehore AOI.
   - Crop type is **NOT** inferred from optical NDVI alone.
   - Phenological stages are **NOT** claimed without validated crop-calendar ground models.
   - Trajectory decline is **NOT** classified as "stress" without calibrated moisture models.

---

## 5. Cloud Filtering Methodology & Limitations

Filtering uses the scene-level metadata property `CLOUDY_PIXEL_PERCENTAGE < 20%`.

### Current Limitations:
1. **Scene-Level Filter**: Filters entire satellite tiles, but localized sub-pixel cirrus clouds or shadows over the AOI can still introduce noise.
2. **Monsoon Gap**: In central India during June–August, persistent high cloud cover ($\ge 39\%$) excludes optical passes. Multi-sensor radar fusion (Sentinel-1 SAR) in Phase 3 will provide all-weather observation continuity.

---

## 6. Ground Truth & Validation Strategy (Roadmap)

To validate crop classification and stress detection in subsequent phases:
- **No Fabricated Labels**: Agricultural labels will not be randomly populated or artificially synthesized.
- **Label Acquisition**: Ground-truth datasets will be structured as:
  ```csv
  field_id,lat,lon,crop_type,season
  F001,23.201,77.082,Wheat,Rabi
  F002,23.195,77.078,Wheat,Rabi
  F003,23.210,77.085,Other,Rabi
  ```
- **Evaluation**: Training and validation splits will use spatial k-fold cross-validation to prevent spatial autocorrelation leakage.
