# AgriN — Scientific Methodology (Phase 1)

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
- **Agronomic Context**: Major soybean, wheat, and pulses agricultural corridor in central India.

---

## 3. Spectral Index Formulation: NDVI

The Normalized Difference Vegetation Index (NDVI) assesses photosynthetic capacity and live green vegetation density:

$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}} = \frac{\text{B8} - \text{B4}}{\text{B8} + \text{B4}}$$

### Bands Used
| Band ID | Name | Central Wavelength | Native Resolution | Role in NDVI |
|---|---|---|---|---|
| **B4** | Red | 665 nm | 10 m | Chlorophyll absorption peak |
| **B8** | NIR | 842 nm | 10 m | Mesophyll cell leaf reflectance |

### Theoretical Interpretation
- **NDVI < 0.15**: Bare soil, fallow fields, water bodies, or uncultivated land.
- **0.15 ≤ NDVI < 0.30**: Sparse or emerging canopy.
- **0.30 ≤ NDVI < 0.50**: Moderate crop canopy development.
- **NDVI ≥ 0.50**: Dense, healthy, photosynthetically active crop vegetation.

---

## 4. Cloud Filtering Methodology & Limitations

In Phase 1, filtering uses the scene-level metadata property `CLOUDY_PIXEL_PERCENTAGE < 20%`.

### Current Limitations:
1. **Scene-Level Filter**: Filters entire satellite tiles, but localized sub-pixel cirrus clouds or shadows over the AOI can still introduce noise.
2. **Phase 2 Evolution**: In subsequent phases, pixel-level cloud masking will be implemented using the Sentinel-2 **QA60** bitmask and **SCL (Scene Classification Layer)** probabilities.

---

## 5. Ground Truth & Validation Strategy (Roadmap)

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
