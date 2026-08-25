# AgriN — Scientific Methodology

## 1. Remote Sensing Data Sources

AgriN integrates dual-sensor Earth observation data from the European Space Agency (ESA) Copernicus constellation via Google Earth Engine:

### 1.1 Sentinel-2 Optical (Surface Reflectance)
- **Collection**: `COPERNICUS/S2_SR_HARMONIZED`
- **Spatial Resolution**: 10 meters (B2, B3, B4, B8)
- **Revisit Cycle**: ~5 days
- **Product Level**: Level-2A Bottom-of-Atmosphere (BOA) surface reflectance.

### 1.2 Sentinel-1 SAR (Synthetic Aperture Radar)
- **Collection**: `COPERNICUS/S1_GRD`
- **Instrument**: C-band Synthetic Aperture Radar ($\sim 5.405\text{ GHz}$, wavelength $\lambda \approx 5.55\text{ cm}$)
- **Product Type**: Ground Range Detected (GRD), High Resolution (GRDH)
- **Acquisition Mode**: Interferometric Wide Swath (`IW`)
- **Polarizations**: Dual-polarization (`VV` and `VH`)
- **Radiometric Calibration**: Calibrated $\sigma^0$ (sigma naught) backscatter in decibels (dB).
- **All-Weather Capability**: Active microwave signals penetrate cloud cover, haze, and rain, enabling unhindered monitoring during the Indian monsoon season (June–August) when optical sensors are blinded.

---

## 2. Pilot Region & Area of Interest (AOI)

- **Location**: Sehore District, Madhya Pradesh, India
- **Coordinates**: Latitude 23.2000°N, Longitude 77.0800°E
- **Area of Interest (AOI)**: A dedicated 2,000-meter test buffer around the pilot center (`~12.5 km²`).
- **Agronomic Context**: Major agricultural belt in central India.

---

## 3. Optical Index Formulation: NDVI (Phase 1 & 2)

The Normalized Difference Vegetation Index (NDVI) assesses photosynthetic capacity and live green canopy density:

$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}} = \frac{\text{B8} - \text{B4}}{\text{B8} + \text{B4}}$$

### Bands Used
| Band ID | Name | Central Wavelength | Native Resolution | Role in NDVI |
|---|---|---|---|---|
| **B4** | Red | 665 nm | 10 m | Chlorophyll absorption peak |
| **B8** | NIR | 842 nm | 10 m | Mesophyll cell leaf reflectance |

---

## 4. Synthetic Aperture Radar (SAR) Backscatter Methodology (Phase 3)

Sentinel-1 backscatter values measure radar microwave interaction with surface geometry, canopy roughness, and soil dielectric properties:

### 4.1 Polarizations & Physical Interactions
- **VV (Vertical-transmit, Vertical-receive)**: Sensitive to surface roughness, soil dielectric permittivity, and vertical plant structural elements.
- **VH (Vertical-transmit, Horizontal-receive)**: Sensitive to volume scattering within the vegetative crop canopy (depolarization caused by multiple scattering among leaves and stems).

### 4.2 SAR Backscatter Ratios
1. **Backscatter Difference (dB domain)**:
   $$\text{VV} - \text{VH} \quad (\text{in dB})$$
2. **Linear Power Ratio**:
   $$\left(\frac{\text{VV}}{\text{VH}}\right)_{\text{linear}} = 10^{\frac{\text{VV} - \text{VH}}{10}}$$
3. **Cross-Polarization Ratio**:
   $$\left(\frac{\text{VH}}{\text{VV}}\right)_{\text{linear}} = 10^{\frac{\text{VH} - \text{VV}}{10}}$$

### 4.3 Scientific Attribution & Constraints:
- **No Direct Moisture Claim**: SAR backscatter and cross-ratios are treated as **unvalidated SAR backscatter features / radar-derived indicators**. They are NOT claimed to directly measure soil moisture or crop water stress without local in-situ sensor calibration and surface roughness decoupling.
- **Server-Side Reduction**: All reductions (`mean`, `min`, `max`, `stdDev`) are computed over the AOI server-side in Google Earth Engine at 10m spatial resolution.

---

## 5. Dual-Tier Time-Series Architecture & Granule Deduplication

Both Sentinel-2 (optical) and Sentinel-1 (SAR) pipelines maintain a standardized dual-tier output:

```
Google Earth Engine Catalog
        ↓
Server-Side AOI Reduction
        ↓
1. Raw Observation Series
   - Preserves complete instrument, orbit, and reprocessing provenance.
   - Sentinel-2: 23 observations | Sentinel-1: 14 observations
        ↓
2. Canonical Daily Agricultural Series
   - Exactly ONE observation per unique calendar date.
   - Deterministic Selection Rule: When multiple valid granules exist for the same calendar date,
     the granule with the latest generation/processing timestamp is deterministically selected.
   - Validated: 0 duplicate calendar dates in canonical output.
```

---

## 6. Multi-Sensor Temporal Coverage Comparison (Sehore AOI)

| Sensor | Collection | Date Window | Usable Observations | Cloud Impact | Key Strength |
|---|---|---|---|---|---|
| **Sentinel-2** | `COPERNICUS/S2_SR_HARMONIZED` | Feb 27 – Aug 26, 2026 | 22 unique dates | Excluded June 10 – Aug 26 due to $\ge 39\%$ monsoon clouds | 10m direct chlorophyll / greenness assessment |
| **Sentinel-1** | `COPERNICUS/S1_GRD` | Feb 27 – Aug 26, 2026 | 14 unique dates | **0% impact** (100% all-weather cloud penetration) | Continuous canopy structure & dielectric tracking across monsoon |

---

## 7. Ground Truth & Model Validation Strategy (Roadmap)

In subsequent phases:
- **No Fabricated Labels**: Field survey coordinates and crop type annotations will be acquired from verifiable sources.
- **Validation**: Spatial k-fold cross-validation will be used for Phase 6 (Crop Classification) and Phase 7 (Moisture Stress) to prevent spatial autocorrelation leakage.
