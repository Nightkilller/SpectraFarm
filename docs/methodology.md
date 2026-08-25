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
- **All-Weather Capability**: Active microwave signals penetrate cloud cover and haze, enabling continuous monitoring during the Indian monsoon season (June–August) when optical sensors are blinded.

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

---

## 5. Optical + SAR Multi-Sensor Fusion Methodology (Phase 4)

Phase 4 fuses the canonical Sentinel-2 optical time series and Sentinel-1 SAR time series into a unified multi-sensor feature matrix:

```
Canonical Sentinel-2 Optical (NDVI)        Canonical Sentinel-1 SAR (VV, VH, VV/VH)
              │                                           │
              └─────────────────────┬─────────────────────┘
                                    │
                                    ▼
                 Nearest-Temporal Multi-Sensor Alignment
                         (Matching Window: ≤ ±5 Days)
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ 1. Multi-Sensor Fused Observation Matrix (fused_features)│
       │    - Pre-monsoon: Synchronized optical + radar pairs   │
       │    - Monsoon: All-weather SAR continuity (optical gaps) │
       │                                                         │
       │ 2. Temporal Aggregated Feature Vector (temporal_summary)│
       │    - Optical: mean, min, max, std, range, slope         │
       │    - SAR: VV mean/min/max/std, VH mean/min/max/std,     │
       │           VV/VH ratio mean, VV-VH difference mean       │
       └─────────────────────────────────────────────────────────┘
```

### 5.1 Alignment Strategy & Zero Data Fabrication:
1. **Anchor-Based Alignment**: For each radar observation pass, the closest optical observation within $\le 5$ calendar days is paired.
2. **Explicit Gap Representation**: During monsoon months where optical imagery is obscured by clouds, the SAR observation is preserved with `optical_date = None` and `ndvi = None`. No fake or interpolated NDVI values are inserted.
3. **Temporal Synchronization Metric**: Every observation pair records `temporal_delta_days` ($|t_{\text{opt}} - t_{\text{sar}}|$) to explicitly quantify temporal lag.

### 5.2 Scientific Constraints:
- **Unvalidated Feature Vector**: This fused dataset represents an **unvalidated multi-sensor feature set**.
- **No Crop Classification Claims**: Fused features will NOT be used to claim crop identification until real ground truth is integrated in Phase 5 and trained in Phase 6.
- **No Moisture Stress Claims**: Fused features will NOT be used to claim water stress until validated in Phase 7.

---

## 6. Multi-Sensor Temporal Coverage Comparison (Sehore AOI)

| Sensor / Product | Collection | Date Window | Total Observations | Aligned Pairs ($\le 5$ days) | Key Role |
|---|---|---|---|---|---|
| **Sentinel-2 (Optical)** | `COPERNICUS/S2_SR_HARMONIZED` | Feb 27 – Aug 26, 2026 | 22 daily passes | 9 pairs | Direct canopy chlorophyll & photosynthetic density |
| **Sentinel-1 (SAR)** | `COPERNICUS/S1_GRD` | Feb 27 – Aug 26, 2026 | 14 daily passes | 9 pairs (+5 monsoon standalone) | Surface roughness, volume scattering & all-weather continuity |
| **Fused Feature Set** | Multi-Sensor Pipeline | Feb 27 – Aug 26, 2026 | **14 fused records** | **9 joint / 5 SAR monsoon** | Unified feature foundation for downstream ML |

---

## 7. Ground Truth & Model Validation Strategy (Roadmap)

- **Phase 5 (Ground Truth)**: Verifiable agricultural coordinates and crop labels (`field_id, lat, lon, crop_type, season`).
- **Phase 6 (Crop Classification)**: Supervised ML (Random Forest) trained on the fused optical + SAR feature matrix using spatial k-fold cross-validation.
- **Phase 7 (Moisture Stress)**: Calibrated moisture-stress modeling evaluated against reference soil moisture datasets.
