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

## 2. Regional Separation & Application Roles

AgriN strictly decouples its live satellite demonstration pilot from its public machine learning reference regions:

| Region Role | Location | Purpose | Ground Truth Status |
|---|---|---|---|
| **Satellite Demonstration Pilot** | **Sehore District, Madhya Pradesh** (`23.20°N, 77.08°E`) | Live multi-sensor optical + SAR feature extraction & index calculation | **0 in-situ records** (No synthetic records fabricated) |
| **Primary ML Benchmark Region** | **Uttar Pradesh, India** | Public-data machine learning experimentation & crop classification training | **7,081 fields** via AgriFieldNet (`EXTERNAL_PUBLIC_DATASET`) |
| **Secondary ML Benchmark Region** | **Bihar, India** | Public-data cross-region model generalization & validation | Evaluated under AgriFieldNet / CSISA |

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
       │ Multi-Sensor Fused Observation Matrix (fused_features)  │
       │                                                         │
       │ 1. observation_type = "FUSED_PAIR" (9 Pairs)            │
       │    - Both optical NDVI and SAR backscatter valid        │
       │    - Synchronization lag ≤ 3 calendar days              │
       │                                                         │
       │ 2. observation_type = "SAR_STANDALONE" (5 Passes)       │
       │    - SAR radar observation only during monsoon clouds   │
       │    - optical_date = None, optical_ndvi = None           │
       │    - Explicitly NOT a fused optical+SAR measurement     │
       └─────────────────────────────────────────────────────────┘
```

---

## 6. Ground Truth Validation & Public ML Strategy (Phase 5)

AgriN enforces a strict ground-truth ingestion gate to ensure that machine learning models (Phase 6) are trained exclusively on validated, non-fabricated reference data.

```
Public Benchmark Dataset (AgriFieldNet UP/Bihar)
                     │
                     ▼
      ┌──────────────────────────────────────────────┐
      │     Ground Truth Validator (Phase 5)         │
      ├──────────────────────────────────────────────┤
      │ 1. Schema & Null Completeness Check          │
      │ 2. Spatial Bounding Box Filter               │
      │ 3. Spatial Duplicate Detection (<15m delta)  │
      │ 4. Class & Season Distribution Audit         │
      │ 5. Spatial Block ID Assignment (Grid Binning)│
      │ 6. Provenance Classification:                │
      │    -> EXTERNAL_PUBLIC_DATASET                │
      └──────────────────────────────────────────────┘
                     │
                     ▼
        Google Cloud Target Architecture
 ┌───────────────────────────────────────────────────────────────┐
 │ Cloud Storage: gs://agrin-ground-truth-506618/external/       │
 │ BigQuery:      agrin-506618.agrin_db.external_agrifieldnet_labels │
 └───────────────────────────────────────────────────────────────┘
```

### 6.1 Strict Attribution Constraints:
- AgriFieldNet data covers **Uttar Pradesh, Bihar, Rajasthan, Odisha** and is classified strictly as **`EXTERNAL_PUBLIC_DATASET`**.
- AgriFieldNet data **does NOT represent Sehore, Madhya Pradesh**.
- Models trained on AgriFieldNet evaluate algorithmic crop classification methodologies in Google Colab / Vertex AI but do not establish localized Sehore field accuracy without local ground truth.
