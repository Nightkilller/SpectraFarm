# AgriN — Public ML Reference Datasets & Strategy

This document details the public agricultural ground-reference datasets evaluated and integrated into AgriN for machine learning pipeline development and benchmarking, while strictly separating them from the Sehore satellite pilot.

---

## 1. Candidate Dataset Comparison

| Evaluation Metric | Candidate 1: AgriFieldNet Competition Dataset (Selected Primary) | Candidate 2: Eastern India Wheat LDS (CIMMYT/CSISA) |
|---|---|---|
| **Curator / Authority** | Radiant Earth Foundation / IDinsight "Data on Demand" (ECAAS) | CIMMYT / Cereal Systems Initiative for South Asia (CSISA) |
| **DOI / Source** | DOI: `10.34911/rdnt.wu92p1` / [`https://source.coop/radiantearth/agrifieldnet-competition`](https://source.coop/radiantearth/agrifieldnet-competition) | Dataverse Handle / DOI: `10.71682/...` |
| **License** | **CC-BY-4.0** (Open Access) | Open Access (CGIAR Open Data) |
| **Geographic Coverage** | **Northern India: Uttar Pradesh, Bihar, Rajasthan, Odisha** | Eastern Uttar Pradesh, Bihar |
| **Field / Record Count** | **7,081 fields** (5,551 train fields, 1,530 test fields across 1,217 tiles) | ~1,000–3,000 surveyed farm plots |
| **Crop Classes** | **13 multi-crop classes** (Wheat, Mustard, Lentil, Green pea, Sugarcane, Garlic, Maize, Gram, Coriander, Potato, Berseem, Rice, Fallow) | Primarily single-crop (**Wheat** agronomic management, inputs, yield) |
| **Field Geometry** | Georeferenced raster masks ($256 \times 256$ px tiles) with pixel-level field boundaries | Point coordinates / farmer survey tables |
| **Remote Sensing Link** | Pre-matched 12-band Sentinel-2 L2A BOA tiles | Survey metadata (requires manual satellite pairing) |
| **Suitability for AgriN ML** | **HIGHEST** (Direct multi-crop classification benchmark) | Moderate (Better suited for crop yield estimation) |

---

## 2. Selection Rationale for AgriN Phase 6

**AgriFieldNet Competition Dataset** is selected as the primary public ML dataset because:
1. **Multi-Crop Granularity**: Contains ground-truth for 13 distinct crop classes, enabling true multi-class supervised classification.
2. **Primary Geographic Alignment**: High density of fields in **Uttar Pradesh** (Primary ML region) and **Bihar** (Secondary ML region).
3. **Sensor Pairing**: Directly georeferenced to Sentinel-2 multi-spectral bands with bounding boxes ready for Sentinel-1 SAR extraction.
4. **Permissive Open Licensing**: `CC-BY-4.0` allows unrestricted research and development.

---

## 3. Strict Geographic Separation Policy

> [!IMPORTANT]
> **Clear Separation of Roles in AgriN**:
> 1. **Sehore Pilot Test AOI (Madhya Pradesh)**:
>    - Role: End-to-end satellite remote sensing pipeline demonstration (Sentinel-1 SAR + Sentinel-2 optical live GEE compute).
>    - Sehore In-Situ Ground Truth Count: **0 records** (Zero synthetic records fabricated).
> 2. **Uttar Pradesh / Bihar (Public ML Reference Region)**:
>    - Role: Public-data ML experimentation, feature engineering benchmarking, and Random Forest classifier training in Google Colab / Vertex AI.
>    - Dataset Status: **`EXTERNAL_PUBLIC_DATASET`** (Must NEVER be labeled `SEHORE_GROUND_TRUTH`).

---

## 4. Cloud Ingestion Workflow (Google Cloud Platform)

```
Public Source: source.coop/radiantearth/agrifieldnet-competition
                                │
                                ▼
         gs://agrin-ground-truth-506618/external/agrifieldnet/
                                │
                                ▼
         AgriN Ground Truth Validator (External Dataset Mode)
                                │
                                ▼
         BigQuery: agrin-506618.agrin_db.external_agrifieldnet_labels
                                │
                                ▼
         Google Colab / Vertex AI Notebook (Cloud ML Training)
                                │
                                ▼
         Trained Model Artifact (.joblib)
                                │
                                ▼
         gs://agrin-models-506618/crop_classifier/
```
