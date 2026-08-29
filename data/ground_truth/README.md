# AgriN — Ground Truth Field Observation Registry

This directory is the local staging container for **authoritative agricultural field surveys and reference crop labels** prior to cloud ingestion into Google Cloud Storage (`gs://agrin-ground-truth-506618/`) and BigQuery (`agrin_db.ground_truth_labels`).

---

## 1. Scientific Integrity Policy

- **ZERO FABRICATION**: Synthetic coordinates or randomly populated crop labels are strictly prohibited.
- **AUTHORITATIVE SOURCES ONLY**: Ground truth must originate from verified in-situ field surveys (e.g. Krishi Vigyan Kendra Sehore, State Department of Agriculture, ICAR-CIAE Bhopal, or verified farmer plot registries).
- **SPATIAL BLOCKING**: Every field record is assigned a `spatial_block_id` by the validator to ensure that spatial k-fold cross-validation in Phase 6 avoids spatial autocorrelation data leakage.

---

## 2. Ingestion CSV Schema (`ground_truth_template.csv`)

| Column Name | Type | Required | Description | Example |
|---|---|---|---|---|
| `field_id` | String | **Yes** | Unique plot identifier | `SEH_2026_F001` |
| `latitude` | Float | **Yes** | WGS84 centroid latitude | `23.2015` |
| `longitude` | Float | **Yes** | WGS84 centroid longitude | `77.0812` |
| `crop_type` | String | **Yes** | Observed crop species | `Wheat` / `Soybean` / `Gram` / `Mustard` / `Other` |
| `season` | String | **Yes** | Agricultural season | `Rabi` / `Kharif` / `Zaid` |
| `reference_date` | Date | **Yes** | Survey observation date (`YYYY-MM-DD`) | `2026-03-10` |
| `source` | String | **Yes** | Authoritative surveying body | `Field Survey - KVK Sehore` |
| `verification_method` | String | **Yes** | Method of ground measurement | `GPS In-Situ Survey` |
| `confidence` | Float | No | Observer confidence (0.0 to 1.0) | `1.0` |

---

## 3. Cloud Target Architecture

- **Object Store**: `gs://agrin-ground-truth-506618/raw_surveys/`
- **Data Warehouse**: Google BigQuery `agrin-506618.agrin_db.ground_truth_labels`
- **Model Training**: Google Colab / Vertex AI reads directly from BigQuery to train Phase 6 Random Forest classifiers.
