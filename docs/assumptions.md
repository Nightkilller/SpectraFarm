# AgriN — Assumptions & Limitations

This document records all known assumptions and limitations. Transparency is
essential for an agricultural decision-support system.

---

## NDVI Thresholds

The NDVI thresholds in `config/thresholds.yaml` are **initial estimates**
based on general remote sensing literature. They have NOT been validated
against ground-truth measurements in the pilot region.

- Bare soil: NDVI < 0.15
- Sparse vegetation: 0.15–0.30
- Moderate vegetation: 0.30–0.50
- Dense vegetation: > 0.50
- Trend threshold: ±0.05 absolute change

**Action required:** Calibrate thresholds against actual field observations.

---

## Stress Classification

The stress indicator is a **satellite-based composite estimate**, NOT a
validated soil-moisture measurement. It should be labeled as:

> "Satellite-based moisture/crop stress indicator"

The indicator combines NDVI, NDWI, and SAR-derived features into a 0–1 score.
Mapping to Healthy/Mild/Moderate/Severe uses configurable thresholds that
have not been physically validated.

---

## Crop Classification

The initial Random Forest model classifies: **Wheat, Rice, Other**.

- Model accuracy targets (~80–90%) are aspirational, not guaranteed
- Actual performance must be measured and reported honestly
- The "Other" class is a catch-all and will have lower precision
- Classification is valid only for the pilot region and season

---

## Sentinel Data

- **Sentinel-2:** 10m resolution, ~5 day revisit. Cloud cover limits usability
- **Sentinel-1:** SAR (cloud-independent) but noisier for vegetation analysis
- Temporal compositing reduces noise but may miss short events
- Atmospheric correction assumed to be handled by SR product (Level-2A)

---

## Gemini AI

- Gemini provides **interpretation, not diagnosis**
- All factual measurements come from the backend pipeline
- Gemini is instructed to never fabricate data, but LLMs can hallucinate
- Advisories should always be verified against local conditions
- Gemini should not be treated as a certified agricultural authority

---

## Geographic Scope

- MVP covers only the **Ludhiana, Punjab** pilot region
- Thresholds, crop classes, and seasons may not transfer to other regions
- Architecture supports multi-region expansion but requires per-region calibration

---

## Demo Data

- Demo data is **synthetic** — it is designed to look realistic but does not
  come from actual satellite observations
- Demo mode is clearly labeled in all outputs
- Demo data must never be mixed with real data pipelines
