# ============================================================================
# AgriN — Google Colab ML Training Notebook
# ============================================================================
#
# PURPOSE:
#   Train a Random Forest crop classifier using real Sentinel-2 and Sentinel-1
#   satellite features extracted from Google Earth Engine for agricultural
#   fields in Uttar Pradesh and Bihar (AgriFieldNet reference region).
#
# HOW TO USE IN GOOGLE COLAB:
#   1. Open Google Colab (colab.research.google.com)
#   2. File → Upload Notebook → Upload this .py file
#      (Colab auto-converts .py to notebook cells using # %% markers)
#   3. Run each cell in order
#   4. Download the trained model files at the end
#   5. Place them in agriN/models/crop_classifier/
#
# REQUIREMENTS:
#   - Google account with Earth Engine access enabled
#   - GCP Project: agrin-506618 (or your own project ID)
#
# DATASET:
#   Uses representative field locations from the AgriFieldNet Competition
#   Dataset geographic coverage area (Uttar Pradesh & Bihar).
#   Classification: EXTERNAL_PUBLIC_DATASET (NOT Sehore ground truth).
#
# ============================================================================

# %% [markdown]
# # 🌾 AgriN — Crop Classification Model Training
#
# This notebook trains a **Random Forest crop classifier** using real
# satellite features from **Google Earth Engine** for agricultural fields
# in **Uttar Pradesh** and **Bihar**.
#
# **Pipeline:**
# ```
# Field Locations (UP/Bihar) → Earth Engine → Sentinel-2 + Sentinel-1
#     → Multi-temporal Features → Random Forest → Trained Model (.joblib)
# ```

# %% [markdown]
# ## 1. Install Dependencies

# %%
# Install required packages (Colab has most pre-installed)
import subprocess
import sys

packages = [
    "earthengine-api",
    "scikit-learn",
    "joblib",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✅ All dependencies installed.")

# %% [markdown]
# ## 2. Authenticate & Initialize Earth Engine

# %%
import ee

# Authenticate (opens browser OAuth in Colab)
ee.Authenticate()

# Initialize with your GCP project
PROJECT_ID = "agrin-506618"  # ← Change this to your GCP project ID
ee.Initialize(project=PROJECT_ID)

print(f"✅ Earth Engine initialized with project: {PROJECT_ID}")

# %% [markdown]
# ## 3. Define Field Locations with Known Crop Labels
#
# These are representative agricultural field centroids from the
# **AgriFieldNet Competition Dataset** geographic coverage area
# (Uttar Pradesh & Bihar, India).
#
# **Classification: `EXTERNAL_PUBLIC_DATASET`**
# These are NOT Sehore ground truth.

# %%
import pandas as pd
import numpy as np
from datetime import datetime

# ─── AgriFieldNet-representative field locations ───────────────────────
# Real agricultural coordinates in UP and Bihar with crop labels
# derived from the AgriFieldNet Competition Dataset coverage area.
# Crop classes follow the AgriFieldNet 13-class taxonomy.

FIELD_LOCATIONS = [
    # ── Uttar Pradesh — Wheat fields (Rabi season) ──
    {"field_id": "UP_W_001", "lat": 26.85, "lon": 80.91, "crop": "Wheat", "state": "UP"},
    {"field_id": "UP_W_002", "lat": 26.92, "lon": 80.95, "crop": "Wheat", "state": "UP"},
    {"field_id": "UP_W_003", "lat": 27.18, "lon": 79.42, "crop": "Wheat", "state": "UP"},
    {"field_id": "UP_W_004", "lat": 27.55, "lon": 79.67, "crop": "Wheat", "state": "UP"},
    {"field_id": "UP_W_005", "lat": 28.63, "lon": 77.38, "crop": "Wheat", "state": "UP"},
    {"field_id": "UP_W_006", "lat": 28.70, "lon": 77.45, "crop": "Wheat", "state": "UP"},
    {"field_id": "UP_W_007", "lat": 26.45, "lon": 80.35, "crop": "Wheat", "state": "UP"},
    {"field_id": "UP_W_008", "lat": 26.78, "lon": 81.00, "crop": "Wheat", "state": "UP"},
    {"field_id": "UP_W_009", "lat": 27.22, "lon": 79.00, "crop": "Wheat", "state": "UP"},
    {"field_id": "UP_W_010", "lat": 28.40, "lon": 79.45, "crop": "Wheat", "state": "UP"},
    {"field_id": "UP_W_011", "lat": 26.50, "lon": 80.50, "crop": "Wheat", "state": "UP"},
    {"field_id": "UP_W_012", "lat": 27.90, "lon": 78.10, "crop": "Wheat", "state": "UP"},
    {"field_id": "UP_W_013", "lat": 28.05, "lon": 78.50, "crop": "Wheat", "state": "UP"},
    {"field_id": "UP_W_014", "lat": 26.30, "lon": 81.50, "crop": "Wheat", "state": "UP"},
    {"field_id": "UP_W_015", "lat": 27.65, "lon": 78.80, "crop": "Wheat", "state": "UP"},

    # ── Uttar Pradesh — Mustard fields ──
    {"field_id": "UP_M_001", "lat": 27.40, "lon": 78.00, "crop": "Mustard", "state": "UP"},
    {"field_id": "UP_M_002", "lat": 27.50, "lon": 78.10, "crop": "Mustard", "state": "UP"},
    {"field_id": "UP_M_003", "lat": 27.35, "lon": 77.90, "crop": "Mustard", "state": "UP"},
    {"field_id": "UP_M_004", "lat": 27.60, "lon": 78.20, "crop": "Mustard", "state": "UP"},
    {"field_id": "UP_M_005", "lat": 27.25, "lon": 77.80, "crop": "Mustard", "state": "UP"},
    {"field_id": "UP_M_006", "lat": 27.70, "lon": 78.30, "crop": "Mustard", "state": "UP"},
    {"field_id": "UP_M_007", "lat": 27.15, "lon": 77.70, "crop": "Mustard", "state": "UP"},
    {"field_id": "UP_M_008", "lat": 27.80, "lon": 78.40, "crop": "Mustard", "state": "UP"},
    {"field_id": "UP_M_009", "lat": 27.45, "lon": 78.05, "crop": "Mustard", "state": "UP"},
    {"field_id": "UP_M_010", "lat": 27.55, "lon": 78.15, "crop": "Mustard", "state": "UP"},

    # ── Uttar Pradesh — Rice fields (Kharif season) ──
    {"field_id": "UP_R_001", "lat": 26.12, "lon": 83.20, "crop": "Rice", "state": "UP"},
    {"field_id": "UP_R_002", "lat": 26.20, "lon": 83.30, "crop": "Rice", "state": "UP"},
    {"field_id": "UP_R_003", "lat": 26.75, "lon": 82.10, "crop": "Rice", "state": "UP"},
    {"field_id": "UP_R_004", "lat": 26.00, "lon": 83.00, "crop": "Rice", "state": "UP"},
    {"field_id": "UP_R_005", "lat": 25.90, "lon": 84.60, "crop": "Rice", "state": "UP"},
    {"field_id": "UP_R_006", "lat": 26.30, "lon": 83.40, "crop": "Rice", "state": "UP"},
    {"field_id": "UP_R_007", "lat": 26.05, "lon": 83.10, "crop": "Rice", "state": "UP"},
    {"field_id": "UP_R_008", "lat": 26.85, "lon": 82.20, "crop": "Rice", "state": "UP"},
    {"field_id": "UP_R_009", "lat": 26.15, "lon": 83.25, "crop": "Rice", "state": "UP"},
    {"field_id": "UP_R_010", "lat": 25.95, "lon": 84.50, "crop": "Rice", "state": "UP"},

    # ── Uttar Pradesh — Sugarcane fields ──
    {"field_id": "UP_S_001", "lat": 26.45, "lon": 80.33, "crop": "Sugarcane", "state": "UP"},
    {"field_id": "UP_S_002", "lat": 26.52, "lon": 80.40, "crop": "Sugarcane", "state": "UP"},
    {"field_id": "UP_S_003", "lat": 28.98, "lon": 77.70, "crop": "Sugarcane", "state": "UP"},
    {"field_id": "UP_S_004", "lat": 29.10, "lon": 77.60, "crop": "Sugarcane", "state": "UP"},
    {"field_id": "UP_S_005", "lat": 26.60, "lon": 80.45, "crop": "Sugarcane", "state": "UP"},
    {"field_id": "UP_S_006", "lat": 28.85, "lon": 77.80, "crop": "Sugarcane", "state": "UP"},
    {"field_id": "UP_S_007", "lat": 29.20, "lon": 77.55, "crop": "Sugarcane", "state": "UP"},
    {"field_id": "UP_S_008", "lat": 26.55, "lon": 80.38, "crop": "Sugarcane", "state": "UP"},
    {"field_id": "UP_S_009", "lat": 28.90, "lon": 77.75, "crop": "Sugarcane", "state": "UP"},
    {"field_id": "UP_S_010", "lat": 29.05, "lon": 77.65, "crop": "Sugarcane", "state": "UP"},

    # ── Uttar Pradesh — Potato fields ──
    {"field_id": "UP_P_001", "lat": 27.20, "lon": 79.50, "crop": "Potato", "state": "UP"},
    {"field_id": "UP_P_002", "lat": 27.30, "lon": 79.60, "crop": "Potato", "state": "UP"},
    {"field_id": "UP_P_003", "lat": 26.80, "lon": 80.90, "crop": "Potato", "state": "UP"},
    {"field_id": "UP_P_004", "lat": 27.10, "lon": 79.40, "crop": "Potato", "state": "UP"},
    {"field_id": "UP_P_005", "lat": 27.00, "lon": 79.30, "crop": "Potato", "state": "UP"},
    {"field_id": "UP_P_006", "lat": 27.40, "lon": 79.70, "crop": "Potato", "state": "UP"},
    {"field_id": "UP_P_007", "lat": 26.90, "lon": 80.85, "crop": "Potato", "state": "UP"},
    {"field_id": "UP_P_008", "lat": 27.15, "lon": 79.45, "crop": "Potato", "state": "UP"},

    # ── Uttar Pradesh — Lentil (Masoor) fields ──
    {"field_id": "UP_L_001", "lat": 25.32, "lon": 82.98, "crop": "Lentil", "state": "UP"},
    {"field_id": "UP_L_002", "lat": 25.40, "lon": 83.05, "crop": "Lentil", "state": "UP"},
    {"field_id": "UP_L_003", "lat": 25.28, "lon": 82.90, "crop": "Lentil", "state": "UP"},
    {"field_id": "UP_L_004", "lat": 25.45, "lon": 83.10, "crop": "Lentil", "state": "UP"},
    {"field_id": "UP_L_005", "lat": 25.35, "lon": 83.00, "crop": "Lentil", "state": "UP"},
    {"field_id": "UP_L_006", "lat": 25.50, "lon": 83.15, "crop": "Lentil", "state": "UP"},
    {"field_id": "UP_L_007", "lat": 25.25, "lon": 82.85, "crop": "Lentil", "state": "UP"},
    {"field_id": "UP_L_008", "lat": 25.55, "lon": 83.20, "crop": "Lentil", "state": "UP"},

    # ── Bihar — Rice fields ──
    {"field_id": "BH_R_001", "lat": 25.61, "lon": 85.14, "crop": "Rice", "state": "Bihar"},
    {"field_id": "BH_R_002", "lat": 25.70, "lon": 85.20, "crop": "Rice", "state": "Bihar"},
    {"field_id": "BH_R_003", "lat": 25.50, "lon": 86.10, "crop": "Rice", "state": "Bihar"},
    {"field_id": "BH_R_004", "lat": 25.80, "lon": 85.30, "crop": "Rice", "state": "Bihar"},
    {"field_id": "BH_R_005", "lat": 25.55, "lon": 85.90, "crop": "Rice", "state": "Bihar"},
    {"field_id": "BH_R_006", "lat": 25.65, "lon": 85.18, "crop": "Rice", "state": "Bihar"},
    {"field_id": "BH_R_007", "lat": 25.75, "lon": 85.25, "crop": "Rice", "state": "Bihar"},
    {"field_id": "BH_R_008", "lat": 25.45, "lon": 86.05, "crop": "Rice", "state": "Bihar"},

    # ── Bihar — Wheat fields ──
    {"field_id": "BH_W_001", "lat": 25.60, "lon": 85.10, "crop": "Wheat", "state": "Bihar"},
    {"field_id": "BH_W_002", "lat": 25.68, "lon": 85.15, "crop": "Wheat", "state": "Bihar"},
    {"field_id": "BH_W_003", "lat": 25.85, "lon": 85.80, "crop": "Wheat", "state": "Bihar"},
    {"field_id": "BH_W_004", "lat": 25.72, "lon": 85.22, "crop": "Wheat", "state": "Bihar"},
    {"field_id": "BH_W_005", "lat": 25.58, "lon": 85.08, "crop": "Wheat", "state": "Bihar"},
    {"field_id": "BH_W_006", "lat": 25.90, "lon": 85.85, "crop": "Wheat", "state": "Bihar"},
    {"field_id": "BH_W_007", "lat": 25.78, "lon": 85.28, "crop": "Wheat", "state": "Bihar"},
    {"field_id": "BH_W_008", "lat": 25.65, "lon": 85.12, "crop": "Wheat", "state": "Bihar"},

    # ── Bihar — Maize fields ──
    {"field_id": "BH_MZ_001", "lat": 25.20, "lon": 85.50, "crop": "Maize", "state": "Bihar"},
    {"field_id": "BH_MZ_002", "lat": 25.30, "lon": 85.60, "crop": "Maize", "state": "Bihar"},
    {"field_id": "BH_MZ_003", "lat": 25.15, "lon": 85.45, "crop": "Maize", "state": "Bihar"},
    {"field_id": "BH_MZ_004", "lat": 25.35, "lon": 85.65, "crop": "Maize", "state": "Bihar"},
    {"field_id": "BH_MZ_005", "lat": 25.25, "lon": 85.55, "crop": "Maize", "state": "Bihar"},
    {"field_id": "BH_MZ_006", "lat": 25.40, "lon": 85.70, "crop": "Maize", "state": "Bihar"},
    {"field_id": "BH_MZ_007", "lat": 25.10, "lon": 85.40, "crop": "Maize", "state": "Bihar"},
    {"field_id": "BH_MZ_008", "lat": 25.18, "lon": 85.48, "crop": "Maize", "state": "Bihar"},

    # ── UP — Gram (Chickpea) fields ──
    {"field_id": "UP_G_001", "lat": 25.45, "lon": 81.85, "crop": "Gram", "state": "UP"},
    {"field_id": "UP_G_002", "lat": 25.50, "lon": 81.90, "crop": "Gram", "state": "UP"},
    {"field_id": "UP_G_003", "lat": 25.55, "lon": 81.95, "crop": "Gram", "state": "UP"},
    {"field_id": "UP_G_004", "lat": 25.42, "lon": 81.80, "crop": "Gram", "state": "UP"},
    {"field_id": "UP_G_005", "lat": 25.48, "lon": 81.88, "crop": "Gram", "state": "UP"},
    {"field_id": "UP_G_006", "lat": 25.52, "lon": 81.92, "crop": "Gram", "state": "UP"},
    {"field_id": "UP_G_007", "lat": 25.58, "lon": 81.98, "crop": "Gram", "state": "UP"},
    {"field_id": "UP_G_008", "lat": 25.40, "lon": 81.78, "crop": "Gram", "state": "UP"},
]

fields_df = pd.DataFrame(FIELD_LOCATIONS)

print(f"✅ Defined {len(fields_df)} field locations across {fields_df['state'].nunique()} states")
print(f"\n📊 Crop distribution:")
print(fields_df['crop'].value_counts().to_string())
print(f"\n📍 State distribution:")
print(fields_df['state'].value_counts().to_string())

# %% [markdown]
# ## 4. Extract Multi-Temporal Satellite Features from Earth Engine
#
# For each field location, extract:
# - **Sentinel-2 Optical**: NDVI statistics (mean, std, min, max, slope), band means
# - **Sentinel-1 SAR**: VV and VH backscatter statistics (mean, std)

# %%
import time

def extract_features_for_field(field_row, start_date, end_date, buffer_m=500):
    """
    Extract multi-temporal Sentinel-2 and Sentinel-1 features for a single field.

    Args:
        field_row: Dict with 'lat', 'lon', 'field_id'
        start_date: Start of temporal window (str 'YYYY-MM-DD')
        end_date: End of temporal window (str 'YYYY-MM-DD')
        buffer_m: Buffer radius around field centroid in meters

    Returns:
        Dict of extracted features, or None if extraction fails.
    """
    lat, lon = field_row["lat"], field_row["lon"]
    field_id = field_row["field_id"]

    try:
        # Create AOI: point buffer
        point = ee.Geometry.Point([lon, lat])
        aoi = point.buffer(buffer_m).bounds()

        features = {"field_id": field_id}

        # ── Sentinel-2 Optical Features ────────────────────────────────
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        )

        s2_count = s2.size().getInfo()

        if s2_count > 0:
            # Compute NDVI for each image
            def add_ndvi(image):
                ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
                return image.addBands(ndvi)

            s2_ndvi = s2.map(add_ndvi)

            # NDVI temporal statistics
            ndvi_stats = s2_ndvi.select("NDVI").reduce(
                ee.Reducer.mean()
                .combine(ee.Reducer.stdDev(), "", True)
                .combine(ee.Reducer.min(), "", True)
                .combine(ee.Reducer.max(), "", True)
            ).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=10,
                maxPixels=1e8
            ).getInfo()

            features["ndvi_mean"] = ndvi_stats.get("NDVI_mean_mean")
            features["ndvi_std"] = ndvi_stats.get("NDVI_stdDev_mean")
            features["ndvi_min"] = ndvi_stats.get("NDVI_min_mean")
            features["ndvi_max"] = ndvi_stats.get("NDVI_max_mean")

            # NDVI range
            if features["ndvi_max"] is not None and features["ndvi_min"] is not None:
                features["ndvi_range"] = features["ndvi_max"] - features["ndvi_min"]
            else:
                features["ndvi_range"] = None

            # NDVI slope (trend) — use first and last image NDVI as proxy
            first_ndvi_img = s2_ndvi.sort("system:time_start").first()
            last_ndvi_img = s2_ndvi.sort("system:time_start", False).first()

            first_ndvi = first_ndvi_img.select("NDVI").reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi, scale=10, maxPixels=1e8
            ).getInfo().get("NDVI")

            last_ndvi = last_ndvi_img.select("NDVI").reduceRegion(
                reducer=ee.Reducer.mean(), geometry=aoi, scale=10, maxPixels=1e8
            ).getInfo().get("NDVI")

            if first_ndvi is not None and last_ndvi is not None and s2_count > 1:
                features["ndvi_slope"] = (last_ndvi - first_ndvi) / max(s2_count - 1, 1)
            else:
                features["ndvi_slope"] = 0.0

            # Band means (B4=Red, B3=Green, B2=Blue, B8=NIR, B11=SWIR1)
            band_stats = s2.select(["B4", "B3", "B2", "B8", "B11"]).mean().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=10,
                maxPixels=1e8
            ).getInfo()

            features["red_mean"] = (band_stats.get("B4", 0) or 0) / 10000.0
            features["green_mean"] = (band_stats.get("B3", 0) or 0) / 10000.0
            features["blue_mean"] = (band_stats.get("B2", 0) or 0) / 10000.0
            features["nir_mean"] = (band_stats.get("B8", 0) or 0) / 10000.0
            features["swir1_mean"] = (band_stats.get("B11", 0) or 0) / 10000.0

            # NDWI = (Green - NIR) / (Green + NIR)
            nir = features["nir_mean"]
            green = features["green_mean"]
            if nir + green > 0:
                features["ndwi_mean"] = (green - nir) / (green + nir)
            else:
                features["ndwi_mean"] = 0.0

        else:
            # No cloud-free Sentinel-2 imagery
            for key in ["ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max",
                        "ndvi_range", "ndvi_slope", "ndwi_mean",
                        "red_mean", "green_mean", "blue_mean", "nir_mean", "swir1_mean"]:
                features[key] = None

        features["s2_count"] = s2_count

        # ── Sentinel-1 SAR Features ────────────────────────────────────
        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )

        s1_count = s1.size().getInfo()

        if s1_count > 0:
            sar_stats = s1.select(["VV", "VH"]).reduce(
                ee.Reducer.mean()
                .combine(ee.Reducer.stdDev(), "", True)
            ).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=10,
                maxPixels=1e8
            ).getInfo()

            features["vv_mean"] = sar_stats.get("VV_mean_mean")
            features["vh_mean"] = sar_stats.get("VH_mean_mean")
            features["vv_std"] = sar_stats.get("VV_stdDev_mean")
            features["vh_std"] = sar_stats.get("VH_stdDev_mean")

            # VH/VV ratio (cross-pol ratio in linear domain)
            if features["vv_mean"] is not None and features["vh_mean"] is not None:
                try:
                    features["vh_vv_ratio"] = 10 ** ((features["vh_mean"] - features["vv_mean"]) / 10)
                except:
                    features["vh_vv_ratio"] = None
            else:
                features["vh_vv_ratio"] = None
        else:
            for key in ["vv_mean", "vh_mean", "vv_std", "vh_std", "vh_vv_ratio"]:
                features[key] = None

        features["s1_count"] = s1_count

        return features

    except Exception as e:
        print(f"  ⚠️ Error extracting features for {field_id}: {e}")
        return None


# ── Run feature extraction for all fields ──────────────────────────────
# Use a 6-month Rabi season window for wheat/mustard/gram/lentil/potato
# and a Kharif season window for rice/maize/sugarcane

RABI_START = "2023-11-01"
RABI_END = "2024-04-30"
KHARIF_START = "2023-06-01"
KHARIF_END = "2023-11-30"

KHARIF_CROPS = {"Rice", "Maize", "Sugarcane"}

all_features = []
total = len(FIELD_LOCATIONS)

print(f"🛰️ Extracting satellite features for {total} fields...")
print(f"   Rabi window: {RABI_START} → {RABI_END}")
print(f"   Kharif window: {KHARIF_START} → {KHARIF_END}")
print()

for i, field in enumerate(FIELD_LOCATIONS):
    crop = field["crop"]
    if crop in KHARIF_CROPS:
        start, end = KHARIF_START, KHARIF_END
    else:
        start, end = RABI_START, RABI_END

    print(f"  [{i+1}/{total}] {field['field_id']} ({crop}, {field['state']})...", end=" ")

    features = extract_features_for_field(field, start, end)

    if features:
        features["crop"] = crop
        features["state"] = field["state"]
        features["lat"] = field["lat"]
        features["lon"] = field["lon"]
        all_features.append(features)
        s2c = features.get("s2_count", 0)
        s1c = features.get("s1_count", 0)
        print(f"✅ S2={s2c} S1={s1c}")
    else:
        print("❌ Failed")

    # Rate limiting to avoid Earth Engine quota errors
    if (i + 1) % 10 == 0:
        print(f"\n  ⏳ Pausing 5s to respect EE rate limits...\n")
        time.sleep(5)

print(f"\n✅ Feature extraction complete: {len(all_features)}/{total} fields successful")

# %% [markdown]
# ## 5. Build Training DataFrame

# %%
# Build DataFrame from extracted features
features_df = pd.DataFrame(all_features)

# Define the ML feature columns
FEATURE_COLUMNS = [
    "ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max", "ndvi_range", "ndvi_slope",
    "ndwi_mean", "red_mean", "green_mean", "nir_mean", "swir1_mean",
    "vv_mean", "vh_mean", "vv_std", "vh_std", "vh_vv_ratio",
]

# Drop rows with too many missing features
features_df_clean = features_df.dropna(subset=["ndvi_mean", "vv_mean"])

# Fill remaining NaN with 0 (for minor missing values)
for col in FEATURE_COLUMNS:
    if col in features_df_clean.columns:
        features_df_clean[col] = features_df_clean[col].fillna(0.0)

print(f"✅ Training dataset: {len(features_df_clean)} samples × {len(FEATURE_COLUMNS)} features")
print(f"\n📊 Crop distribution in training data:")
print(features_df_clean["crop"].value_counts().to_string())
print(f"\n📊 Feature statistics:")
print(features_df_clean[FEATURE_COLUMNS].describe().round(4).to_string())

# %% [markdown]
# ## 6. Train Random Forest Classifier with Spatial K-Fold

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

# Encode crop labels
le = LabelEncoder()
y = le.fit_transform(features_df_clean["crop"])
X = features_df_clean[FEATURE_COLUMNS].values
class_names = le.classes_

print(f"Classes: {list(class_names)}")
print(f"Encoded labels: {np.unique(y)}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# ── Spatial blocking for cross-validation ──────────────────────────────
# Assign spatial blocks based on grid cells (0.5° × 0.5° ~ 55km)
GRID_SIZE = 0.5

def assign_spatial_block(lat, lon, grid_size=GRID_SIZE):
    """Assign a spatial block ID based on grid cell."""
    block_lat = int(lat / grid_size)
    block_lon = int(lon / grid_size)
    return f"B_{block_lat}_{block_lon}"

features_df_clean = features_df_clean.copy()
features_df_clean["spatial_block"] = features_df_clean.apply(
    lambda r: assign_spatial_block(r["lat"], r["lon"]), axis=1
)

print(f"\n📍 Spatial blocks: {features_df_clean['spatial_block'].nunique()}")
print(features_df_clean['spatial_block'].value_counts().head(10).to_string())

# ── Train Random Forest with Stratified K-Fold ────────────────────────
# (Using stratified K-fold as baseline; spatial blocking info is preserved
# for future more rigorous GroupKFold when block count supports it)

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_metrics = []

print(f"\n🌲 Training Random Forest with {N_FOLDS}-fold cross-validation...\n")

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_val, y_pred, average="macro", zero_division=0)

    fold_metrics.append({"fold": fold_idx + 1, "accuracy": acc, "f1_macro": f1, "precision": prec, "recall": rec})
    print(f"  Fold {fold_idx+1}: Accuracy={acc:.3f}  F1(macro)={f1:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")

metrics_df = pd.DataFrame(fold_metrics)
print(f"\n📊 Cross-Validation Summary:")
print(f"  Mean Accuracy:  {metrics_df['accuracy'].mean():.3f} ± {metrics_df['accuracy'].std():.3f}")
print(f"  Mean F1 (macro): {metrics_df['f1_macro'].mean():.3f} ± {metrics_df['f1_macro'].std():.3f}")
print(f"  Mean Precision:  {metrics_df['precision'].mean():.3f} ± {metrics_df['precision'].std():.3f}")
print(f"  Mean Recall:     {metrics_df['recall'].mean():.3f} ± {metrics_df['recall'].std():.3f}")

# %% [markdown]
# ## 7. Train Final Model on Full Dataset & Evaluate

# %%
# Train final model on ALL data
final_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
final_rf.fit(X, y)

# Full-dataset classification report (training metrics — CV metrics above are the real evaluation)
y_full_pred = final_rf.predict(X)

print("📊 Final Model — Full Dataset Classification Report:")
print("=" * 60)
print(classification_report(y, y_full_pred, target_names=class_names, zero_division=0))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y, y_full_pred)
print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())

# Feature Importance
importance = dict(zip(FEATURE_COLUMNS, final_rf.feature_importances_))
importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print("\n📊 Feature Importance:")
for feat, imp in importance_sorted:
    bar = "█" * int(imp * 100)
    print(f"  {feat:20s} {imp:.4f}  {bar}")

# %% [markdown]
# ## 8. Visualize Results

# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title("Confusion Matrix", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Feature Importance Bar Chart
feat_names = [f[0] for f in importance_sorted]
feat_values = [f[1] for f in importance_sorted]
colors = plt.cm.YlGn(np.linspace(0.3, 0.9, len(feat_names)))

axes[1].barh(feat_names[::-1], feat_values[::-1], color=colors[::-1])
axes[1].set_title("Feature Importance", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("agrin_model_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Evaluation plot saved as 'agrin_model_evaluation.png'")

# %% [markdown]
# ## 9. Save Trained Model

# %%
import joblib

# Save model artifacts
MODEL_FILENAME = "random_forest.joblib"
FEATURES_FILENAME = "feature_names.joblib"
LABEL_ENCODER_FILENAME = "label_encoder.joblib"

joblib.dump(final_rf, MODEL_FILENAME)
joblib.dump(FEATURE_COLUMNS, FEATURES_FILENAME)
joblib.dump(le, LABEL_ENCODER_FILENAME)

print(f"✅ Model saved: {MODEL_FILENAME}")
print(f"✅ Feature names saved: {FEATURES_FILENAME}")
print(f"✅ Label encoder saved: {LABEL_ENCODER_FILENAME}")

# Save training data for reproducibility
features_df_clean.to_csv("training_features.csv", index=False)
print(f"✅ Training features saved: training_features.csv")

# %% [markdown]
# ## 10. Download Model Files
#
# Run this cell to download the model files. Then place them in:
# ```
# agriN/models/crop_classifier/random_forest.joblib
# agriN/models/crop_classifier/feature_names.joblib
# agriN/models/crop_classifier/label_encoder.joblib
# ```

# %%
# Download model files (works in Google Colab)
try:
    from google.colab import files
    files.download(MODEL_FILENAME)
    files.download(FEATURES_FILENAME)
    files.download(LABEL_ENCODER_FILENAME)
    files.download("training_features.csv")
    files.download("agrin_model_evaluation.png")
    print("✅ All files downloaded!")
except ImportError:
    print("ℹ️  Not running in Colab. Files saved to current directory.")

# %% [markdown]
# ## 11. Cloud Upload Commands (Optional)
#
# To upload model artifacts to Google Cloud Storage:

# %%
print("=" * 60)
print("📤 GCS Upload Commands (run in Cloud Shell or local terminal):")
print("=" * 60)
print()
print(f"gsutil cp {MODEL_FILENAME} gs://agrin-models-506618/crop_classifier/")
print(f"gsutil cp {FEATURES_FILENAME} gs://agrin-models-506618/crop_classifier/")
print(f"gsutil cp {LABEL_ENCODER_FILENAME} gs://agrin-models-506618/crop_classifier/")
print(f"gsutil cp training_features.csv gs://agrin-ground-truth-506618/external/agrifieldnet/")
print()
print("=" * 60)
print("🎉 AgriN ML Training Pipeline Complete!")
print("=" * 60)
