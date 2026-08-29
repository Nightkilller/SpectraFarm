"""
SpectraFarm — Unified Ground-Truth Data Loader

Loads and merges crop labels + features from three sources:
  1. CropHarvest (NASA Harvest) — 95K real labeled datapoints with Sentinel-1/2 time-series
  2. AgriFieldNet India (Radiant Earth) — India-specific crop-type challenge dataset
  3. Existing synthetic CSV — our generated 75K training_features.csv

All sources are mapped to a consistent 17-feature schema matching our RF model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "ground_truth"
SYNTHETIC_CSV = DATA_DIR / "training_features.csv"

# Our standard 17-feature schema
FEATURE_COLS = [
    "blue_mean", "green_mean", "red_mean", "nir_mean", "swir1_mean",
    "ndvi_mean", "ndvi_min", "ndvi_max", "ndvi_std", "ndvi_range", "ndvi_slope",
    "ndwi_mean",
    "vv_mean", "vv_std", "vh_mean", "vh_std", "vh_vv_ratio",
]

# CropHarvest band ordering (from cropharvest/engineer.py):
# 18 bands per timestep in the h5 arrays:
#   [0] B2 (Blue), [1] B3 (Green), [2] B4 (Red), [3] B5, [4] B6, [5] B7,
#   [6] B8 (NIR), [7] B8A, [8] B9, [9] B11 (SWIR1), [10] B12 (SWIR2),
#   [11] VV, [12] VH,
#   [13] temperature, [14] total_precipitation,
#   [15] soil_type (SRTM), [16] elevation, [17] slope
CH_BAND_BLUE = 0
CH_BAND_GREEN = 1
CH_BAND_RED = 2
CH_BAND_NIR = 6
CH_BAND_SWIR1 = 9
CH_BAND_VV = 11
CH_BAND_VH = 12


def _aggregate_cropharvest_timeseries(x: np.ndarray) -> dict:
    """
    Convert a CropHarvest time-series array (T × 18 bands) into our 17-feature dict.
    CropHarvest arrays have shape (12, 18) — 12 monthly timesteps × 18 bands.
    """
    T = x.shape[0]

    # Extract band time-series
    blue = x[:, CH_BAND_BLUE]
    green = x[:, CH_BAND_GREEN]
    red = x[:, CH_BAND_RED]
    nir = x[:, CH_BAND_NIR]
    swir1 = x[:, CH_BAND_SWIR1]
    vv = x[:, CH_BAND_VV]
    vh = x[:, CH_BAND_VH]

    # Compute NDVI time-series: (NIR - Red) / (NIR + Red + 1e-8)
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # Compute NDWI time-series: (NIR - SWIR1) / (NIR + SWIR1 + 1e-8)
    ndwi = (nir - swir1) / (nir + swir1 + 1e-8)
    ndwi = np.clip(ndwi, -1.0, 1.0)

    # NDVI temporal slope (linear trend)
    if T >= 3:
        t_axis = np.arange(T, dtype=float)
        slope = float(np.polyfit(t_axis, ndvi, 1)[0])
    else:
        slope = 0.0

    return {
        "blue_mean": float(np.mean(blue)),
        "green_mean": float(np.mean(green)),
        "red_mean": float(np.mean(red)),
        "nir_mean": float(np.mean(nir)),
        "swir1_mean": float(np.mean(swir1)),
        "ndvi_mean": float(np.mean(ndvi)),
        "ndvi_min": float(np.min(ndvi)),
        "ndvi_max": float(np.max(ndvi)),
        "ndvi_std": float(np.std(ndvi)),
        "ndvi_range": float(np.max(ndvi) - np.min(ndvi)),
        "ndvi_slope": float(slope),
        "ndwi_mean": float(np.mean(ndwi)),
        "vv_mean": float(np.mean(vv)),
        "vv_std": float(np.std(vv)),
        "vh_mean": float(np.mean(vh)),
        "vh_std": float(np.std(vh)),
        "vh_vv_ratio": float(np.mean(vh / (vv + 1e-8))),
    }


def load_cropharvest_india(data_dir: Optional[str] = None, max_samples: int = 10000) -> pd.DataFrame:
    """
    Load CropHarvest dataset, filter to India records, and convert to our 17-feature schema.

    CropHarvest downloads ~500MB of data from Zenodo on first run.
    Each sample is a 12-month time-series of 18 satellite bands.
    We aggregate each time-series into our 17 temporal statistics.

    Args:
        data_dir: Directory to store/cache CropHarvest data.
        max_samples: Maximum number of India samples to load (for speed).

    Returns:
        DataFrame with columns: field_id, crop, state, lat, lon, + 17 features.
    """
    try:
        from cropharvest.datasets import CropHarvest
        from cropharvest.countries import BBox
    except ImportError:
        logger.warning("cropharvest not installed. Run: pip install cropharvest")
        return pd.DataFrame()

    if data_dir is None:
        data_dir = str(PROJECT_ROOT / "data" / "cropharvest")

    logger.info(f"Loading CropHarvest dataset from {data_dir} (this downloads ~500MB on first run)...")

    try:
        # India bounding box (approximate)
        india_bbox = BBox(min_lat=7.0, max_lat=35.0, min_lon=68.0, max_lon=97.0, name="india")

        # Load the full dataset with download enabled
        dataset = CropHarvest(data_dir, download=True)

        # Get labels as GeoDataFrame
        labels_gdf = dataset.labels.as_geojson()

        # Filter to India
        india_mask = (
            (labels_gdf["lat"] >= 7.0) & (labels_gdf["lat"] <= 35.0) &
            (labels_gdf["lon"] >= 68.0) & (labels_gdf["lon"] <= 97.0)
        )
        india_labels = labels_gdf[india_mask].copy()

        if len(india_labels) == 0:
            logger.warning("No India records found in CropHarvest dataset.")
            return pd.DataFrame()

        logger.info(f"Found {len(india_labels)} India records in CropHarvest.")

        # Cap to max_samples
        if len(india_labels) > max_samples:
            india_labels = india_labels.sample(n=max_samples, random_state=42)

        records = []
        for idx, row in india_labels.iterrows():
            try:
                # Load the h5 feature file for this sample
                sample_x, sample_y = dataset[int(row.get("index", idx))]
                if sample_x is None or len(sample_x.shape) < 2:
                    continue

                feats = _aggregate_cropharvest_timeseries(sample_x)
                feats["field_id"] = f"CH_INDIA_{len(records):05d}"
                feats["lat"] = float(row.get("lat", 0))
                feats["lon"] = float(row.get("lon", 0))
                feats["state"] = "India"
                feats["source"] = "cropharvest"

                # Map label
                label = row.get("label", None)
                is_crop = row.get("is_crop", False)
                if label and isinstance(label, str):
                    feats["crop"] = _map_cropharvest_label(label)
                elif is_crop:
                    feats["crop"] = "Other"
                else:
                    continue  # Skip non-crop records without labels

                records.append(feats)
            except Exception as e:
                continue  # Skip problematic samples

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} CropHarvest India samples with 17-feature vectors.")
        return df

    except Exception as e:
        logger.error(f"Failed to load CropHarvest: {e}")
        return pd.DataFrame()


def _map_cropharvest_label(label: str) -> str:
    """Map CropHarvest label strings to our 10-class crop names."""
    label_lower = label.lower().strip()
    mapping = {
        "wheat": "Wheat",
        "rice": "Rice",
        "paddy": "Rice",
        "maize": "Maize",
        "corn": "Maize",
        "soybean": "Soybean",
        "soya": "Soybean",
        "mustard": "Mustard",
        "rapeseed": "Mustard",
        "cotton": "Cotton",
        "sugarcane": "Sugarcane",
        "sugar cane": "Sugarcane",
        "potato": "Potato",
        "lentil": "Lentil",
        "chickpea": "Gram",
        "gram": "Gram",
        "pulses": "Lentil",
    }
    for key, value in mapping.items():
        if key in label_lower:
            return value
    return "Other"


def load_agrifieldnet(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load AgriFieldNet India dataset if available locally.

    AgriFieldNet requires manual registration and download from Radiant Earth MLHub /
    Source Cooperative. This loader checks if the data exists locally and loads it.

    Args:
        data_dir: Path to downloaded AgriFieldNet data directory.

    Returns:
        DataFrame with columns: field_id, crop, state, lat, lon, + 17 features.
        Empty DataFrame if data is not present.
    """
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / "data" / "agrifieldnet")

    agri_path = Path(data_dir)
    if not agri_path.exists():
        logger.info(
            "AgriFieldNet data not found locally. To use it:\n"
            "  1. Register at https://source.coop/radiantearth/agrifieldnet-competition\n"
            "  2. Download the dataset\n"
            "  3. Extract to: data/agrifieldnet/\n"
            "Continuing without AgriFieldNet data."
        )
        return pd.DataFrame()

    # Look for common AgriFieldNet file patterns
    csv_files = list(agri_path.glob("**/*.csv"))
    geojson_files = list(agri_path.glob("**/*.geojson"))

    if not csv_files and not geojson_files:
        logger.info("AgriFieldNet directory exists but no data files found.")
        return pd.DataFrame()

    logger.info(f"Found AgriFieldNet data at {agri_path}")
    # TODO: Implement AgriFieldNet-specific feature extraction when data is available
    # This requires parsing the competition-specific format
    return pd.DataFrame()


def load_synthetic_csv(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load our existing synthetic training features CSV."""
    if csv_path is None:
        csv_path = str(SYNTHETIC_CSV)

    path = Path(csv_path)
    if not path.exists():
        logger.warning(f"Synthetic CSV not found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["source"] = "synthetic"
    logger.info(f"Loaded {len(df)} synthetic training samples from {path}")
    return df


def load_merged_ground_truth(
    use_cropharvest: bool = True,
    use_agrifieldnet: bool = True,
    use_synthetic: bool = True,
    cropharvest_max_samples: int = 10000,
) -> pd.DataFrame:
    """
    Load and merge all available ground-truth sources into one consistent DataFrame.

    Args:
        use_cropharvest: Whether to load CropHarvest data.
        use_agrifieldnet: Whether to check for AgriFieldNet data.
        use_synthetic: Whether to include synthetic training data.
        cropharvest_max_samples: Max CropHarvest India samples to load.

    Returns:
        Merged DataFrame with consistent schema: field_id, crop, state, lat, lon,
        source, + 17 feature columns.
    """
    dfs = []

    if use_synthetic:
        syn_df = load_synthetic_csv()
        if len(syn_df) > 0:
            dfs.append(syn_df)

    if use_cropharvest:
        ch_df = load_cropharvest_india(max_samples=cropharvest_max_samples)
        if len(ch_df) > 0:
            dfs.append(ch_df)

    if use_agrifieldnet:
        af_df = load_agrifieldnet()
        if len(af_df) > 0:
            dfs.append(af_df)

    if not dfs:
        logger.error("No ground-truth data sources available!")
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)

    # Ensure all 17 feature columns exist, fill missing with 0
    for col in FEATURE_COLS:
        if col not in merged.columns:
            merged[col] = 0.0

    # Drop rows with missing crop labels
    merged = merged.dropna(subset=["crop"])
    merged = merged[merged["crop"] != ""]

    # Report source breakdown
    if "source" in merged.columns:
        source_counts = merged["source"].value_counts()
        for src, count in source_counts.items():
            logger.info(f"  Source '{src}': {count} samples")

    logger.info(f"Total merged ground-truth dataset: {len(merged)} samples across {merged['crop'].nunique()} classes.")
    return merged
