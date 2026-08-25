"""
AgriN — Feature Extraction

Extracts ML-ready features from satellite observations.
Computes temporal statistics (mean, std, min, max, trend) for each band/index.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.data.schemas import SatelliteObservation


def extract_optical_features(
    observations: list[SatelliteObservation],
) -> dict[str, float]:
    """
    Extract temporal optical features from Sentinel-2 observations.

    Returns a flat dict of features suitable for ML input:
      ndvi_mean, ndvi_std, ndvi_min, ndvi_max, ndvi_trend,
      ndwi_mean, red_mean, green_mean, blue_mean, nir_mean,
      swir1_mean, swir2_mean
    """
    features: dict[str, float] = {}

    # Filter to Sentinel-2 observations with NDVI
    s2_obs = [o for o in observations if o.satellite == "Sentinel-2" and o.ndvi is not None]

    if not s2_obs:
        return features

    # NDVI temporal statistics
    ndvi_vals = np.array([o.ndvi for o in s2_obs])
    features["ndvi_mean"] = float(np.mean(ndvi_vals))
    features["ndvi_std"] = float(np.std(ndvi_vals))
    features["ndvi_min"] = float(np.min(ndvi_vals))
    features["ndvi_max"] = float(np.max(ndvi_vals))
    features["ndvi_range"] = float(np.max(ndvi_vals) - np.min(ndvi_vals))

    # NDVI trend (linear slope over time)
    if len(ndvi_vals) >= 3:
        x = np.arange(len(ndvi_vals), dtype=float)
        slope = np.polyfit(x, ndvi_vals, 1)[0]
        features["ndvi_trend"] = float(slope)
    else:
        features["ndvi_trend"] = 0.0

    # NDWI
    ndwi_vals = [o.ndwi for o in s2_obs if o.ndwi is not None]
    if ndwi_vals:
        features["ndwi_mean"] = float(np.mean(ndwi_vals))

    # Band means
    for band in ["red", "green", "blue", "nir", "swir1", "swir2"]:
        vals = [getattr(o, band) for o in s2_obs if getattr(o, band) is not None]
        if vals:
            features[f"{band}_mean"] = float(np.mean(vals))

    return features


def extract_sar_features(
    observations: list[SatelliteObservation],
) -> dict[str, float]:
    """
    Extract temporal SAR features from Sentinel-1 observations.

    Returns: vv_mean, vh_mean, vh_vv_ratio_mean, vv_std, vh_std
    """
    features: dict[str, float] = {}

    s1_obs = [o for o in observations if o.satellite == "Sentinel-1"]

    if not s1_obs:
        return features

    vv_vals = [o.vv for o in s1_obs if o.vv is not None]
    vh_vals = [o.vh for o in s1_obs if o.vh is not None]
    ratio_vals = [o.vh_vv_ratio for o in s1_obs if o.vh_vv_ratio is not None]

    if vv_vals:
        features["vv_mean"] = float(np.mean(vv_vals))
        features["vv_std"] = float(np.std(vv_vals))

    if vh_vals:
        features["vh_mean"] = float(np.mean(vh_vals))
        features["vh_std"] = float(np.std(vh_vals))

    if ratio_vals:
        features["vh_vv_ratio_mean"] = float(np.mean(ratio_vals))

    return features


def combine_features(
    optical: dict[str, float],
    sar: dict[str, float],
) -> dict[str, float]:
    """Merge optical and SAR features into a single feature dict."""
    combined = {}
    combined.update(optical)
    combined.update(sar)
    return combined
