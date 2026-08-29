"""
AgriN — Geospatial Indices Module

Calculates spectral vegetation indices from satellite observations.
NDVI = (NIR - Red) / (NIR + Red)
Using Sentinel-2 B8 (842 nm, NIR) and B4 (665 nm, Red).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def calculate_ndvi(image: Any) -> Any:
    """
    Calculate Normalized Difference Vegetation Index (NDVI) on an Earth Engine Sentinel-2 image.

    Formula:
        NDVI = (B8 - B4) / (B8 + B4)

    Args:
        image: ee.Image containing bands 'B8' (NIR) and 'B4' (Red).

    Returns:
        ee.Image with added 'NDVI' band, preserving original image metadata.
    """
    # Compute normalized difference: (B8 - B4) / (B8 + B4)
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    
    # Add NDVI band to the image while preserving metadata/system properties
    return image.addBands(ndvi)


def calculate_ndwi(image: Any) -> Any:
    """
    Calculate Normalized Difference Water Index (NDWI) on an Earth Engine Sentinel-2 image.

    Formula:
        NDWI = (B3 - B8) / (B3 + B8)  (McFeeters 1996)
        or (B8 - B11) / (B8 + B11)     (Gao 1996 / NDII for vegetation water content)

    Here using B8 (NIR) and B11 (SWIR1) for canopy moisture estimation.
    """
    ndwi = image.normalizedDifference(["B8", "B11"]).rename("NDWI")
    return image.addBands(ndwi)


def compute_vci(ndvi_current: float, ndvi_min: float, ndvi_max: float) -> float:
    """
    Compute Vegetation Condition Index (VCI).

    VCI is a standard drought monitoring metric that normalizes NDVI against
    historical min/max for the pixel/area, producing a 0-100% score.

    Formula:
        VCI = ((NDVI_current - NDVI_min) / (NDVI_max - NDVI_min)) × 100

    Interpretation (NOAA / FAO standard):
        VCI > 60%  → Healthy / No Stress
        30-60%     → Mild to Moderate Stress
        VCI < 30%  → Severe Stress / Drought

    Args:
        ndvi_current: Current NDVI observation value.
        ndvi_min: Historical minimum NDVI for this location/season.
        ndvi_max: Historical maximum NDVI for this location/season.

    Returns:
        VCI percentage (0.0 to 100.0).
    """
    denom = ndvi_max - ndvi_min
    if denom <= 0.01:
        # Avoid division by zero — if no seasonal variation, return 50%
        return 50.0
    vci = ((ndvi_current - ndvi_min) / denom) * 100.0
    return max(0.0, min(100.0, vci))


def classify_vci_stress(vci_pct: float) -> str:
    """
    Classify moisture stress level from VCI percentage.

    Returns:
        "Healthy", "Mild Stress", or "Severe Stress"
    """
    if vci_pct > 60.0:
        return "Healthy"
    elif vci_pct >= 30.0:
        return "Mild Stress"
    else:
        return "Severe Stress"


def compute_evi(nir: float, red: float, blue: float) -> float:
    """
    Compute Enhanced Vegetation Index (EVI).

    EVI is more robust than NDVI in dense canopy conditions and resists
    atmospheric noise using the blue band for correction.

    Formula:
        EVI = 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)

    Args:
        nir: Near-Infrared reflectance (Sentinel-2 B8).
        red: Red reflectance (Sentinel-2 B4).
        blue: Blue reflectance (Sentinel-2 B2).

    Returns:
        EVI value (typically -1.0 to 1.0).
    """
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    if abs(denom) < 1e-8:
        return 0.0
    evi = 2.5 * (nir - red) / denom
    return max(-1.0, min(1.0, evi))
