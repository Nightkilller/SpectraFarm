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
