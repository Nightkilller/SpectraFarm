"""
AgriN — Google Earth Engine Geospatial Client

Handles Earth Engine initialization, AOI definition, Sentinel-2 retrieval,
metadata extraction, and regional index statistics for the Sehore pilot region.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Optional

from src.config.settings import get_settings
from src.geospatial.indices import calculate_ndvi

logger = logging.getLogger(__name__)

_ee = None
_gee_initialized = False


def init_earth_engine(project: Optional[str] = None) -> bool:
    """
    Initialize Google Earth Engine using the configured project.

    Args:
        project: Google Cloud project ID (defaults to settings.gee_project / GEE_PROJECT)

    Returns:
        bool: True if initialization succeeded, False otherwise.
    """
    global _ee, _gee_initialized

    if _gee_initialized and _ee is not None:
        return True

    settings = get_settings()
    target_project = project or settings.gee_project or "agrin-506618"

    try:
        import ee
        ee.Initialize(project=target_project)
        _ee = ee
        _gee_initialized = True
        logger.info(f"[INFO] Earth Engine initialized successfully with project: {target_project}")
        return True
    except ImportError:
        logger.error("[ERROR] earthengine-api is not installed.")
        _gee_initialized = False
        return False
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize Earth Engine with project '{target_project}': {e}")
        _gee_initialized = False
        return False


def get_ee_module():
    """Return the initialized ee module or None."""
    if not _gee_initialized:
        init_earth_engine()
    return _ee


def get_sehore_aoi(use_buffer: bool = True) -> Any:
    """
    Construct the Sehore Pilot Test AOI geometry.

    Returns:
        ee.Geometry: Point buffer or BoundingBox geometry representing the pilot AOI.
    """
    ee = get_ee_module()
    if not ee:
        raise RuntimeError("Earth Engine is not initialized.")

    settings = get_settings()
    region = settings.pilot_region

    if use_buffer:
        center_lat = region.get("center_lat", 23.2000)
        center_lon = region.get("center_lon", 77.0800)
        buffer_dist = region.get("buffer_meters", 2000)
        point = ee.Geometry.Point([center_lon, center_lat])
        return point.buffer(buffer_dist).bounds()
    else:
        bbox = region.get("bbox", {})
        return ee.Geometry.Rectangle([
            bbox.get("min_lon", 77.0600),
            bbox.get("min_lat", 23.1800),
            bbox.get("max_lon", 77.1000),
            bbox.get("max_lat", 23.2200),
        ])


def query_sentinel2_imagery(
    aoi: Any,
    start_date: str | date,
    end_date: str | date,
    max_cloud_percentage: float = 20.0,
) -> Any:
    """
    Query the Sentinel-2 Surface Reflectance Harmonized collection filtered by AOI,
    temporal date window, and cloudy pixel threshold.

    Collection: COPERNICUS/S2_SR_HARMONIZED
    """
    ee = get_ee_module()
    if not ee:
        raise RuntimeError("Earth Engine is not initialized.")

    settings = get_settings()
    collection_name = settings.sentinel2_config.get("collection", "COPERNICUS/S2_SR_HARMONIZED")

    return (
        ee.ImageCollection(collection_name)
        .filterBounds(aoi)
        .filterDate(str(start_date), str(end_date))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_percentage))
        .sort("CLOUDY_PIXEL_PERCENTAGE")  # Prioritize lowest cloud cover
    )


def extract_image_metadata(image: Any) -> dict[str, Any]:
    """
    Extract essential scientific and acquisition metadata from a Sentinel-2 image.
    """
    ee = get_ee_module()
    if not ee:
        raise RuntimeError("Earth Engine is not initialized.")

    info = image.getInfo()
    props = info.get("properties", {})
    
    timestamp_ms = props.get("system:time_start", 0)
    obs_date = date.fromtimestamp(timestamp_ms / 1000) if timestamp_ms else None

    bands = [b.get("id") for b in info.get("bands", [])]

    return {
        "id": info.get("id", "Unknown"),
        "date": str(obs_date) if obs_date else "Unknown",
        "cloud_percentage": props.get("CLOUDY_PIXEL_PERCENTAGE", None),
        "spacecraft": props.get("SPACECRAFT_NAME", "Sentinel-2"),
        "processing_baseline": props.get("PROCESSING_BASELINE", "Unknown"),
        "bands": bands,
    }


def compute_aoi_ndvi_statistics(
    image: Any,
    aoi: Any,
    scale: int = 10,
) -> dict[str, float]:
    """
    Calculate summary statistics (min, mean, max) of the NDVI band over the AOI.
    """
    ee = get_ee_module()
    if not ee:
        raise RuntimeError("Earth Engine is not initialized.")

    # Ensure NDVI band exists
    image_with_ndvi = calculate_ndvi(image)
    ndvi_band = image_with_ndvi.select("NDVI")

    # Combine min, mean, max reducers
    combined_reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.min(), "", True)
        .combine(ee.Reducer.max(), "", True)
        .combine(ee.Reducer.stdDev(), "", True)
    )

    stats = ndvi_band.reduceRegion(
        reducer=combined_reducer,
        geometry=aoi,
        scale=scale,
        maxPixels=1e8,
    ).getInfo()

    return {
        "min": round(stats.get("NDVI_min", 0.0), 4) if stats.get("NDVI_min") is not None else None,
        "mean": round(stats.get("NDVI_mean", 0.0), 4) if stats.get("NDVI_mean") is not None else None,
        "max": round(stats.get("NDVI_max", 0.0), 4) if stats.get("NDVI_max") is not None else None,
        "stdDev": round(stats.get("NDVI_stdDev", 0.0), 4) if stats.get("NDVI_stdDev") is not None else None,
    }
