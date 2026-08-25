"""
AgriN — Google Earth Engine Client

Handles GEE authentication, Sentinel-2/1 data retrieval, cloud filtering,
temporal compositing, and index calculation.

Falls back to demo data if GEE is not configured.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Optional

from src.config.settings import get_settings
from src.data.schemas import BoundingBox, DataSource, SatelliteObservation

logger = logging.getLogger(__name__)

# GEE is optional — import lazily
_ee = None
_gee_initialized = False


def _init_gee() -> bool:
    """Attempt to initialize Google Earth Engine. Returns True if successful."""
    global _ee, _gee_initialized

    if _gee_initialized:
        return _ee is not None

    settings = get_settings()
    project = settings.gee_project

    if not project:
        logger.warning("GEE_PROJECT not set — satellite service will use demo data.")
        _gee_initialized = True
        return False

    try:
        import ee
        ee.Initialize(project=project)
        _ee = ee
        _gee_initialized = True
        logger.info(f"Google Earth Engine initialized with project: {project}")
        return True
    except ImportError:
        logger.warning("earthengine-api not installed. Run: pip install earthengine-api")
        _gee_initialized = True
        return False
    except Exception as e:
        logger.error(f"Failed to initialize GEE: {e}")
        _gee_initialized = True
        return False


def is_gee_available() -> bool:
    """Check if GEE is configured and available."""
    return _init_gee()


def get_sentinel2_observations(
    bbox: BoundingBox,
    start_date: date,
    end_date: date,
    farm_id: str,
    max_cloud_cover: int = 20,
) -> list[SatelliteObservation]:
    """
    Retrieve and process Sentinel-2 observations for the given AOI and date range.

    If GEE is unavailable, returns empty list (caller should use demo data).
    """
    if not _init_gee():
        logger.info("GEE unavailable — returning empty. Use demo data instead.")
        return []

    ee = _ee
    settings = get_settings()

    try:
        # Define AOI
        aoi = ee.Geometry.Rectangle([
            bbox.min_lon, bbox.min_lat,
            bbox.max_lon, bbox.max_lat,
        ])

        # Get Sentinel-2 collection
        collection = (
            ee.ImageCollection(settings.sentinel2_config["collection"])
            .filterBounds(aoi)
            .filterDate(str(start_date), str(end_date))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover))
            .sort("system:time_start")
        )

        count = collection.size().getInfo()
        logger.info(f"Found {count} Sentinel-2 images")

        if count == 0:
            return []

        # Process each image
        observations = []
        image_list = collection.toList(min(count, 24))  # Limit to 24 images

        for i in range(min(count, 24)):
            try:
                image = ee.Image(image_list.get(i))
                obs = _process_s2_image(image, aoi, farm_id)
                if obs:
                    observations.append(obs)
            except Exception as e:
                logger.warning(f"Failed to process S2 image {i}: {e}")
                continue

        return observations

    except Exception as e:
        logger.error(f"Sentinel-2 retrieval failed: {e}")
        return []


def get_sentinel1_observations(
    bbox: BoundingBox,
    start_date: date,
    end_date: date,
    farm_id: str,
) -> list[SatelliteObservation]:
    """
    Retrieve Sentinel-1 SAR observations for the given AOI and date range.
    """
    if not _init_gee():
        return []

    ee = _ee
    settings = get_settings()

    try:
        aoi = ee.Geometry.Rectangle([
            bbox.min_lon, bbox.min_lat,
            bbox.max_lon, bbox.max_lat,
        ])

        s1_config = settings.sentinel1_config
        collection = (
            ee.ImageCollection(s1_config["collection"])
            .filterBounds(aoi)
            .filterDate(str(start_date), str(end_date))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VV"
            ))
            .filter(ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VH"
            ))
            .sort("system:time_start")
        )

        count = collection.size().getInfo()
        logger.info(f"Found {count} Sentinel-1 images")

        if count == 0:
            return []

        observations = []
        image_list = collection.toList(min(count, 24))

        for i in range(min(count, 24)):
            try:
                image = ee.Image(image_list.get(i))
                obs = _process_s1_image(image, aoi, farm_id)
                if obs:
                    observations.append(obs)
            except Exception as e:
                logger.warning(f"Failed to process S1 image {i}: {e}")
                continue

        return observations

    except Exception as e:
        logger.error(f"Sentinel-1 retrieval failed: {e}")
        return []


def _process_s2_image(
    image: Any, aoi: Any, farm_id: str
) -> Optional[SatelliteObservation]:
    """Extract band values and indices from a single Sentinel-2 image."""
    ee = _ee

    # Calculate NDVI and NDWI
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")

    combined = image.addBands(ndvi).addBands(ndwi)

    # Get mean values over AOI
    stats = combined.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,
        maxPixels=1e8,
    ).getInfo()

    # Get image date
    timestamp = image.get("system:time_start").getInfo()
    obs_date = date.fromtimestamp(timestamp / 1000)

    cloud_cover = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()

    # Scale reflectance values (SR product uses scale factor of 10000)
    scale = 10000.0

    return SatelliteObservation(
        observation_date=obs_date,
        satellite="Sentinel-2",
        farm_id=farm_id,
        ndvi=round(stats.get("NDVI", 0), 4) if stats.get("NDVI") else None,
        ndwi=round(stats.get("NDWI", 0), 4) if stats.get("NDWI") else None,
        red=round(stats.get("B4", 0) / scale, 4) if stats.get("B4") else None,
        green=round(stats.get("B3", 0) / scale, 4) if stats.get("B3") else None,
        blue=round(stats.get("B2", 0) / scale, 4) if stats.get("B2") else None,
        nir=round(stats.get("B8", 0) / scale, 4) if stats.get("B8") else None,
        swir1=round(stats.get("B11", 0) / scale, 4) if stats.get("B11") else None,
        swir2=round(stats.get("B12", 0) / scale, 4) if stats.get("B12") else None,
        cloud_cover=round(cloud_cover, 1) if cloud_cover else None,
        data_source=DataSource.LIVE,
    )


def _process_s1_image(
    image: Any, aoi: Any, farm_id: str
) -> Optional[SatelliteObservation]:
    """Extract VV/VH values from a single Sentinel-1 image."""
    ee = _ee

    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,
        maxPixels=1e8,
    ).getInfo()

    timestamp = image.get("system:time_start").getInfo()
    obs_date = date.fromtimestamp(timestamp / 1000)

    vv = stats.get("VV")
    vh = stats.get("VH")

    return SatelliteObservation(
        observation_date=obs_date,
        satellite="Sentinel-1",
        farm_id=farm_id,
        vv=round(vv, 2) if vv else None,
        vh=round(vh, 2) if vh else None,
        vh_vv_ratio=round(vh - vv, 2) if (vv and vh) else None,
        data_source=DataSource.LIVE,
    )
