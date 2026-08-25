"""
AgriN — Multi-Temporal Sentinel-2 NDVI Time Series Module

Extracts multi-temporal NDVI trajectories from Sentinel-2 Surface Reflectance
observations over a configured AOI.

Scientific constraints:
- Uses server-side Earth Engine reduction (no large raster downloads).
- Computes per-observation NDVI statistics (min, mean, max, stdDev) over AOI.
- Filters by AOI, Date range, and CLOUDY_PIXEL_PERCENTAGE.
- Does not fabricate missing observations.
- Does not infer crop phenology or crop type from NDVI alone.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd

from src.config.settings import get_settings
from src.data.schemas import DataSource, NDVITimeSeries, NDVITimeSeriesPoint
from src.geospatial.gee_client import get_ee_module, query_sentinel2_imagery
from src.geospatial.indices import calculate_ndvi

logger = logging.getLogger(__name__)


def extract_ndvi_timeseries(
    aoi: Any,
    start_date: date | str,
    end_date: date | str,
    max_cloud_percentage: float = 20.0,
    aoi_name: str = "Sehore Pilot Test AOI",
    scale: int = 10,
    max_observations: int = 50,
) -> NDVITimeSeries:
    """
    Extract a real multi-temporal NDVI time series from Earth Engine.

    Args:
        aoi: ee.Geometry representing the area of interest.
        start_date: Start of observation window (YYYY-MM-DD or date).
        end_date: End of observation window (YYYY-MM-DD or date).
        max_cloud_percentage: Filter for scene cloud cover (e.g. 20.0%).
        aoi_name: Descriptive name of the AOI.
        scale: Spatial resolution in meters for reduction (default 10m).
        max_observations: Safety limit on total images to process.

    Returns:
        NDVITimeSeries: Validated Pydantic container with chronological points.
    """
    ee = get_ee_module()
    if not ee:
        raise RuntimeError("Google Earth Engine is not initialized.")

    start_d = start_date if isinstance(start_date, date) else datetime.strptime(str(start_date), "%Y-%m-%d").date()
    end_d = end_date if isinstance(end_date, date) else datetime.strptime(str(end_date), "%Y-%m-%d").date()

    # Query Sentinel-2 collection (COPERNICUS/S2_SR_HARMONIZED)
    collection = query_sentinel2_imagery(
        aoi=aoi,
        start_date=start_d,
        end_date=end_d,
        max_cloud_percentage=max_cloud_percentage,
    )

    # Sort chronologically (ascending) for time series
    collection = collection.sort("system:time_start", True)

    total_available = collection.size().getInfo()
    logger.info(f"[INFO] Sentinel-2 collection contains {total_available} candidate images.")

    if total_available == 0:
        logger.warning("[WARNING] No Sentinel-2 observations found for the specified criteria.")
        return NDVITimeSeries(
            aoi_name=aoi_name,
            start_date=start_d,
            end_date=end_d,
            cloud_threshold=max_cloud_percentage,
            observations_count=0,
            points=[],
            data_source=DataSource.LIVE,
        )

    # Map reduction across the image collection server-side
    def _compute_stats_per_image(img):
        # Calculate NDVI band: (B8 - B4) / (B8 + B4)
        with_ndvi = calculate_ndvi(img)
        ndvi_band = with_ndvi.select("NDVI")

        reducer = (
            ee.Reducer.mean()
            .combine(ee.Reducer.min(), "", True)
            .combine(ee.Reducer.max(), "", True)
            .combine(ee.Reducer.stdDev(), "", True)
        )

        stats = ndvi_band.reduceRegion(
            reducer=reducer,
            geometry=aoi,
            scale=scale,
            maxPixels=1e8,
        )

        # Set calculated properties on the feature
        return ee.Feature(
            None,
            {
                "image_id": img.get("system:id"),
                "timestamp": img.get("system:time_start"),
                "cloud_percentage": img.get("CLOUDY_PIXEL_PERCENTAGE"),
                "NDVI_min": stats.get("NDVI_min"),
                "NDVI_mean": stats.get("NDVI_mean"),
                "NDVI_max": stats.get("NDVI_max"),
                "NDVI_stdDev": stats.get("NDVI_stdDev"),
            },
        )

    # Convert image collection to feature collection with NDVI properties
    limited_collection = collection.limit(max_observations)
    stats_fc = limited_collection.map(_compute_stats_per_image)

    # Fetch structured properties
    features_info = stats_fc.getInfo().get("features", [])

    points: list[NDVITimeSeriesPoint] = []
    for feat in features_info:
        props = feat.get("properties", {})
        ts_ms = props.get("timestamp")
        if not ts_ms:
            continue

        obs_date = date.fromtimestamp(ts_ms / 1000)
        mean_v = props.get("NDVI_mean")
        min_v = props.get("NDVI_min")
        max_v = props.get("NDVI_max")

        # Skip observations where reduction yielded no valid unmasked pixels
        if mean_v is None or min_v is None or max_v is None:
            continue

        # Enforce physical bounding [-1.0, +1.0]
        point = NDVITimeSeriesPoint(
            observation_date=obs_date,
            image_id=str(props.get("image_id", "Unknown")),
            cloud_percentage=round(float(props.get("cloud_percentage", 0.0)), 2),
            min_ndvi=round(float(min_v), 4),
            mean_ndvi=round(float(mean_v), 4),
            max_ndvi=round(float(max_v), 4),
            stdDev_ndvi=round(float(props.get("NDVI_stdDev", 0.0)), 4) if props.get("NDVI_stdDev") is not None else None,
            data_source=DataSource.LIVE,
        )
        points.append(point)

    return NDVITimeSeries(
        aoi_name=aoi_name,
        start_date=start_d,
        end_date=end_d,
        cloud_threshold=max_cloud_percentage,
        observations_count=len(points),
        points=points,
        data_source=DataSource.LIVE,
    )


def timeseries_to_dataframe(timeseries: NDVITimeSeries) -> pd.DataFrame:
    """Convert an NDVITimeSeries object to a pandas DataFrame."""
    if not timeseries.points:
        return pd.DataFrame(columns=[
            "date", "image_id", "cloud_percentage", "min_ndvi", "mean_ndvi", "max_ndvi", "stdDev_ndvi"
        ])

    records = [
        {
            "date": p.observation_date,
            "image_id": p.image_id,
            "cloud_percentage": p.cloud_percentage,
            "min_ndvi": p.min_ndvi,
            "mean_ndvi": p.mean_ndvi,
            "max_ndvi": p.max_ndvi,
            "stdDev_ndvi": p.stdDev_ndvi,
        }
        for p in timeseries.points
    ]
    return pd.DataFrame(records)
