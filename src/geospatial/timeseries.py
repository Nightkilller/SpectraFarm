"""
AgriN — Multi-Temporal Sentinel-2 NDVI Time Series Module

Extracts multi-temporal NDVI trajectories from Sentinel-2 Surface Reflectance
observations over a configured AOI.

Supports two tiers of time-series representations:
1. Raw Observation Level: Every valid Sentinel-2 granule processed (preserves full provenance).
2. Canonical Agricultural Daily Level: Exactly one observation per calendar date, deterministically
   selecting the latest processing generation granule when same-date passes occur.
"""

from __future__ import annotations

import logging
from collections import defaultdict
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
    Extract raw multi-temporal Sentinel-2 NDVI observations from Earth Engine.

    Returns:
        NDVITimeSeries: Validated Pydantic container with raw chronological observations.
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

        return ee.Feature(
            None,
            {
                "image_id": img.get("system:id"),
                "timestamp": img.get("system:time_start"),
                "generation_time": img.get("GENERATION_TIME"),
                "cloud_percentage": img.get("CLOUDY_PIXEL_PERCENTAGE"),
                "NDVI_min": stats.get("NDVI_min"),
                "NDVI_mean": stats.get("NDVI_mean"),
                "NDVI_max": stats.get("NDVI_max"),
                "NDVI_stdDev": stats.get("NDVI_stdDev"),
            },
        )

    limited_collection = collection.limit(max_observations)
    stats_fc = limited_collection.map(_compute_stats_per_image)

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

        if mean_v is None or min_v is None or max_v is None:
            continue

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


def deduplicate_to_canonical(timeseries: NDVITimeSeries) -> NDVITimeSeries:
    """
    Produce a canonical agricultural time series with exactly ONE observation per calendar date.

    Deterministic Rule:
    For any calendar date with multiple valid Sentinel-2 granules (e.g. reprocessing passes),
    select the granule with the latest generation/processing timestamp (parsed from image ID or
    deterministic ID order).

    Validation:
    Strictly asserts that no duplicate calendar dates exist in the output.
    """
    if not timeseries.points:
        return timeseries

    # Group points by observation_date
    grouped: dict[date, list[NDVITimeSeriesPoint]] = defaultdict(list)
    for p in timeseries.points:
        grouped[p.observation_date].append(p)

    canonical_points: list[NDVITimeSeriesPoint] = []

    # Sort calendar dates chronologically
    for obs_date in sorted(grouped.keys()):
        candidates = grouped[obs_date]
        if len(candidates) == 1:
            canonical_points.append(candidates[0])
        else:
            # Deterministic selection: choose granule with highest/latest processing timestamp
            # Copernicus S2 ID format: ..._<SensingTime>_<GenerationTime>_<TileID>
            selected = max(candidates, key=lambda p: p.image_id)
            logger.info(
                f"[DEDUPLICATE] Date {obs_date} has {len(candidates)} granules. "
                f"Selected latest generation granule: {selected.image_id}"
            )
            canonical_points.append(selected)

    # Validation: Ensure 100% unique calendar dates
    dates_list = [p.observation_date for p in canonical_points]
    assert len(dates_list) == len(set(dates_list)), "Canonical time series contains duplicate calendar dates!"

    return NDVITimeSeries(
        aoi_name=timeseries.aoi_name,
        start_date=timeseries.start_date,
        end_date=timeseries.end_date,
        cloud_threshold=timeseries.cloud_threshold,
        observations_count=len(canonical_points),
        points=canonical_points,
        data_source=timeseries.data_source,
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
