"""
AgriN — Multi-Temporal Sentinel-1 SAR Module (Phase 3)

Extracts and computes multi-temporal Synthetic Aperture Radar (SAR) backscatter
features (VV, VH, VV/VH ratio) from Sentinel-1 Ground Range Detected (GRD)
observations over a configured AOI.

Scientific constraints:
- Uses server-side Earth Engine reduction (no client-side raster downloads).
- Operates on calibrated gamma/sigma naught backscatter in decibels (dB).
- Computes linear power cross-ratio 10^((VV-VH)/10) and dB difference (VV - VH).
- All-weather C-band microwave observation (cloud-penetrating).
- Does NOT claim SAR directly measures soil moisture or crop water stress without ground calibration.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd

from src.config.settings import get_settings
from src.data.schemas import DataSource, SARTimeSeries, SARTimeSeriesPoint
from src.geospatial.gee_client import get_ee_module

logger = logging.getLogger(__name__)


def query_sentinel1_imagery(
    aoi: Any,
    start_date: date | str,
    end_date: date | str,
    orbit_pass: str = "DESCENDING",
    instrument_mode: str = "IW",
) -> Any:
    """
    Query the Sentinel-1 GRD collection filtered by AOI, date range, orbit direction,
    and polarization mode.

    Collection: COPERNICUS/S1_GRD
    """
    ee = get_ee_module()
    if not ee:
        raise RuntimeError("Google Earth Engine is not initialized.")

    settings = get_settings()
    collection_name = settings.sentinel1_config.get("collection", "COPERNICUS/S1_GRD")

    start_str = str(start_date)
    end_str = str(end_date)

    query = (
        ee.ImageCollection(collection_name)
        .filterBounds(aoi)
        .filterDate(start_str, end_str)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", instrument_mode))
    )

    if orbit_pass:
        query = query.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))

    return query.sort("system:time_start", True)


def calculate_sar_features(image: Any) -> Any:
    """
    Compute SAR ratio and difference bands on an Earth Engine Sentinel-1 image.

    In COPERNICUS/S1_GRD, VV and VH are provided in decibels (dB):
    - Linear power ratio: VV_VH_ratio = 10^((VV - VH) / 10)
    - Cross-polarization ratio: VH_VV_ratio = 10^((VH - VV) / 10)
    - Difference in dB: VV_minus_VH = VV - VH

    Returns:
        ee.Image: Image with added 'VV_VH_ratio', 'VH_VV_ratio', and 'VV_minus_VH' bands.
    """
    ee = get_ee_module()
    if not ee:
        raise RuntimeError("Google Earth Engine is not initialized.")

    vv = image.select("VV")
    vh = image.select("VH")

    # Difference in dB: VV - VH
    diff_db = vv.subtract(vh).rename("VV_minus_VH")

    # Linear power ratio: 10^((VV - VH) / 10)
    vv_vh_ratio = ee.Image(10.0).pow(diff_db.divide(10.0)).rename("VV_VH_ratio")

    # Cross ratio: 10^((VH - VV) / 10)
    vh_vv_ratio = ee.Image(10.0).pow(vh.subtract(vv).divide(10.0)).rename("VH_VV_ratio")

    return image.addBands([diff_db, vv_vh_ratio, vh_vv_ratio])


def extract_sar_timeseries(
    aoi: Any,
    start_date: date | str,
    end_date: date | str,
    aoi_name: str = "Sehore Pilot Test AOI",
    orbit_pass: str = "DESCENDING",
    instrument_mode: str = "IW",
    scale: int = 10,
    max_observations: int = 50,
) -> SARTimeSeries:
    """
    Extract multi-temporal Sentinel-1 SAR observations from Earth Engine.

    Returns:
        SARTimeSeries: Validated container with raw chronological SAR observations.
    """
    ee = get_ee_module()
    if not ee:
        raise RuntimeError("Google Earth Engine is not initialized.")

    start_d = start_date if isinstance(start_date, date) else datetime.strptime(str(start_date), "%Y-%m-%d").date()
    end_d = end_date if isinstance(end_date, date) else datetime.strptime(str(end_date), "%Y-%m-%d").date()

    collection = query_sentinel1_imagery(
        aoi=aoi,
        start_date=start_d,
        end_date=end_d,
        orbit_pass=orbit_pass,
        instrument_mode=instrument_mode,
    )

    total_available = collection.size().getInfo()
    logger.info(f"[INFO] Sentinel-1 GRD collection contains {total_available} candidate observations.")

    if total_available == 0:
        logger.warning("[WARNING] No Sentinel-1 observations found for the specified criteria.")
        return SARTimeSeries(
            aoi_name=aoi_name,
            start_date=start_d,
            end_date=end_d,
            orbit_pass=orbit_pass,
            instrument_mode=instrument_mode,
            observations_count=0,
            points=[],
            data_source=DataSource.LIVE,
        )

    def _compute_sar_stats_per_image(img):
        with_features = calculate_sar_features(img)

        reducer = (
            ee.Reducer.mean()
            .combine(ee.Reducer.min(), "", True)
            .combine(ee.Reducer.max(), "", True)
            .combine(ee.Reducer.stdDev(), "", True)
        )

        bands_to_reduce = with_features.select(["VV", "VH", "VV_VH_ratio", "VH_VV_ratio", "VV_minus_VH"])

        stats = bands_to_reduce.reduceRegion(
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
                "orbit_pass": img.get("orbitProperties_pass"),
                "relative_orbit": img.get("relativeOrbitNumber_start"),
                "instrument_mode": img.get("instrumentMode"),
                "VV_min": stats.get("VV_min"),
                "VV_mean": stats.get("VV_mean"),
                "VV_max": stats.get("VV_max"),
                "VV_stdDev": stats.get("VV_stdDev"),
                "VH_min": stats.get("VH_min"),
                "VH_mean": stats.get("VH_mean"),
                "VH_max": stats.get("VH_max"),
                "VH_stdDev": stats.get("VH_stdDev"),
                "VV_VH_ratio_mean": stats.get("VV_VH_ratio_mean"),
                "VH_VV_ratio_mean": stats.get("VH_VV_ratio_mean"),
                "VV_minus_VH_mean": stats.get("VV_minus_VH_mean"),
            },
        )

    limited_collection = collection.limit(max_observations)
    stats_fc = limited_collection.map(_compute_sar_stats_per_image)

    features_info = stats_fc.getInfo().get("features", [])

    points: list[SARTimeSeriesPoint] = []
    for feat in features_info:
        props = feat.get("properties", {})
        ts_ms = props.get("timestamp")
        if not ts_ms:
            continue

        obs_date = date.fromtimestamp(ts_ms / 1000)
        vv_mean = props.get("VV_mean")
        vh_mean = props.get("VH_mean")

        if vv_mean is None or vh_mean is None:
            continue

        point = SARTimeSeriesPoint(
            observation_date=obs_date,
            image_id=str(props.get("image_id", "Unknown")),
            orbit_pass=str(props.get("orbit_pass", orbit_pass)),
            relative_orbit=int(props["relative_orbit"]) if props.get("relative_orbit") is not None else None,
            instrument_mode=str(props.get("instrument_mode", instrument_mode)),
            min_vv=round(float(props.get("VV_min", 0.0)), 4),
            mean_vv=round(float(vv_mean), 4),
            max_vv=round(float(props.get("VV_max", 0.0)), 4),
            stdDev_vv=round(float(props.get("VV_stdDev", 0.0)), 4) if props.get("VV_stdDev") is not None else None,
            min_vh=round(float(props.get("VH_min", 0.0)), 4),
            mean_vh=round(float(vh_mean), 4),
            max_vh=round(float(props.get("VH_max", 0.0)), 4),
            stdDev_vh=round(float(props.get("VH_stdDev", 0.0)), 4) if props.get("VH_stdDev") is not None else None,
            mean_vv_vh_ratio=round(float(props.get("VV_VH_ratio_mean", 1.0)), 4),
            mean_vh_vv_ratio=round(float(props.get("VH_VV_ratio_mean", 0.0)), 4) if props.get("VH_VV_ratio_mean") is not None else None,
            mean_vv_minus_vh_db=round(float(props.get("VV_minus_VH_mean", 0.0)), 4) if props.get("VV_minus_VH_mean") is not None else None,
            data_source=DataSource.LIVE,
        )
        points.append(point)

    return SARTimeSeries(
        aoi_name=aoi_name,
        start_date=start_d,
        end_date=end_d,
        orbit_pass=orbit_pass,
        instrument_mode=instrument_mode,
        observations_count=len(points),
        points=points,
        data_source=DataSource.LIVE,
    )


def deduplicate_sar_to_canonical(timeseries: SARTimeSeries) -> SARTimeSeries:
    """
    Produce a canonical agricultural SAR time series with exactly ONE observation per calendar date.

    Deterministic Rule:
    For any calendar date with multiple valid Sentinel-1 passes, select the observation with
    the highest image_id sorting order (or latest generation time).

    Validation:
    Strictly asserts that zero duplicate calendar dates exist in output.
    """
    if not timeseries.points:
        return timeseries

    grouped: dict[date, list[SARTimeSeriesPoint]] = defaultdict(list)
    for p in timeseries.points:
        grouped[p.observation_date].append(p)

    canonical_points: list[SARTimeSeriesPoint] = []
    for obs_date in sorted(grouped.keys()):
        candidates = grouped[obs_date]
        if len(candidates) == 1:
            canonical_points.append(candidates[0])
        else:
            selected = max(candidates, key=lambda p: p.image_id)
            logger.info(
                f"[DEDUPLICATE SAR] Date {obs_date} has {len(candidates)} passes. "
                f"Selected: {selected.image_id}"
            )
            canonical_points.append(selected)

    dates_list = [p.observation_date for p in canonical_points]
    assert len(dates_list) == len(set(dates_list)), "Canonical SAR time series contains duplicate calendar dates!"

    return SARTimeSeries(
        aoi_name=timeseries.aoi_name,
        start_date=timeseries.start_date,
        end_date=timeseries.end_date,
        orbit_pass=timeseries.orbit_pass,
        instrument_mode=timeseries.instrument_mode,
        observations_count=len(canonical_points),
        points=canonical_points,
        data_source=timeseries.data_source,
    )


def sar_timeseries_to_dataframe(timeseries: SARTimeSeries) -> pd.DataFrame:
    """Convert a SARTimeSeries object to a pandas DataFrame."""
    if not timeseries.points:
        return pd.DataFrame(columns=[
            "date", "image_id", "orbit_pass", "relative_orbit", "mean_vv_db", "mean_vh_db",
            "vv_vh_ratio_linear", "vv_minus_vh_db"
        ])

    records = [
        {
            "date": p.observation_date,
            "image_id": p.image_id,
            "orbit_pass": p.orbit_pass,
            "relative_orbit": p.relative_orbit,
            "min_vv_db": p.min_vv,
            "mean_vv_db": p.mean_vv,
            "max_vv_db": p.max_vv,
            "stdDev_vv": p.stdDev_vv,
            "min_vh_db": p.min_vh,
            "mean_vh_db": p.mean_vh,
            "max_vh_db": p.max_vh,
            "stdDev_vh": p.stdDev_vh,
            "vv_vh_ratio_linear": p.mean_vv_vh_ratio,
            "vh_vv_ratio_linear": p.mean_vh_vv_ratio,
            "vv_minus_vh_db": p.mean_vv_minus_vh_db,
        }
        for p in timeseries.points
    ]
    return pd.DataFrame(records)
