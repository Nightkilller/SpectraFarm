"""
SpectraFarm — Unified Satellite Data Acquisition Layer

This module is the SINGLE ENTRY POINT for all satellite data in the dashboard.
It attempts to fetch REAL data from Google Earth Engine first, and gracefully
falls back to demo/synthetic data when GEE is unavailable.

Pipeline:
    1. Try to initialize Google Earth Engine
    2. If GEE is available -> query real Sentinel-2 + Sentinel-1 data
    3. If GEE is unavailable -> fall back to demo_data generators
    4. Convert GEE time-series to SatelliteObservation schema objects

This replaces the direct import of demo_data in app.py.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np

from src.config.settings import get_settings
from src.data.schemas import DataSource, SatelliteObservation

logger = logging.getLogger(__name__)

# ── In-memory caches (avoid re-fetching landcover per rerun) ──────────────
_LANDCOVER_CACHE: dict[str, dict] = {}
_PARCEL_LC_CACHE: dict[str, list] = {}


def _try_init_gee() -> bool:
    """Attempt to initialize Google Earth Engine. Returns True on success."""
    try:
        from src.geospatial.gee_client import is_gee_available
        return is_gee_available()
    except ImportError:
        logger.warning("[SAT] earthengine-api not installed — using demo data.")
        return False
    except Exception as e:
        logger.warning(f"[SAT] GEE initialization failed: {e} — using demo data.")
        return False


def fetch_optical_observations(
    lat: float,
    lon: float,
    farm_id: str,
    buffer_m: int = 1000,
    lookback_months: int = 6,
    end_date: Optional[date] = None,
) -> list[SatelliteObservation]:
    """
    Fetch Sentinel-2 optical observations for a location.

    Tries Google Earth Engine first. Falls back to demo data if GEE is unavailable.

    Returns:
        List of SatelliteObservation objects with NDVI, NDWI, band values.
    """
    if end_date is None:
        end_date = date.today()
    start_date = end_date - timedelta(days=lookback_months * 30)

    if _try_init_gee():
        try:
            return _fetch_real_optical(lat, lon, farm_id, buffer_m, start_date, end_date)
        except Exception as e:
            logger.error(f"[SAT] Real optical fetch failed: {e} — falling back to demo data.")

    # Fallback to demo data
    from src.data.demo_data import generate_ndvi_timeseries
    logger.info(f"[SAT] Using DEMO optical data for farm {farm_id}")
    return generate_ndvi_timeseries(farm_id)


def fetch_sar_observations(
    lat: float,
    lon: float,
    farm_id: str,
    buffer_m: int = 1000,
    lookback_months: int = 6,
    end_date: Optional[date] = None,
) -> list[SatelliteObservation]:
    """
    Fetch Sentinel-1 SAR observations for a location.

    Tries Google Earth Engine first. Falls back to demo data if GEE is unavailable.

    Returns:
        List of SatelliteObservation objects with VV, VH, VH/VV ratio.
    """
    if end_date is None:
        end_date = date.today()
    start_date = end_date - timedelta(days=lookback_months * 30)

    if _try_init_gee():
        try:
            return _fetch_real_sar(lat, lon, farm_id, buffer_m, start_date, end_date)
        except Exception as e:
            logger.error(f"[SAT] Real SAR fetch failed: {e} — falling back to demo data.")

    # Fallback to demo data
    from src.data.demo_data import generate_sar_observations
    logger.info(f"[SAT] Using DEMO SAR data for farm {farm_id}")
    return generate_sar_observations(farm_id)


def get_data_source_status() -> dict:
    """Return current data source status for the dashboard."""
    gee_available = _try_init_gee()
    return {
        "gee_available": gee_available,
        "data_source": "LIVE (Google Earth Engine)" if gee_available else "DEMO (Synthetic Data)",
        "source_enum": DataSource.LIVE if gee_available else DataSource.DEMO,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ESA WorldCover 10m Land-Cover Masking
# ═══════════════════════════════════════════════════════════════════════════

def get_cropland_mask(geometry=None) -> Any:
    """
    Load ESA WorldCover 10m Land Cover dataset and construct a binary Cropland mask.

    ESA WorldCover Classes:
      - 40: Cropland (Included -> 1)
      - 50: Built-up (Excluded -> 0)
      - 80: Permanent Water (Excluded -> 0)
      - 60: Bare / Sparse (Excluded -> 0)
      - 70: Snow and Ice (Excluded -> 0)
      - 10: Trees, 20: Shrubland, 30: Grassland

    Returns:
        ee.Image: A binary mask where 1 = Cropland (class 40), 0 = Non-cropland.
    """
    from src.geospatial.gee_client import get_ee_module
    ee = get_ee_module()
    if not ee:
        return None

    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
    cropland_mask = worldcover.select("Map").eq(40)
    if geometry is not None:
        cropland_mask = cropland_mask.clip(geometry)
    return cropland_mask


def get_landcover_summary(lat: float, lon: float, buffer_m: int = 500) -> dict[str, Any]:
    """
    Summarize ESA WorldCover 10m land cover distribution within a circular AOI buffer.
    Cached in-memory to prevent repeated GEE network latency.
    """
    cache_key = (round(lat, 4), round(lon, 4), int(buffer_m))
    if cache_key in _LANDCOVER_CACHE:
        return _LANDCOVER_CACHE[cache_key]

    if not _try_init_gee():
        res = {
            "cropland_pct": 88.0,
            "builtup_pct": 4.0,
            "water_pct": 1.0,
            "tree_pct": 7.0,
            "is_predominantly_cropland": True,
            "warning": None,
        }
        _LANDCOVER_CACHE[cache_key] = res
        return res

    try:
        from src.geospatial.gee_client import get_ee_module, get_dynamic_aoi
        ee = get_ee_module()
        aoi = get_dynamic_aoi(lat, lon, buffer_m)
        worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()

        hist = worldcover.select("Map").reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=aoi,
            scale=10,
            maxPixels=1e8,
        ).getInfo().get("Map", {})

        total_pixels = sum(hist.values())
        if total_pixels == 0:
            res = {
                "cropland_pct": 100.0,
                "builtup_pct": 0.0,
                "water_pct": 0.0,
                "tree_pct": 0.0,
                "is_predominantly_cropland": True,
                "warning": None,
            }
            _LANDCOVER_CACHE[cache_key] = res
            return res

        crop_px = hist.get("40", 0)
        built_px = hist.get("50", 0)
        water_px = hist.get("80", 0)
        tree_px = hist.get("10", 0)

        crop_pct = round((crop_px / total_pixels) * 100, 1)
        built_pct = round((built_px / total_pixels) * 100, 1)
        water_pct = round((water_px / total_pixels) * 100, 1)
        tree_pct = round((tree_px / total_pixels) * 100, 1)

        is_predom = crop_pct >= 40.0 and built_pct < 45.0
        warning = None
        if not is_predom:
            warning = (
                f"High Non-Agricultural Landcover: Only {crop_pct:.1f}% of this buffer is active cropland "
                f"({built_pct:.1f}% built-up / urban). Results may be unreliable."
            )

        res = {
            "cropland_pct": crop_pct,
            "builtup_pct": built_pct,
            "water_pct": water_pct,
            "tree_pct": tree_pct,
            "is_predominantly_cropland": is_predom,
            "warning": warning,
        }
        _LANDCOVER_CACHE[cache_key] = res
        return res
    except Exception as e:
        logger.warning(f"[SAT] Landcover summary failed: {e}")
        res = {
            "cropland_pct": 85.0,
            "builtup_pct": 5.0,
            "water_pct": 1.0,
            "tree_pct": 9.0,
            "is_predominantly_cropland": True,
            "warning": None,
        }
        _LANDCOVER_CACHE[cache_key] = res
        return res


def sample_parcel_landcover(coords_list: list[tuple[float, float]]) -> dict[int, dict]:
    """
    Batch-sample ESA WorldCover class for a list of (lat, lon) parcel centroids.
    Cached in-memory for instant rendering.
    """
    if not coords_list:
        return {}

    cache_key = tuple((round(lat, 4), round(lon, 4)) for lat, lon in coords_list[:5])
    if cache_key in _PARCEL_LC_CACHE:
        return _PARCEL_LC_CACHE[cache_key]

    if not _try_init_gee():
        res = {
            idx: {"code": 40, "name": "Cropland", "is_cropland": True}
            for idx in range(len(coords_list))
        }
        _PARCEL_LC_CACHE[cache_key] = res
        return res

    try:
        from src.geospatial.gee_client import get_ee_module
        ee = get_ee_module()
        worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()

        features = [
            ee.Feature(ee.Geometry.Point([lon, lat]), {"parcel_idx": idx})
            for idx, (lat, lon) in enumerate(coords_list)
        ]
        fc = ee.FeatureCollection(features)
        sampled = worldcover.select("Map").reduceRegions(
            collection=fc,
            reducer=ee.Reducer.first(),
            scale=10,
        ).getInfo()

        class_names = {
            10: "Tree cover",
            20: "Shrubland",
            30: "Grassland",
            40: "Cropland",
            50: "Built-up / Settlement",
            60: "Bare / Sparse",
            70: "Snow / Ice",
            80: "Water Body",
            90: "Herbaceous Wetland",
        }

        results = {}
        for f in sampled.get("features", []):
            p_id = f.get("properties", {}).get("parcel_idx")
            code = f.get("properties", {}).get("first", 40)
            if code is None:
                code = 40
            else:
                code = int(code)
            
            name = class_names.get(code, "Other")
            is_crop = (code == 40)
            results[p_id] = {
                "code": code,
                "name": name,
                "is_cropland": is_crop,
            }
        _PARCEL_LC_CACHE[cache_key] = results
        return results
    except Exception as e:
        logger.warning(f"[SAT] Batch parcel landcover sampling failed: {e}")
        res = {
            idx: {"code": 40, "name": "Cropland", "is_cropland": True}
            for idx in range(len(coords_list))
        }
        _PARCEL_LC_CACHE[cache_key] = res
        return res


# ═══════════════════════════════════════════════════════════════════════════
# Real GEE Data Fetchers (Private)
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_real_optical(
    lat: float,
    lon: float,
    farm_id: str,
    buffer_m: int,
    start_date: date,
    end_date: date,
) -> list[SatelliteObservation]:
    """
    Fetch REAL Sentinel-2 optical observations from Google Earth Engine.

    Uses the existing GEE timeseries module to query COPERNICUS/S2_SR_HARMONIZED,
    then converts GEE NDVITimeSeries -> SatelliteObservation schema objects.
    """
    from src.geospatial.gee_client import get_dynamic_aoi
    from src.geospatial.timeseries import extract_ndvi_timeseries, deduplicate_to_canonical
    from src.geospatial.indices import calculate_ndvi, calculate_ndwi

    logger.info(f"[SAT] Fetching REAL Sentinel-2 data for {lat:.4f}°N, {lon:.4f}°E (buffer={buffer_m}m)")

    aoi = get_dynamic_aoi(lat, lon, buffer_m)

    # Extract multi-temporal NDVI time series from GEE
    ndvi_ts = extract_ndvi_timeseries(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        max_cloud_percentage=25.0,
        aoi_name=f"Farm_{farm_id}",
        scale=10,
        max_observations=30,
    )

    # Deduplicate to one observation per calendar date
    ndvi_ts = deduplicate_to_canonical(ndvi_ts)

    if ndvi_ts.observations_count == 0:
        logger.warning("[SAT] No Sentinel-2 observations found from GEE — falling back to demo.")
        from src.data.demo_data import generate_ndvi_timeseries
        return generate_ndvi_timeseries(farm_id)

    logger.info(f"[SAT] Retrieved {ndvi_ts.observations_count} real Sentinel-2 observations from GEE.")

    # Now also fetch band-level statistics for each date using a batch approach
    band_stats = _fetch_optical_band_stats(aoi, start_date, end_date)

    # Convert NDVITimeSeries points -> SatelliteObservation objects
    observations: list[SatelliteObservation] = []
    for point in ndvi_ts.points:
        # Match band stats by date if available
        date_key = str(point.observation_date)
        bands = band_stats.get(date_key, {})

        obs = SatelliteObservation(
            observation_date=point.observation_date,
            satellite="Sentinel-2",
            farm_id=farm_id,
            ndvi=point.mean_ndvi,
            ndwi=bands.get("ndwi_mean"),
            red=bands.get("red_mean"),
            green=bands.get("green_mean"),
            blue=bands.get("blue_mean"),
            nir=bands.get("nir_mean"),
            swir1=bands.get("swir1_mean"),
            swir2=bands.get("swir2_mean"),
            cloud_cover=point.cloud_percentage,
            data_source=DataSource.LIVE,
        )
        observations.append(obs)

    return observations


def _fetch_optical_band_stats(
    aoi,
    start_date: date,
    end_date: date,
) -> dict[str, dict[str, float]]:
    """
    Fetch per-date optical band statistics from GEE using batch reduction.

    Returns a dict keyed by date string, containing band means for each date.
    """
    try:
        from src.geospatial.gee_client import get_ee_module, query_sentinel2_imagery
        from src.geospatial.indices import calculate_ndvi, calculate_ndwi

        ee = get_ee_module()
        if not ee:
            return {}

        collection = query_sentinel2_imagery(aoi, start_date, end_date, max_cloud_percentage=25.0)

        # Add vegetation indices
        def add_indices(img):
            ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndwi = img.normalizedDifference(["B8", "B11"]).rename("NDWI")
            return img.addBands([ndvi, ndwi])

        collection = collection.map(add_indices)

        # Load ESA WorldCover Cropland mask
        worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
        cropland_mask = worldcover.select("Map").eq(40)

        # Compute per-image band means over the AOI (strictly over cropland pixels)
        def extract_band_means(img):
            bands = img.select(["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI"]).updateMask(cropland_mask)
            stats = bands.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=10,
                maxPixels=1e8,
            )
            return ee.Feature(None, {
                "timestamp": img.get("system:time_start"),
                "blue_mean": ee.Number(stats.get("B2")).divide(10000),
                "green_mean": ee.Number(stats.get("B3")).divide(10000),
                "red_mean": ee.Number(stats.get("B4")).divide(10000),
                "nir_mean": ee.Number(stats.get("B8")).divide(10000),
                "swir1_mean": ee.Number(stats.get("B11")).divide(10000),
                "swir2_mean": ee.Number(stats.get("B12")).divide(10000),
                "ndvi_mean": stats.get("NDVI"),
                "ndwi_mean": stats.get("NDWI"),
            })

        stats_fc = collection.limit(30).map(extract_band_means)
        features = stats_fc.getInfo().get("features", [])

        result: dict[str, dict[str, float]] = {}
        for feat in features:
            props = feat.get("properties", {})
            ts_ms = props.get("timestamp")
            if not ts_ms:
                continue

            obs_date = date.fromtimestamp(ts_ms / 1000)
            date_key = str(obs_date)

            band_data = {}
            for key in ["blue_mean", "green_mean", "red_mean", "nir_mean",
                         "swir1_mean", "swir2_mean", "ndvi_mean", "ndwi_mean"]:
                val = props.get(key)
                if val is not None:
                    band_data[key] = round(float(val), 4)

            result[date_key] = band_data

        logger.info(f"[SAT] Extracted band statistics for {len(result)} dates from GEE.")
        return result

    except Exception as e:
        logger.warning(f"[SAT] Band-level stats extraction failed: {e}")
        return {}


def _fetch_real_sar(
    lat: float,
    lon: float,
    farm_id: str,
    buffer_m: int,
    start_date: date,
    end_date: date,
) -> list[SatelliteObservation]:
    """
    Fetch REAL Sentinel-1 SAR observations from Google Earth Engine.

    Uses the existing GEE sar module to query COPERNICUS/S1_GRD,
    then converts SARTimeSeries -> SatelliteObservation schema objects.
    """
    from src.geospatial.gee_client import get_dynamic_aoi
    from src.geospatial.sar import extract_sar_timeseries, deduplicate_sar_to_canonical

    logger.info(f"[SAT] Fetching REAL Sentinel-1 SAR data for {lat:.4f}°N, {lon:.4f}°E")

    aoi = get_dynamic_aoi(lat, lon, buffer_m)

    sar_ts = extract_sar_timeseries(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        aoi_name=f"Farm_{farm_id}",
        scale=10,
        max_observations=30,
    )

    sar_ts = deduplicate_sar_to_canonical(sar_ts)

    if sar_ts.observations_count == 0:
        logger.warning("[SAT] No Sentinel-1 SAR observations found from GEE — falling back to demo.")
        from src.data.demo_data import generate_sar_observations
        return generate_sar_observations(farm_id)

    logger.info(f"[SAT] Retrieved {sar_ts.observations_count} real Sentinel-1 SAR observations from GEE.")

    observations: list[SatelliteObservation] = []
    for point in sar_ts.points:
        obs = SatelliteObservation(
            observation_date=point.observation_date,
            satellite="Sentinel-1",
            farm_id=farm_id,
            vv=point.mean_vv,
            vh=point.mean_vh,
            vh_vv_ratio=point.mean_vh_vv_ratio,
            data_source=DataSource.LIVE,
        )
        observations.append(obs)

    return observations
