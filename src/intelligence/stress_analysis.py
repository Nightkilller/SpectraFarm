"""
AgriN — Stress Detection Module (Guarded / Uncalibrated)

Classifies crop stress from satellite indicators.

IMPORTANT: Thresholds are currently UNCALIBRATED placeholders.
The production stress-classification path is explicitly guarded and will
warn that thresholds are uncalibrated until regional ground truth calibration
is completed.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np

from src.config.settings import get_settings
from src.data.schemas import (
    DataSource,
    HealthTrend,
    SatelliteObservation,
    StressAssessment,
    StressLevel,
)
from src.geospatial.indices import classify_vci_stress, compute_vci

logger = logging.getLogger(__name__)


def assess_stress(
    observations: list[SatelliteObservation],
    farm_id: str,
    allow_uncalibrated: bool = True,
) -> StressAssessment:
    """
    Compute a stress assessment from satellite observations.

    Guarded: If thresholds in config/thresholds.yaml are marked `calibrated: false`,
    the assessment is strictly flagged as uncalibrated/demo indicator and
    never represented as a validated production soil-moisture measurement.
    """
    settings = get_settings()
    is_calibrated = settings.stress_thresholds.get("calibrated", False)

    if not is_calibrated:
        logger.warning(
            "[GUARD] Moisture stress thresholds are UNCALIBRATED placeholders. "
            "Stress classification is indicative/experimental only."
        )

    # Filter to optical observations with NDVI
    s2_obs = sorted(
        [o for o in observations if o.satellite == "Sentinel-2" and o.ndvi is not None],
        key=lambda o: o.observation_date,
    )

    if not s2_obs:
        logger.warning("No optical observations available for stress assessment")
        return _fallback_assessment(farm_id)

    # If uncalibrated, force data_source to DEMO/unvalidated
    data_source = DataSource.DEMO if not is_calibrated else DataSource.LIVE

    # Current and previous NDVI
    ndvi_current = s2_obs[-1].ndvi
    ndvi_previous = s2_obs[-2].ndvi if len(s2_obs) >= 2 else None

    # Compute VCI (Vegetation Condition Index)
    ndvi_vals = [o.ndvi for o in s2_obs if o.ndvi is not None]
    ndvi_min_hist = min(ndvi_vals) if ndvi_vals else 0.1
    ndvi_max_hist = max(ndvi_vals) if ndvi_vals else 0.8
    # Ensure realistic range even with few observations
    ndvi_min_hist = min(ndvi_min_hist, settings.ndvi_thresholds.get("bare_soil_max", 0.15))
    ndvi_max_hist = max(ndvi_max_hist, settings.ndvi_thresholds.get("dense_vegetation_min", 0.70))

    vci_pct = compute_vci(ndvi_current, ndvi_min_hist, ndvi_max_hist)
    vci_stress = classify_vci_stress(vci_pct)

    # NDVI trend
    trend = _compute_trend(s2_obs, settings.ndvi_thresholds.get("trend_threshold", 0.05))

    # Composite stress indicator (experimental)
    indicator = _compute_stress_indicator(s2_obs, settings)

    # Classify stress level using placeholder bounds
    stress_thresholds = settings.stress_thresholds
    if indicator >= stress_thresholds.get("healthy_min", 0.7):
        stress_level = StressLevel.HEALTHY
    elif indicator >= stress_thresholds.get("mild_min", 0.5):
        stress_level = StressLevel.MILD
    elif indicator >= stress_thresholds.get("moderate_min", 0.3):
        stress_level = StressLevel.MODERATE
    else:
        stress_level = StressLevel.SEVERE

    return StressAssessment(
        farm_id=farm_id,
        stress_level=stress_level,
        indicator_value=round(indicator, 3),
        assessment_date=s2_obs[-1].observation_date,
        trend=trend,
        ndvi_current=round(ndvi_current, 4),
        ndvi_previous=round(ndvi_previous, 4) if ndvi_previous is not None else None,
        confidence=0.0 if not is_calibrated else _estimate_confidence(len(s2_obs)),
        vci_percentage=round(vci_pct, 1),
        vci_stress_level=vci_stress,
        data_source=data_source,
    )


def _compute_trend(
    observations: list[SatelliteObservation],
    threshold: float,
) -> HealthTrend:
    """Determine vegetation health trend from NDVI time series."""
    if len(observations) < 3:
        return HealthTrend.STABLE

    ndvi_vals = np.array([o.ndvi for o in observations])
    x = np.arange(len(ndvi_vals), dtype=float)
    slope = np.polyfit(x, ndvi_vals, 1)[0]

    if slope > threshold:
        return HealthTrend.IMPROVING
    elif slope < -threshold:
        return HealthTrend.DECLINING
    else:
        return HealthTrend.STABLE


def _compute_stress_indicator(
    observations: list[SatelliteObservation],
    settings,
) -> float:
    """
    Compute a composite stress indicator (0–1).
    """
    ndvi_thresholds = settings.ndvi_thresholds

    # NDVI score
    current_ndvi = observations[-1].ndvi
    ndvi_floor = ndvi_thresholds.get("bare_soil_max", 0.15)
    ndvi_ceil = ndvi_thresholds.get("dense_vegetation_min", 0.5)
    ndvi_score = (current_ndvi - ndvi_floor) / max(ndvi_ceil - ndvi_floor, 0.01)
    ndvi_score = max(0.0, min(1.0, ndvi_score))

    # Trend score
    if len(observations) >= 3:
        ndvi_vals = np.array([o.ndvi for o in observations])
        x = np.arange(len(ndvi_vals), dtype=float)
        slope = np.polyfit(x, ndvi_vals, 1)[0]
        trend_score = (slope + 0.05) / 0.10
        trend_score = max(0.0, min(1.0, trend_score))
    else:
        trend_score = 0.5

    # NDWI score
    ndwi_vals = [o.ndwi for o in observations if o.ndwi is not None]
    if ndwi_vals:
        mean_ndwi = np.mean(ndwi_vals)
        ndwi_threshold = settings.ndwi_thresholds.get("water_stress_threshold", 0.0)
        ndwi_ceil = settings.ndwi_thresholds.get("adequate_moisture_min", 0.1)
        ndwi_score = (mean_ndwi - ndwi_threshold) / max(ndwi_ceil - ndwi_threshold, 0.01)
        ndwi_score = max(0.0, min(1.0, ndwi_score))
    else:
        ndwi_score = 0.5

    indicator = 0.60 * ndvi_score + 0.25 * trend_score + 0.15 * ndwi_score
    return max(0.0, min(1.0, indicator))


def _estimate_confidence(num_observations: int) -> float:
    if num_observations >= 8:
        return 0.85
    elif num_observations >= 5:
        return 0.70
    elif num_observations >= 3:
        return 0.55
    else:
        return 0.35


def _fallback_assessment(farm_id: str) -> StressAssessment:
    return StressAssessment(
        farm_id=farm_id,
        stress_level=StressLevel.MODERATE,
        indicator_value=0.5,
        assessment_date=date.today(),
        trend=HealthTrend.STABLE,
        confidence=0.0,
        data_source=DataSource.DEMO,
    )
