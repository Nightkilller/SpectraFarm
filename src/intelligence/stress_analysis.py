"""
AgriN — Stress Detection

Classifies crop stress from satellite indicators.  Uses a transparent,
configurable rules-based approach combining NDVI, NDWI, and trend data.

This is a satellite-based crop stress INDICATOR, NOT a validated physical
soil-moisture measurement.
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

logger = logging.getLogger(__name__)


def assess_stress(
    observations: list[SatelliteObservation],
    farm_id: str,
) -> StressAssessment:
    """
    Compute a stress assessment from satellite observations.

    The indicator is a composite score (0–1, where 1 = best/healthiest)
    derived from:
    - Current NDVI relative to thresholds
    - NDVI trend (improving/declining)
    - NDWI (water content indicator)

    All thresholds are loaded from config/thresholds.yaml.
    """
    settings = get_settings()

    # Filter to optical observations with NDVI
    s2_obs = sorted(
        [o for o in observations if o.satellite == "Sentinel-2" and o.ndvi is not None],
        key=lambda o: o.observation_date,
    )

    if not s2_obs:
        logger.warning("No optical observations available for stress assessment")
        return _fallback_assessment(farm_id)

    # Determine data source
    data_source = DataSource.DEMO if all(o.data_source == DataSource.DEMO for o in s2_obs) else DataSource.LIVE

    # Current and previous NDVI
    ndvi_current = s2_obs[-1].ndvi
    ndvi_previous = s2_obs[-2].ndvi if len(s2_obs) >= 2 else None

    # NDVI trend
    trend = _compute_trend(s2_obs, settings.ndvi_thresholds["trend_threshold"])

    # Composite stress indicator
    indicator = _compute_stress_indicator(s2_obs, settings)

    # Classify stress level
    stress_thresholds = settings.stress_thresholds
    if indicator >= stress_thresholds["healthy_min"]:
        stress_level = StressLevel.HEALTHY
    elif indicator >= stress_thresholds["mild_min"]:
        stress_level = StressLevel.MILD
    elif indicator >= stress_thresholds["moderate_min"]:
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
        confidence=_estimate_confidence(len(s2_obs)),
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

    Components:
    - NDVI score (60% weight): current NDVI normalized to thresholds
    - NDVI trend score (25% weight): positive trend = less stress
    - NDWI score (15% weight): higher NDWI = less water stress
    """
    ndvi_thresholds = settings.ndvi_thresholds

    # --- NDVI score ---
    current_ndvi = observations[-1].ndvi
    # Normalize: bare_soil_max → 0.0, dense_vegetation_min → 1.0
    ndvi_floor = ndvi_thresholds["bare_soil_max"]
    ndvi_ceil = ndvi_thresholds["dense_vegetation_min"]
    ndvi_score = (current_ndvi - ndvi_floor) / max(ndvi_ceil - ndvi_floor, 0.01)
    ndvi_score = max(0.0, min(1.0, ndvi_score))

    # --- Trend score ---
    if len(observations) >= 3:
        ndvi_vals = np.array([o.ndvi for o in observations])
        x = np.arange(len(ndvi_vals), dtype=float)
        slope = np.polyfit(x, ndvi_vals, 1)[0]
        # Normalize slope: -0.05 → 0.0, +0.05 → 1.0
        trend_score = (slope + 0.05) / 0.10
        trend_score = max(0.0, min(1.0, trend_score))
    else:
        trend_score = 0.5

    # --- NDWI score ---
    ndwi_vals = [o.ndwi for o in observations if o.ndwi is not None]
    if ndwi_vals:
        mean_ndwi = np.mean(ndwi_vals)
        ndwi_threshold = settings.ndwi_thresholds["water_stress_threshold"]
        ndwi_ceil = settings.ndwi_thresholds["adequate_moisture_min"]
        ndwi_score = (mean_ndwi - ndwi_threshold) / max(ndwi_ceil - ndwi_threshold, 0.01)
        ndwi_score = max(0.0, min(1.0, ndwi_score))
    else:
        ndwi_score = 0.5

    # Weighted composite
    indicator = 0.60 * ndvi_score + 0.25 * trend_score + 0.15 * ndwi_score

    return max(0.0, min(1.0, indicator))


def _estimate_confidence(num_observations: int) -> float:
    """Estimate confidence based on number of observations."""
    if num_observations >= 8:
        return 0.85
    elif num_observations >= 5:
        return 0.70
    elif num_observations >= 3:
        return 0.55
    else:
        return 0.35


def _fallback_assessment(farm_id: str) -> StressAssessment:
    """Return a low-confidence assessment when no data is available."""
    return StressAssessment(
        farm_id=farm_id,
        stress_level=StressLevel.MODERATE,
        indicator_value=0.5,
        assessment_date=date.today(),
        trend=HealthTrend.STABLE,
        confidence=0.1,
        data_source=DataSource.DEMO,
    )
