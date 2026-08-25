"""
AgriN — Demo Data Generator

Generates realistic but SYNTHETIC agricultural data for development and
demonstration purposes.  All data produced by this module is clearly tagged
with DataSource.DEMO.

IMPORTANT: This data does NOT come from real satellite observations.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import Optional

from src.data.schemas import (
    Advisory,
    BoundingBox,
    CropPrediction,
    CropType,
    DataSource,
    Farm,
    FarmAnalysis,
    HealthTrend,
    SatelliteObservation,
    StressAssessment,
    StressLevel,
)


def get_demo_farm() -> Farm:
    """Return the pilot demo farm for Ludhiana, Punjab."""
    return Farm(
        farm_id="pilot_01",
        name="Pilot Farm — Ludhiana, Punjab",
        latitude=30.9,
        longitude=75.85,
        bbox=BoundingBox(min_lat=30.85, max_lat=30.95, min_lon=75.80, max_lon=75.90),
        area_ha=42.5,
        crop=CropType.WHEAT,
        season="rabi",
        data_source=DataSource.DEMO,
    )


def generate_ndvi_timeseries(
    farm_id: str = "pilot_01",
    num_observations: int = 12,
    end_date: Optional[date] = None,
    trend: str = "declining",
) -> list[SatelliteObservation]:
    """
    Generate a realistic NDVI time series.

    The curve follows a rough crop growth pattern:
    - Early growth: rising NDVI
    - Peak: high NDVI
    - Late season: declining or stable

    Args:
        farm_id: Farm identifier
        num_observations: Number of temporal observations
        end_date: Last observation date (defaults to today)
        trend: Overall trend — "improving", "stable", or "declining"
    """
    if end_date is None:
        end_date = date.today()

    observations = []
    base_interval = 16  # ~16 days between Sentinel-2 revisits

    # Generate dates going backwards from end_date
    dates = []
    for i in range(num_observations):
        obs_date = end_date - timedelta(days=i * base_interval)
        dates.append(obs_date)
    dates.reverse()

    # Generate NDVI values following a crop growth curve with specified trend
    random.seed(42)  # Reproducible demo data
    for i, obs_date in enumerate(dates):
        progress = i / max(num_observations - 1, 1)

        if trend == "declining":
            # Peak mid-season, then decline
            base_ndvi = 0.75 - 0.25 * (progress - 0.4) ** 2 if progress < 0.5 else 0.75 - 0.3 * (progress - 0.4)
        elif trend == "improving":
            # Steady rise
            base_ndvi = 0.35 + 0.35 * progress
        else:
            # Stable mid-range
            base_ndvi = 0.55 + 0.05 * math.sin(progress * math.pi * 2)

        # Add realistic noise
        ndvi = max(0.1, min(0.95, base_ndvi + random.uniform(-0.03, 0.03)))
        ndwi = max(-0.3, min(0.5, ndvi * 0.4 + random.uniform(-0.05, 0.05) - 0.1))
        cloud_cover = random.uniform(0, 15)

        observations.append(
            SatelliteObservation(
                observation_date=obs_date,
                satellite="Sentinel-2",
                farm_id=farm_id,
                ndvi=round(ndvi, 4),
                ndwi=round(ndwi, 4),
                red=round(random.uniform(0.03, 0.08), 4),
                green=round(random.uniform(0.04, 0.10), 4),
                blue=round(random.uniform(0.02, 0.06), 4),
                nir=round(ndvi * 0.3 + random.uniform(0.05, 0.15), 4),
                swir1=round(random.uniform(0.10, 0.25), 4),
                swir2=round(random.uniform(0.05, 0.15), 4),
                cloud_cover=round(cloud_cover, 1),
                data_source=DataSource.DEMO,
            )
        )

    return observations


def generate_sar_observations(
    farm_id: str = "pilot_01",
    num_observations: int = 12,
    end_date: Optional[date] = None,
) -> list[SatelliteObservation]:
    """Generate synthetic Sentinel-1 SAR observations."""
    if end_date is None:
        end_date = date.today()

    random.seed(43)
    observations = []
    base_interval = 12  # SAR revisit

    for i in range(num_observations):
        obs_date = end_date - timedelta(days=i * base_interval)
        vv = round(-10 + random.uniform(-4, 2), 2)
        vh = round(-17 + random.uniform(-4, 2), 2)

        observations.append(
            SatelliteObservation(
                observation_date=obs_date,
                satellite="Sentinel-1",
                farm_id=farm_id,
                vv=vv,
                vh=vh,
                vh_vv_ratio=round(vh - vv, 2),
                data_source=DataSource.DEMO,
            )
        )

    observations.reverse()
    return observations


def get_demo_crop_prediction(farm_id: str = "pilot_01") -> CropPrediction:
    """Return a demo crop classification result."""
    return CropPrediction(
        farm_id=farm_id,
        predicted_crop=CropType.WHEAT,
        confidence=0.91,
        model_version="rf_demo_v0.1",
        prediction_date=date.today(),
        feature_importance={
            "ndvi_mean": 0.28,
            "ndvi_std": 0.15,
            "nir_mean": 0.13,
            "swir1_mean": 0.11,
            "vv_mean": 0.09,
            "vh_mean": 0.08,
            "ndwi_mean": 0.07,
            "red_mean": 0.05,
            "green_mean": 0.04,
        },
        data_source=DataSource.DEMO,
    )


def get_demo_stress_assessment(farm_id: str = "pilot_01") -> StressAssessment:
    """Return a demo stress assessment."""
    return StressAssessment(
        farm_id=farm_id,
        stress_level=StressLevel.MODERATE,
        indicator_value=0.45,
        assessment_date=date.today(),
        trend=HealthTrend.DECLINING,
        ndvi_current=0.52,
        ndvi_previous=0.65,
        confidence=0.75,
        data_source=DataSource.DEMO,
    )


def get_demo_farm_analysis() -> FarmAnalysis:
    """Return a complete demo FarmAnalysis object."""
    farm = get_demo_farm()
    observations = generate_ndvi_timeseries(farm.farm_id)
    prediction = get_demo_crop_prediction(farm.farm_id)
    stress = get_demo_stress_assessment(farm.farm_id)

    return FarmAnalysis(
        farm=farm,
        crop_prediction=prediction,
        stress_assessment=stress,
        recent_observations=observations,
        ndvi_current=stress.ndvi_current,
        ndvi_previous=stress.ndvi_previous,
        ndvi_trend=HealthTrend.DECLINING,
        observation_date=date.today(),
        data_source=DataSource.DEMO,
    )


def get_demo_advisory(farm_id: str = "pilot_01", language: str = "en") -> Advisory:
    """Return a demo advisory."""
    if language == "hi":
        return Advisory(
            farm_id=farm_id,
            language="hi",
            observations_summary=(
                "आपकी गेहूँ की फसल में उपग्रह-आधारित संकेतकों के अनुसार मध्यम तनाव दिख रहा है। "
                "वनस्पति स्वास्थ्य सूचकांक (NDVI) पिछले अवलोकनों की तुलना में गिरा है "
                "(0.65 से 0.52)।"
            ),
            advisory_text=(
                "🌾 फसल की स्थिति: आपकी गेहूँ की फसल में मध्यम तनाव के संकेत हैं।\n\n"
                "📊 अवलोकन: वनस्पति स्वास्थ्य में गिरावट का रुझान है।\n\n"
                "💧 संभावित कारण: यह पैटर्न पानी की कमी या अन्य फसल तनाव कारकों से "
                "जुड़ा हो सकता है।\n\n"
                "✅ सुझाव: कृपया खेत की नमी और स्थानीय फसल की स्थिति की जांच करें। "
                "सिंचाई का निर्णय लेने से पहले मिट्टी की नमी की पुष्टि करें।\n\n"
                "⚠️ सीमा: उपग्रह संकेतक अकेले तनाव का सटीक कारण नहीं बता सकते। "
                "कृपया स्थानीय कृषि विशेषज्ञ से परामर्श लें।"
            ),
            data_source=DataSource.DEMO,
        )

    return Advisory(
        farm_id=farm_id,
        language="en",
        observations_summary=(
            "Your wheat crop is showing moderate satellite-based stress. "
            "The vegetation health index (NDVI) has declined compared to previous "
            "observations (from 0.65 to 0.52)."
        ),
        advisory_text=(
            "🌾 Crop Status: Your wheat crop is showing signs of moderate stress.\n\n"
            "📊 Observation: Vegetation health has been declining over recent observations.\n\n"
            "💧 Possible Interpretation: The observed pattern may be associated with "
            "water stress or other crop stress factors.\n\n"
            "✅ Suggested Action: Check field moisture and local crop conditions. "
            "Verify soil moisture before making irrigation decisions.\n\n"
            "⚠️ Limitation: Satellite indicators alone cannot identify the exact cause "
            "of stress. Please verify conditions in the field and consult local "
            "agricultural experts for specific treatment decisions."
        ),
        data_source=DataSource.DEMO,
    )


# ── Crop classification map demo data ──────────────────────────────────────

def generate_demo_crop_map_data(
    center_lat: float = 30.9,
    center_lon: float = 75.85,
    grid_size: int = 20,
) -> list[dict]:
    """
    Generate a grid of crop classification results for map display.

    Returns a list of dicts with lat, lon, crop, confidence for each cell.
    """
    random.seed(44)
    cell_size = 0.001  # ~100m cells
    results = []

    for i in range(grid_size):
        for j in range(grid_size):
            lat = center_lat - (grid_size / 2 - i) * cell_size
            lon = center_lon - (grid_size / 2 - j) * cell_size

            # Create spatial clusters for realism
            dist_from_center = math.sqrt((i - grid_size / 2) ** 2 + (j - grid_size / 2) ** 2)

            if dist_from_center < grid_size * 0.3:
                crop = "wheat"
                confidence = round(random.uniform(0.82, 0.96), 2)
            elif dist_from_center < grid_size * 0.45:
                crop = "rice"
                confidence = round(random.uniform(0.75, 0.92), 2)
            else:
                crop = random.choice(["wheat", "rice", "other"])
                confidence = round(random.uniform(0.55, 0.85), 2)

            results.append({
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "crop": crop,
                "confidence": confidence,
            })

    return results


def generate_demo_stress_map_data(
    center_lat: float = 30.9,
    center_lon: float = 75.85,
    grid_size: int = 20,
) -> list[dict]:
    """
    Generate a grid of stress levels for map display.
    """
    random.seed(45)
    cell_size = 0.001
    results = []

    for i in range(grid_size):
        for j in range(grid_size):
            lat = center_lat - (grid_size / 2 - i) * cell_size
            lon = center_lon - (grid_size / 2 - j) * cell_size

            # Stress gradient: more stress towards one corner
            stress_factor = (i + j) / (2 * grid_size) + random.uniform(-0.15, 0.15)
            stress_factor = max(0, min(1, stress_factor))

            if stress_factor < 0.3:
                level = "healthy"
            elif stress_factor < 0.5:
                level = "mild"
            elif stress_factor < 0.7:
                level = "moderate"
            else:
                level = "severe"

            results.append({
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "stress_level": level,
                "indicator": round(1 - stress_factor, 3),
            })

    return results
