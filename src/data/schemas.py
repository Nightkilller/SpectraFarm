"""
AgriN — Data Schemas

Pydantic models for all core entities.  These enforce validation and provide
a single source of truth for the data structures exchanged between services.

Key design decisions:
- Optional fields for data that may not be available in demo/early phases.
- Enums for constrained values (stress levels, trends, data sources).
- All schemas are immutable by default (frozen=True on critical types).
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class CropType(str, Enum):
    WHEAT = "wheat"
    RICE = "rice"
    OTHER = "other"


class StressLevel(str, Enum):
    HEALTHY = "healthy"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class HealthTrend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


class DataSource(str, Enum):
    """Whether data came from real processing or demo/mock mode."""
    LIVE = "live"
    DEMO = "demo"


# ═══════════════════════════════════════════════════════════════════════════
# Farm / Field
# ═══════════════════════════════════════════════════════════════════════════

class BoundingBox(BaseModel):
    """Geographic bounding box in decimal degrees."""
    min_lat: float = Field(..., ge=-90, le=90)
    max_lat: float = Field(..., ge=-90, le=90)
    min_lon: float = Field(..., ge=-180, le=180)
    max_lon: float = Field(..., ge=-180, le=180)

    @field_validator("max_lat")
    @classmethod
    def max_lat_gte_min(cls, v: float, info) -> float:
        if "min_lat" in info.data and v < info.data["min_lat"]:
            raise ValueError("max_lat must be >= min_lat")
        return v

    @field_validator("max_lon")
    @classmethod
    def max_lon_gte_min(cls, v: float, info) -> float:
        if "min_lon" in info.data and v < info.data["min_lon"]:
            raise ValueError("max_lon must be >= min_lon")
        return v


class Farm(BaseModel):
    """A farm / pilot field."""
    farm_id: str
    name: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    bbox: BoundingBox
    area_ha: Optional[float] = Field(None, ge=0, description="Area in hectares")
    crop: Optional[CropType] = None
    season: Optional[str] = None
    data_source: DataSource = DataSource.DEMO


# ═══════════════════════════════════════════════════════════════════════════
# Satellite Observation
# ═══════════════════════════════════════════════════════════════════════════

class SatelliteObservation(BaseModel):
    """A processed satellite observation for a field."""
    observation_date: date
    satellite: str = Field(..., description="e.g. 'Sentinel-2', 'Sentinel-1'")
    farm_id: str

    # Optical indices (Sentinel-2)
    ndvi: Optional[float] = Field(None, ge=-1, le=1)
    ndwi: Optional[float] = Field(None, ge=-1, le=1)
    red: Optional[float] = None
    green: Optional[float] = None
    blue: Optional[float] = None
    nir: Optional[float] = None
    swir1: Optional[float] = None
    swir2: Optional[float] = None

    # SAR features (Sentinel-1)
    vv: Optional[float] = None  # dB
    vh: Optional[float] = None  # dB
    vh_vv_ratio: Optional[float] = None

    cloud_cover: Optional[float] = Field(None, ge=0, le=100)
    data_source: DataSource = DataSource.DEMO


# ═══════════════════════════════════════════════════════════════════════════
# Crop Prediction
# ═══════════════════════════════════════════════════════════════════════════

class CropPrediction(BaseModel):
    """Output of the crop classification model."""
    farm_id: str
    predicted_crop: CropType
    confidence: float = Field(..., ge=0, le=1, description="Prediction probability")
    model_version: str = "rf_v0.1"
    prediction_date: date
    feature_importance: Optional[dict[str, float]] = None
    data_source: DataSource = DataSource.DEMO


# ═══════════════════════════════════════════════════════════════════════════
# Stress Assessment
# ═══════════════════════════════════════════════════════════════════════════

class StressAssessment(BaseModel):
    """Satellite-based crop stress indicator (NOT validated soil moisture)."""
    farm_id: str
    stress_level: StressLevel
    indicator_value: float = Field(
        ..., ge=0, le=1,
        description="Composite stress indicator (0=worst, 1=best)"
    )
    assessment_date: date
    trend: HealthTrend
    ndvi_current: Optional[float] = Field(None, ge=-1, le=1)
    ndvi_previous: Optional[float] = Field(None, ge=-1, le=1)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    data_source: DataSource = DataSource.DEMO


# ═══════════════════════════════════════════════════════════════════════════
# Farm Analysis  (Agricultural Intelligence Layer output)
# ═══════════════════════════════════════════════════════════════════════════

class FarmAnalysis(BaseModel):
    """
    Structured agricultural intelligence — the single object that Gemini
    receives.  All factual measurements come from the backend; Gemini only
    interprets and explains.
    """
    farm: Farm
    crop_prediction: Optional[CropPrediction] = None
    stress_assessment: Optional[StressAssessment] = None
    recent_observations: list[SatelliteObservation] = Field(default_factory=list)

    # Summary fields for Gemini prompt construction
    ndvi_current: Optional[float] = Field(None, ge=-1, le=1)
    ndvi_previous: Optional[float] = Field(None, ge=-1, le=1)
    ndvi_trend: Optional[HealthTrend] = None
    observation_date: Optional[date] = None

    # Weather context (Phase 7 — optional)
    weather: Optional[dict] = None

    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    data_source: DataSource = DataSource.DEMO


# ═══════════════════════════════════════════════════════════════════════════
# Advisory (Gemini output)
# ═══════════════════════════════════════════════════════════════════════════

class Advisory(BaseModel):
    """Gemini-generated advisory for a farm."""
    farm_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    language: str = "en"

    # Content
    observations_summary: str
    advisory_text: str
    limitations: str = (
        "Satellite indicators alone cannot identify the exact cause of "
        "crop stress. Please verify conditions in the field and consult "
        "local agricultural experts for specific treatment decisions."
    )

    # Provenance
    model_version: str = "gemini-1.5-flash"
    data_source: DataSource = DataSource.DEMO
