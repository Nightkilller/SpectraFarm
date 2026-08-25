"""
Tests for AgriN data schemas.

Validates that Pydantic models enforce constraints, accept valid data,
and reject invalid data correctly.
"""

from datetime import date, datetime

import pytest

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


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_bbox():
    return BoundingBox(min_lat=30.85, max_lat=30.95, min_lon=75.80, max_lon=75.90)


@pytest.fixture
def sample_farm(sample_bbox):
    return Farm(
        farm_id="pilot_01",
        name="Pilot Farm 01",
        latitude=30.9,
        longitude=75.85,
        bbox=sample_bbox,
        area_ha=25.0,
        crop=CropType.WHEAT,
        season="rabi",
        data_source=DataSource.DEMO,
    )


@pytest.fixture
def sample_observation():
    return SatelliteObservation(
        observation_date=date(2025, 2, 15),
        satellite="Sentinel-2",
        farm_id="pilot_01",
        ndvi=0.65,
        ndwi=0.12,
        cloud_cover=8.0,
        data_source=DataSource.DEMO,
    )


@pytest.fixture
def sample_prediction():
    return CropPrediction(
        farm_id="pilot_01",
        predicted_crop=CropType.WHEAT,
        confidence=0.91,
        prediction_date=date(2025, 2, 20),
        data_source=DataSource.DEMO,
    )


@pytest.fixture
def sample_stress():
    return StressAssessment(
        farm_id="pilot_01",
        stress_level=StressLevel.MODERATE,
        indicator_value=0.45,
        assessment_date=date(2025, 2, 20),
        trend=HealthTrend.DECLINING,
        ndvi_current=0.55,
        ndvi_previous=0.65,
        data_source=DataSource.DEMO,
    )


# ═══════════════════════════════════════════════════════════════════════════
# BoundingBox Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBoundingBox:
    def test_valid(self, sample_bbox):
        assert sample_bbox.min_lat == 30.85
        assert sample_bbox.max_lon == 75.90

    def test_inverted_lat_rejected(self):
        with pytest.raises(ValueError, match="max_lat must be >= min_lat"):
            BoundingBox(min_lat=31.0, max_lat=30.0, min_lon=75.0, max_lon=76.0)

    def test_inverted_lon_rejected(self):
        with pytest.raises(ValueError, match="max_lon must be >= min_lon"):
            BoundingBox(min_lat=30.0, max_lat=31.0, min_lon=76.0, max_lon=75.0)

    def test_out_of_range_lat(self):
        with pytest.raises(ValueError):
            BoundingBox(min_lat=-100, max_lat=30, min_lon=75, max_lon=76)


# ═══════════════════════════════════════════════════════════════════════════
# Farm Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFarm:
    def test_valid(self, sample_farm):
        assert sample_farm.farm_id == "pilot_01"
        assert sample_farm.crop == CropType.WHEAT
        assert sample_farm.data_source == DataSource.DEMO

    def test_optional_fields(self, sample_bbox):
        farm = Farm(
            farm_id="f2",
            name="Minimal Farm",
            latitude=28.5,
            longitude=77.0,
            bbox=sample_bbox,
        )
        assert farm.crop is None
        assert farm.area_ha is None

    def test_negative_area_rejected(self, sample_bbox):
        with pytest.raises(ValueError):
            Farm(
                farm_id="f3",
                name="Bad Area",
                latitude=28.5,
                longitude=77.0,
                bbox=sample_bbox,
                area_ha=-10,
            )


# ═══════════════════════════════════════════════════════════════════════════
# SatelliteObservation Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSatelliteObservation:
    def test_valid(self, sample_observation):
        assert sample_observation.ndvi == 0.65
        assert sample_observation.satellite == "Sentinel-2"

    def test_ndvi_range(self):
        with pytest.raises(ValueError):
            SatelliteObservation(
                observation_date=date(2025, 1, 1),
                satellite="Sentinel-2",
                farm_id="f1",
                ndvi=1.5,  # Invalid: must be <= 1
            )

    def test_sar_fields(self):
        obs = SatelliteObservation(
            observation_date=date(2025, 1, 1),
            satellite="Sentinel-1",
            farm_id="f1",
            vv=-12.5,
            vh=-18.0,
            vh_vv_ratio=-5.5,
        )
        assert obs.vv == -12.5
        assert obs.ndvi is None  # SAR doesn't produce NDVI


# ═══════════════════════════════════════════════════════════════════════════
# CropPrediction Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCropPrediction:
    def test_valid(self, sample_prediction):
        assert sample_prediction.predicted_crop == CropType.WHEAT
        assert sample_prediction.confidence == 0.91

    def test_confidence_range(self):
        with pytest.raises(ValueError):
            CropPrediction(
                farm_id="f1",
                predicted_crop=CropType.RICE,
                confidence=1.5,  # Invalid: must be <= 1
                prediction_date=date(2025, 1, 1),
            )

    def test_feature_importance(self):
        pred = CropPrediction(
            farm_id="f1",
            predicted_crop=CropType.WHEAT,
            confidence=0.88,
            prediction_date=date(2025, 1, 1),
            feature_importance={"ndvi_mean": 0.35, "vv_mean": 0.20},
        )
        assert "ndvi_mean" in pred.feature_importance


# ═══════════════════════════════════════════════════════════════════════════
# StressAssessment Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStressAssessment:
    def test_valid(self, sample_stress):
        assert sample_stress.stress_level == StressLevel.MODERATE
        assert sample_stress.trend == HealthTrend.DECLINING

    def test_indicator_range(self):
        with pytest.raises(ValueError):
            StressAssessment(
                farm_id="f1",
                stress_level=StressLevel.HEALTHY,
                indicator_value=1.5,  # Invalid: must be <= 1
                assessment_date=date(2025, 1, 1),
                trend=HealthTrend.STABLE,
            )


# ═══════════════════════════════════════════════════════════════════════════
# FarmAnalysis Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFarmAnalysis:
    def test_full_analysis(self, sample_farm, sample_prediction,
                           sample_stress, sample_observation):
        analysis = FarmAnalysis(
            farm=sample_farm,
            crop_prediction=sample_prediction,
            stress_assessment=sample_stress,
            recent_observations=[sample_observation],
            ndvi_current=0.55,
            ndvi_previous=0.65,
            ndvi_trend=HealthTrend.DECLINING,
            observation_date=date(2025, 2, 20),
        )
        assert analysis.farm.farm_id == "pilot_01"
        assert analysis.crop_prediction.predicted_crop == CropType.WHEAT
        assert analysis.data_source == DataSource.DEMO
        assert len(analysis.recent_observations) == 1

    def test_minimal_analysis(self, sample_farm):
        """FarmAnalysis should work with only the farm — early phases."""
        analysis = FarmAnalysis(farm=sample_farm)
        assert analysis.crop_prediction is None
        assert analysis.stress_assessment is None
        assert analysis.recent_observations == []


# ═══════════════════════════════════════════════════════════════════════════
# Advisory Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAdvisory:
    def test_valid(self):
        advisory = Advisory(
            farm_id="pilot_01",
            language="en",
            observations_summary="NDVI has declined over the past 3 observations.",
            advisory_text="Consider checking field moisture levels.",
        )
        assert advisory.farm_id == "pilot_01"
        assert "Satellite indicators" in advisory.limitations

    def test_hindi(self):
        advisory = Advisory(
            farm_id="pilot_01",
            language="hi",
            observations_summary="NDVI पिछले 3 अवलोकनों में गिरा है।",
            advisory_text="कृपया खेत की नमी की स्थिति जांचें।",
        )
        assert advisory.language == "hi"


# ═══════════════════════════════════════════════════════════════════════════
# Enum Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEnums:
    def test_crop_type_values(self):
        assert CropType.WHEAT.value == "wheat"
        assert CropType.RICE.value == "rice"
        assert CropType.OTHER.value == "other"

    def test_stress_level_values(self):
        assert StressLevel.HEALTHY.value == "healthy"
        assert StressLevel.SEVERE.value == "severe"

    def test_health_trend_values(self):
        assert HealthTrend.IMPROVING.value == "improving"
        assert HealthTrend.DECLINING.value == "declining"

    def test_data_source_values(self):
        assert DataSource.LIVE.value == "live"
        assert DataSource.DEMO.value == "demo"
