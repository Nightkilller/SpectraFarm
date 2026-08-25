"""Tests for demo data, features, stress detection, and farm analysis."""

from datetime import date

import pytest

from src.data.demo_data import (
    generate_demo_crop_map_data,
    generate_demo_stress_map_data,
    generate_ndvi_timeseries,
    generate_sar_observations,
    get_demo_advisory,
    get_demo_crop_prediction,
    get_demo_farm,
    get_demo_farm_analysis,
    get_demo_stress_assessment,
)
from src.data.schemas import (
    CropType,
    DataSource,
    HealthTrend,
    StressLevel,
)
from src.features.feature_extraction import (
    combine_features,
    extract_optical_features,
    extract_sar_features,
)
from src.intelligence.stress_analysis import assess_stress


# ═══════════════════════════════════════════════════════════════════════════
# Demo Data Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDemoData:
    def test_demo_farm(self):
        farm = get_demo_farm()
        assert farm.farm_id == "pilot_01"
        assert farm.data_source == DataSource.DEMO

    def test_ndvi_timeseries(self):
        obs = generate_ndvi_timeseries()
        assert len(obs) == 12
        for o in obs:
            assert o.data_source == DataSource.DEMO
            assert -1 <= o.ndvi <= 1

    def test_ndvi_timeseries_trends(self):
        dec = generate_ndvi_timeseries(trend="declining")
        imp = generate_ndvi_timeseries(trend="improving")
        # Declining: later values should generally be lower
        assert dec[-1].ndvi < dec[3].ndvi
        # Improving: later values should generally be higher
        assert imp[-1].ndvi > imp[0].ndvi

    def test_sar_observations(self):
        obs = generate_sar_observations()
        assert len(obs) == 12
        for o in obs:
            assert o.satellite == "Sentinel-1"
            assert o.vv is not None
            assert o.vh is not None

    def test_demo_prediction(self):
        pred = get_demo_crop_prediction()
        assert pred.predicted_crop == CropType.WHEAT
        assert pred.data_source == DataSource.DEMO

    def test_demo_stress(self):
        stress = get_demo_stress_assessment()
        assert stress.stress_level == StressLevel.MODERATE

    def test_demo_farm_analysis(self):
        analysis = get_demo_farm_analysis()
        assert analysis.farm.farm_id == "pilot_01"
        assert analysis.crop_prediction is not None
        assert analysis.stress_assessment is not None
        assert len(analysis.recent_observations) > 0

    def test_demo_advisory_en(self):
        adv = get_demo_advisory(language="en")
        assert adv.language == "en"
        assert len(adv.advisory_text) > 50

    def test_demo_advisory_hi(self):
        adv = get_demo_advisory(language="hi")
        assert adv.language == "hi"
        assert len(adv.advisory_text) > 50

    def test_crop_map_data(self):
        data = generate_demo_crop_map_data(grid_size=10)
        assert len(data) == 100
        crops = {d["crop"] for d in data}
        assert "wheat" in crops

    def test_stress_map_data(self):
        data = generate_demo_stress_map_data(grid_size=10)
        assert len(data) == 100
        levels = {d["stress_level"] for d in data}
        assert len(levels) > 1


# ═══════════════════════════════════════════════════════════════════════════
# Feature Extraction Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureExtraction:
    def test_optical_features(self):
        obs = generate_ndvi_timeseries()
        features = extract_optical_features(obs)
        assert "ndvi_mean" in features
        assert "ndvi_std" in features
        assert "ndvi_trend" in features
        assert 0 < features["ndvi_mean"] < 1

    def test_sar_features(self):
        obs = generate_sar_observations()
        features = extract_sar_features(obs)
        assert "vv_mean" in features
        assert "vh_mean" in features

    def test_combine_features(self):
        opt = extract_optical_features(generate_ndvi_timeseries())
        sar = extract_sar_features(generate_sar_observations())
        combined = combine_features(opt, sar)
        assert "ndvi_mean" in combined
        assert "vv_mean" in combined

    def test_empty_observations(self):
        features = extract_optical_features([])
        assert features == {}


# ═══════════════════════════════════════════════════════════════════════════
# Stress Analysis Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStressAnalysis:
    def test_stress_from_demo_data(self):
        obs = generate_ndvi_timeseries(trend="declining")
        stress = assess_stress(obs, "test_farm")
        assert stress.stress_level in [
            StressLevel.HEALTHY, StressLevel.MILD,
            StressLevel.MODERATE, StressLevel.SEVERE,
        ]
        assert 0 <= stress.indicator_value <= 1
        assert stress.ndvi_current is not None

    def test_healthy_crop(self):
        obs = generate_ndvi_timeseries(trend="improving")
        stress = assess_stress(obs, "test_farm")
        # Improving trend with decent NDVI should not be severe
        assert stress.stress_level != StressLevel.SEVERE

    def test_empty_observations_fallback(self):
        stress = assess_stress([], "test_farm")
        assert stress.confidence == 0.1  # Low confidence fallback
