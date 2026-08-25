"""Tests for the config loader."""

import pytest
from src.config.settings import Settings, get_settings, reset_settings


class TestSettings:
    def setup_method(self):
        reset_settings()

    def test_load_settings(self):
        s = get_settings()
        assert s.app_name == "AgriN"
        assert s.app_version == "0.1.0"

    def test_mode_setting(self):
        s = get_settings()
        assert s.mode in ["live", "demo"]

    def test_pilot_region(self):
        s = get_settings()
        region = s.pilot_region
        assert region["district"] == "Sehore"
        assert region["state"] == "Madhya Pradesh"
        assert region["center_lat"] == 23.2000
        assert region["center_lon"] == 77.0800

    def test_crop_classes(self):
        s = get_settings()
        assert len(s.crop_classes) == 3
        assert s.crop_ids == ["wheat", "rice", "other"]

    def test_ndvi_thresholds(self):
        s = get_settings()
        t = s.ndvi_thresholds
        assert "bare_soil_max" in t
        assert "dense_vegetation_min" in t
        assert t["trend_threshold"] > 0

    def test_stress_thresholds(self):
        s = get_settings()
        t = s.stress_thresholds
        assert t["healthy_min"] > t["mild_min"] > t["moderate_min"]

    def test_languages(self):
        s = get_settings()
        langs = s.languages
        codes = [l["code"] for l in langs]
        assert "en" in codes
        assert "hi" in codes

    def test_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
