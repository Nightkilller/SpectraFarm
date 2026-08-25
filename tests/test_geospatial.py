"""
Unit tests for AgriN Geospatial module (Earth Engine & Indices).
"""

import pytest
from unittest.mock import MagicMock, patch
from src.config.settings import get_settings, reset_settings
from src.geospatial.indices import calculate_ndvi, calculate_ndwi


class TestGeospatialConfiguration:
    def setup_method(self):
        reset_settings()

    def test_sehore_pilot_config(self):
        settings = get_settings()
        region = settings.pilot_region
        assert region["district"] == "Sehore"
        assert region["state"] == "Madhya Pradesh"
        assert region["country"] == "India"
        assert region["center_lat"] == 23.2000
        assert region["center_lon"] == 77.0800
        assert region["buffer_meters"] == 2000

    def test_sentinel2_collection_config(self):
        settings = get_settings()
        s2_cfg = settings.sentinel2_config
        assert s2_cfg["collection"] == "COPERNICUS/S2_SR_HARMONIZED"
        assert "B4" in s2_cfg["bands"]
        assert "B8" in s2_cfg["bands"]
        assert s2_cfg["cloud_cover_max"] <= 30


class TestIndicesLogic:
    def test_calculate_ndvi_calls_normalized_difference(self):
        mock_image = MagicMock()
        mock_norm_diff = MagicMock()
        mock_image.normalizedDifference.return_value = mock_norm_diff
        mock_renamed = MagicMock()
        mock_norm_diff.rename.return_value = mock_renamed
        mock_final_image = MagicMock()
        mock_image.addBands.return_value = mock_final_image

        result = calculate_ndvi(mock_image)

        mock_image.normalizedDifference.assert_called_once_with(["B8", "B4"])
        mock_norm_diff.rename.assert_called_once_with("NDVI")
        mock_image.addBands.assert_called_once_with(mock_renamed)
        assert result == mock_final_image

    def test_calculate_ndwi_calls_normalized_difference(self):
        mock_image = MagicMock()
        mock_norm_diff = MagicMock()
        mock_image.normalizedDifference.return_value = mock_norm_diff
        mock_renamed = MagicMock()
        mock_norm_diff.rename.return_value = mock_renamed
        mock_final_image = MagicMock()
        mock_image.addBands.return_value = mock_final_image

        result = calculate_ndwi(mock_image)

        mock_image.normalizedDifference.assert_called_once_with(["B8", "B11"])
        mock_norm_diff.rename.assert_called_once_with("NDWI")
        mock_image.addBands.assert_called_once_with(mock_renamed)
        assert result == mock_final_image
