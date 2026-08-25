"""
Unit tests for AgriN Multi-Temporal NDVI Time Series Module.
"""

from datetime import date
import pytest
from src.data.schemas import DataSource, NDVITimeSeries, NDVITimeSeriesPoint
from src.geospatial.timeseries import timeseries_to_dataframe


class TestNDVITimeSeriesSchemas:
    def test_timeseries_point_valid(self):
        point = NDVITimeSeriesPoint(
            observation_date=date(2026, 3, 1),
            image_id="COPERNICUS/S2_SR_HARMONIZED/test_img_01",
            cloud_percentage=5.2,
            min_ndvi=-0.05,
            mean_ndvi=0.42,
            max_ndvi=0.88,
            stdDev_ndvi=0.15,
            data_source=DataSource.LIVE,
        )
        assert point.mean_ndvi == 0.42
        assert point.data_source == DataSource.LIVE
        assert point.observation_date == date(2026, 3, 1)

    def test_timeseries_point_out_of_bounds_rejected(self):
        with pytest.raises(ValueError):
            NDVITimeSeriesPoint(
                observation_date=date(2026, 3, 1),
                image_id="test_id",
                cloud_percentage=0.0,
                min_ndvi=-1.5,  # Invalid: < -1.0
                mean_ndvi=0.5,
                max_ndvi=0.8,
            )

    def test_timeseries_container(self):
        point1 = NDVITimeSeriesPoint(
            observation_date=date(2026, 3, 1),
            image_id="img1",
            cloud_percentage=0.0,
            min_ndvi=0.1,
            mean_ndvi=0.4,
            max_ndvi=0.8,
        )
        point2 = NDVITimeSeriesPoint(
            observation_date=date(2026, 3, 15),
            image_id="img2",
            cloud_percentage=1.0,
            min_ndvi=0.05,
            mean_ndvi=0.35,
            max_ndvi=0.75,
        )
        ts = NDVITimeSeries(
            aoi_name="Sehore Pilot Test AOI",
            start_date=date(2026, 3, 1),
            end_date=date(2026, 3, 15),
            cloud_threshold=20.0,
            observations_count=2,
            points=[point1, point2],
            data_source=DataSource.LIVE,
        )
        assert ts.observations_count == 2
        assert len(ts.points) == 2

    def test_timeseries_to_dataframe(self):
        point = NDVITimeSeriesPoint(
            observation_date=date(2026, 3, 1),
            image_id="img1",
            cloud_percentage=2.5,
            min_ndvi=0.0,
            mean_ndvi=0.35,
            max_ndvi=0.85,
            stdDev_ndvi=0.12,
        )
        ts = NDVITimeSeries(
            aoi_name="Test AOI",
            start_date=date(2026, 3, 1),
            end_date=date(2026, 3, 1),
            cloud_threshold=20.0,
            observations_count=1,
            points=[point],
        )
        df = timeseries_to_dataframe(ts)
        assert len(df) == 1
        assert "mean_ndvi" in df.columns
        assert df.iloc[0]["mean_ndvi"] == 0.35
        assert df.iloc[0]["cloud_percentage"] == 2.5
