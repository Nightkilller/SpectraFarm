"""
Unit tests for AgriN Multi-Temporal NDVI Time Series Module.
"""

from datetime import date
import pytest
from src.data.schemas import DataSource, NDVITimeSeries, NDVITimeSeriesPoint
from src.geospatial.timeseries import deduplicate_to_canonical, timeseries_to_dataframe


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
                min_ndvi=-1.5,
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

    def test_deduplicate_to_canonical_resolves_same_date(self):
        # Two granules on the same calendar date (2026-03-11)
        granule_earlier = NDVITimeSeriesPoint(
            observation_date=date(2026, 3, 11),
            image_id="COPERNICUS/S2_SR_HARMONIZED/20260311T051651_20260311T052224_T43QGF",
            cloud_percentage=0.0,
            min_ndvi=-0.2596,
            mean_ndvi=0.2956,
            max_ndvi=0.9685,
        )
        granule_later = NDVITimeSeriesPoint(
            observation_date=date(2026, 3, 11),
            image_id="COPERNICUS/S2_SR_HARMONIZED/20260311T051651_20260311T053230_T43QGF",
            cloud_percentage=0.0,
            min_ndvi=-0.2262,
            mean_ndvi=0.2929,
            max_ndvi=0.9661,
        )
        point_other_date = NDVITimeSeriesPoint(
            observation_date=date(2026, 3, 16),
            image_id="COPERNICUS/S2_SR_HARMONIZED/20260316T051649_20260316T052630_T43QGF",
            cloud_percentage=0.0,
            min_ndvi=-0.0510,
            mean_ndvi=0.2588,
            max_ndvi=0.9040,
        )

        raw_ts = NDVITimeSeries(
            aoi_name="Sehore Pilot Test AOI",
            start_date=date(2026, 3, 1),
            end_date=date(2026, 3, 20),
            cloud_threshold=20.0,
            observations_count=3,
            points=[granule_earlier, granule_later, point_other_date],
        )

        canonical_ts = deduplicate_to_canonical(raw_ts)

        # 3 raw observations become 2 canonical daily observations
        assert canonical_ts.observations_count == 2
        assert len(canonical_ts.points) == 2

        # Verify exact one observation per calendar date
        dates = [p.observation_date for p in canonical_ts.points]
        assert len(dates) == len(set(dates))

        # Verify the deterministic selection picked the latest processing timestamp (053230)
        obs_0311 = [p for p in canonical_ts.points if p.observation_date == date(2026, 3, 11)][0]
        assert "053230" in obs_0311.image_id
        assert obs_0311.mean_ndvi == 0.2929

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
