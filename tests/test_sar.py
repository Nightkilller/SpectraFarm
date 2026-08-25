"""
Unit tests for AgriN Sentinel-1 SAR Module (Phase 3).
"""

from datetime import date
import pytest
from src.data.schemas import DataSource, SARTimeSeries, SARTimeSeriesPoint
from src.geospatial.sar import deduplicate_sar_to_canonical, sar_timeseries_to_dataframe


class TestSARSchemas:
    def test_sar_point_valid(self):
        point = SARTimeSeriesPoint(
            observation_date=date(2026, 3, 5),
            image_id="COPERNICUS/S1_GRD/test_s1_01",
            orbit_pass="DESCENDING",
            relative_orbit=63,
            instrument_mode="IW",
            min_vv=-22.5,
            mean_vv=-11.2,
            max_vv=-3.1,
            stdDev_vv=2.1,
            min_vh=-28.4,
            mean_vh=-18.6,
            max_vh=-10.2,
            stdDev_vh=2.3,
            mean_vv_vh_ratio=5.5,
            mean_vh_vv_ratio=0.18,
            mean_vv_minus_vh_db=7.4,
            data_source=DataSource.LIVE,
        )
        assert point.mean_vv == -11.2
        assert point.mean_vh == -18.6
        assert point.orbit_pass == "DESCENDING"
        assert point.data_source == DataSource.LIVE

    def test_sar_timeseries_container(self):
        point1 = SARTimeSeriesPoint(
            observation_date=date(2026, 3, 5),
            image_id="s1_1",
            orbit_pass="DESCENDING",
            min_vv=-20.0,
            mean_vv=-10.0,
            max_vv=-4.0,
            min_vh=-25.0,
            mean_vh=-17.0,
            max_vh=-11.0,
            mean_vv_vh_ratio=5.0,
        )
        point2 = SARTimeSeriesPoint(
            observation_date=date(2026, 3, 17),
            image_id="s1_2",
            orbit_pass="DESCENDING",
            min_vv=-21.0,
            mean_vv=-11.0,
            max_vv=-5.0,
            min_vh=-26.0,
            mean_vh=-18.0,
            max_vh=-12.0,
            mean_vv_vh_ratio=5.0,
        )
        ts = SARTimeSeries(
            aoi_name="Sehore Pilot Test AOI",
            start_date=date(2026, 3, 1),
            end_date=date(2026, 3, 30),
            orbit_pass="DESCENDING",
            instrument_mode="IW",
            observations_count=2,
            points=[point1, point2],
            data_source=DataSource.LIVE,
        )
        assert ts.observations_count == 2
        assert len(ts.points) == 2

    def test_deduplicate_sar_resolves_same_date(self):
        # Two SAR passes on the same calendar date
        pass1 = SARTimeSeriesPoint(
            observation_date=date(2026, 3, 5),
            image_id="COPERNICUS/S1_GRD/S1A_IW_GRDH_20260305T004523_01",
            orbit_pass="DESCENDING",
            min_vv=-20.0,
            mean_vv=-10.5,
            max_vv=-4.0,
            min_vh=-25.0,
            mean_vh=-17.5,
            max_vh=-11.0,
            mean_vv_vh_ratio=5.0,
        )
        pass2 = SARTimeSeriesPoint(
            observation_date=date(2026, 3, 5),
            image_id="COPERNICUS/S1_GRD/S1A_IW_GRDH_20260305T004523_02",
            orbit_pass="DESCENDING",
            min_vv=-20.2,
            mean_vv=-10.3,
            max_vv=-4.1,
            min_vh=-25.1,
            mean_vh=-17.3,
            max_vh=-11.1,
            mean_vv_vh_ratio=5.0,
        )
        pass3 = SARTimeSeriesPoint(
            observation_date=date(2026, 3, 17),
            image_id="COPERNICUS/S1_GRD/S1A_IW_GRDH_20260317T004523_01",
            orbit_pass="DESCENDING",
            min_vv=-21.0,
            mean_vv=-11.0,
            max_vv=-5.0,
            min_vh=-26.0,
            mean_vh=-18.0,
            max_vh=-12.0,
            mean_vv_vh_ratio=5.0,
        )

        raw_ts = SARTimeSeries(
            aoi_name="Sehore Pilot Test AOI",
            start_date=date(2026, 3, 1),
            end_date=date(2026, 3, 30),
            orbit_pass="DESCENDING",
            instrument_mode="IW",
            observations_count=3,
            points=[pass1, pass2, pass3],
        )

        canonical_ts = deduplicate_sar_to_canonical(raw_ts)

        assert canonical_ts.observations_count == 2
        assert len(canonical_ts.points) == 2

        dates = [p.observation_date for p in canonical_ts.points]
        assert len(dates) == len(set(dates))
        assert canonical_ts.points[0].image_id == "COPERNICUS/S1_GRD/S1A_IW_GRDH_20260305T004523_02"

    def test_sar_timeseries_to_dataframe(self):
        point = SARTimeSeriesPoint(
            observation_date=date(2026, 3, 5),
            image_id="test_s1",
            orbit_pass="DESCENDING",
            relative_orbit=63,
            min_vv=-20.0,
            mean_vv=-10.0,
            max_vv=-4.0,
            min_vh=-25.0,
            mean_vh=-17.0,
            max_vh=-11.0,
            mean_vv_vh_ratio=5.0,
            mean_vv_minus_vh_db=7.0,
        )
        ts = SARTimeSeries(
            aoi_name="Test AOI",
            start_date=date(2026, 3, 5),
            end_date=date(2026, 3, 5),
            observations_count=1,
            points=[point],
        )
        df = sar_timeseries_to_dataframe(ts)
        assert len(df) == 1
        assert "mean_vv_db" in df.columns
        assert "mean_vh_db" in df.columns
        assert "vv_vh_ratio_linear" in df.columns
        assert df.iloc[0]["mean_vv_db"] == -10.0
        assert df.iloc[0]["relative_orbit"] == 63
