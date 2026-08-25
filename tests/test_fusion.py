"""
Unit tests for AgriN Optical + SAR Multi-Sensor Fusion Module (Phase 4).
"""

from datetime import date
import pytest
from src.data.schemas import (
    DataSource,
    FusedFeatureDataset,
    FusedObservationPair,
    NDVITimeSeries,
    NDVITimeSeriesPoint,
    ObservationType,
    SARTimeSeries,
    SARTimeSeriesPoint,
    TemporalFeatureVector,
)
from src.features.fusion import (
    compute_temporal_feature_vector,
    fuse_optical_sar_timeseries,
    fused_dataset_to_dataframe,
    temporal_summary_to_dataframe,
)


class TestMultiSensorFusion:
    def test_fuse_optical_sar_timeseries_aligned(self):
        # Create optical points (2026-03-01, 2026-03-06, 2026-03-16)
        opt_p1 = NDVITimeSeriesPoint(
            observation_date=date(2026, 3, 1),
            image_id="opt_1",
            cloud_percentage=0.0,
            min_ndvi=-0.1,
            mean_ndvi=0.35,
            max_ndvi=0.9,
        )
        opt_p2 = NDVITimeSeriesPoint(
            observation_date=date(2026, 3, 6),
            image_id="opt_2",
            cloud_percentage=0.0,
            min_ndvi=-0.1,
            mean_ndvi=0.32,
            max_ndvi=0.9,
        )
        opt_p3 = NDVITimeSeriesPoint(
            observation_date=date(2026, 3, 16),
            image_id="opt_3",
            cloud_percentage=0.0,
            min_ndvi=-0.05,
            mean_ndvi=0.26,
            max_ndvi=0.88,
        )

        opt_ts = NDVITimeSeries(
            aoi_name="Sehore Pilot Test AOI",
            start_date=date(2026, 3, 1),
            end_date=date(2026, 3, 20),
            cloud_threshold=20.0,
            observations_count=3,
            points=[opt_p1, opt_p2, opt_p3],
        )

        # Create SAR points (2026-03-05 [1 day from opt_2], 2026-03-17 [1 day from opt_3])
        sar_p1 = SARTimeSeriesPoint(
            observation_date=date(2026, 3, 5),
            image_id="sar_1",
            orbit_pass="DESCENDING",
            min_vv=-20.0,
            mean_vv=-10.4,
            max_vv=-3.0,
            min_vh=-25.0,
            mean_vh=-17.8,
            max_vh=-10.0,
            mean_vv_vh_ratio=10.3,
            mean_vv_minus_vh_db=7.4,
        )
        sar_p2 = SARTimeSeriesPoint(
            observation_date=date(2026, 3, 17),
            image_id="sar_2",
            orbit_pass="DESCENDING",
            min_vv=-20.0,
            mean_vv=-10.5,
            max_vv=-3.0,
            min_vh=-25.0,
            mean_vh=-18.3,
            max_vh=-10.0,
            mean_vv_vh_ratio=14.7,
            mean_vv_minus_vh_db=7.8,
        )

        sar_ts = SARTimeSeries(
            aoi_name="Sehore Pilot Test AOI",
            start_date=date(2026, 3, 1),
            end_date=date(2026, 3, 20),
            observations_count=2,
            points=[sar_p1, sar_p2],
        )

        fused = fuse_optical_sar_timeseries(opt_ts, sar_ts, max_temporal_delta_days=5)

        assert fused.aligned_pairs_count == 2
        assert len(fused.pairs) == 2

        # Pair 1 on 2026-03-05 should align with 2026-03-06 (lag: 1 day)
        p1 = fused.pairs[0]
        assert p1.target_date == date(2026, 3, 5)
        assert p1.observation_type == ObservationType.FUSED_PAIR
        assert p1.optical_date == date(2026, 3, 6)
        assert p1.temporal_delta_days == 1
        assert p1.ndvi == 0.32
        assert p1.vv_db == -10.4

        # Pair 2 on 2026-03-17 should align with 2026-03-16 (lag: 1 day)
        p2 = fused.pairs[1]
        assert p2.target_date == date(2026, 3, 17)
        assert p2.observation_type == ObservationType.FUSED_PAIR
        assert p2.optical_date == date(2026, 3, 16)
        assert p2.temporal_delta_days == 1
        assert p2.ndvi == 0.26
        assert p2.vv_db == -10.5

    def test_fuse_optical_sar_timeseries_monsoon_gap(self):
        # Optical series with no observations in July
        opt_ts = NDVITimeSeries(
            aoi_name="Sehore Pilot Test AOI",
            start_date=date(2026, 6, 1),
            end_date=date(2026, 7, 30),
            cloud_threshold=20.0,
            observations_count=1,
            points=[
                NDVITimeSeriesPoint(
                    observation_date=date(2026, 6, 1),
                    image_id="opt_1",
                    cloud_percentage=10.0,
                    min_ndvi=-0.1,
                    mean_ndvi=0.17,
                    max_ndvi=0.8,
                )
            ],
        )

        # SAR series with observation on 2026-07-10 (>5 days from June 1)
        sar_ts = SARTimeSeries(
            aoi_name="Sehore Pilot Test AOI",
            start_date=date(2026, 6, 1),
            end_date=date(2026, 7, 30),
            observations_count=1,
            points=[
                SARTimeSeriesPoint(
                    observation_date=date(2026, 7, 10),
                    image_id="sar_july",
                    orbit_pass="DESCENDING",
                    min_vv=-21.0,
                    mean_vv=-8.6,
                    max_vv=-3.0,
                    min_vh=-25.0,
                    mean_vh=-16.8,
                    max_vh=-10.0,
                    mean_vv_vh_ratio=10.9,
                )
            ],
        )

        fused = fuse_optical_sar_timeseries(opt_ts, sar_ts, max_temporal_delta_days=5)
        assert fused.aligned_pairs_count == 1
        p = fused.pairs[0]
        assert p.observation_type == ObservationType.SAR_STANDALONE
        assert p.optical_date is None
        assert p.ndvi is None
        assert p.vv_db == -8.6

    def test_observation_type_semantics(self):
        # Valid FUSED_PAIR
        fused_p = FusedObservationPair(
            pair_id="PAIR_01",
            target_date=date(2026, 3, 5),
            observation_type=ObservationType.FUSED_PAIR,
            optical_date=date(2026, 3, 6),
            ndvi=0.32,
            cloud_percentage=0.0,
            sar_date=date(2026, 3, 5),
            sar_image_id="s1",
            vv_db=-10.4,
            vh_db=-17.8,
            vv_vh_ratio_linear=10.3,
            temporal_delta_days=1,
        )
        assert fused_p.observation_type == ObservationType.FUSED_PAIR
        assert fused_p.ndvi is not None
        assert fused_p.optical_date is not None
        assert fused_p.sar_date is not None

        # Valid SAR_STANDALONE
        standalone_p = FusedObservationPair(
            pair_id="PAIR_02",
            target_date=date(2026, 7, 10),
            observation_type=ObservationType.SAR_STANDALONE,
            optical_date=None,
            ndvi=None,
            cloud_percentage=None,
            sar_date=date(2026, 7, 10),
            sar_image_id="s2",
            vv_db=-8.6,
            vh_db=-16.8,
            vv_vh_ratio_linear=10.9,
            temporal_delta_days=31,
        )
        assert standalone_p.observation_type == ObservationType.SAR_STANDALONE
        assert standalone_p.optical_date is None
        assert standalone_p.ndvi is None

    def test_compute_temporal_feature_vector(self):
        opt_ts = NDVITimeSeries(
            aoi_name="Test AOI",
            start_date=date(2026, 3, 1),
            end_date=date(2026, 3, 20),
            cloud_threshold=20.0,
            observations_count=2,
            points=[
                NDVITimeSeriesPoint(
                    observation_date=date(2026, 3, 1),
                    image_id="o1",
                    cloud_percentage=0.0,
                    min_ndvi=0.1,
                    mean_ndvi=0.4,
                    max_ndvi=0.8,
                ),
                NDVITimeSeriesPoint(
                    observation_date=date(2026, 3, 15),
                    image_id="o2",
                    cloud_percentage=0.0,
                    min_ndvi=0.1,
                    mean_ndvi=0.3,
                    max_ndvi=0.7,
                ),
            ],
        )

        sar_ts = SARTimeSeries(
            aoi_name="Test AOI",
            start_date=date(2026, 3, 1),
            end_date=date(2026, 3, 20),
            observations_count=2,
            points=[
                SARTimeSeriesPoint(
                    observation_date=date(2026, 3, 1),
                    image_id="s1",
                    orbit_pass="DESCENDING",
                    min_vv=-20.0,
                    mean_vv=-10.0,
                    max_vv=-4.0,
                    min_vh=-25.0,
                    mean_vh=-18.0,
                    max_vh=-12.0,
                    mean_vv_vh_ratio=12.0,
                    mean_vv_minus_vh_db=8.0,
                ),
                SARTimeSeriesPoint(
                    observation_date=date(2026, 3, 15),
                    image_id="s2",
                    orbit_pass="DESCENDING",
                    min_vv=-22.0,
                    mean_vv=-12.0,
                    max_vv=-5.0,
                    min_vh=-27.0,
                    mean_vh=-20.0,
                    max_vh=-13.0,
                    mean_vv_vh_ratio=14.0,
                    mean_vv_minus_vh_db=8.0,
                ),
            ],
        )

        vec = compute_temporal_feature_vector(opt_ts, sar_ts, aligned_pairs_count=2)
        assert vec.ndvi_mean == 0.35
        assert vec.vv_mean_db == -11.0
        assert vec.vh_mean_db == -19.0
        assert vec.vv_vh_ratio_mean == 13.0
        assert vec.aligned_pairs_count == 2
        assert vec.status == "UNVALIDATED MULTI-SENSOR FEATURE VECTOR"

    def test_dataframe_converters(self):
        pair = FusedObservationPair(
            pair_id="PAIR_01",
            target_date=date(2026, 3, 5),
            observation_type=ObservationType.FUSED_PAIR,
            optical_date=date(2026, 3, 6),
            ndvi=0.32,
            cloud_percentage=0.0,
            sar_date=date(2026, 3, 5),
            sar_image_id="s1",
            vv_db=-10.4,
            vh_db=-17.8,
            vv_vh_ratio_linear=10.3,
            temporal_delta_days=1,
        )
        vec = TemporalFeatureVector(
            aoi_name="Test",
            start_date=date(2026, 3, 1),
            end_date=date(2026, 3, 20),
            optical_obs_count=1,
            ndvi_mean=0.32,
            sar_obs_count=1,
            vv_mean_db=-10.4,
            vh_mean_db=-17.8,
            aligned_pairs_count=1,
        )
        fused = FusedFeatureDataset(
            aoi_name="Test",
            start_date=date(2026, 3, 1),
            end_date=date(2026, 3, 20),
            aligned_pairs_count=1,
            pairs=[pair],
            temporal_summary=vec,
        )

        df_pairs = fused_dataset_to_dataframe(fused)
        assert len(df_pairs) == 1
        assert "observation_type" in df_pairs.columns
        assert df_pairs.iloc[0]["observation_type"] == "FUSED_PAIR"
        assert "optical_ndvi" in df_pairs.columns
        assert "sar_vv_db" in df_pairs.columns

        df_summary = temporal_summary_to_dataframe(vec)
        assert len(df_summary) == 1
        assert df_summary.iloc[0]["ndvi_mean"] == 0.32
