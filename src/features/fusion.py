"""
AgriN — Optical + SAR Multi-Sensor Fusion Module (Phase 4)

Fuses canonical Sentinel-2 optical NDVI time series and Sentinel-1 SAR backscatter
time series into a unified multi-sensor agricultural feature dataset.

Features:
1. Nearest-temporal cross-sensor observation alignment (within configurable delta days).
2. Per-observation multi-sensor records (Optical NDVI + SAR VV/VH).
3. Temporal aggregated feature vector for machine learning readiness (Phase 6/7).
4. Strictly preserves provenance (LIVE DERIVED FROM SATELLITE) without fabricating observations.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.data.schemas import (
    DataSource,
    FusedFeatureDataset,
    FusedObservationPair,
    NDVITimeSeries,
    ObservationType,
    SARTimeSeries,
    TemporalFeatureVector,
)

logger = logging.getLogger(__name__)


def fuse_optical_sar_timeseries(
    optical_ts: NDVITimeSeries,
    sar_ts: SARTimeSeries,
    max_temporal_delta_days: int = 5,
) -> FusedFeatureDataset:
    """
    Temporally fuse canonical Sentinel-2 optical observations and Sentinel-1 SAR observations.

    Alignment Strategy:
    - Iterates through all unique acquisition dates in both time series.
    - For each SAR pass (all-weather anchor), identifies the closest optical pass within max_temporal_delta_days.
    - Captures temporal lag (temporal_delta_days) to quantify cross-sensor synchronization.
    - Explicitly tags each row as FUSED_PAIR (both sensors valid) or SAR_STANDALONE (optical cloud gap).
    - Calculates joint aggregate statistical vector across the full observation window.

    Args:
        optical_ts: Canonical NDVITimeSeries from Sentinel-2.
        sar_ts: Canonical SARTimeSeries from Sentinel-1.
        max_temporal_delta_days: Maximum allowable calendar day difference for pairing.

    Returns:
        FusedFeatureDataset: Structured Pydantic model containing aligned observation pairs and temporal summary.
    """
    aoi_name = optical_ts.aoi_name or sar_ts.aoi_name
    start_date = min(optical_ts.start_date, sar_ts.start_date)
    end_date = max(optical_ts.end_date, sar_ts.end_date)

    pairs: list[FusedObservationPair] = []

    # Map SAR observations to nearest optical observation
    for idx, sar_pt in enumerate(sar_ts.points):
        sar_d = sar_pt.observation_date

        # Find closest optical point
        best_opt = None
        min_delta = float("inf")

        for opt_pt in optical_ts.points:
            delta = abs((opt_pt.observation_date - sar_d).days)
            if delta < min_delta:
                min_delta = delta
                best_opt = opt_pt

        pair_id = f"PAIR_{sar_d.strftime('%Y%m%d')}_{idx+1:02d}"

        if best_opt is not None and min_delta <= max_temporal_delta_days:
            pair = FusedObservationPair(
                pair_id=pair_id,
                target_date=sar_d,
                observation_type=ObservationType.FUSED_PAIR,
                optical_date=best_opt.observation_date,
                optical_image_id=best_opt.image_id,
                ndvi=best_opt.mean_ndvi,
                cloud_percentage=best_opt.cloud_percentage,
                sar_date=sar_d,
                sar_image_id=sar_pt.image_id,
                vv_db=sar_pt.mean_vv,
                vh_db=sar_pt.mean_vh,
                vv_vh_ratio_linear=sar_pt.mean_vv_vh_ratio,
                vv_minus_vh_db=sar_pt.mean_vv_minus_vh_db,
                temporal_delta_days=int(min_delta),
                data_source=DataSource.LIVE,
            )
        else:
            # All-weather SAR observation without concurrent cloud-free optical match (e.g. monsoon)
            pair = FusedObservationPair(
                pair_id=pair_id,
                target_date=sar_d,
                observation_type=ObservationType.SAR_STANDALONE,
                optical_date=None,
                optical_image_id=None,
                ndvi=None,
                cloud_percentage=None,
                sar_date=sar_d,
                sar_image_id=sar_pt.image_id,
                vv_db=sar_pt.mean_vv,
                vh_db=sar_pt.mean_vh,
                vv_vh_ratio_linear=sar_pt.mean_vv_vh_ratio,
                vv_minus_vh_db=sar_pt.mean_vv_minus_vh_db,
                temporal_delta_days=int(min_delta) if min_delta != float("inf") else 999,
                data_source=DataSource.LIVE,
            )

        pairs.append(pair)

    # Compute Temporal Aggregated Feature Vector
    summary = compute_temporal_feature_vector(optical_ts, sar_ts, len([p for p in pairs if p.ndvi is not None]))

    return FusedFeatureDataset(
        aoi_name=aoi_name,
        start_date=start_date,
        end_date=end_date,
        aligned_pairs_count=len(pairs),
        pairs=pairs,
        temporal_summary=summary,
        data_source=DataSource.LIVE,
    )


def compute_temporal_feature_vector(
    optical_ts: NDVITimeSeries,
    sar_ts: SARTimeSeries,
    aligned_pairs_count: int,
) -> TemporalFeatureVector:
    """Compute summary statistics and trends across the complete time series window."""
    aoi_name = optical_ts.aoi_name or sar_ts.aoi_name
    start_date = min(optical_ts.start_date, sar_ts.start_date)
    end_date = max(optical_ts.end_date, sar_ts.end_date)

    # Optical metrics
    ndvi_vals = [p.mean_ndvi for p in optical_ts.points]
    if ndvi_vals:
        ndvi_arr = np.array(ndvi_vals)
        ndvi_mean = round(float(np.mean(ndvi_arr)), 4)
        ndvi_min = round(float(np.min(ndvi_arr)), 4)
        ndvi_max = round(float(np.max(ndvi_arr)), 4)
        ndvi_std = round(float(np.std(ndvi_arr)), 4)
        ndvi_range = round(float(ndvi_max - ndvi_min), 4)

        if len(ndvi_vals) >= 3:
            x = np.arange(len(ndvi_vals), dtype=float)
            slope = float(np.polyfit(x, ndvi_vals, 1)[0])
            ndvi_slope = round(slope, 6)
        else:
            ndvi_slope = 0.0
    else:
        ndvi_mean = ndvi_min = ndvi_max = ndvi_std = ndvi_range = ndvi_slope = None

    # SAR metrics
    vv_vals = [p.mean_vv for p in sar_ts.points]
    vh_vals = [p.mean_vh for p in sar_ts.points]
    ratio_vals = [p.mean_vv_vh_ratio for p in sar_ts.points]
    diff_vals = [p.mean_vv_minus_vh_db for p in sar_ts.points if p.mean_vv_minus_vh_db is not None]

    if vv_vals:
        vv_arr = np.array(vv_vals)
        vv_mean_db = round(float(np.mean(vv_arr)), 2)
        vv_min_db = round(float(np.min(vv_arr)), 2)
        vv_max_db = round(float(np.max(vv_arr)), 2)
        vv_std_db = round(float(np.std(vv_arr)), 2)
    else:
        vv_mean_db = vv_min_db = vv_max_db = vv_std_db = None

    if vh_vals:
        vh_arr = np.array(vh_vals)
        vh_mean_db = round(float(np.mean(vh_arr)), 2)
        vh_min_db = round(float(np.min(vh_arr)), 2)
        vh_max_db = round(float(np.max(vh_arr)), 2)
        vh_std_db = round(float(np.std(vh_arr)), 2)
    else:
        vh_mean_db = vh_min_db = vh_max_db = vh_std_db = None

    vv_vh_ratio_mean = round(float(np.mean(ratio_vals)), 2) if ratio_vals else None
    vv_minus_vh_mean_db = round(float(np.mean(diff_vals)), 2) if diff_vals else None

    return TemporalFeatureVector(
        aoi_name=aoi_name,
        start_date=start_date,
        end_date=end_date,
        optical_obs_count=optical_ts.observations_count,
        ndvi_mean=ndvi_mean,
        ndvi_min=ndvi_min,
        ndvi_max=ndvi_max,
        ndvi_std=ndvi_std,
        ndvi_range=ndvi_range,
        ndvi_slope=ndvi_slope,
        sar_obs_count=sar_ts.observations_count,
        vv_mean_db=vv_mean_db,
        vv_min_db=vv_min_db,
        vv_max_db=vv_max_db,
        vv_std_db=vv_std_db,
        vh_mean_db=vh_mean_db,
        vh_min_db=vh_min_db,
        vh_max_db=vh_max_db,
        vh_std_db=vh_std_db,
        vv_vh_ratio_mean=vv_vh_ratio_mean,
        vv_minus_vh_mean_db=vv_minus_vh_mean_db,
        aligned_pairs_count=aligned_pairs_count,
        data_source=DataSource.LIVE,
        status="UNVALIDATED MULTI-SENSOR FEATURE VECTOR",
    )


def fused_dataset_to_dataframe(fused_dataset: FusedFeatureDataset) -> pd.DataFrame:
    """Convert fused observation pairs to a pandas DataFrame."""
    records = []
    for p in fused_dataset.pairs:
        obs_type = p.observation_type.value if hasattr(p.observation_type, "value") else str(p.observation_type)
        records.append(
            {
                "pair_id": p.pair_id,
                "target_date": p.target_date,
                "observation_type": obs_type,
                "optical_date": p.optical_date,
                "optical_ndvi": p.ndvi,
                "optical_cloud_pct": p.cloud_percentage,
                "sar_date": p.sar_date,
                "sar_vv_db": p.vv_db,
                "sar_vh_db": p.vh_db,
                "sar_vv_vh_ratio": p.vv_vh_ratio_linear,
                "sar_vv_minus_vh_db": p.vv_minus_vh_db,
                "temporal_delta_days": p.temporal_delta_days,
            }
        )
    return pd.DataFrame(records)


def temporal_summary_to_dataframe(summary: TemporalFeatureVector) -> pd.DataFrame:
    """Convert the temporal feature summary vector to a single-row DataFrame."""
    return pd.DataFrame([summary.model_dump()])
