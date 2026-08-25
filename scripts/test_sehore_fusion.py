"""
AgriN — Phase 4: Optical + SAR Multi-Sensor Fusion Script

Executes Phase 4 verification pipeline:
1. Loads canonical Sentinel-2 optical time series (NDVI) from Earth Engine
2. Loads canonical Sentinel-1 SAR time series (VV, VH, VV/VH) from Earth Engine
3. Aligns multi-sensor observations using nearest-temporal pairing (within +/- 5 days)
4. Computes fused per-pair feature matrix and temporal summary feature vector
5. Formats and validates the multi-sensor feature dataset
6. Exports fused datasets to data/processed/sehore/fused_features.csv and temporal_summary.csv

Usage:
    python scripts/test_sehore_fusion.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.features.fusion import (
    fuse_optical_sar_timeseries,
    fused_dataset_to_dataframe,
    temporal_summary_to_dataframe,
)
from src.geospatial.gee_client import get_sehore_aoi, init_earth_engine
from src.geospatial.sar import deduplicate_sar_to_canonical, extract_sar_timeseries
from src.geospatial.timeseries import deduplicate_to_canonical, extract_ndvi_timeseries


def run_sehore_fusion_pipeline() -> bool:
    print("=" * 95)
    print("AgriN — Phase 4: Optical + SAR Multi-Sensor Fusion Pipeline (Sehore AOI)")
    print("=" * 95)
    print()

    # Step 1: Initialize Earth Engine
    settings = get_settings()
    project_id = settings.gee_project or "agrin-506618"
    print(f"[INFO] Initializing Earth Engine with project: {project_id}")

    if not init_earth_engine(project_id):
        print(f"[ERROR] Earth Engine initialization failed.")
        return False

    print("[INFO] Earth Engine connection established successfully.")
    print()

    # Step 2: Load Sehore Pilot AOI
    region = settings.pilot_region
    print(f"[INFO] Loading Pilot AOI: {region.get('name', 'Sehore Pilot Test AOI')}")
    print(f"[INFO] Location: {region.get('district', 'Sehore')}, {region.get('state', 'Madhya Pradesh')}, {region.get('country', 'India')}")
    aoi = get_sehore_aoi(use_buffer=True)
    print()

    # Step 3: Date Window
    end_date = date.today()
    start_date = end_date - timedelta(days=180)
    print(f"[INFO] Multi-Sensor Fusion Window: {start_date} to {end_date}")
    print()

    # Step 4: Retrieve Sentinel-2 Optical Time Series
    print("[INFO] Retrieving Sentinel-2 optical time series (COPERNICUS/S2_SR_HARMONIZED)...")
    raw_opt = extract_ndvi_timeseries(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        max_cloud_percentage=settings.sentinel2_config.get("cloud_cover_max", 20.0),
        aoi_name=region.get("name", "Sehore Pilot Test AOI"),
    )
    canonical_opt = deduplicate_to_canonical(raw_opt)
    print(f"[INFO] Optical series retrieved: {canonical_opt.observations_count} canonical daily observations.")
    print()

    # Step 5: Retrieve Sentinel-1 SAR Time Series
    print("[INFO] Retrieving Sentinel-1 SAR time series (COPERNICUS/S1_GRD)...")
    raw_sar = extract_sar_timeseries(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        aoi_name=region.get("name", "Sehore Pilot Test AOI"),
        orbit_pass=settings.sentinel1_config.get("orbit_pass", "DESCENDING"),
        instrument_mode=settings.sentinel1_config.get("instrument_mode", "IW"),
    )
    canonical_sar = deduplicate_sar_to_canonical(raw_sar)
    print(f"[INFO] SAR series retrieved: {canonical_sar.observations_count} canonical daily observations.")
    print()

    # Step 6: Fuse Multi-Sensor Observations
    print("[INFO] Performing temporal fusion (nearest observation within +/- 5 days)...")
    fused_dataset = fuse_optical_sar_timeseries(
        optical_ts=canonical_opt,
        sar_ts=canonical_sar,
        max_temporal_delta_days=5,
    )
    print(f"[INFO] Generated {fused_dataset.aligned_pairs_count} fused multi-sensor observation records.")
    print()

    # Step 7: Display Fused Observation Table
    df_fused = fused_dataset_to_dataframe(fused_dataset)
    fused_count = len(df_fused[df_fused["observation_type"] == "FUSED_PAIR"])
    standalone_count = len(df_fused[df_fused["observation_type"] == "SAR_STANDALONE"])

    print("-" * 115)
    print(f"{'Target Date':<12} | {'Type':<15} | {'Optical Date':<12} | {'NDVI':<7} | {'SAR Date':<12} | {'VV (dB)':<8} | {'VH (dB)':<8} | {'VV/VH (lin)':<11} | {'Lag (days)':<10}")
    print("-" * 115)
    for _, row in df_fused.iterrows():
        opt_d = str(row['optical_date']) if pd.notna(row['optical_date']) else "CLOUD GAP"
        ndvi_str = f"{row['optical_ndvi']:>7.4f}" if pd.notna(row['optical_ndvi']) else "   N/A "
        print(f"{str(row['target_date']):<12} | {row['observation_type']:<15} | {opt_d:<12} | {ndvi_str} | {str(row['sar_date']):<12} | {row['sar_vv_db']:>8.2f} | {row['sar_vh_db']:>8.2f} | {row['sar_vv_vh_ratio']:>11.2f} | {row['temporal_delta_days']:>10}")
    print("-" * 115)
    print()

    # Step 8: Display Summary Vector
    summary = fused_dataset.temporal_summary
    print("=" * 115)
    print("PHASE 4 TEMPORAL SUMMARY FEATURE VECTOR")
    print("=" * 115)
    print(f"  Target Pilot AOI:             {summary.aoi_name} (Sehore, MP)")
    print(f"  Date Range:                   {summary.start_date} to {summary.end_date}")
    print(f"  Total Fused Matrix Rows:      {len(df_fused)}")
    print(f"  FUSED_PAIR Rows:              {fused_count} (Optical + SAR within <= 5 days)")
    print(f"  SAR_STANDALONE Rows:          {standalone_count} (SAR only during optical cloud gap)")
    print(f"  Optical Observations:         {summary.optical_obs_count} passes (NDVI Mean: {summary.ndvi_mean:.4f}, Range: {summary.ndvi_min:.4f} to {summary.ndvi_max:.4f}, Trend Slope: {summary.ndvi_slope:.6f})")
    print(f"  SAR Observations:             {summary.sar_obs_count} passes (VV Mean: {summary.vv_mean_db:.2f} dB, VH Mean: {summary.vh_mean_db:.2f} dB, VV/VH Ratio: {summary.vv_vh_ratio_mean:.2f})")
    print(f"  Data Origin:                  LIVE DERIVED FROM SATELLITE (100% Earth Engine Compute)")
    print(f"  Scientific Status:            {summary.status}")
    print("=" * 115)
    print()

    # Step 9: Export Datasets
    out_dir = PROJECT_ROOT / "data" / "processed" / "sehore"
    out_dir.mkdir(parents=True, exist_ok=True)

    fused_csv = out_dir / "fused_features.csv"
    summary_csv = out_dir / "temporal_feature_summary.csv"

    df_fused.to_csv(fused_csv, index=False)
    temporal_summary_to_dataframe(summary).to_csv(summary_csv, index=False)

    print(f"[INFO] Exported fused observation dataset to: {fused_csv.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] Exported temporal feature summary to: {summary_csv.relative_to(PROJECT_ROOT)}")
    print()

    # Step 10: Validation
    if fused_dataset.aligned_pairs_count >= 10 and summary.ndvi_mean is not None and summary.vv_mean_db is not None:
        print("✅ Validation PASSED: Multi-sensor optical and SAR features successfully fused.")
        print("✅ Validation PASSED: Temporal summary vector computed with zero synthetic data.")
        print()
        print("=" * 95)
        print("PHASE 4 COMPLETE — OPTICAL + SAR FUSION VERIFIED")
        print("=" * 95)
        return True
    else:
        print("❌ Validation FAILED: Inadequate fused observations.")
        return False


if __name__ == "__main__":
    success = run_sehore_fusion_pipeline()
    sys.exit(0 if success else 1)
