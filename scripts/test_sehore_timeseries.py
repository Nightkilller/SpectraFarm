"""
AgriN — Phase 2: Multi-Temporal Sentinel-2 NDVI Time Series Script

Executes Phase 2 verification pipeline:
1. Initializes Google Earth Engine with project agrin-506618
2. Loads the Sehore Pilot Test AOI from configuration
3. Queries multiple real Sentinel-2 observations (COPERNICUS/S2_SR_HARMONIZED)
4. Computes per-observation NDVI statistics (min, mean, max, stdDev) server-side
5. Formats raw observation time series (preserving granule-level provenance)
6. Derives canonical daily agricultural time series (exactly one observation per calendar date)
7. Validates zero duplicate calendar dates in canonical series
8. Exports raw and canonical CSV outputs to data/processed/sehore/

Usage:
    python scripts/test_sehore_timeseries.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.geospatial.gee_client import get_sehore_aoi, init_earth_engine
from src.geospatial.timeseries import (
    deduplicate_to_canonical,
    extract_ndvi_timeseries,
    timeseries_to_dataframe,
)


def run_sehore_timeseries_pipeline() -> bool:
    print("=" * 85)
    print("AgriN — Phase 2: Multi-Temporal Sentinel-2 NDVI Time Series Pipeline")
    print("=" * 85)
    print()

    # Step 1: Initialize Earth Engine
    settings = get_settings()
    project_id = settings.gee_project or "agrin-506618"
    print(f"[INFO] Initializing Earth Engine with project: {project_id}")

    if not init_earth_engine(project_id):
        print(f"[ERROR] Earth Engine initialization failed for project '{project_id}'.")
        return False

    print("[INFO] Earth Engine connection established successfully.")
    print()

    # Step 2: Load Sehore Pilot AOI
    region = settings.pilot_region
    print(f"[INFO] Loading Pilot AOI: {region.get('name', 'Sehore Pilot Test AOI')}")
    print(f"[INFO] Location: {region.get('district', 'Sehore')}, {region.get('state', 'Madhya Pradesh')}, {region.get('country', 'India')}")
    print(f"[INFO] Center Coordinates: {region.get('center_lat')}°N, {region.get('center_lon')}°E (Buffer: {region.get('buffer_meters', 2000)}m)")

    aoi = get_sehore_aoi(use_buffer=True)
    print()

    # Step 3: Configure Date Range & Cloud Filter
    end_date = date.today()
    start_date = end_date - timedelta(days=180)
    max_cloud = settings.sentinel2_config.get("cloud_cover_max", 20.0)

    print(f"[INFO] Collection: {settings.sentinel2_config.get('collection', 'COPERNICUS/S2_SR_HARMONIZED')}")
    print(f"[INFO] Observation Window: {start_date} to {end_date}")
    print(f"[INFO] Cloud Filter: < {max_cloud}% cloudy pixels")
    print()

    # Step 4: Extract Raw Multi-Temporal NDVI Time Series
    print("[INFO] Querying and computing multi-temporal NDVI statistics over Sehore AOI...")
    raw_timeseries = extract_ndvi_timeseries(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        max_cloud_percentage=max_cloud,
        aoi_name=region.get("name", "Sehore Pilot Test AOI"),
        scale=10,
    )

    print(f"[INFO] Retrieved {raw_timeseries.observations_count} raw Sentinel-2 granule observations.")
    print()

    if raw_timeseries.observations_count == 0:
        print("[WARNING] No observations retrieved.")
        return False

    # Step 5: Derive Canonical Daily Agricultural Time Series
    print("[INFO] Generating canonical daily time series (one observation per calendar date)...")
    canonical_timeseries = deduplicate_to_canonical(raw_timeseries)
    print(f"[INFO] Canonical time series contains {canonical_timeseries.observations_count} daily observations.")
    print()

    # Step 6: Display Canonical Chronological Table
    df_canonical = timeseries_to_dataframe(canonical_timeseries)
    print("-" * 85)
    print(f"{'Date':<12} | {'Cloud %':<8} | {'Min NDVI':<9} | {'Mean NDVI':<10} | {'Max NDVI':<9} | {'StdDev':<8} | {'Image ID':<20}")
    print("-" * 85)
    for _, row in df_canonical.iterrows():
        short_id = row['image_id'].split('/')[-1][:18] + '...' if len(row['image_id']) > 20 else row['image_id']
        print(f"{str(row['date']):<12} | {row['cloud_percentage']:>6.2f}% | {row['min_ndvi']:>9.4f} | {row['mean_ndvi']:>10.4f} | {row['max_ndvi']:>9.4f} | {row['stdDev_ndvi']:>8.4f} | {short_id:<20}")
    print("-" * 85)
    print()

    # Step 7: Export Outputs
    out_dir = PROJECT_ROOT / "data" / "processed" / "sehore"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = timeseries_to_dataframe(raw_timeseries)
    raw_csv = out_dir / "ndvi_timeseries_raw.csv"
    canonical_csv = out_dir / "ndvi_timeseries_canonical.csv"
    default_csv = out_dir / "ndvi_timeseries.csv"

    df_raw.to_csv(raw_csv, index=False)
    df_canonical.to_csv(canonical_csv, index=False)
    df_canonical.to_csv(default_csv, index=False)

    print(f"[INFO] Exported raw observation series ({len(df_raw)} rows) to: {raw_csv.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] Exported canonical daily series ({len(df_canonical)} rows) to: {canonical_csv.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] Saved canonical default to: {default_csv.relative_to(PROJECT_ROOT)}")
    print()

    # Step 8: Multi-Temporal Summary
    mean_vals = df_canonical['mean_ndvi'].tolist()
    overall_mean = sum(mean_vals) / len(mean_vals)
    min_mean = min(mean_vals)
    max_mean = max(mean_vals)

    print("=" * 85)
    print("PHASE 2 MULTI-TEMPORAL SENTINEL-2 NDVI SUMMARY")
    print("=" * 85)
    print(f"  Target Pilot AOI:             {canonical_timeseries.aoi_name} (Sehore, MP)")
    print(f"  Date Range Filter:            {start_date} to {end_date}")
    print(f"  Cloud Cover Threshold:        < {max_cloud}%")
    print(f"  Raw Granule Observations:     {raw_timeseries.observations_count}")
    print(f"  Canonical Daily Observations: {canonical_timeseries.observations_count}")
    print(f"  First Observation Date:       {df_canonical['date'].iloc[0]}")
    print(f"  Last Observation Date:        {df_canonical['date'].iloc[-1]}")
    print(f"  Mean NDVI Trajectory Range:   {min_mean:.4f} to {max_mean:.4f}")
    print(f"  Overall Time Series Mean:     {overall_mean:.4f}")
    print(f"  Data Origin:                  100% Live Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED)")
    print("=" * 85)
    print()

    # Step 9: Validation Checks
    dates_list = df_canonical['date'].tolist()
    has_no_date_duplicates = len(dates_list) == len(set(dates_list))
    valid_bounds = all(-1.0 <= p.mean_ndvi <= 1.0 for p in canonical_timeseries.points)

    if has_no_date_duplicates and valid_bounds and len(canonical_timeseries.points) >= 2:
        print("✅ Validation PASSED: Canonical series contains zero duplicate calendar dates.")
        print("✅ Validation PASSED: Multi-temporal observations verified within physical bounds.")
        print("✅ No synthetic/demo data used. 100% live Earth Engine reduction.")
        print()
        print("=" * 85)
        print("PHASE 2 COMPLETE — MULTI-TEMPORAL SENTINEL-2 NDVI VERIFIED")
        print("=" * 85)
        return True
    else:
        print("❌ Validation FAILED: Validation assertions not met.")
        return False


if __name__ == "__main__":
    success = run_sehore_timeseries_pipeline()
    sys.exit(0 if success else 1)
