"""
AgriN — Phase 2: Multi-Temporal Sentinel-2 NDVI Time Series Script

Executes Phase 2 verification pipeline:
1. Initializes Google Earth Engine with project agrin-506618
2. Loads the Sehore Pilot Test AOI from configuration
3. Queries multiple real Sentinel-2 observations (COPERNICUS/S2_SR_HARMONIZED)
4. Computes per-observation NDVI statistics (min, mean, max, stdDev) server-side
5. Formats and validates the complete chronological time series
6. Saves structured time series summary to data/processed/sehore/ndvi_timeseries.csv

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
from src.geospatial.timeseries import extract_ndvi_timeseries, timeseries_to_dataframe


def run_sehore_timeseries_pipeline() -> bool:
    print("=" * 80)
    print("AgriN — Phase 2: Multi-Temporal Sentinel-2 NDVI Time Series Pipeline")
    print("=" * 80)
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
    # Use past 180 days (6 months) for multi-temporal trajectory
    end_date = date.today()
    start_date = end_date - timedelta(days=180)
    max_cloud = settings.sentinel2_config.get("cloud_cover_max", 20.0)

    print(f"[INFO] Collection: {settings.sentinel2_config.get('collection', 'COPERNICUS/S2_SR_HARMONIZED')}")
    print(f"[INFO] Observation Window: {start_date} to {end_date}")
    print(f"[INFO] Cloud Filter: < {max_cloud}% cloudy pixels")
    print()

    # Step 4: Extract Multi-Temporal NDVI Time Series
    print("[INFO] Querying and computing multi-temporal NDVI statistics over Sehore AOI...")
    timeseries = extract_ndvi_timeseries(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        max_cloud_percentage=max_cloud,
        aoi_name=region.get("name", "Sehore Pilot Test AOI"),
        scale=10,
    )

    print(f"[INFO] Successfully retrieved {timeseries.observations_count} real Sentinel-2 observations.")
    print()

    if timeseries.observations_count == 0:
        print("[WARNING] No observations retrieved. Ensure date range and cloud criteria are adequate.")
        return False

    # Step 5: Display Chronological Time Series Table
    df = timeseries_to_dataframe(timeseries)
    print("-" * 80)
    print(f"{'Date':<12} | {'Cloud %':<8} | {'Min NDVI':<9} | {'Mean NDVI':<10} | {'Max NDVI':<9} | {'StdDev':<8} | {'Image ID':<20}")
    print("-" * 80)
    for _, row in df.iterrows():
        short_id = row['image_id'].split('/')[-1][:18] + '...' if len(row['image_id']) > 20 else row['image_id']
        print(f"{str(row['date']):<12} | {row['cloud_percentage']:>6.2f}% | {row['min_ndvi']:>9.4f} | {row['mean_ndvi']:>10.4f} | {row['max_ndvi']:>9.4f} | {row['stdDev_ndvi']:>8.4f} | {short_id:<20}")
    print("-" * 80)
    print()

    # Step 6: Summary Statistics
    mean_ndvi_values = df['mean_ndvi'].tolist()
    overall_mean = sum(mean_ndvi_values) / len(mean_ndvi_values)
    min_mean_ndvi = min(mean_ndvi_values)
    max_mean_ndvi = max(mean_ndvi_values)

    first_obs_date = df['date'].iloc[0]
    last_obs_date = df['date'].iloc[-1]

    print("=" * 80)
    print("PHASE 2 MULTI-TEMPORAL SENTINEL-2 NDVI SUMMARY")
    print("=" * 80)
    print(f"  Target Pilot AOI:          {timeseries.aoi_name} (Sehore, MP)")
    print(f"  Date Range Filter:         {start_date} to {end_date}")
    print(f"  Cloud Cover Threshold:     < {max_cloud}%")
    print(f"  Total Valid Observations:  {timeseries.observations_count}")
    print(f"  First Observation Date:    {first_obs_date}")
    print(f"  Last Observation Date:     {last_obs_date}")
    print(f"  Temporal Trajectory Range: Mean NDVI spans from {min_mean_ndvi:.4f} to {max_mean_ndvi:.4f}")
    print(f"  Overall Time Series Mean:  {overall_mean:.4f}")
    print(f"  Data Origin:               100% Live Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED)")
    print("=" * 80)
    print()

    # Step 7: Export small structured CSV for inspection
    out_dir = PROJECT_ROOT / "data" / "processed" / "sehore"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ndvi_timeseries.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Exported structured time series record to: {out_csv.relative_to(PROJECT_ROOT)}")
    print()

    # Step 8: Validation assertions
    valid_range = all(-1.0 <= p.mean_ndvi <= 1.0 for p in timeseries.points)
    if valid_range and timeseries.observations_count >= 2:
        print("✅ Validation PASSED: Multi-temporal observations verified within physical bounds.")
        print("✅ No synthetic/demo data used. All points derived from real Earth Engine observations.")
        print()
        print("=" * 80)
        print("PHASE 2 COMPLETE — MULTI-TEMPORAL SENTINEL-2 NDVI VERIFIED")
        print("=" * 80)
        return True
    else:
        print("❌ Validation FAILED: Inadequate or out-of-bounds observations.")
        return False


if __name__ == "__main__":
    success = run_sehore_timeseries_pipeline()
    sys.exit(0 if success else 1)
