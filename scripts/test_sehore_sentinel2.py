"""
AgriN — Real Sentinel-2 + Sehore Pilot Test Script

Executes Phase 1 verification pipeline:
1. Initializes Google Earth Engine with project agrin-506618
2. Loads the Sehore Pilot Test AOI from configuration
3. Queries Sentinel-2 Surface Reflectance collection (COPERNICUS/S2_SR_HARMONIZED)
4. Filters by AOI, Date range, and Cloud percentage
5. Selects the highest quality observation and extracts acquisition metadata
6. Computes real NDVI using (B8 - B4) / (B8 + B4)
7. Computes regional NDVI summary statistics (Min, Mean, Max, StdDev) over the Sehore AOI

Usage:
    python scripts/test_sehore_sentinel2.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.geospatial.gee_client import (
    compute_aoi_ndvi_statistics,
    extract_image_metadata,
    get_ee_module,
    get_sehore_aoi,
    init_earth_engine,
    query_sentinel2_imagery,
)
from src.geospatial.indices import calculate_ndvi


def run_sehore_sentinel2_test() -> bool:
    print("=" * 70)
    print("AgriN — Phase 1: Real Sentinel-2 + Sehore Earth Engine Verification")
    print("=" * 70)
    print()

    # Step 1: Initialize Earth Engine
    settings = get_settings()
    project_id = settings.gee_project or "agrin-506618"
    print(f"[INFO] Initializing Google Earth Engine...")
    print(f"[INFO] Project: {project_id}")

    if not init_earth_engine(project_id):
        print(f"[ERROR] Failed to initialize Earth Engine for project '{project_id}'.")
        print("        Ensure 'earthengine authenticate' has been completed.")
        return False

    ee = get_ee_module()
    print("[INFO] Earth Engine connection established successfully.")
    print()

    # Step 2: Load Sehore Pilot AOI
    region = settings.pilot_region
    print(f"[INFO] Loading Pilot AOI: {region.get('name', 'Sehore Pilot Test AOI')}")
    print(f"[INFO] Location: {region.get('district', 'Sehore')}, {region.get('state', 'Madhya Pradesh')}, {region.get('country', 'India')}")
    print(f"[INFO] Center Coordinates: {region.get('center_lat')}°N, {region.get('center_lon')}°E (Buffer: {region.get('buffer_meters', 2000)}m)")

    aoi = get_sehore_aoi(use_buffer=True)
    aoi_bounds = aoi.bounds().getInfo()
    print(f"[INFO] AOI Bounding Geometry: {aoi_bounds.get('coordinates', [])[0][:2]}...")
    print()

    # Step 3: Query Sentinel-2 Collection
    end_date = date.today()
    # Search within recent months (e.g. past 6 months to ensure clear imagery)
    start_date = end_date - timedelta(days=180)
    max_cloud = settings.sentinel2_config.get("cloud_cover_max", 20.0)

    print(f"[INFO] Querying Sentinel-2 Surface Reflectance (COPERNICUS/S2_SR_HARMONIZED)")
    print(f"[INFO] Temporal Filter: {start_date} to {end_date}")
    print(f"[INFO] Cloud Filter: < {max_cloud}% cloudy pixels")

    collection = query_sentinel2_imagery(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        max_cloud_percentage=max_cloud,
    )

    image_count = collection.size().getInfo()
    print(f"[INFO] Sentinel-2 observations found: {image_count}")
    print()

    if image_count == 0:
        print("[WARNING] No suitable Sentinel-2 imagery found for the selected AOI/date/cloud criteria.")
        print("          Consider expanding date window or adjusting cloud threshold.")
        return False

    # Step 4: Select optimal image & Extract Metadata
    selected_image = ee.Image(collection.first())
    metadata = extract_image_metadata(selected_image)

    print("-" * 50)
    print("Selected Sentinel-2 Image Metadata")
    print("-" * 50)
    print(f"  Image ID:          {metadata['id']}")
    print(f"  Acquisition Date:  {metadata['date']}")
    print(f"  Cloud Percentage:  {metadata['cloud_percentage']:.2f}%")
    print(f"  Spacecraft:        {metadata['spacecraft']}")
    print(f"  Processing Level:  Surface Reflectance (Level-2A Harmonized)")
    print(f"  Spectral Bands:    B4 (Red, 665nm), B8 (NIR, 842nm), B2, B3, B11, B12")
    print()

    # Step 5: Compute NDVI
    print("[INFO] Computing Normalized Difference Vegetation Index (NDVI)...")
    print("       Formula: NDVI = (B8 - B4) / (B8 + B4)")
    image_with_ndvi = calculate_ndvi(selected_image)

    # Step 6: Compute Summary Statistics over Sehore AOI
    print(f"[INFO] Calculating regional NDVI statistics over Sehore AOI at 10m resolution...")
    stats = compute_aoi_ndvi_statistics(image_with_ndvi, aoi=aoi, scale=10)

    print("-" * 50)
    print("Sehore AOI — Real Satellite NDVI Statistics")
    print("-" * 50)
    print(f"  Minimum NDVI:      {stats['min']}")
    print(f"  Mean NDVI:         {stats['mean']}")
    print(f"  Maximum NDVI:      {stats['max']}")
    print(f"  Std Deviation:     {stats['stdDev']}")
    print("-" * 50)
    print()

    # Step 7: Validation Checks
    if stats["mean"] is not None and -1.0 <= stats["mean"] <= 1.0:
        print("✅ Validation PASSED: NDVI mean is within theoretical physical range [-1.0, +1.0].")
        print("✅ Real satellite data pipeline verified from Earth Engine to AgriN.")
        print()
        print("=" * 70)
        print("PHASE 1 COMPLETE — REAL SATELLITE DATA VERIFIED")
        print("=" * 70)
        return True
    else:
        print("❌ Validation FAILED: NDVI values out of expected bounds.")
        return False


if __name__ == "__main__":
    success = run_sehore_sentinel2_test()
    sys.exit(0 if success else 1)
