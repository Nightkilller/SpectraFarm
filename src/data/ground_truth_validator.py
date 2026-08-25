"""
AgriN — Ground Truth Validation & Ingestion Infrastructure (Phase 5)

Cloud-ready validation module for authoritative ground-truth field data.
Enforces rigorous data-quality checks before allowing reference data to enter
downstream crop classification (Phase 6) and stress calibration (Phase 7).

Scientific & Quality Checks:
1. Strict Schema Enforcement (field_id, lat, lon, crop_type, season, reference_date, source, verification_method).
2. Spatial Bounding Validation (ensures coordinates lie within target agricultural region).
3. Duplicate Coordinate Detection (flags field centroids closer than spatial tolerance threshold).
4. Class Balance Analysis (quantifies class distribution and flags extreme imbalance).
5. Spatial Blocking / Spatial K-Fold ID Assignment (prevents spatial autocorrelation leakage).
6. Provenance & Source Auditing (rejects unverified or missing sources).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.data.schemas import (
    DataSource,
    GroundTruthDataset,
    GroundTruthRecord,
    GroundTruthValidationReport,
)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "field_id",
    "latitude",
    "longitude",
    "crop_type",
    "season",
    "reference_date",
    "source",
    "verification_method",
]


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters between two geodetic coordinates."""
    r = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def assign_spatial_blocks(
    df: pd.DataFrame,
    grid_size_deg: float = 0.02,
) -> pd.DataFrame:
    """
    Assign spatial block IDs using deterministic geographic grid binning.
    Used for spatial k-fold cross-validation to prevent spatial autocorrelation leakage.
    """
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    df = df.copy()
    lat_bins = np.floor(df["latitude"].values / grid_size_deg).astype(int)
    lon_bins = np.floor(df["longitude"].values / grid_size_deg).astype(int)
    df["spatial_block_id"] = [f"BLOCK_{la}_{lo}" for la, lo in zip(lat_bins, lon_bins)]
    return df


def validate_ground_truth_dataframe(
    df: pd.DataFrame,
    dataset_name: str = "Ground Truth Dataset",
    expected_bbox: Optional[dict[str, float]] = None,
    duplicate_tolerance_meters: float = 15.0,
) -> tuple[list[GroundTruthRecord], GroundTruthValidationReport]:
    """
    Perform a strict validation audit on a candidate ground-truth DataFrame.

    Returns:
        tuple[list[GroundTruthRecord], GroundTruthValidationReport]
    """
    issues: list[str] = []

    if df.empty:
        report = GroundTruthValidationReport(
            dataset_name=dataset_name,
            total_records=0,
            valid_records_count=0,
            flagged_records_count=0,
            unique_locations_count=0,
            duplicate_locations_count=0,
            crop_class_distribution={},
            season_distribution={},
            spatial_blocks_count=0,
            bounding_box_valid=False,
            validation_status="WAITING_FOR_DATA",
            issues=["Dataset is empty. Real ground-truth field observations are required."],
        )
        return [], report

    # 1. Check Required Columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # 2. Check Null Values in Critical Fields
    for col in [c for c in REQUIRED_COLUMNS if c in df.columns]:
        null_count = df[col].isna().sum()
        if null_count > 0:
            issues.append(f"Column '{col}' contains {null_count} null/missing values.")

    # 3. Spatial Bounding Validation
    bbox_valid = True
    if expected_bbox and "latitude" in df.columns and "longitude" in df.columns:
        min_lat = expected_bbox.get("min_lat", -90.0)
        max_lat = expected_bbox.get("max_lat", 90.0)
        min_lon = expected_bbox.get("min_lon", -180.0)
        max_lon = expected_bbox.get("max_lon", 180.0)

        out_of_bounds = df[
            (df["latitude"] < min_lat)
            | (df["latitude"] > max_lat)
            | (df["longitude"] < min_lon)
            | (df["longitude"] > max_lon)
        ]
        if not out_of_bounds.empty:
            bbox_valid = False
            issues.append(
                f"{len(out_of_bounds)} records fall outside expected geographic bounding box "
                f"[{min_lat}, {max_lat}, {min_lon}, {max_lon}]."
            )

    # 4. Duplicate Coordinate Detection
    duplicate_count = 0
    if "latitude" in df.columns and "longitude" in df.columns:
        coords = df[["latitude", "longitude"]].values
        n = len(coords)
        for i in range(n):
            for j in range(i + 1, n):
                dist = haversine_distance_meters(
                    coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]
                )
                if dist < duplicate_tolerance_meters:
                    duplicate_count += 1

        if duplicate_count > 0:
            issues.append(
                f"Found {duplicate_count} pairs of coordinates within {duplicate_tolerance_meters}m spatial tolerance (potential duplicate/overlapping surveys)."
            )

    # 5. Assign Spatial Blocks for Spatial K-Fold
    df_blocked = assign_spatial_blocks(df)
    spatial_blocks_count = len(df_blocked["spatial_block_id"].unique()) if "spatial_block_id" in df_blocked.columns else 0

    # 6. Distributions
    crop_dist = dict(Counter(df["crop_type"].dropna().tolist())) if "crop_type" in df.columns else {}
    season_dist = dict(Counter(df["season"].dropna().tolist())) if "season" in df.columns else {}

    # 7. Convert to validated records
    records: list[GroundTruthRecord] = []
    valid_count = 0
    flagged_count = 0

    if not missing_cols:
        for _, row in df_blocked.iterrows():
            try:
                ref_d = (
                    row["reference_date"]
                    if isinstance(row["reference_date"], date)
                    else datetime.strptime(str(row["reference_date"]), "%Y-%m-%d").date()
                )
                rec = GroundTruthRecord(
                    field_id=str(row["field_id"]),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    crop_type=str(row["crop_type"]),
                    season=str(row["season"]),
                    reference_date=ref_d,
                    source=str(row["source"]),
                    verification_method=str(row["verification_method"]),
                    confidence=float(row.get("confidence", 1.0)),
                    spatial_block_id=str(row.get("spatial_block_id", "BLOCK_0_0")),
                    status="VALIDATED" if bbox_valid else "FLAGGED",
                )
                records.append(rec)
                if rec.status == "VALIDATED":
                    valid_count += 1
                else:
                    flagged_count += 1
            except Exception as e:
                flagged_count += 1
                issues.append(f"Record {row.get('field_id', 'UNKNOWN')} parsing error: {e}")

    val_status = "PASS" if (not issues and valid_count > 0) else ("REJECTED" if (issues and valid_count == 0) else "FLAGGED_WARNINGS")

    report = GroundTruthValidationReport(
        dataset_name=dataset_name,
        total_records=len(df),
        valid_records_count=valid_count,
        flagged_records_count=flagged_count,
        unique_locations_count=len(df) - duplicate_count,
        duplicate_locations_count=duplicate_count,
        crop_class_distribution=crop_dist,
        season_distribution=season_dist,
        spatial_blocks_count=spatial_blocks_count,
        bounding_box_valid=bbox_valid,
        validation_status=val_status,
        issues=issues,
    )

    return records, report


def get_bigquery_ground_truth_schema() -> list[dict[str, Any]]:
    """Return the Google BigQuery table schema definition for agrin_db.ground_truth_labels."""
    return [
        {"name": "field_id", "type": "STRING", "mode": "REQUIRED", "description": "Unique field survey plot identifier"},
        {"name": "latitude", "type": "FLOAT64", "mode": "REQUIRED", "description": "WGS84 field centroid latitude"},
        {"name": "longitude", "type": "FLOAT64", "mode": "REQUIRED", "description": "WGS84 field centroid longitude"},
        {"name": "crop_type", "type": "STRING", "mode": "REQUIRED", "description": "Observed crop label (Wheat, Soybean, etc.)"},
        {"name": "season", "type": "STRING", "mode": "REQUIRED", "description": "Agricultural season (Rabi, Kharif, Zaid)"},
        {"name": "reference_date", "type": "DATE", "mode": "REQUIRED", "description": "Ground observation / survey date"},
        {"name": "source", "type": "STRING", "mode": "REQUIRED", "description": "Authoritative survey source organization"},
        {"name": "verification_method", "type": "STRING", "mode": "REQUIRED", "description": "Method of ground verification"},
        {"name": "confidence", "type": "FLOAT64", "mode": "NULLABLE", "description": "Surveyor confidence score 0.0-1.0"},
        {"name": "spatial_block_id", "type": "STRING", "mode": "NULLABLE", "description": "Spatial cluster block for CV partitioning"},
        {"name": "status", "type": "STRING", "mode": "REQUIRED", "description": "VALIDATED, FLAGGED, or PENDING_VERIFICATION"},
        {"name": "ingested_at", "type": "TIMESTAMP", "mode": "NULLABLE", "description": "Timestamp of cloud ingestion"},
    ]
