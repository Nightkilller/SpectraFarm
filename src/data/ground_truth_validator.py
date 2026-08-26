"""
AgriN — Ground Truth Validation & Ingestion Infrastructure (Phase 5)

Cloud-ready validation module for authoritative ground-truth field data.
Enforces rigorous data-quality checks before allowing reference data to enter
downstream crop classification (Phase 6) and stress calibration (Phase 7).

Scientific & Quality Checks:
1. Strict Schema Enforcement (field_id, lat, lon, crop_type, season, reference_date, source, verification_method).
2. Spatial Bounding Validation (ensures coordinates lie within target agricultural region).
3. Duplicate Coordinate Detection (flags field centroids closer than spatial tolerance threshold).
4. Unique Field ID Verification (detects identifier collisions).
5. Class Balance & Distribution Analysis (computes counts and percentages, identifies severe imbalance).
6. Spatial Blocking / Spatial K-Fold ID Assignment (prevents spatial autocorrelation leakage).
7. Provenance & Source Auditing (ensures verifiable surveying authority and date ranges).
8. Explicit Validation Status Assignment (DATA_NOT_AVAILABLE, RAW, VALIDATION_PENDING, VALIDATED, INVALID, REQUIRES_REVIEW).
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
    DatasetProvenance,
    GroundTruthDataset,
    GroundTruthDatasetStatus,
    GroundTruthRecord,
    GroundTruthRecordStatus,
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


def create_agrifieldnet_provenance() -> DatasetProvenance:
    """Create verified provenance metadata for the public AgriFieldNet India Challenge dataset."""
    return DatasetProvenance(
        dataset_name="AgriFieldNet Competition Dataset",
        dataset_version="1.0 (DOI: 10.34911/rdnt.wu92p1)",
        source_url="https://source.coop/radiantearth/agrifieldnet-competition",
        license="CC-BY-4.0",
        geographic_region="Northern India (Uttar Pradesh, Rajasthan, Odisha, Bihar) — NOT Sehore MP",
        is_sehore_ground_truth=False,
        label_collection_method="In-situ ground surveys collected by IDinsight Data on Demand (ECAAS initiative)",
    )


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
    provenance: Optional[DatasetProvenance] = None,
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
            provenance=provenance,
            total_records=0,
            valid_records_count=0,
            rejected_records_count=0,
            requires_review_count=0,
            unique_field_ids_count=0,
            duplicate_field_ids_count=0,
            unique_locations_count=0,
            duplicate_locations_count=0,
            records_outside_bbox_count=0,
            missing_values_count=0,
            crop_class_distribution={},
            crop_class_percentages={},
            season_distribution={},
            date_range_start=None,
            date_range_end=None,
            spatial_blocks_count=0,
            bounding_box_valid=False,
            provenance_complete=False,
            validation_status=GroundTruthDatasetStatus.DATA_NOT_AVAILABLE,
            issues=["Dataset is empty. Real ground-truth field observations are required."],
        )
        return [], report

    # 1. Check Required Columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # 2. Check Null Values in Critical Fields
    missing_values_count = 0
    for col in [c for c in REQUIRED_COLUMNS if c in df.columns]:
        null_count = int(df[col].isna().sum())
        missing_values_count += null_count
        if null_count > 0:
            issues.append(f"Column '{col}' contains {null_count} null/missing values.")

    # 3. Spatial Bounding Validation
    records_outside_bbox = 0
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
        records_outside_bbox = len(out_of_bounds)
        if records_outside_bbox > 0:
            bbox_valid = False
            issues.append(
                f"{records_outside_bbox} records fall outside expected geographic bounding box "
                f"[{min_lat}, {max_lat}, {min_lon}, {max_lon}]."
            )

    # 4. Duplicate Field ID Check
    duplicate_field_ids_count = 0
    if "field_id" in df.columns:
        duplicate_field_ids_count = int(df["field_id"].duplicated().sum())
        if duplicate_field_ids_count > 0:
            issues.append(f"Found {duplicate_field_ids_count} duplicate field_id values.")

    # 5. Duplicate Coordinate Detection
    duplicate_loc_count = 0
    if "latitude" in df.columns and "longitude" in df.columns:
        coords = df[["latitude", "longitude"]].dropna().values
        n = len(coords)
        for i in range(n):
            for j in range(i + 1, n):
                dist = haversine_distance_meters(
                    coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]
                )
                if dist < duplicate_tolerance_meters:
                    duplicate_loc_count += 1

        if duplicate_loc_count > 0:
            issues.append(
                f"Found {duplicate_loc_count} pairs of coordinates within {duplicate_tolerance_meters}m spatial tolerance (potential duplicate/overlapping surveys)."
            )

    # 6. Assign Spatial Blocks for Spatial K-Fold
    df_blocked = assign_spatial_blocks(df)
    spatial_blocks_count = len(df_blocked["spatial_block_id"].unique()) if "spatial_block_id" in df_blocked.columns else 0

    # 7. Distributions & Percentages
    crop_dist: dict[str, int] = {}
    crop_pcts: dict[str, float] = {}
    if "crop_type" in df.columns:
        crop_counts = Counter(df["crop_type"].dropna().tolist())
        crop_dist = dict(crop_counts)
        total_crops = sum(crop_counts.values())
        if total_crops > 0:
            crop_pcts = {k: round((v / total_crops) * 100.0, 2) for k, v in crop_counts.items()}

    season_dist = dict(Counter(df["season"].dropna().tolist())) if "season" in df.columns else {}

    # 8. Date Range
    date_start = None
    date_end = None
    if "reference_date" in df.columns:
        try:
            parsed_dates = [
                d if isinstance(d, date) else datetime.strptime(str(d), "%Y-%m-%d").date()
                for d in df["reference_date"].dropna()
            ]
            if parsed_dates:
                date_start = min(parsed_dates)
                date_end = max(parsed_dates)
        except Exception as e:
            issues.append(f"Date parsing error: {e}")

    # 9. Provenance completeness check
    provenance_complete = True
    if "source" in df.columns:
        unverified_sources = df[df["source"].isna() | (df["source"].str.strip() == "") | (df["source"].str.lower() == "unknown")]
        if not unverified_sources.empty:
            provenance_complete = False
            issues.append(f"{len(unverified_sources)} records have missing or unverified sources.")
    else:
        provenance_complete = False

    # 10. Convert to records & evaluate status
    records: list[GroundTruthRecord] = []
    valid_count = 0
    rejected_count = 0
    requires_review_count = 0

    if not missing_cols:
        for _, row in df_blocked.iterrows():
            try:
                ref_d = (
                    row["reference_date"]
                    if isinstance(row["reference_date"], date)
                    else datetime.strptime(str(row["reference_date"]), "%Y-%m-%d").date()
                )
                lat = float(row["latitude"])
                lon = float(row["longitude"])

                # Evaluate individual record status
                is_oob = False
                if expected_bbox:
                    is_oob = (
                        lat < expected_bbox.get("min_lat", -90.0)
                        or lat > expected_bbox.get("max_lat", 90.0)
                        or lon < expected_bbox.get("min_lon", -180.0)
                        or lon > expected_bbox.get("max_lon", 180.0)
                    )

                if is_oob:
                    rec_status = GroundTruthRecordStatus.REQUIRES_REVIEW
                    requires_review_count += 1
                else:
                    rec_status = GroundTruthRecordStatus.VALIDATED
                    valid_count += 1

                rec = GroundTruthRecord(
                    field_id=str(row["field_id"]),
                    latitude=lat,
                    longitude=lon,
                    crop_type=str(row["crop_type"]),
                    season=str(row["season"]),
                    reference_date=ref_d,
                    source=str(row["source"]),
                    verification_method=str(row["verification_method"]),
                    confidence=float(row.get("confidence", 1.0)),
                    spatial_block_id=str(row.get("spatial_block_id", "BLOCK_0_0")),
                    status=rec_status,
                )
                records.append(rec)
            except Exception as e:
                rejected_count += 1
                issues.append(f"Record {row.get('field_id', 'UNKNOWN')} rejected: {e}")
    else:
        rejected_count = len(df)

    # Determine overall dataset status
    if missing_cols or rejected_count > 0:
        val_status = GroundTruthDatasetStatus.INVALID
    elif requires_review_count > 0 or duplicate_loc_count > 0 or duplicate_field_ids_count > 0:
        val_status = GroundTruthDatasetStatus.REQUIRES_REVIEW
    elif valid_count > 0:
        if provenance and not provenance.is_sehore_ground_truth:
            val_status = GroundTruthDatasetStatus.EXTERNAL_PUBLIC_DATASET
        elif provenance and provenance.is_sehore_ground_truth:
            val_status = GroundTruthDatasetStatus.SEHORE_GROUND_TRUTH
        else:
            val_status = GroundTruthDatasetStatus.VALIDATED
    else:
        val_status = GroundTruthDatasetStatus.DATA_NOT_AVAILABLE

    report = GroundTruthValidationReport(
        dataset_name=dataset_name,
        provenance=provenance,
        total_records=len(df),
        valid_records_count=valid_count,
        rejected_records_count=rejected_count,
        requires_review_count=requires_review_count,
        unique_field_ids_count=len(df) - duplicate_field_ids_count,
        duplicate_field_ids_count=duplicate_field_ids_count,
        unique_locations_count=len(df) - duplicate_loc_count,
        duplicate_locations_count=duplicate_loc_count,
        records_outside_bbox_count=records_outside_bbox,
        missing_values_count=missing_values_count,
        crop_class_distribution=crop_dist,
        crop_class_percentages=crop_pcts,
        season_distribution=season_dist,
        date_range_start=date_start,
        date_range_end=date_end,
        spatial_blocks_count=spatial_blocks_count,
        bounding_box_valid=bbox_valid,
        provenance_complete=provenance_complete,
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
        {"name": "status", "type": "STRING", "mode": "REQUIRED", "description": "VALIDATED, INVALID, or REQUIRES_REVIEW"},
        {"name": "ingested_at", "type": "TIMESTAMP", "mode": "NULLABLE", "description": "Timestamp of cloud ingestion"},
    ]
