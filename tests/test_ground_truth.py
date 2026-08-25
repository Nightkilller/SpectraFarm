"""
Unit tests for AgriN Ground Truth Validation & Ingestion Module (Phase 5).
"""

from datetime import date
import pandas as pd
import pytest

from src.data.ground_truth_validator import (
    assign_spatial_blocks,
    get_bigquery_ground_truth_schema,
    haversine_distance_meters,
    validate_ground_truth_dataframe,
)
from src.data.schemas import GroundTruthRecord


class TestGroundTruthValidation:
    def test_haversine_distance(self):
        # Coordinates ~111 meters apart in latitude (0.001 deg ~ 111m)
        dist = haversine_distance_meters(23.2000, 77.0800, 23.2010, 77.0800)
        assert 100.0 < dist < 120.0

    def test_valid_ground_truth_dataframe(self):
        df = pd.DataFrame([
            {
                "field_id": "SEH_001",
                "latitude": 23.2010,
                "longitude": 77.0810,
                "crop_type": "Wheat",
                "season": "Rabi",
                "reference_date": "2026-03-10",
                "source": "Field Survey - KVK Sehore",
                "verification_method": "GPS Survey",
                "confidence": 1.0,
            },
            {
                "field_id": "SEH_002",
                "latitude": 23.2050,
                "longitude": 77.0850,
                "crop_type": "Mustard",
                "season": "Rabi",
                "reference_date": "2026-03-10",
                "source": "Field Survey - KVK Sehore",
                "verification_method": "GPS Survey",
                "confidence": 0.95,
            },
        ])

        bbox = {"min_lat": 23.0, "max_lat": 23.4, "min_lon": 76.8, "max_lon": 77.3}
        records, report = validate_ground_truth_dataframe(df, dataset_name="Test Valid", expected_bbox=bbox)

        assert report.validation_status == "PASS"
        assert report.valid_records_count == 2
        assert report.flagged_records_count == 0
        assert report.duplicate_locations_count == 0
        assert report.crop_class_distribution == {"Wheat": 1, "Mustard": 1}
        assert len(records) == 2
        assert records[0].crop_type == "Wheat"
        assert records[0].status == "VALIDATED"

    def test_missing_required_column_rejected(self):
        # Missing 'crop_type'
        df = pd.DataFrame([
            {
                "field_id": "SEH_001",
                "latitude": 23.2010,
                "longitude": 77.0810,
                "season": "Rabi",
                "reference_date": "2026-03-10",
                "source": "KVK",
                "verification_method": "GPS",
            }
        ])

        records, report = validate_ground_truth_dataframe(df, dataset_name="Test Missing Col")
        assert report.validation_status == "REJECTED"
        assert any("crop_type" in issue for issue in report.issues)

    def test_out_of_bounds_coordinate_flagged(self):
        df = pd.DataFrame([
            {
                "field_id": "OUT_001",
                "latitude": 15.0000,  # Far outside Sehore (23.2°N)
                "longitude": 77.0810,
                "crop_type": "Wheat",
                "season": "Rabi",
                "reference_date": "2026-03-10",
                "source": "KVK",
                "verification_method": "GPS",
            }
        ])

        bbox = {"min_lat": 23.0, "max_lat": 23.4, "min_lon": 76.8, "max_lon": 77.3}
        records, report = validate_ground_truth_dataframe(df, dataset_name="Test OOB", expected_bbox=bbox)

        assert report.bounding_box_valid is False
        assert any("bounding box" in issue for issue in report.issues)

    def test_duplicate_coordinate_detection(self):
        # Two records with virtually identical coordinates (1 meter apart)
        df = pd.DataFrame([
            {
                "field_id": "DUP_001",
                "latitude": 23.200000,
                "longitude": 77.080000,
                "crop_type": "Wheat",
                "season": "Rabi",
                "reference_date": "2026-03-10",
                "source": "KVK",
                "verification_method": "GPS",
            },
            {
                "field_id": "DUP_002",
                "latitude": 23.200005,  # ~0.5 meter delta
                "longitude": 77.080000,
                "crop_type": "Wheat",
                "season": "Rabi",
                "reference_date": "2026-03-10",
                "source": "KVK",
                "verification_method": "GPS",
            },
        ])

        records, report = validate_ground_truth_dataframe(df, dataset_name="Test DUP", duplicate_tolerance_meters=15.0)
        assert report.duplicate_locations_count == 1
        assert any("spatial tolerance" in issue for issue in report.issues)

    def test_spatial_block_assignment(self):
        df = pd.DataFrame([
            {"latitude": 23.2010, "longitude": 77.0810},
            {"latitude": 23.2450, "longitude": 77.0810},
        ])
        blocked = assign_spatial_blocks(df, grid_size_deg=0.02)
        assert "spatial_block_id" in blocked.columns
        # The two points separated by >0.04 deg lat should have different block IDs
        assert blocked.iloc[0]["spatial_block_id"] != blocked.iloc[1]["spatial_block_id"]

    def test_empty_dataframe_waiting_status(self):
        empty_df = pd.DataFrame()
        records, report = validate_ground_truth_dataframe(empty_df, dataset_name="Empty")
        assert report.validation_status == "WAITING_FOR_DATA"
        assert report.total_records == 0
        assert len(records) == 0

    def test_bigquery_schema_definition(self):
        schema = get_bigquery_ground_truth_schema()
        assert isinstance(schema, list)
        assert len(schema) >= 8
        col_names = [col["name"] for col in schema]
        assert "field_id" in col_names
        assert "latitude" in col_names
        assert "longitude" in col_names
        assert "crop_type" in col_names
        assert "spatial_block_id" in col_names
