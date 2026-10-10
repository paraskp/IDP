"""
Tests for Phase 10: Results Export and Summary Statistics.
"""

import json
from pathlib import Path
import pytest

from core.results.exporter import export_to_csv, export_to_json
from core.results.statistics import compute_summary_statistics


def test_export_to_csv(tmp_path: Path):
    """Verifies CSV generation format and schema."""
    csv_file = tmp_path / "test_out.csv"
    records = [
        {
            "timestamp": 1.250,
            "ego_speed": 62.5,
            "vehicle_id": 7,
            "vehicle_class": "car",
            "vehicle_speed": 68.2,
            "relative_speed": 5.7,
            "direction": "same_direction",
            "confidence": 0.84,
        }
    ]

    export_to_csv(records, csv_file)
    assert csv_file.exists()

    content = csv_file.read_text(encoding="utf-8")
    assert "timestamp,ego_speed,vehicle_id,vehicle_class" in content
    assert "68.2" in content
    assert "same_direction" in content


def test_export_to_json(tmp_path: Path):
    """Verifies JSON generation and valid schema."""
    json_file = tmp_path / "test_out.json"
    data = {
        "status": "success",
        "ego_average_speed": 58.4,
        "vehicles_detected": 3,
    }

    export_to_json(data, json_file)
    assert json_file.exists()

    with open(json_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["ego_average_speed"] == 58.4


def test_summary_statistics():
    """Verifies summary statistics calculation."""
    ego_records = [{"speed_kmh": 60.0}, {"speed_kmh": 80.0}]
    veh_records = [
        {"vehicle_id": 1, "vehicle_speed": 70.0, "confidence": 0.8},
        {"vehicle_id": 2, "vehicle_speed": 90.0, "confidence": 0.9},
    ]

    stats = compute_summary_statistics(duration_sec=5.0, ego_speed_records=ego_records, vehicle_records=veh_records)
    assert stats["duration_seconds"] == 5.0
    assert stats["total_unique_vehicles"] == 2
    assert stats["ego_vehicle"]["average_speed_kmh"] == 70.0
    assert stats["ego_vehicle"]["max_speed_kmh"] == 80.0
    assert stats["surrounding_vehicles"]["average_speed_kmh"] == 80.0
    assert stats["confidence"]["mean_confidence"] == 0.85
