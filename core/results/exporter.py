"""
Results exporter module (Phase 10).
Exports analysis data to standard CSV and JSON formats, and produces
comprehensive statistical summaries.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def export_to_csv(records: List[Dict[str, Any]], output_path: Union[str, Path]) -> Path:
    """
    Exports speed analysis records to CSV.
    Format:
    timestamp,ego_speed,vehicle_id,vehicle_class,vehicle_speed,relative_speed,direction,confidence
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "ego_speed",
        "vehicle_id",
        "vehicle_class",
        "vehicle_speed",
        "relative_speed",
        "direction",
        "confidence",
    ]

    with open(out_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "timestamp": round(r.get("timestamp", 0.0), 3),
                "ego_speed": r.get("ego_speed"),
                "vehicle_id": r.get("vehicle_id", ""),
                "vehicle_class": r.get("vehicle_class", ""),
                "vehicle_speed": r.get("vehicle_speed"),
                "relative_speed": r.get("relative_speed"),
                "direction": r.get("direction", ""),
                "confidence": r.get("confidence", 0.0),
            })

    return out_file


def export_to_json(data: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Exports full structured analysis report to JSON."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return out_file
