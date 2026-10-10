"""
Vehicle detection and tracking subsystem (Phases 6 and 7).
"""

from core.detection.vehicle_detector import (
    COCO_VEHICLE_CLASSES,
    BaseVehicleDetector,
    Detection,
    SyntheticVehicleDetector,
    YOLOVehicleDetector,
)
from core.detection.tracker import (
    ByteVehicleTracker,
    TrackState,
    TrajectoryPoint,
    VehicleTrack,
    compute_iou,
)

__all__ = [
    "COCO_VEHICLE_CLASSES",
    "BaseVehicleDetector",
    "Detection",
    "SyntheticVehicleDetector",
    "YOLOVehicleDetector",
    "ByteVehicleTracker",
    "TrackState",
    "TrajectoryPoint",
    "VehicleTrack",
    "compute_iou",
]
