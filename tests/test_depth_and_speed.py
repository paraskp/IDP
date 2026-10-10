"""
Tests for Phases 8 and 9: Depth Estimation and Surrounding Vehicle Speed.
"""

from pathlib import Path
import pytest

from core.calibration import calibrate_scene
from core.depth.monocular_depth import MonocularDepthEstimator
from core.depth.road_plane import RoadPlaneDepthEstimator
from core.detection.vehicle_detector import Detection
from core.detection.tracker import VehicleTrack
from core.motion.ego_speed import EgoSpeedResult
from core.speed.confidence import ConfidenceEngine
from core.speed.vehicle_speed import VehicleDirection, VehicleSpeedEstimator
from core.video.reader import VideoReader


def test_depth_estimation_synthetic(sample_video_30fps: Path):
    """Verifies physical depth and lateral estimation for detected vehicle."""
    with VideoReader(sample_video_30fps) as reader:
        frame = reader.read_frame().frame

    calib = calibrate_scene([frame])
    depth_est = MonocularDepthEstimator(
        camera_model=calib.camera_model,
        road_geometry=calib.road_geometry,
        camera_height_meters=1.35,
    )

    # Lead car detection in synthetic scene
    det = Detection(
        x1=750.0, y1=380.0, x2=870.0, y2=470.0,
        confidence=0.9, class_id=2, class_name="car"
    )

    d3d = depth_est.estimate_depth(det)
    assert d3d is not None
    assert 2.0 < d3d.Z < 100.0  # Plausible longitudinal depth
    assert d3d.X > 0.0          # Car is on the right side of the ego vehicle
    assert d3d.confidence > 0.3


def test_vehicle_speed_estimation():
    """Verifies vehicle speed calculation, relative velocity, and direction classification."""
    from unittest.mock import MagicMock
    from core.depth.monocular_depth import VehicleDepth3D

    mock_depth_est = MagicMock()
    # Simulate a car moving forward ahead of us at 20 m (t=0.0), 20.5 m (t=0.1), 21.0 m (t=0.2)
    # Moving away at dZ/dt = +5.0 m/s relative to ego vehicle
    depth_sequence = [
        VehicleDepth3D(X=2.0, Y=-1.35, Z=20.0, ground_contact_z=20.0, dimensional_z=20.0, confidence=0.8, scale_source="test"),
        VehicleDepth3D(X=2.0, Y=-1.35, Z=20.5, ground_contact_z=20.5, dimensional_z=20.5, confidence=0.8, scale_source="test"),
        VehicleDepth3D(X=2.0, Y=-1.35, Z=21.0, ground_contact_z=21.0, dimensional_z=21.0, confidence=0.8, scale_source="test"),
    ]

    speed_est = VehicleSpeedEstimator(mock_depth_est, history_window_seconds=1.0)

    det = Detection(x1=700, y1=400, x2=800, y2=480, confidence=0.9, class_id=2, class_name="car")
    track = VehicleTrack(track_id=42, initial_detection=det, timestamp=0.0, frame_idx=0)

    # Ego vehicle is traveling at 15.0 m/s (54 km/h)
    ego_speed = EgoSpeedResult(
        timestamp=0.2,
        frame_index=2,
        speed_kmh=54.0,
        speed_mps=15.0,
        acceleration_mps2=0.0,
        confidence=0.85,
        scale_source="test",
        is_reliable=True,
    )

    times = [0.0, 0.1, 0.2]
    res = None
    for i, t in enumerate(times):
        mock_depth_est.estimate_depth.return_value = depth_sequence[i]
        track.update(det, timestamp=t, frame_idx=i)
        res = speed_est.estimate_vehicle_speed(track, timestamp=t, ego_speed=ego_speed)

    assert res is not None
    assert res.is_reliable is True
    # Relative speed = 5.0 m/s = 18.0 km/h
    assert pytest.approx(res.relative_speed_kmh, abs=2.0) == 18.0
    # Absolute speed = 15.0 + 5.0 = 20.0 m/s = 72.0 km/h
    assert pytest.approx(res.speed_kmh, abs=3.0) == 72.0
    assert res.direction == VehicleDirection.SAME_DIRECTION


def test_confidence_engine():
    """Verifies multi-factor reliability scoring."""
    report = ConfidenceEngine.evaluate(
        optical_flow_confidence=0.85,
        detection_confidence=0.90,
        tracker_hits=10,
        depth_confidence=0.80,
        scale_source="Road Plane Geometry",
        is_calibrated=True,
    )
    assert report.overall_confidence >= 0.70
    assert report.is_statistically_sound is True
    assert report.quality_grade in ("HIGH", "MEDIUM")
