"""
Tests for Phases 6 and 7: Vehicle Detection and Multi-Object Tracking.
"""

from pathlib import Path
import pytest

from core.detection.vehicle_detector import Detection, SyntheticVehicleDetector
from core.detection.tracker import ByteVehicleTracker, TrackState, compute_iou
from core.video.reader import VideoReader


def test_iou_computation():
    """Verifies IoU computation across various geometric box overlaps."""
    # Identical boxes -> IoU = 1.0
    b1 = (10.0, 10.0, 50.0, 50.0)
    assert pytest.approx(compute_iou(b1, b1), abs=1e-4) == 1.0

    # Non-overlapping disjoint boxes -> IoU = 0.0
    b2 = (100.0, 100.0, 150.0, 150.0)
    assert compute_iou(b1, b2) == 0.0

    # 50% width overlap
    b3 = (30.0, 10.0, 70.0, 50.0)
    # Area1 = 40*40=1600, Area3 = 40*40=1600, Intersection = 20*40=800, Union = 3200-800=2400 -> IoU = 800/2400 = 0.3333
    assert pytest.approx(compute_iou(b1, b3), abs=1e-3) == (1.0 / 3.0)


def test_synthetic_vehicle_detector(sample_video_30fps: Path):
    """Verifies SyntheticVehicleDetector detects the lead vehicle in synthetic footage."""
    with VideoReader(sample_video_30fps) as reader:
        frame = reader.read_frame().frame

    detector = SyntheticVehicleDetector()
    dets = detector.detect(frame)

    assert len(dets) >= 1
    car_det = dets[0]
    assert isinstance(car_det, Detection)
    assert car_det.class_name == "car"
    assert car_det.confidence > 0.8
    # Lead car in synthetic scene is positioned in the right half of the image
    assert car_det.center[0] > 600
    assert car_det.center[1] > 300


def test_byte_vehicle_tracker_persistence(sample_video_30fps: Path):
    """Verifies persistent vehicle ID tracking across consecutive frames."""
    with VideoReader(sample_video_30fps) as reader:
        frames = [reader.read_frame().frame for _ in range(8)]

    detector = SyntheticVehicleDetector()
    tracker = ByteVehicleTracker(min_hits_to_confirm=2)

    assigned_ids = []
    for i, frame in enumerate(frames):
        dets = detector.detect(frame)
        active_tracks = tracker.update(dets, timestamp=i / 30.0, frame_idx=i)
        if active_tracks:
            assigned_ids.append(active_tracks[0].track_id)

    # After initial confirmation (frame 2+), the tracked vehicle ID must remain constant
    assert len(assigned_ids) >= 5
    first_id = assigned_ids[0]
    for tid in assigned_ids:
        assert tid == first_id  # Track ID persists without flickering!
