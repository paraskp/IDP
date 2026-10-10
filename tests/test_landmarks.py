"""
Tests for Stationary Road Landmark & Marking Detection and Speed Anchoring.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from core.detection.landmark_detector import (
    ClassicalRoadMarkingDetector,
    HybridLandmarkDetector,
    RoadLandmark,
    YOLOLandmarkDetector,
)
from core.motion.landmark_speed import LandmarkSpeedTracker
from core.motion.optical_flow import FlowMatch
from core.video.reader import VideoReader


def test_classical_road_marking_detector(sample_video_30fps: Path):
    """Verifies that classical detector detects road markings on test video frames."""
    with VideoReader(sample_video_30fps) as reader:
        frame = reader.read_frame().frame

    detector = ClassicalRoadMarkingDetector()
    landmarks = detector.detect(frame)

    assert isinstance(landmarks, list)
    for lm in landmarks:
        assert isinstance(lm, RoadLandmark)
        assert lm.landmark_id > 0
        assert lm.class_name in ["lane_dash", "road_marking", "pole", "curb"]
        x1, y1, x2, y2 = lm.bbox
        assert x1 < x2 and y1 < y2
        assert 0.0 <= lm.confidence <= 1.0
        assert len(lm.ground_contact_pt) == 2


def test_hybrid_landmark_detector_fallback(sample_video_30fps: Path):
    """Verifies that HybridLandmarkDetector falls back to classical detector when model is absent."""
    with VideoReader(sample_video_30fps) as reader:
        frame = reader.read_frame().frame

    hybrid = HybridLandmarkDetector(yolo_model_path="models/non_existent_weights.pt")
    assert not hybrid.is_yolo_active

    landmarks = hybrid.detect(frame)
    assert isinstance(landmarks, list)


def test_landmark_feature_filtering():
    """Verifies filtering optical flow matches strictly against detected stationary landmarks."""
    tracker = LandmarkSpeedTracker()

    # Create dummy landmarks
    lm1 = RoadLandmark(
        landmark_id=1,
        class_name="lane_dash",
        bbox=(100.0, 200.0, 120.0, 250.0),
        confidence=0.9,
        ground_contact_pt=(110.0, 250.0),
    )
    lm2 = RoadLandmark(
        landmark_id=2,
        class_name="road_marking",
        bbox=(300.0, 200.0, 350.0, 220.0),
        confidence=0.85,
        ground_contact_pt=(325.0, 220.0),
    )
    landmarks = [lm1, lm2]

    # Create matches: some inside landmarks, some outside on the road/sky
    inside_matches = [
        FlowMatch(pt_prev=(105.0, 210.0), pt_curr=(105.0, 215.0), displacement=(0, 5), magnitude=5.0),
        FlowMatch(pt_prev=(115.0, 240.0), pt_curr=(115.0, 245.0), displacement=(0, 5), magnitude=5.0),
        FlowMatch(pt_prev=(310.0, 205.0), pt_curr=(310.0, 210.0), displacement=(0, 5), magnitude=5.0),
        FlowMatch(pt_prev=(340.0, 215.0), pt_curr=(340.0, 220.0), displacement=(0, 5), magnitude=5.0),
    ]
    outside_matches = [
        FlowMatch(pt_prev=(50.0, 50.0), pt_curr=(50.0, 55.0), displacement=(0, 5), magnitude=5.0),  # sky/vehicle
        FlowMatch(pt_prev=(500.0, 500.0), pt_curr=(500.0, 505.0), displacement=(0, 5), magnitude=5.0),
    ]

    all_matches = inside_matches + outside_matches
    filtered = tracker.filter_features_to_landmarks(all_matches, landmarks)

    # Filtered matches should only contain the 4 inside matches
    assert len(filtered) == 4
    for m in filtered:
        assert m in inside_matches


def test_dash_passage_frequency_speed():
    """Verifies that regular passage of lane dashes computes valid frequency speed."""
    # Standard 3.0m dash + 4.5m gap = 7.5m cycle
    tracker = LandmarkSpeedTracker(standard_dash_length_m=3.0, standard_dash_gap_m=4.5)

    # Simulate 5 dashes passing at 2.0 Hz (period = 0.5s)
    # Expected speed = 2.0 Hz * 7.5m = 15.0 m/s = 54.0 km/h
    dt = 0.1
    estimates = []
    dash_lm = [
        RoadLandmark(
            landmark_id=1,
            class_name="lane_dash",
            bbox=(200.0, 300.0, 220.0, 350.0),
            confidence=0.9,
            ground_contact_pt=(210.0, 350.0),
        )
    ]

    # Feed frames over 2.0 seconds
    for step in range(21):
        t = step * dt
        # Dash appears every 0.5 seconds (steps 0, 5, 10, 15, 20)
        curr_lms = dash_lm if (step % 5 == 0) else []
        est = tracker.update_dash_frequency(curr_lms, timestamp=t, dt=dt)
        estimates.append(est)

    last_est = estimates[-1]
    assert last_est.dash_frequency_hz is not None
    assert pytest.approx(last_est.dash_frequency_hz, abs=0.2) == 2.0
    assert last_est.frequency_speed_kmh is not None
    assert pytest.approx(last_est.frequency_speed_kmh, abs=5.0) == 54.0
