"""
Tests for Phases 3, 4, and 5: Stationary Feature Tracking, Visual Odometry, and Ego Speed.
"""

from pathlib import Path
import numpy as np
import pytest

from core.calibration import calibrate_scene
from core.motion.feature_detection import StationaryFeatureDetector
from core.motion.optical_flow import SparseOpticalFlowTracker
from core.motion.visual_odometry import VisualOdometry
from core.motion.ego_speed import EgoSpeedEstimator, EgoSpeedKalmanFilter
from core.video.reader import VideoReader


def test_stationary_feature_detector(sample_video_30fps: Path):
    """Verifies feature detection and dynamic box exclusion."""
    with VideoReader(sample_video_30fps) as reader:
        frame = reader.read_frame().frame

    detector = StationaryFeatureDetector(max_corners=100)
    # Detect without exclusions
    pts = detector.detect(frame)
    assert len(pts) > 10

    # Exclude a large box covering the road center
    h, w = frame.shape[:2]
    blocked_box = [(int(w * 0.2), int(h * 0.4), int(w * 0.8), int(h * 0.9))]
    pts_blocked = detector.detect(frame, dynamic_boxes=blocked_box)

    # All returned points must be outside the blocked box
    for p in pts_blocked:
        x, y = p[0, 0], p[0, 1]
        in_box = (blocked_box[0][0] <= x <= blocked_box[0][2]) and (blocked_box[0][1] <= y <= blocked_box[0][3])
        assert not in_box


def test_sparse_optical_flow_tracking(sample_video_30fps: Path):
    """Verifies bidirectional optical flow tracking across two consecutive frames."""
    with VideoReader(sample_video_30fps) as reader:
        f0 = reader.read_frame().frame
        f1 = reader.read_frame().frame

    gray0 = np.mean(f0, axis=2).astype(np.uint8)
    gray1 = np.mean(f1, axis=2).astype(np.uint8)

    detector = StationaryFeatureDetector(max_corners=150)
    pts0 = detector.detect(f0)

    tracker = SparseOpticalFlowTracker()
    result = tracker.track(gray0, gray1, pts0)

    assert result.confidence > 0.0
    assert len(result.inlier_matches) > 0
    assert result.tracking_ratio > 0.0

    # Test debug visualization drawing
    debug_img = tracker.draw_debug_flow(f1, result)
    assert debug_img.shape == f1.shape


def test_ego_speed_kalman_filter():
    """Verifies Kalman filter stability and convergence."""
    kf = EgoSpeedKalmanFilter()
    dt = 1.0 / 30.0

    # Constant 16.67 m/s (60 km/h) input with simulated noise
    speeds = [16.67 + np.sin(i) * 1.5 for i in range(30)]
    filtered_speeds = []
    for s in speeds:
        v, a = kf.update(dt, s)
        filtered_speeds.append(v)

    # Filtered speed should converge near 16.67 m/s
    assert pytest.approx(filtered_speeds[-1], abs=1.5) == 16.67


def test_ego_speed_estimator_on_synthetic_stream(sample_video_30fps: Path):
    """Verifies end-to-end ego speed estimation on synthetic video."""
    with VideoReader(sample_video_30fps) as reader:
        frames = [reader.read_frame().frame for _ in range(10)]

    calib = calibrate_scene(frames[:5])
    ego_estimator = EgoSpeedEstimator(
        camera_model=calib.camera_model,
        road_geometry=calib.road_geometry,
        camera_height_meters=1.35,
    )

    detector = StationaryFeatureDetector(max_corners=200)
    tracker = SparseOpticalFlowTracker()

    prev_gray = np.mean(frames[0], axis=2).astype(np.uint8)
    results = []

    for i in range(1, len(frames)):
        curr_gray = np.mean(frames[i], axis=2).astype(np.uint8)
        pts = detector.detect(frames[i - 1])
        track_res = tracker.track(prev_gray, curr_gray, pts)
        speed_res = ego_estimator.estimate_speed(
            frame_idx=i,
            timestamp=i / 30.0,
            tracking_result=track_res,
        )
        results.append(speed_res)
        prev_gray = curr_gray

    assert len(results) == 9
    # Results must include valid timestamp and metadata
    for r in results:
        assert r.timestamp > 0
        assert r.confidence >= 0.0
        assert len(r.scale_source) > 0
