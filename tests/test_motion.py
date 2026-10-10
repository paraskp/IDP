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


def test_road_grid_projection_and_squares_passed(sample_video_30fps: Path):
    """Verifies RoadGrid metric projection and squares-per-unit-time speed derivation."""
    from core.motion.road_grid import RoadGrid

    with VideoReader(sample_video_30fps) as reader:
        frames = [reader.read_frame().frame for _ in range(5)]

    calib = calibrate_scene(frames)
    grid = RoadGrid(
        camera_model=calib.camera_model,
        road_geometry=calib.road_geometry,
        camera_height_meters=1.35,
        square_size_meters=1.0,
    )

    # 1. Test Invertibility between ground and image plane
    x_test, z_test = 0.0, 15.0  # Center lane, 15 meters ahead
    pix = grid.ground_to_pixel(x_test, z_test)
    assert pix is not None
    u, v = pix

    ground_back = grid.project_pixel_to_ground(u, v)
    assert ground_back is not None
    assert pytest.approx(ground_back[0], abs=0.05) == x_test
    assert pytest.approx(ground_back[1], abs=0.05) == z_test

    # 2. Test squares traversed calculation for vehicle traveling at 20 m/s (72 km/h)
    dt = 0.1  # 100 ms
    speed_target_mps = 20.0
    dz = speed_target_mps * dt  # 2.0 meters forward displacement = 2.0 grid squares

    # Generate synthetic road points moving towards vehicle
    pts_prev = []
    pts_curr = []
    for x_m in [-1.5, -0.5, 0.5, 1.5]:
        for z_m in [10.0, 15.0, 20.0, 25.0]:
            p0 = grid.ground_to_pixel(x_m, z_m)
            p1 = grid.ground_to_pixel(x_m, z_m - dz)
            if p0 is not None and p1 is not None:
                pts_prev.append(p0)
                pts_curr.append(p1)

    pts_prev_arr = np.array(pts_prev, dtype=np.float32)
    pts_curr_arr = np.array(pts_curr, dtype=np.float32)

    motion_res = grid.calculate_squares_passed(pts_prev_arr, pts_curr_arr, dt=dt)
    assert motion_res.is_valid
    assert pytest.approx(motion_res.squares_passed_per_frame, abs=0.1) == 2.0
    assert pytest.approx(motion_res.squares_passed_per_second, abs=1.0) == 20.0
    assert pytest.approx(motion_res.raw_speed_mps, abs=1.0) == 20.0
    assert pytest.approx(motion_res.raw_speed_kmh, abs=3.6) == 72.0


def test_road_grid_overlay_and_mask(sample_video_30fps: Path):
    """Verifies RoadGrid overlay generation and drivable road mask creation."""
    from core.motion.road_grid import RoadGrid

    with VideoReader(sample_video_30fps) as reader:
        frames = [reader.read_frame().frame for _ in range(5)]

    calib = calibrate_scene(frames)
    grid = RoadGrid(
        camera_model=calib.camera_model,
        road_geometry=calib.road_geometry,
        camera_height_meters=1.35,
        square_size_meters=1.0,
    )

    overlay_data = grid.generate_grid_overlay((720, 1280))
    assert len(overlay_data["longitudinal_lines"]) > 0
    assert len(overlay_data["transversal_lines"]) > 0
    assert len(overlay_data["grid_cells"]) > 0

    mask = grid.create_road_grid_mask((720, 1280))
    assert mask.shape == (720, 1280)
    assert np.count_nonzero(mask) > 1000

