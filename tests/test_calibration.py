"""
Tests for Phase 2: Automatic Camera and Scene Calibration.
"""

import math
from pathlib import Path
import numpy as np
import pytest

from core.calibration.camera import CameraExtrinsics, CameraIntrinsics, CameraModel
from core.calibration.distortion import DistortionModel
from core.calibration.fov import calculate_fov_from_focal, estimate_camera_intrinsics
from core.calibration.road_geometry import (
    create_road_roi_mask,
    estimate_road_geometry,
)
from core.calibration.vanishing_point import (
    LineSegment,
    VanishingPointResult,
    compute_intersection,
    detect_line_segments,
    estimate_vanishing_point_ransac,
)
from core.calibration import calibrate_scene
from core.video.reader import VideoReader


def test_line_segment_properties():
    """Verifies LineSegment length, angle, and homogeneous line."""
    seg = LineSegment(0.0, 0.0, 10.0, 10.0)
    assert pytest.approx(seg.length, rel=1e-3) == math.sqrt(200)
    assert pytest.approx(seg.angle_deg, abs=0.1) == 45.0
    assert seg.midpoint == (5.0, 5.0)

    homo = seg.to_homogeneous_line()
    assert len(homo) == 3
    # Any point on the line dot product with line coefficients should be ~0
    p = np.array([5.0, 5.0, 1.0])
    assert pytest.approx(float(p @ homo), abs=1e-5) == 0.0


def test_line_intersection():
    """Verifies 2D intersection of two non-parallel line segments."""
    l1 = LineSegment(0.0, 0.0, 10.0, 10.0)
    l2 = LineSegment(0.0, 10.0, 10.0, 0.0)
    pt = compute_intersection(l1, l2)
    assert pt is not None
    assert pytest.approx(pt[0], abs=1e-3) == 5.0
    assert pytest.approx(pt[1], abs=1e-3) == 5.0


def test_camera_intrinsics_projection():
    """Verifies CameraModel 3D point projection and ray casting."""
    intrinsics = CameraIntrinsics(
        fx=1000.0,
        fy=1000.0,
        cx=640.0,
        cy=360.0,
        width=1280,
        height=720,
        confidence=1.0,
    )
    extrinsics = CameraExtrinsics(pitch_rad=0.0, roll_rad=0.0, yaw_rad=0.0)
    model = CameraModel(intrinsics, extrinsics)

    # Point straight ahead at Z = 10m
    pt3d = np.array([0.0, 0.0, 10.0])
    u, v = model.project_3d_to_pixel(pt3d)
    assert pytest.approx(u, abs=1e-3) == 640.0
    assert pytest.approx(v, abs=1e-3) == 360.0

    # Ray cast from principal point should point along optical axis [0, 0, 1]
    ray = model.pixel_to_ray(640.0, 360.0)
    assert pytest.approx(ray[0], abs=1e-3) == 0.0
    assert pytest.approx(ray[1], abs=1e-3) == 0.0
    assert pytest.approx(ray[2], abs=1e-3) == 1.0


def test_fov_honest_reporting_unknown():
    """Verifies that without geometric constraints, FOV is marked unknown with low confidence."""
    intrinsics, fov_est = estimate_camera_intrinsics(width=1280, height=720)
    assert fov_est.fov_h_deg is None
    assert fov_est.is_measured is False
    assert fov_est.confidence < 0.25
    assert "insufficient geometric constraints" in fov_est.reason.lower()


def test_fov_user_calibrated():
    """Verifies user specified FOV takes precedence with high confidence."""
    intrinsics, fov_est = estimate_camera_intrinsics(
        width=1280, height=720, user_fov_deg=110.0
    )
    assert fov_est.fov_h_deg == 110.0
    assert fov_est.confidence > 0.9
    assert fov_est.is_measured is True
    assert fov_est.scale_source == "User Specified"


def test_road_roi_mask():
    """Verifies binary road ROI mask generation."""
    mask = create_road_roi_mask((720, 1280), (640.0, 324.0))
    assert mask.shape == (720, 1280)
    assert mask.dtype == np.uint8
    # Bottom center should be inside mask (255)
    assert mask[710, 640] == 255
    # Sky region should be outside mask (0)
    assert mask[100, 640] == 0


def test_scene_calibration_on_synthetic_video(sample_video_30fps: Path):
    """Verifies automatic scene calibration on the synthetic dashcam video."""
    with VideoReader(sample_video_30fps) as reader:
        frames = [reader.read_frame().frame for _ in range(5)]

    calib = calibrate_scene(frames)
    assert calib is not None
    # Vanishing point should be detected near synthetic road convergence (~640, ~324)
    assert calib.vanishing_point.is_valid is True
    assert 550 < calib.vanishing_point.x < 730
    assert 250 < calib.vanishing_point.y < 420
    assert calib.road_geometry.is_valid is True
    assert calib.confidence > 0.3
    assert len(calib.summary()) > 0
