"""
Ego vehicle speed estimation module (Phase 5).
Resolves monocular metric scale using road plane geometry and IPM ground mapping.
Applies Kalman filtering for physically stable speed and acceleration tracking.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import numpy as np

from core.calibration.camera import CameraModel
from core.calibration.road_geometry import RoadGeometry
from core.motion.optical_flow import TrackingFrameResult
from core.motion.road_grid import GridMotionResult, RoadGrid
from core.motion.visual_odometry import PoseEstimate


@dataclass(frozen=True)
class EgoSpeedResult:
    """Estimated ego vehicle speed at a specific frame/timestamp."""
    timestamp: float
    frame_index: int
    speed_kmh: Optional[float]
    speed_mps: Optional[float]
    acceleration_mps2: Optional[float]
    confidence: float
    scale_source: str
    is_reliable: bool
    squares_passed_per_second: Optional[float] = None
    squares_passed_per_frame: Optional[float] = None
    grid_square_size_m: float = 1.0
    reason: Optional[str] = None


class EgoSpeedKalmanFilter:
    """1D Kalman filter for vehicle longitudinal motion (position, velocity, acceleration)."""

    def __init__(self, process_noise: float = 0.5, measurement_noise: float = 2.0):
        # State: [position_m, velocity_mps, acceleration_mps2]
        self.x = np.zeros(3, dtype=np.float64)
        self.P = np.eye(3, dtype=np.float64) * 10.0
        self.Q_base = process_noise
        self.R = measurement_noise
        self.initialized = False

    def update(self, dt: float, measured_speed_mps: Optional[float]) -> Tuple[float, float]:
        """
        Updates the Kalman filter with a new speed measurement.
        Returns: (filtered_speed_mps, acceleration_mps2).
        """
        if dt <= 0:
            return float(self.x[1]), float(self.x[2])

        # State transition matrix F
        F = np.array([
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Process noise covariance Q
        q = self.Q_base
        Q = np.array([
            [0.25 * dt**4, 0.5 * dt**3, 0.5 * dt**2],
            [0.5 * dt**3, dt**2, dt],
            [0.5 * dt**2, dt, 1.0],
        ], dtype=np.float64) * q

        # Predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Prevent negative speed for forward driving
        if self.x[1] < 0:
            self.x[1] = 0.0

        if measured_speed_mps is not None and not math.isnan(measured_speed_mps):
            if not self.initialized:
                self.x[1] = measured_speed_mps
                self.x[2] = 0.0
                self.initialized = True
            else:
                # Direct velocity measurement update
                z = float(measured_speed_mps)
                y = z - float(self.x[1])
                S = float(self.P[1, 1]) + self.R
                K = self.P[:, 1] / S
                self.x = self.x + K * y
                self.P = self.P - np.outer(K, self.P[1, :])

        vel = max(0.0, float(self.x[1]))
        acc = float(self.x[2])
        return vel, acc


class EgoSpeedEstimator:
    """
    Estimates forward velocity of the ego vehicle using road-plane homography
    and optical flow integration.
    """

    def __init__(
        self,
        camera_model: CameraModel,
        road_geometry: RoadGeometry,
        camera_height_meters: float = 1.35,  # Nominal standard car mount height
        square_size_meters: float = 1.0,     # Imaginary road grid square size in meters
        is_user_calibrated_scale: bool = False,
    ):
        self.camera_model = camera_model
        self.road_geom = road_geometry
        self.camera_height = camera_height_meters
        self.square_size = square_size_meters
        self.is_user_calibrated = is_user_calibrated_scale
        self.filter = EgoSpeedKalmanFilter()
        self.last_timestamp: Optional[float] = None
        self.road_grid = RoadGrid(
            camera_model=camera_model,
            road_geometry=road_geometry,
            camera_height_meters=camera_height_meters,
            square_size_meters=square_size_meters,
        )

    def estimate_speed(
        self,
        frame_idx: int,
        timestamp: float,
        tracking_result: TrackingFrameResult,
        pose_estimate: Optional[PoseEstimate] = None,
    ) -> EgoSpeedResult:
        """
        Estimates metric ego speed for the current frame using the imaginary road grid.
        """
        dt = (timestamp - self.last_timestamp) if self.last_timestamp is not None else (1.0 / 30.0)
        self.last_timestamp = timestamp

        inliers = tracking_result.inlier_matches

        # Calculate motion across the imaginary road grid
        prev_pts = np.array([m.pt_prev for m in inliers], dtype=np.float32) if inliers else np.empty((0, 2), dtype=np.float32)
        curr_pts = np.array([m.pt_curr for m in inliers], dtype=np.float32) if inliers else np.empty((0, 2), dtype=np.float32)

        grid_res = self.road_grid.calculate_squares_passed(prev_pts, curr_pts, dt)

        raw_speed_mps: Optional[float] = None
        squares_frame: Optional[float] = None
        squares_sec: Optional[float] = None

        if grid_res.is_valid:
            raw_speed_mps = grid_res.raw_speed_mps
            squares_frame = grid_res.squares_passed_per_frame
            squares_sec = grid_res.squares_passed_per_second
        elif inliers:
            # Fallback point-by-point calculation if grid criteria wasn't fully met
            h_cam = self.camera_height
            fy = self.camera_model.intrinsics.fy
            vy = self.road_geom.vanishing_point[1]
            metric_speeds: List[float] = []
            for m in inliers:
                x0, y0 = m.pt_prev
                x1, y1 = m.pt_curr
                if y0 <= vy + 15 or y1 <= vy + 15:
                    continue
                z0 = (fy * h_cam) / (y0 - vy)
                z1 = (fy * h_cam) / (y1 - vy)
                dz = z0 - z1
                if 0 < dz < (120.0 * dt):
                    metric_speeds.append(dz / dt)
            if len(metric_speeds) >= 3:
                raw_speed_mps = float(np.median(metric_speeds))
                squares_sec = round(raw_speed_mps / self.square_size, 2)
                squares_frame = round((raw_speed_mps * dt) / self.square_size, 3)

        # Filter with Kalman model
        filtered_mps, acc = self.filter.update(dt, raw_speed_mps)
        filtered_kmh = filtered_mps * 3.6

        # Assess scale validity and confidence
        if not self.road_geom.is_valid:
            return EgoSpeedResult(
                timestamp=timestamp,
                frame_index=frame_idx,
                speed_kmh=None,
                speed_mps=None,
                acceleration_mps2=None,
                confidence=0.15,
                scale_source="None (uncalibrated scene)",
                is_reliable=False,
                squares_passed_per_second=None,
                squares_passed_per_frame=None,
                grid_square_size_m=self.square_size,
                reason="insufficient road plane constraints to recover metric scale",
            )

        scale_source = (
            "User Calibrated Ground Plane"
            if self.is_user_calibrated
            else f"Road Grid (Nominal Cam Height: {self.camera_height:.2f}m, Sq: {self.square_size:.1f}m)"
        )

        base_conf = (
            grid_res.confidence * 0.7 + tracking_result.confidence * 0.3
            if grid_res.is_valid
            else tracking_result.confidence * 0.5
        )
        if self.is_user_calibrated:
            base_conf = min(0.95, base_conf + 0.15)
        else:
            base_conf = min(0.88, base_conf)

        is_reliable = (grid_res.is_valid or raw_speed_mps is not None) and base_conf >= 0.35

        return EgoSpeedResult(
            timestamp=timestamp,
            frame_index=frame_idx,
            speed_kmh=round(filtered_kmh, 1) if is_reliable else None,
            speed_mps=round(filtered_mps, 2) if is_reliable else None,
            acceleration_mps2=round(acc, 2) if is_reliable else None,
            confidence=round(base_conf, 3),
            scale_source=scale_source,
            is_reliable=is_reliable,
            squares_passed_per_second=squares_sec if is_reliable else None,
            squares_passed_per_frame=squares_frame if is_reliable else None,
            grid_square_size_m=self.square_size,
            reason=None if is_reliable else "sparse road grid feature tracking",
        )
