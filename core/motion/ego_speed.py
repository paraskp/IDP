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
        is_user_calibrated_scale: bool = False,
    ):
        self.camera_model = camera_model
        self.road_geom = road_geometry
        self.camera_height = camera_height_meters
        self.is_user_calibrated = is_user_calibrated_scale
        self.filter = EgoSpeedKalmanFilter()
        self.last_timestamp: Optional[float] = None

    def estimate_speed(
        self,
        frame_idx: int,
        timestamp: float,
        tracking_result: TrackingFrameResult,
        pose_estimate: Optional[PoseEstimate] = None,
    ) -> EgoSpeedResult:
        """
        Estimates metric ego speed for the current frame.
        """
        dt = (timestamp - self.last_timestamp) if self.last_timestamp is not None else (1.0 / 30.0)
        self.last_timestamp = timestamp

        inliers = tracking_result.inlier_matches
        h_cam = self.camera_height
        fy = self.camera_model.intrinsics.fy
        vy = self.road_geom.vanishing_point[1]

        # Calculate metric physical displacements of road features
        metric_speeds: List[float] = []

        for m in inliers:
            x0, y0 = m.pt_prev
            x1, y1 = m.pt_curr

            # Only consider features in lower road region below horizon
            if y0 <= vy + 15 or y1 <= vy + 15:
                continue

            # Ground plane depth: Z = (fy * h_cam) / (y - vy)
            z0 = (fy * h_cam) / (y0 - vy)
            z1 = (fy * h_cam) / (y1 - vy)

            dz = z0 - z1  # Forward displacement in meters
            if 0 < dz < (120.0 * dt):  # Plausible physical displacement (<120 m/s)
                speed_sample = dz / dt
                metric_speeds.append(speed_sample)

        raw_speed_mps: Optional[float] = None
        if len(metric_speeds) >= 3:
            raw_speed_mps = float(np.median(metric_speeds))

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
                reason="insufficient road plane constraints to recover metric scale",
            )

        scale_source = (
            "User Calibrated Ground Plane"
            if self.is_user_calibrated
            else f"Road Geometry (Nominal Cam Height: {self.camera_height:.2f}m)"
        )

        base_conf = tracking_result.confidence * 0.5 + (0.4 if len(metric_speeds) >= 5 else 0.2)
        if self.is_user_calibrated:
            base_conf = min(0.95, base_conf + 0.15)
        else:
            base_conf = min(0.85, base_conf)

        is_reliable = len(metric_speeds) >= 3 and base_conf >= 0.4

        return EgoSpeedResult(
            timestamp=timestamp,
            frame_index=frame_idx,
            speed_kmh=round(filtered_kmh, 1) if is_reliable else None,
            speed_mps=round(filtered_mps, 2) if is_reliable else None,
            acceleration_mps2=round(acc, 2) if is_reliable else None,
            confidence=round(base_conf, 3),
            scale_source=scale_source,
            is_reliable=is_reliable,
            reason=None if is_reliable else "sparse road feature tracking",
        )
