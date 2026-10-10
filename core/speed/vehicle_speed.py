"""
Vehicle speed and trajectory estimation module (Phase 9).
Combines ego motion, vehicle tracking, and 3D depth geometry to estimate
relative and absolute speeds, direction, and trajectory.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.depth.monocular_depth import MonocularDepthEstimator, VehicleDepth3D
from core.detection.tracker import VehicleTrack
from core.motion.ego_speed import EgoSpeedResult


class VehicleDirection(Enum):
    SAME_DIRECTION = "same_direction"
    OPPOSITE_DIRECTION = "opposite_direction"
    CROSSING = "crossing"
    STATIONARY = "stationary"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class VehicleSpeedResult:
    """Estimated physical speed, direction, and confidence for a tracked vehicle."""
    track_id: int
    class_name: str
    speed_kmh: Optional[float]
    speed_mps: Optional[float]
    relative_speed_kmh: Optional[float]
    relative_speed_mps: Optional[float]
    direction: VehicleDirection
    depth_meters: Optional[float]
    lateral_meters: Optional[float]
    confidence: float
    scale_source: str
    estimation_method: str
    is_reliable: bool
    reason: Optional[str] = None


class VehicleSpeedEstimator:
    """
    Computes absolute and relative speeds of surrounding vehicles by integrating
    depth trajectories over time and fusing them with ego-vehicle velocity.
    """

    def __init__(
        self,
        depth_estimator: MonocularDepthEstimator,
        history_window_seconds: float = 0.5,
    ):
        self.depth_estimator = depth_estimator
        self.window_sec = history_window_seconds
        # track_id -> List of (timestamp, X, Z)
        self._depth_histories: Dict[int, List[Tuple[float, float, float]]] = {}

    def estimate_vehicle_speed(
        self,
        track: VehicleTrack,
        timestamp: float,
        ego_speed: Optional[EgoSpeedResult] = None,
    ) -> VehicleSpeedResult:
        """
        Estimates the physical speed and trajectory direction of a single tracked vehicle.
        """
        if not track.trajectory:
            return self._empty_result(track, "no trajectory points available")

        # Get latest detection from track
        latest_pt = track.trajectory[-1]
        x1, y1, x2, y2 = latest_pt.bbox

        from core.detection.vehicle_detector import Detection
        dummy_det = Detection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=track.confidence,
            class_id=0,
            class_name=track.class_name,
        )

        depth_3d = self.depth_estimator.estimate_depth(dummy_det)
        if depth_3d is None:
            return self._empty_result(track, "depth estimation failed or point above horizon")

        tid = track.track_id
        if tid not in self._depth_histories:
            self._depth_histories[tid] = []

        history = self._depth_histories[tid]
        history.append((timestamp, depth_3d.X, depth_3d.Z))

        # Keep only points within history window
        cutoff = timestamp - self.window_sec
        history = [h for h in history if h[0] >= cutoff]
        self._depth_histories[tid] = history

        # Require at least 3 historical observations across > 0.15s
        if len(history) < 3 or (history[-1][0] - history[0][0]) < 0.15:
            return VehicleSpeedResult(
                track_id=tid,
                class_name=track.class_name,
                speed_kmh=None,
                speed_mps=None,
                relative_speed_kmh=None,
                relative_speed_mps=None,
                direction=VehicleDirection.UNKNOWN,
                depth_meters=depth_3d.Z,
                lateral_meters=depth_3d.X,
                confidence=round(depth_3d.confidence * 0.4, 2),
                scale_source=depth_3d.scale_source,
                estimation_method="depth_trajectory_differencing",
                is_reliable=False,
                reason="accumulating trajectory history (need >= 3 observations)",
            )

        # Compute relative velocities via linear regression over time window
        times = np.array([h[0] for h in history])
        xs = np.array([h[1] for h in history])
        zs = np.array([h[2] for h in history])

        t_mean = np.mean(times)
        dt_var = np.sum((times - t_mean) ** 2)
        if dt_var < 1e-4:
            return self._empty_result(track, "insufficient temporal variance")

        # Relative velocity: dZ/dt, dX/dt
        v_rel_z = float(np.sum((times - t_mean) * (zs - np.mean(zs))) / dt_var)
        v_rel_x = float(np.sum((times - t_mean) * (xs - np.mean(xs))) / dt_var)

        v_ego_mps = (
            ego_speed.speed_mps
            if (ego_speed and ego_speed.is_reliable and ego_speed.speed_mps is not None)
            else 0.0
        )

        # Absolute forward velocity = relative forward velocity + ego forward velocity
        v_abs_z = v_rel_z + v_ego_mps
        v_abs_x = v_rel_x

        abs_speed_mps = math.hypot(v_abs_x, v_abs_z)
        abs_speed_kmh = abs_speed_mps * 3.6
        rel_speed_mps = math.hypot(v_rel_x, v_rel_z)
        rel_speed_kmh = rel_speed_mps * 3.6

        # Classify direction
        if abs_speed_kmh < 4.0:
            direction = VehicleDirection.STATIONARY
        elif abs(v_abs_x) > 1.5 * abs(v_abs_z):
            direction = VehicleDirection.CROSSING
        elif v_abs_z > 0:
            direction = VehicleDirection.SAME_DIRECTION
        else:
            direction = VehicleDirection.OPPOSITE_DIRECTION

        # Confidence assessment
        track_factor = min(1.0, track.hits / 10.0)
        conf = min(0.92, (depth_3d.confidence * 0.5) + (track_factor * 0.3) + (0.2 if ego_speed and ego_speed.is_reliable else 0.0))

        # Check physical feasibility (< 220 km/h)
        if abs_speed_kmh > 220.0:
            return VehicleSpeedResult(
                track_id=tid,
                class_name=track.class_name,
                speed_kmh=None,
                speed_mps=None,
                relative_speed_kmh=None,
                relative_speed_mps=None,
                direction=direction,
                depth_meters=depth_3d.Z,
                lateral_meters=depth_3d.X,
                confidence=0.2,
                scale_source=depth_3d.scale_source,
                estimation_method="depth_trajectory_differencing",
                is_reliable=False,
                reason="speed estimate exceeded physical limits (>220 km/h), likely depth noise",
            )

        return VehicleSpeedResult(
            track_id=tid,
            class_name=track.class_name,
            speed_kmh=round(abs_speed_kmh, 1),
            speed_mps=round(abs_speed_mps, 2),
            relative_speed_kmh=round(rel_speed_kmh, 1),
            relative_speed_mps=round(rel_speed_mps, 2),
            direction=direction,
            depth_meters=round(depth_3d.Z, 2),
            lateral_meters=round(depth_3d.X, 2),
            confidence=round(conf, 2),
            scale_source=depth_3d.scale_source,
            estimation_method="fused_depth_and_ego_odometry",
            is_reliable=conf >= 0.4,
            reason=None,
        )

    def _empty_result(self, track: VehicleTrack, reason: str) -> VehicleSpeedResult:
        return VehicleSpeedResult(
            track_id=track.track_id,
            class_name=track.class_name,
            speed_kmh=None,
            speed_mps=None,
            relative_speed_kmh=None,
            relative_speed_mps=None,
            direction=VehicleDirection.UNKNOWN,
            depth_meters=None,
            lateral_meters=None,
            confidence=0.1,
            scale_source="None",
            estimation_method="none",
            is_reliable=False,
            reason=reason,
        )
