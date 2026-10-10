"""
Road geometry and camera orientation estimation module.
Estimates camera pitch, roll, horizon line, and ground plane orientation
from the estimated vanishing point and road boundaries.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import cv2
import numpy as np

from core.calibration.camera import CameraExtrinsics, CameraIntrinsics
from core.calibration.vanishing_point import VanishingPointResult


@dataclass(frozen=True)
class RoadGeometry:
    """Represents the estimated road plane and horizon geometry."""
    horizon_y: float
    pitch_rad: float
    roll_rad: float
    ground_normal: np.ndarray  # 3D unit normal vector in camera frame
    vanishing_point: Tuple[float, float]
    confidence: float
    is_valid: bool

    @property
    def pitch_deg(self) -> float:
        return math.degrees(self.pitch_rad)

    @property
    def roll_deg(self) -> float:
        return math.degrees(self.roll_rad)


def estimate_road_geometry(
    vanishing_point: VanishingPointResult,
    intrinsics: CameraIntrinsics,
) -> RoadGeometry:
    """
    Computes road horizon, camera pitch, roll, and ground normal vector from the vanishing point.

    Args:
        vanishing_point: Detected longitudinal vanishing point.
        intrinsics: Camera intrinsics model.

    Returns:
        RoadGeometry dataclass.
    """
    vx, vy = vanishing_point.x, vanishing_point.y
    cx, cy = intrinsics.cx, intrinsics.cy
    fx, fy = intrinsics.fx, intrinsics.fy

    # Camera pitch: angle between optical axis and ground plane
    # If VP is below cy, camera is tilted downwards (positive pitch in our coordinate system)
    pitch_rad = math.atan2(vy - cy, fy)

    # Camera roll: lateral tilt
    roll_rad = -math.atan2(vx - cx, fx)

    # In camera coordinates, ground normal is roughly [0, -1, 0] when level.
    # Applying rotation for pitch:
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)

    # Normal vector perpendicular to the road plane
    nx = -sr * cp
    ny = -cr * cp
    nz = sp
    normal = np.array([nx, ny, nz], dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal /= norm

    conf = vanishing_point.confidence if vanishing_point.is_valid else 0.1

    return RoadGeometry(
        horizon_y=vy,
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
        ground_normal=normal,
        vanishing_point=(vx, vy),
        confidence=conf,
        is_valid=vanishing_point.is_valid,
    )


def create_road_roi_mask(
    image_shape: Tuple[int, int],
    vanishing_point: Tuple[float, float],
    top_width_ratio: float = 0.15,
    bottom_width_ratio: float = 0.95,
) -> np.ndarray:
    """
    Creates a binary mask (uint8, 255 inside road, 0 outside) representing
    the trapezoidal drivable road corridor from the vanishing point to image bottom.
    """
    h, w = image_shape[:2]
    vx, vy = vanishing_point

    top_y = int(max(0, min(h - 1, vy + 10)))
    top_half_w = int(w * top_width_ratio * 0.5)
    bottom_half_w = int(w * bottom_width_ratio * 0.5)
    center_x = int(max(0, min(w - 1, vx)))

    pts = np.array(
        [
            [max(0, center_x - top_half_w), top_y],
            [min(w - 1, center_x + top_half_w), top_y],
            [min(w - 1, center_x + bottom_half_w), h - 1],
            [max(0, center_x - bottom_half_w), h - 1],
        ],
        dtype=np.int32,
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask
