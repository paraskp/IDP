"""
Camera intrinsics, extrinsics, and geometric models.
Handles projection, unprojection, and coordinate transformations between
2D image plane and 3D camera/world coordinates.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    Pinhole camera intrinsic parameters.
    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    fov_h_deg: Optional[float] = None
    fov_v_deg: Optional[float] = None
    fov_diag_deg: Optional[float] = None
    confidence: float = 0.0
    estimation_method: str = "unknown"

    @property
    def matrix(self) -> np.ndarray:
        """Returns 3x3 intrinsic matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def inv_matrix(self) -> np.ndarray:
        """Returns 3x3 inverse intrinsic matrix K^-1."""
        return np.linalg.inv(self.matrix)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "fov_h_deg": self.fov_h_deg,
            "fov_v_deg": self.fov_v_deg,
            "fov_diag_deg": self.fov_diag_deg,
            "confidence": self.confidence,
            "estimation_method": self.estimation_method,
        }


@dataclass(frozen=True)
class CameraExtrinsics:
    """
    Camera orientation relative to the road ground plane.
    Pitch: camera tilt up/down relative to horizontal ground.
    Roll: camera rotation around optical axis.
    Yaw: lateral camera azimuth offset.
    Camera height: height above ground plane in meters (if calibrated).
    """
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    yaw_rad: float = 0.0
    height_meters: Optional[float] = None
    confidence: float = 0.0
    estimation_method: str = "unknown"

    @property
    def pitch_deg(self) -> float:
        return math.degrees(self.pitch_rad)

    @property
    def roll_deg(self) -> float:
        return math.degrees(self.roll_rad)

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw_rad)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        3x3 rotation matrix R from world (ground) coordinates to camera coordinates.
        Applying R_roll * R_pitch * R_yaw.
        """
        cp, sp = math.cos(self.pitch_rad), math.sin(self.pitch_rad)
        cr, sr = math.cos(self.roll_rad), math.sin(self.roll_rad)
        cy, sy = math.cos(self.yaw_rad), math.sin(self.yaw_rad)

        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        return Rz @ Rx @ Ry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "yaw_deg": self.yaw_deg,
            "height_meters": self.height_meters,
            "confidence": self.confidence,
            "estimation_method": self.estimation_method,
        }


class CameraModel:
    """
    Combines camera intrinsics and extrinsics to project 3D points
    to image coordinates and project 2D image coordinates to ground plane (IPM).
    """

    def __init__(self, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def project_3d_to_pixel(self, pt3d_camera: np.ndarray) -> Tuple[float, float]:
        """Projects a 3D point in camera frame [X, Y, Z] to [u, v] pixel coordinate."""
        if pt3d_camera[2] <= 0:
            return float("nan"), float("nan")
        p = self.intrinsics.matrix @ pt3d_camera
        return float(p[0] / p[2]), float(p[1] / p[2])

    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """
        Converts pixel coordinate [u, v] into a normalized 3D ray in camera frame.
        """
        homog = np.array([u, v, 1.0], dtype=np.float64)
        ray = self.intrinsics.inv_matrix @ homog
        norm = np.linalg.norm(ray)
        return ray / norm if norm > 0 else ray
