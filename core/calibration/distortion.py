"""
Lens distortion modeling and correction module.
Supports radial (k1, k2, k3) and tangential (p1, p2) lens distortion models.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from core.calibration.camera import CameraIntrinsics


@dataclass(frozen=True)
class DistortionModel:
    """Lens distortion coefficients for OpenCV camera model."""
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    is_calibrated: bool = False

    @property
    def coefficients(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    def undistort_image(self, image: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
        """Undistorts an image using camera matrix K and distortion coefficients."""
        if not self.is_calibrated or (self.k1 == 0.0 and self.k2 == 0.0):
            return image
        return cv2.undistort(image, intrinsics.matrix, self.coefficients)

    def undistort_points(self, points_2d: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
        """
        Undistorts an array of 2D pixel coordinates of shape (N, 2).
        """
        if not self.is_calibrated or (self.k1 == 0.0 and self.k2 == 0.0):
            return points_2d
        pts = points_2d.reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.undistortPoints(pts, intrinsics.matrix, self.coefficients, P=intrinsics.matrix)
        return undistorted.reshape(-1, 2)
