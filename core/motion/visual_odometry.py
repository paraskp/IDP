"""
Monocular visual odometry module (Phase 4).
Estimates relative camera rotation (R) and translation direction (t)
from stationary feature correspondences using Essential matrix decomposition.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from core.calibration.camera import CameraIntrinsics
from core.motion.optical_flow import TrackingFrameResult


@dataclass(frozen=True)
class PoseEstimate:
    """Estimated relative camera pose between consecutive frames."""
    rotation_matrix: np.ndarray      # 3x3 R
    translation_vector: np.ndarray   # 3x1 unit translation direction t
    pitch_change_rad: float
    yaw_change_rad: float
    roll_change_rad: float
    inlier_count: int
    inlier_ratio: float
    confidence: float
    is_valid: bool


class VisualOdometry:
    """
    Computes camera egomotion between frame pairs using the Essential Matrix.
    """

    def __init__(self, intrinsics: CameraIntrinsics, min_inliers: int = 15):
        self.intrinsics = intrinsics
        self.min_inliers = min_inliers

    def estimate_motion(
        self,
        tracking_result: TrackingFrameResult,
    ) -> PoseEstimate:
        """
        Estimates relative camera motion from inlier matches.

        Args:
            tracking_result: Result from optical flow tracking.

        Returns:
            PoseEstimate containing R, t, Euler angle changes, and confidence.
        """
        inliers = tracking_result.inlier_matches
        if len(inliers) < self.min_inliers:
            return self._fallback_pose(len(inliers), 0.0, "insufficient inlier matches")

        pts_prev = np.array([m.pt_prev for m in inliers], dtype=np.float64)
        pts_curr = np.array([m.pt_curr for m in inliers], dtype=np.float64)

        K = self.intrinsics.matrix

        # Compute Essential Matrix using RANSAC
        E, mask = cv2.findEssentialMat(
            pts_curr,
            pts_prev,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is None or E.shape != (3, 3):
            return self._fallback_pose(len(inliers), 0.0, "essential matrix estimation failed")

        # Recover pose [R, t] from Essential Matrix
        num_inliers, R, t, pose_mask = cv2.recoverPose(
            E, pts_curr, pts_prev, K, mask=mask
        )

        if num_inliers < self.min_inliers:
            return self._fallback_pose(num_inliers, 0.0, "too few cheirality inliers")

        # Extract Euler angle deltas (pitch, yaw, roll) from rotation matrix
        sy = np.hypot(R[0, 0], R[1, 0])
        singular = sy < 1e-6
        if not singular:
            pitch = float(np.arctan2(R[2, 1], R[2, 2]))
            yaw = float(np.arctan2(-R[2, 0], sy))
            roll = float(np.arctan2(R[1, 0], R[0, 0]))
        else:
            pitch = float(np.arctan2(-R[1, 2], R[1, 1]))
            yaw = float(np.arctan2(-R[2, 0], sy))
            roll = 0.0

        inlier_ratio = num_inliers / len(inliers)
        conf = min(1.0, (inlier_ratio * 0.6) + min(0.4, num_inliers / 60.0))

        return PoseEstimate(
            rotation_matrix=R,
            translation_vector=t,
            pitch_change_rad=pitch,
            yaw_change_rad=yaw,
            roll_change_rad=roll,
            inlier_count=num_inliers,
            inlier_ratio=inlier_ratio,
            confidence=round(conf, 4),
            is_valid=True,
        )

    def _fallback_pose(self, count: int, conf: float, reason: str) -> PoseEstimate:
        return PoseEstimate(
            rotation_matrix=np.eye(3, dtype=np.float64),
            translation_vector=np.array([[0.0], [0.0], [1.0]], dtype=np.float64),
            pitch_change_rad=0.0,
            yaw_change_rad=0.0,
            roll_change_rad=0.0,
            inlier_count=count,
            inlier_ratio=0.0,
            confidence=conf,
            is_valid=False,
        )
