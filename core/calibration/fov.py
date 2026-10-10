"""
Field of View (FOV) and focal length estimation module.
Strictly differentiates measured, estimated, and assumed camera intrinsics.
Never fabricates camera parameters when geometric constraints are insufficient.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import numpy as np

from core.calibration.camera import CameraIntrinsics
from core.calibration.vanishing_point import VanishingPointResult


@dataclass(frozen=True)
class FOVEstimate:
    """Result of FOV and focal length analysis."""
    fov_h_deg: Optional[float]
    fov_v_deg: Optional[float]
    fov_diag_deg: Optional[float]
    focal_length_px: Optional[float]
    confidence: float
    is_measured: bool
    scale_source: str
    reason: Optional[str] = None


def estimate_focal_from_orthogonal_vps(
    vp1: Tuple[float, float],
    vp2: Tuple[float, float],
    principal_point: Tuple[float, float],
) -> Optional[float]:
    """
    Computes focal length from two mutually orthogonal vanishing points:
    (vp1 - c)^T * (vp2 - c) = -f^2
    """
    cx, cy = principal_point
    v1x, v1y = vp1
    v2x, v2y = vp2

    val = -((v1x - cx) * (v2x - cx) + (v1y - cy) * (v2y - cy))
    if val > 0:
        return math.sqrt(val)
    return None


def calculate_fov_from_focal(
    focal_px: float,
    width: int,
    height: int,
) -> Tuple[float, float, float]:
    """
    Calculates horizontal, vertical, and diagonal FOV in degrees from focal length in pixels.
    """
    fov_h = 2.0 * math.degrees(math.atan(width / (2.0 * focal_px)))
    fov_v = 2.0 * math.degrees(math.atan(height / (2.0 * focal_px)))
    diag = math.hypot(width, height)
    fov_diag = 2.0 * math.degrees(math.atan(diag / (2.0 * focal_px)))
    return (round(fov_h, 2), round(fov_v, 2), round(fov_diag, 2))


def estimate_camera_intrinsics(
    width: int,
    height: int,
    longitudinal_vp: Optional[VanishingPointResult] = None,
    transverse_vp: Optional[VanishingPointResult] = None,
    user_fov_deg: Optional[float] = None,
) -> Tuple[CameraIntrinsics, FOVEstimate]:
    """
    Estimates camera intrinsics based strictly on available geometric constraints.

    Priority:
    1. User specified / known calibration (Highest confidence, explicit source).
    2. Two orthogonal vanishing points (Measured from scene geometry).
    3. Single vanishing point / insufficient constraints (FOV marked unknown, low confidence).
    """
    cx = width / 2.0
    cy = height / 2.0

    # Case 1: User explicitly supplied FOV
    if user_fov_deg is not None and user_fov_deg > 0:
        focal_px = (width / 2.0) / math.tan(math.radians(user_fov_deg / 2.0))
        fov_h, fov_v, fov_d = calculate_fov_from_focal(focal_px, width, height)
        fov_est = FOVEstimate(
            fov_h_deg=fov_h,
            fov_v_deg=fov_v,
            fov_diag_deg=fov_d,
            focal_length_px=focal_px,
            confidence=0.95,
            is_measured=True,
            scale_source="User Specified",
        )
        intrinsics = CameraIntrinsics(
            fx=focal_px,
            fy=focal_px,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            fov_h_deg=fov_h,
            fov_v_deg=fov_v,
            fov_diag_deg=fov_d,
            confidence=0.95,
            estimation_method="user_specified",
        )
        return intrinsics, fov_est

    # Case 2: Two orthogonal vanishing points
    if (
        longitudinal_vp is not None
        and longitudinal_vp.is_valid
        and transverse_vp is not None
        and transverse_vp.is_valid
    ):
        focal_px = estimate_focal_from_orthogonal_vps(
            (longitudinal_vp.x, longitudinal_vp.y),
            (transverse_vp.x, transverse_vp.y),
            (cx, cy),
        )
        if focal_px is not None and 100.0 < focal_px < 5000.0:
            fov_h, fov_v, fov_d = calculate_fov_from_focal(focal_px, width, height)
            conf = min(longitudinal_vp.confidence, transverse_vp.confidence) * 0.85
            fov_est = FOVEstimate(
                fov_h_deg=fov_h,
                fov_v_deg=fov_v,
                fov_diag_deg=fov_d,
                focal_length_px=focal_px,
                confidence=conf,
                is_measured=True,
                scale_source="Orthogonal Vanishing Points",
            )
            intrinsics = CameraIntrinsics(
                fx=focal_px,
                fy=focal_px,
                cx=cx,
                cy=cy,
                width=width,
                height=height,
                fov_h_deg=fov_h,
                fov_v_deg=fov_v,
                fov_diag_deg=fov_d,
                confidence=conf,
                estimation_method="orthogonal_vanishing_points",
            )
            return intrinsics, fov_est

    # Case 3: Insufficient constraints (Scientifically honest reporting)
    # A nominal fallback focal length is recorded internally for geometric ray projection,
    # but the FOV status is explicitly marked as UNKNOWN with low confidence.
    nominal_focal_px = (width / 2.0) / math.tan(math.radians(55.0))  # ~110 deg nominal prior
    fov_est = FOVEstimate(
        fov_h_deg=None,
        fov_v_deg=None,
        fov_diag_deg=None,
        focal_length_px=None,
        confidence=0.15,
        is_measured=False,
        scale_source="None",
        reason="insufficient geometric constraints (requires orthogonal vanishing points or manual calibration)",
    )
    intrinsics = CameraIntrinsics(
        fx=nominal_focal_px,
        fy=nominal_focal_px,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        fov_h_deg=None,
        fov_v_deg=None,
        fov_diag_deg=None,
        confidence=0.15,
        estimation_method="uncalibrated_nominal_prior",
    )
    return intrinsics, fov_est
