"""
Automatic camera and scene calibration subsystem (Phase 2).
Performs vanishing point estimation, road geometry recovery, FOV estimation,
and builds structured calibration models without fabricating unknown parameters.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from core.calibration.camera import CameraExtrinsics, CameraIntrinsics, CameraModel
from core.calibration.distortion import DistortionModel
from core.calibration.fov import (
    FOVEstimate,
    calculate_fov_from_focal,
    estimate_camera_intrinsics,
)
from core.calibration.road_geometry import (
    RoadGeometry,
    create_road_roi_mask,
    estimate_road_geometry,
)
from core.calibration.vanishing_point import (
    LineSegment,
    VanishingPointResult,
    detect_line_segments,
    estimate_vanishing_point_ransac,
)


@dataclass(frozen=True)
class CalibrationResult:
    """Consolidated result of camera and scene calibration."""
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    distortion: DistortionModel
    vanishing_point: VanishingPointResult
    road_geometry: RoadGeometry
    fov_estimate: FOVEstimate
    confidence: float
    is_metric_scale_available: bool
    scale_source: str
    camera_model: CameraModel

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intrinsics": self.intrinsics.to_dict(),
            "extrinsics": self.extrinsics.to_dict(),
            "vanishing_point": {
                "x": self.vanishing_point.x,
                "y": self.vanishing_point.y,
                "confidence": self.vanishing_point.confidence,
                "is_valid": self.vanishing_point.is_valid,
            },
            "road_geometry": {
                "horizon_y": self.road_geometry.horizon_y,
                "pitch_deg": self.road_geometry.pitch_deg,
                "roll_deg": self.road_geometry.roll_deg,
                "confidence": self.road_geometry.confidence,
            },
            "fov": {
                "fov_h_deg": self.fov_estimate.fov_h_deg,
                "confidence": self.fov_estimate.confidence,
                "is_measured": self.fov_estimate.is_measured,
                "scale_source": self.scale_source,
                "reason": self.fov_estimate.reason,
            },
            "overall_confidence": self.confidence,
            "metric_scale_available": self.is_metric_scale_available,
        }

    def summary(self) -> str:
        """Returns a human-readable summary of calibration findings."""
        lines = [
            "--- CAMERA & SCENE CALIBRATION ---",
            f"Vanishing Point:  ({self.vanishing_point.x:.1f}, {self.vanishing_point.y:.1f}) [Conf: {self.vanishing_point.confidence * 100:.1f}%]",
            f"Camera Pitch:     {self.road_geometry.pitch_deg:.2f}°",
            f"Camera Roll:      {self.road_geometry.roll_deg:.2f}°",
        ]
        if self.fov_estimate.fov_h_deg is not None:
            lines.append(f"Estimated FOV (H): {self.fov_estimate.fov_h_deg:.1f}° [Conf: {self.fov_estimate.confidence * 100:.1f}%]")
        else:
            lines.append(f"Estimated FOV:    Unknown (Reason: {self.fov_estimate.reason})")
        lines.append(f"Metric Scale:     {'Available' if self.is_metric_scale_available else 'Unavailable / Insufficient constraints'}")
        lines.append(f"Scale Source:     {self.scale_source}")
        lines.append(f"Overall Conf:     {self.confidence * 100:.1f}%")
        return "\n".join(lines)


def calibrate_scene(
    sample_frames: List[np.ndarray],
    user_fov_deg: Optional[float] = None,
    user_camera_height: Optional[float] = None,
) -> CalibrationResult:
    """
    Performs automatic scene calibration across one or more representative sample frames.

    Args:
        sample_frames: List of BGR image frames (typically first 10-30 frames).
        user_fov_deg: Optional user-specified FOV in degrees (manual calibration mode).
        user_camera_height: Optional known camera mount height above ground (meters).

    Returns:
        Structured CalibrationResult.
    """
    if not sample_frames:
        raise ValueError("calibrate_scene requires at least one sample frame.")

    h, w = sample_frames[0].shape[:2]

    # Accumulate line segments across sample frames to find stable vanishing point
    all_lines: List[LineSegment] = []
    for frame in sample_frames:
        lines = detect_line_segments(frame)
        all_lines.extend(lines)

    # Estimate longitudinal vanishing point
    vp = estimate_vanishing_point_ransac(all_lines, (h, w))

    # Estimate camera intrinsics and FOV
    intrinsics, fov_est = estimate_camera_intrinsics(
        width=w,
        height=h,
        longitudinal_vp=vp,
        user_fov_deg=user_fov_deg,
    )

    # Estimate road geometry & camera orientation
    road_geom = estimate_road_geometry(vp, intrinsics)

    # Build extrinsics
    extrinsics = CameraExtrinsics(
        pitch_rad=road_geom.pitch_rad,
        roll_rad=road_geom.roll_rad,
        yaw_rad=0.0,
        height_meters=user_camera_height,
        confidence=road_geom.confidence,
        estimation_method="vanishing_point_horizon",
    )

    distortion = DistortionModel(is_calibrated=False)
    camera_model = CameraModel(intrinsics, extrinsics)

    # Determine metric scale availability
    scale_available = False
    scale_source = "None"
    if user_camera_height is not None and fov_est.fov_h_deg is not None:
        scale_available = True
        scale_source = "Known Camera Height + FOV"
    elif fov_est.is_measured:
        scale_available = True
        scale_source = fov_est.scale_source
    elif road_geom.is_valid:
        scale_source = "Road Geometry (Relative/Geometric only)"
    else:
        scale_source = "None (insufficient constraints)"

    overall_conf = round(vp.confidence * 0.5 + road_geom.confidence * 0.3 + fov_est.confidence * 0.2, 4)

    return CalibrationResult(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        distortion=distortion,
        vanishing_point=vp,
        road_geometry=road_geom,
        fov_estimate=fov_est,
        confidence=overall_conf,
        is_metric_scale_available=scale_available,
        scale_source=scale_source,
        camera_model=camera_model,
    )


__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "CameraModel",
    "DistortionModel",
    "FOVEstimate",
    "calculate_fov_from_focal",
    "estimate_camera_intrinsics",
    "RoadGeometry",
    "create_road_roi_mask",
    "estimate_road_geometry",
    "LineSegment",
    "VanishingPointResult",
    "detect_line_segments",
    "estimate_vanishing_point_ransac",
    "CalibrationResult",
    "calibrate_scene",
]
