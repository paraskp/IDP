"""
Road plane geometry and Inverse Perspective Mapping (IPM) depth estimation (Phase 8).
Computes physical 3D ground coordinates from 2D pixel contact points.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import numpy as np

from core.calibration.camera import CameraModel
from core.calibration.road_geometry import RoadGeometry


@dataclass(frozen=True)
class GroundPoint3D:
    """3D point on the road surface in camera coordinate frame (meters)."""
    X: float  # Lateral (positive right)
    Y: float  # Vertical (negative down to ground)
    Z: float  # Longitudinal depth (positive forward)
    confidence: float
    method: str


class RoadPlaneDepthEstimator:
    """
    Estimates vehicle physical distance (Z) and lateral position (X)
    by projecting the vehicle bottom contact patch onto the calibrated road plane.
    """

    def __init__(
        self,
        camera_model: CameraModel,
        road_geometry: RoadGeometry,
        camera_height_meters: float = 1.35,
    ):
        self.camera_model = camera_model
        self.road_geom = road_geometry
        self.camera_height = camera_height_meters

    def estimate_ground_point(
        self,
        u_bottom: float,
        v_bottom: float,
    ) -> Optional[GroundPoint3D]:
        """
        Projects pixel coordinate (u, v) representing ground contact point
        onto the 3D ground plane.

        Args:
            u_bottom: Horizontal pixel coordinate of tire-road contact.
            v_bottom: Vertical pixel coordinate of tire-road contact.

        Returns:
            GroundPoint3D in meters, or None if point is above horizon.
        """
        vy = self.road_geom.vanishing_point[1]
        vx = self.road_geom.vanishing_point[0]
        fy = self.camera_model.intrinsics.fy
        fx = self.camera_model.intrinsics.fx
        cx = self.camera_model.intrinsics.cx
        h_cam = self.camera_height

        # Must be strictly below horizon/vanishing point
        delta_y = v_bottom - vy
        if delta_y <= 5.0:
            # Contact point is at or above horizon (infinite or invalid depth)
            return None

        # Depth Z along optical axis: Z = (fy * h_cam) / (v - vy)
        depth_z = (fy * h_cam) / delta_y
        if depth_z <= 0.5 or depth_z > 250.0:
            return None

        # Lateral distance X: X = (u - cx) * Z / fx
        lateral_x = ((u_bottom - cx) * depth_z) / fx

        # Vertical height Y is negative ground plane (-h_cam)
        vertical_y = -h_cam

        # Confidence decays with distance squared (perspective compression)
        dist_factor = max(0.1, 1.0 - (depth_z / 150.0))
        conf = round(self.road_geom.confidence * dist_factor, 3)

        return GroundPoint3D(
            X=round(lateral_x, 3),
            Y=round(vertical_y, 3),
            Z=round(depth_z, 3),
            confidence=conf,
            method="road_plane_contact_projection",
        )
