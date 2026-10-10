"""
Monocular depth fusion module (Phase 8).
Combines ground plane projection with physical vehicle dimension priors.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from core.calibration.camera import CameraModel
from core.calibration.road_geometry import RoadGeometry
from core.depth.road_plane import GroundPoint3D, RoadPlaneDepthEstimator
from core.detection.vehicle_detector import Detection


# Typical real-world vehicle dimensions in meters (width, height, length)
VEHICLE_DIMENSION_PRIORS: Dict[str, Tuple[float, float, float]] = {
    "car": (1.82, 1.48, 4.5),
    "truck": (2.50, 3.40, 8.0),
    "bus": (2.55, 3.20, 11.0),
    "motorcycle": (0.85, 1.20, 2.1),
}


@dataclass(frozen=True)
class VehicleDepth3D:
    """Estimated 3D position and depth of a vehicle."""
    X: float  # Lateral distance in meters (0 = directly ahead, >0 = right, <0 = left)
    Y: float  # Vertical distance in meters
    Z: float  # Longitudinal depth in meters
    ground_contact_z: Optional[float]
    dimensional_z: Optional[float]
    confidence: float
    scale_source: str


class MonocularDepthEstimator:
    """
    Estimates vehicle 3D distance by fusing road plane projection
    and class-specific dimensional geometry.
    """

    def __init__(
        self,
        camera_model: CameraModel,
        road_geometry: RoadGeometry,
        camera_height_meters: float = 1.35,
    ):
        self.camera_model = camera_model
        self.road_geom = road_geometry
        self.road_plane_est = RoadPlaneDepthEstimator(
            camera_model, road_geometry, camera_height_meters
        )

    def estimate_depth(self, detection: Detection) -> Optional[VehicleDepth3D]:
        """
        Estimates 3D coordinates for a detected vehicle.
        """
        fx = self.camera_model.intrinsics.fx
        cx = self.camera_model.intrinsics.cx
        u_bottom, v_bottom = detection.bottom_center

        # Method 1: Road plane contact depth
        ground_pt = self.road_plane_est.estimate_ground_point(u_bottom, v_bottom)
        z_ground = ground_pt.Z if ground_pt is not None else None

        # Method 2: Known vehicle dimensional prior
        cls_name = detection.class_name.lower()
        prior = VEHICLE_DIMENSION_PRIORS.get(cls_name, VEHICLE_DIMENSION_PRIORS["car"])
        expected_width_m = prior[0]

        w_box = detection.width
        z_dim = None
        if w_box > 5.0 and fx > 0:
            z_dim = (fx * expected_width_m) / w_box

        # Fusion logic
        final_z = None
        conf = 0.0
        scale_source = "None"

        if z_ground is not None and z_dim is not None:
            # Check consistency between ground and dimensional depth
            ratio = min(z_ground, z_dim) / max(z_ground, z_dim)
            if ratio > 0.5:
                # Highly consistent: weighted average (60% ground plane, 40% dimensions)
                final_z = 0.6 * z_ground + 0.4 * z_dim
                conf = min(0.90, (ground_pt.confidence * 0.7) + (ratio * 0.3))
                scale_source = "Road Plane + Dimensional Prior Fusion"
            else:
                # Discrepancy: trust ground plane if vehicle is near, else dimensions
                if z_ground < 40.0:
                    final_z = z_ground
                    conf = ground_pt.confidence * 0.8
                    scale_source = "Road Plane Projection"
                else:
                    final_z = z_dim
                    conf = 0.5
                    scale_source = "Vehicle Dimensional Prior"
        elif z_ground is not None:
            final_z = z_ground
            conf = ground_pt.confidence
            scale_source = "Road Plane Projection"
        elif z_dim is not None:
            final_z = z_dim
            conf = 0.5
            scale_source = "Vehicle Dimensional Prior"

        if final_z is None or final_z <= 0.5 or final_z > 250.0:
            return None

        # Compute lateral position X
        lateral_x = ((detection.center[0] - cx) * final_z) / fx

        return VehicleDepth3D(
            X=round(lateral_x, 3),
            Y=round(ground_pt.Y if ground_pt else -1.35, 3),
            Z=round(final_z, 3),
            ground_contact_z=round(z_ground, 3) if z_ground else None,
            dimensional_z=round(z_dim, 3) if z_dim else None,
            confidence=round(conf, 3),
            scale_source=scale_source,
        )
