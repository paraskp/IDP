"""
Road Grid Module.
Implements an imaginary metric grid projected on the road plane (Phase 5 enhancement).
Calculates vehicle speed by measuring how many grid squares the road surface passes
beneath the vehicle per unit time (frame / second).
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from core.calibration.camera import CameraModel
from core.calibration.road_geometry import RoadGeometry


@dataclass(frozen=True)
class GridMotionResult:
    """Telemetry of motion measured across the imaginary road grid."""
    squares_passed_per_frame: float
    squares_passed_per_second: float
    raw_speed_mps: float
    raw_speed_kmh: float
    square_size_m: float
    num_grid_features: int
    confidence: float
    is_valid: bool


class RoadGrid:
    """
    Imaginary metric grid laid out on the road plane in front of the vehicle.
    Translates pixel optical flow displacements into physical grid squares traversed.
    """

    def __init__(
        self,
        camera_model: CameraModel,
        road_geometry: RoadGeometry,
        camera_height_meters: float = 1.35,
        square_size_meters: float = 1.0,
        lateral_range_meters: Tuple[float, float] = (-3.0, 3.0),
        longitudinal_range_meters: Tuple[float, float] = (3.0, 35.0),
    ):
        self.camera_model = camera_model
        self.road_geom = road_geometry
        self.camera_height = max(0.5, float(camera_height_meters))
        self.square_size = max(0.2, float(square_size_meters))
        self.x_min, self.x_max = lateral_range_meters
        self.z_min, self.z_max = longitudinal_range_meters

    def project_pixel_to_ground(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """
        Projects 2D image pixel (u, v) on the road surface onto 3D ground coordinates (X, Z) in meters.
        Returns: (X, Z) or None if point is at or above horizon.
        """
        vy = self.road_geom.vanishing_point[1]
        delta_y = v - vy
        if delta_y <= 5.0:
            return None

        fy = self.camera_model.intrinsics.fy
        fx = self.camera_model.intrinsics.fx
        cx = self.camera_model.intrinsics.cx
        h = self.camera_height

        z = (fy * h) / delta_y
        if z < 0.5 or z > 150.0:
            return None

        x = ((u - cx) * z) / fx
        return x, z

    def ground_to_pixel(self, x: float, z: float) -> Optional[Tuple[float, float]]:
        """
        Projects 3D ground coordinate (X, Z) in meters onto 2D image coordinates (u, v).
        Returns: (u, v) or None if Z <= 0.
        """
        if z <= 0.2:
            return None

        fy = self.camera_model.intrinsics.fy
        fx = self.camera_model.intrinsics.fx
        cx = self.camera_model.intrinsics.cx
        vy = self.road_geom.vanishing_point[1]
        h = self.camera_height

        v = vy + (fy * h) / z
        u = cx + (fx * x) / z
        return u, v

    def project_pixel_to_grid(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """
        Projects 2D image pixel to continuous grid square coordinates:
        (grid_x, grid_z) = (X / square_size, Z / square_size).
        """
        ground_pt = self.project_pixel_to_ground(u, v)
        if ground_pt is None:
            return None
        x, z = ground_pt
        return (x / self.square_size, z / self.square_size)

    def calculate_squares_passed(
        self,
        prev_points: np.ndarray,
        curr_points: np.ndarray,
        dt: float,
    ) -> GridMotionResult:
        """
        Calculates how many grid squares the road surface has traversed between two frames.

        Args:
            prev_points: Nx2 array of points in previous frame (x, y).
            curr_points: Nx2 array of points in current frame (x, y).
            dt: Time delta between frames in seconds.

        Returns:
            GridMotionResult with square counts and derived velocities.
        """
        if dt <= 0:
            dt = 1.0 / 30.0

        if len(prev_points) == 0 or len(curr_points) == 0:
            return GridMotionResult(
                squares_passed_per_frame=0.0,
                squares_passed_per_second=0.0,
                raw_speed_mps=0.0,
                raw_speed_kmh=0.0,
                square_size_m=self.square_size,
                num_grid_features=0,
                confidence=0.0,
                is_valid=False,
            )

        sq_diffs: List[float] = []

        for p0, p1 in zip(prev_points, curr_points):
            u0, v0 = float(p0[0]), float(p0[1])
            u1, v1 = float(p1[0]), float(p1[1])

            # Convert both to grid coordinates
            g0 = self.project_pixel_to_grid(u0, v0)
            g1 = self.project_pixel_to_grid(u1, v1)

            if g0 is None or g1 is None:
                continue

            gx0, gz0 = g0
            gx1, gz1 = g1

            # Only consider points within reasonable lateral and longitudinal road bounds
            z_m0 = gz0 * self.square_size
            x_m0 = gx0 * self.square_size
            if not (self.z_min <= z_m0 <= self.z_max and self.x_min <= x_m0 <= self.x_max):
                continue

            # In forward motion, objects move closer: Z_prev > Z_curr => gz0 > gz1
            delta_sq = gz0 - gz1

            # Physical sanity check: max plausible forward speed ~150 km/h (~42 m/s)
            max_sq_per_frame = (45.0 * dt) / self.square_size
            if 0.0 < delta_sq <= max_sq_per_frame:
                sq_diffs.append(delta_sq)

        if len(sq_diffs) < 3:
            return GridMotionResult(
                squares_passed_per_frame=0.0,
                squares_passed_per_second=0.0,
                raw_speed_mps=0.0,
                raw_speed_kmh=0.0,
                square_size_m=self.square_size,
                num_grid_features=len(sq_diffs),
                confidence=0.1,
                is_valid=False,
            )

        # Robust statistics: median and IQR filtering
        sq_array = np.array(sq_diffs, dtype=np.float64)
        median_sq = float(np.median(sq_array))
        q25, q75 = np.percentile(sq_array, [25, 75])
        iqr = q75 - q25
        inliers = sq_array[(sq_array >= q25 - 1.5 * iqr) & (sq_array <= q75 + 1.5 * iqr)]

        final_sq_frame = float(np.median(inliers)) if len(inliers) > 0 else median_sq
        sq_per_sec = final_sq_frame / dt
        speed_mps = sq_per_sec * self.square_size
        speed_kmh = speed_mps * 3.6

        # Confidence based on inlier count and spread
        std_val = float(np.std(inliers)) if len(inliers) > 1 else 1.0
        consistency = max(0.2, 1.0 - min(1.0, std_val / (final_sq_frame + 1e-4)))
        count_factor = min(1.0, len(inliers) / 15.0)
        conf = round(float(0.4 * consistency + 0.6 * count_factor), 3)

        return GridMotionResult(
            squares_passed_per_frame=round(final_sq_frame, 3),
            squares_passed_per_second=round(sq_per_sec, 2),
            raw_speed_mps=round(speed_mps, 2),
            raw_speed_kmh=round(speed_kmh, 1),
            square_size_m=self.square_size,
            num_grid_features=len(inliers),
            confidence=conf,
            is_valid=True,
        )

    def generate_grid_overlay(
        self,
        image_shape: Tuple[int, int],
        max_depth_lines: int = 25,
    ) -> Dict[str, Any]:
        """
        Generates 2D coordinates for rendering the perspective road grid on the camera frame.

        Returns:
            Dictionary containing:
                - longitudinal_lines: list of [((u1, v1), (u2, v2))]
                - transversal_lines: list of [((u1, v1), (u2, v2))]
                - grid_cells: list of 4-point quadrilaterals for grid squares
        """
        h_img, w_img = image_shape[:2]
        long_lines = []
        trans_lines = []
        grid_cells = []

        # Discrete ground coordinates
        x_steps = np.arange(self.x_min, self.x_max + 1e-4, self.square_size)
        z_steps = np.arange(self.z_min, min(self.z_max, self.z_min + max_depth_lines * self.square_size) + 1e-4, self.square_size)

        # 1. Longitudinal lines (along Z, converging towards vanishing point)
        for x in x_steps:
            p_near = self.ground_to_pixel(x, self.z_min)
            p_far = self.ground_to_pixel(x, z_steps[-1])
            if p_near is not None and p_far is not None:
                u0, v0 = int(round(p_near[0])), int(round(p_near[1]))
                u1, v1 = int(round(p_far[0])), int(round(p_far[1]))
                if 0 <= v0 < h_img * 1.5 or 0 <= v1 < h_img * 1.5:
                    long_lines.append(((u0, v0), (u1, v1)))

        # 2. Transversal lines (along X, perpendicular to driving direction)
        for z in z_steps:
            p_left = self.ground_to_pixel(self.x_min, z)
            p_right = self.ground_to_pixel(self.x_max, z)
            if p_left is not None and p_right is not None:
                u0, v0 = int(round(p_left[0])), int(round(p_left[1]))
                u1, v1 = int(round(p_right[0])), int(round(p_right[1]))
                if 0 <= v0 < h_img and 0 <= v1 < h_img:
                    trans_lines.append(((u0, v0), (u1, v1)))

        # 3. Grid squares (quadrilaterals)
        for i in range(len(x_steps) - 1):
            for j in range(len(z_steps) - 1):
                p_bl = self.ground_to_pixel(x_steps[i], z_steps[j])
                p_br = self.ground_to_pixel(x_steps[i + 1], z_steps[j])
                p_tr = self.ground_to_pixel(x_steps[i + 1], z_steps[j + 1])
                p_tl = self.ground_to_pixel(x_steps[i], z_steps[j + 1])

                if all(p is not None for p in [p_bl, p_br, p_tr, p_tl]):
                    pts = np.array([p_bl, p_br, p_tr, p_tl], dtype=np.int32)
                    # Keep if on image
                    if np.any((pts[:, 1] >= 0) & (pts[:, 1] < h_img)):
                        grid_cells.append(pts)

        return {
            "longitudinal_lines": long_lines,
            "transversal_lines": trans_lines,
            "grid_cells": grid_cells,
        }

    def create_road_grid_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Creates a binary mask (uint8) of the road grid region on the image
        to guide feature extraction exclusively inside the grid.
        """
        h_img, w_img = image_shape[:2]
        mask = np.zeros((h_img, w_img), dtype=np.uint8)

        p_bl = self.ground_to_pixel(self.x_min, self.z_min)
        p_br = self.ground_to_pixel(self.x_max, self.z_min)
        p_tr = self.ground_to_pixel(self.x_max, self.z_max)
        p_tl = self.ground_to_pixel(self.x_min, self.z_max)

        if all(p is not None for p in [p_bl, p_br, p_tr, p_tl]):
            poly = np.array([p_bl, p_br, p_tr, p_tl], dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)

        return mask
