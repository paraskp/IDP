"""
Vanishing point detection and geometric estimation module.
Uses line segment detection (LSD / Hough) with RANSAC consensus to detect
the principal longitudinal vanishing point on dashcam footage.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class LineSegment:
    """Represents a 2D directed line segment [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def angle_deg(self) -> float:
        """Angle in degrees [-90, +90] relative to horizontal."""
        angle = math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        return angle

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_homogeneous_line(self) -> np.ndarray:
        """Computes 2D homogeneous line coefficients [a, b, c] where ax + by + c = 0."""
        p1 = np.array([self.x1, self.y1, 1.0])
        p2 = np.array([self.x2, self.y2, 1.0])
        line = np.cross(p1, p2)
        norm = math.hypot(line[0], line[1])
        return line / norm if norm > 0 else line


@dataclass(frozen=True)
class VanishingPointResult:
    """Result of vanishing point estimation."""
    x: float
    y: float
    confidence: float
    inlier_count: int
    total_lines: int
    is_valid: bool
    method: str = "RANSAC_line_intersection"
    reason: Optional[str] = None


def detect_line_segments(
    image: np.ndarray,
    min_length: float = 30.0,
    min_angle_deg: float = 12.0,
    max_angle_deg: float = 85.0,
) -> List[LineSegment]:
    """
    Detects and filters line segments in the frame that are potential road/perspective lines.
    Rejects near-horizontal lines (angle < min_angle_deg) and near-vertical noise.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Enhance contrast and suppress noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Probabilistic Hough Transform
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=int(min_length),
        maxLineGap=20,
    )

    if raw_lines is None:
        return []

    valid_lines: List[LineSegment] = []
    for line in raw_lines:
        coords = line.ravel()
        if len(coords) < 4:
            continue
        x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
        seg = LineSegment(x1, y1, x2, y2)
        abs_angle = abs(seg.angle_deg)

        # Retain diagonal perspective lines converging forward
        if min_angle_deg <= abs_angle <= max_angle_deg and seg.length >= min_length:
            valid_lines.append(seg)

    return valid_lines


def compute_intersection(line1: LineSegment, line2: LineSegment) -> Optional[Tuple[float, float]]:
    """Calculates intersection point of two line segments in 2D plane."""
    l1 = line1.to_homogeneous_line()
    l2 = line2.to_homogeneous_line()
    pt = np.cross(l1, l2)

    if abs(pt[2]) < 1e-6:
        return None  # Parallel lines

    x = pt[0] / pt[2]
    y = pt[1] / pt[2]
    return (x, y)


def estimate_vanishing_point_ransac(
    lines: List[LineSegment],
    image_shape: Tuple[int, int],
    max_iterations: int = 300,
    angle_thresh_deg: float = 2.5,
    random_seed: int = 42,
) -> VanishingPointResult:
    """
    Estimates the principal vanishing point using RANSAC consensus over line pairs.

    Args:
        lines: List of detected LineSegment objects.
        image_shape: (height, width) of the image.
        max_iterations: Maximum RANSAC hypothesis iterations.
        angle_thresh_deg: Max angular tolerance for inlier lines.
        random_seed: Reproducibility seed.

    Returns:
        VanishingPointResult with coordinates and confidence.
    """
    h, w = image_shape
    if len(lines) < 2:
        return VanishingPointResult(
            x=w / 2.0,
            y=h / 2.0,
            confidence=0.0,
            inlier_count=0,
            total_lines=len(lines),
            is_valid=False,
            reason="insufficient line segments detected (need at least 2)",
        )

    rng = np.random.default_rng(random_seed)
    best_inliers = 0
    best_vp = (w / 2.0, h * 0.45)
    thresh_rad = math.radians(angle_thresh_deg)

    # Valid VP search bounds (around road horizon)
    min_x, max_x = -w * 0.5, w * 1.5
    min_y, max_y = 0.0, h * 0.85

    for _ in range(max_iterations):
        idx1, idx2 = rng.choice(len(lines), size=2, replace=False)
        l1, l2 = lines[idx1], lines[idx2]

        # Require reasonable angular disparity between the two lines
        if abs(l1.angle_deg - l2.angle_deg) < 10.0:
            continue

        pt = compute_intersection(l1, l2)
        if pt is None:
            continue

        vx, vy = pt
        if not (min_x <= vx <= max_x and min_y <= vy <= max_y):
            continue

        # Count inliers
        inliers = 0
        for seg in lines:
            mx, my = seg.midpoint
            vec_to_vp = (vx - mx, vy - my)
            norm = math.hypot(vec_to_vp[0], vec_to_vp[1])
            if norm < 1e-3:
                continue

            dir_line = (seg.x2 - seg.x1, seg.y2 - seg.y1)
            norm_dir = math.hypot(dir_line[0], dir_line[1])
            if norm_dir < 1e-3:
                continue

            cos_sim = abs((dir_line[0] * vec_to_vp[0] + dir_line[1] * vec_to_vp[1]) / (norm_dir * norm))
            cos_sim = min(1.0, max(-1.0, cos_sim))
            angle_err = math.acos(cos_sim)

            if angle_err <= thresh_rad:
                inliers += 1

        if inliers > best_inliers:
            best_inliers = inliers
            best_vp = (vx, vy)

    inlier_ratio = best_inliers / len(lines) if len(lines) > 0 else 0.0
    confidence = min(1.0, max(0.0, (inlier_ratio * 0.7) + min(0.3, best_inliers / 30.0)))
    is_valid = best_inliers >= 4 and confidence >= 0.25

    return VanishingPointResult(
        x=float(best_vp[0]),
        y=float(best_vp[1]),
        confidence=round(confidence, 4),
        inlier_count=best_inliers,
        total_lines=len(lines),
        is_valid=is_valid,
        reason=None if is_valid else "low inlier consensus for vanishing point",
    )
