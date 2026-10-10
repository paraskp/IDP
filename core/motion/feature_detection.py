"""
Stationary feature detection module.
Detects salient, stable environmental features (road markings, barriers, signs, poles)
while masking out dynamic regions and sky.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class FeaturePoint:
    """Represents a 2D feature point with quality score."""
    x: float
    y: float
    score: float

    @property
    def pt(self) -> Tuple[float, float]:
        return (self.x, self.y)


class StationaryFeatureDetector:
    """
    Detects high-gradient stationary features suitable for optical flow
    and visual odometry tracking.
    """

    def __init__(
        self,
        max_corners: int = 500,
        quality_level: float = 0.02,
        min_distance: float = 12.0,
        block_size: int = 7,
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size

    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        dynamic_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Detects Shi-Tomasi corners in stationary areas.

        Args:
            image: Input BGR or grayscale image.
            mask: Optional binary mask (255 = keep, 0 = ignore).
            dynamic_boxes: Optional list of [x1, y1, x2, y2] bounding boxes of moving objects to exclude.

        Returns:
            np.ndarray of shape (N, 1, 2) with float32 coordinates.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape[:2]
        effective_mask = np.ones((h, w), dtype=np.uint8) * 255 if mask is None else mask.copy()

        # Exclude dynamic object bounding boxes
        if dynamic_boxes:
            for x1, y1, x2, y2 in dynamic_boxes:
                # Add margin around detected moving vehicles
                bx1 = max(0, int(x1 - 5))
                by1 = max(0, int(y1 - 5))
                bx2 = min(w - 1, int(x2 + 5))
                by2 = min(h - 1, int(y2 + 5))
                cv2.rectangle(effective_mask, (bx1, by1), (bx2, by2), 0, -1)

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=effective_mask,
            blockSize=self.block_size,
        )

        if corners is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return corners.astype(np.float32)
