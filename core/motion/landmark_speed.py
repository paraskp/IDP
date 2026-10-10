"""
Landmark Speed and Feature Filtering Module.
Anchors feature tracking and velocity estimation exclusively to stationary
road markings, lane dashes, and infrastructure to eliminate moving traffic interference.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from core.detection.landmark_detector import RoadLandmark
from core.motion.optical_flow import FlowMatch


@dataclass(frozen=True)
class LandmarkSpeedEstimate:
    """Estimated speed and anchoring metrics derived from stationary landmarks."""
    num_anchored_features: int
    num_detected_landmarks: int
    dash_frequency_hz: Optional[float]
    frequency_speed_mps: Optional[float]
    frequency_speed_kmh: Optional[float]
    confidence: float


class LandmarkSpeedTracker:
    """
    Filters optical flow features against detected stationary landmarks
    and computes lane-dash passage frequency metrics.
    """

    def __init__(
        self,
        standard_dash_length_m: float = 3.0,
        standard_dash_gap_m: float = 4.5,
    ):
        self.dash_cycle_m = standard_dash_length_m + standard_dash_gap_m
        self.last_dash_timestamps: List[float] = []
        self.dash_count = 0

    def filter_features_to_landmarks(
        self,
        matches: List[FlowMatch],
        landmarks: List[RoadLandmark],
        padding_px: int = 12,
    ) -> List[FlowMatch]:
        """
        Filters optical flow matches, keeping only features that fall inside
        or immediately adjacent to detected stationary landmarks.

        Args:
            matches: List of optical flow matches between frames.
            landmarks: List of detected stationary landmarks.
            padding_px: Pixel margin around bounding boxes.

        Returns:
            Filtered list of FlowMatch instances.
        """
        if not landmarks or not matches:
            return matches

        filtered: List[FlowMatch] = []
        boxes = [
            (
                l.bbox[0] - padding_px,
                l.bbox[1] - padding_px,
                l.bbox[2] + padding_px,
                l.bbox[3] + padding_px,
            )
            for l in landmarks
        ]

        for m in matches:
            x, y = m.pt_prev
            # Check if (x, y) is inside any landmark bounding box
            for bx1, by1, bx2, by2 in boxes:
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    filtered.append(m)
                    break

        # If too few points fall strictly on landmarks, return original to avoid starving
        return filtered if len(filtered) >= 4 else matches

    def update_dash_frequency(
        self,
        landmarks: List[RoadLandmark],
        timestamp: float,
        dt: float,
    ) -> LandmarkSpeedEstimate:
        """
        Tracks passage of dashed lane markings over time to estimate
        frequency-based longitudinal velocity.
        """
        lane_dashes = [l for l in landmarks if l.class_name == "lane_dash"]
        num_dashes = len(lane_dashes)

        # Clean timestamps older than 3.0 seconds
        self.last_dash_timestamps = [
            t for t in self.last_dash_timestamps if (timestamp - t) <= 3.0
        ]

        if num_dashes > 0:
            self.last_dash_timestamps.append(timestamp)

        freq_hz: Optional[float] = None
        speed_mps: Optional[float] = None
        speed_kmh: Optional[float] = None
        conf = 0.0

        if len(self.last_dash_timestamps) >= 4:
            duration = self.last_dash_timestamps[-1] - self.last_dash_timestamps[0]
            if duration >= 0.5:
                freq_hz = (len(self.last_dash_timestamps) - 1) / duration
                speed_mps = freq_hz * self.dash_cycle_m
                speed_kmh = speed_mps * 3.6
                conf = min(0.85, 0.4 + len(self.last_dash_timestamps) * 0.05)

        return LandmarkSpeedEstimate(
            num_anchored_features=0,
            num_detected_landmarks=len(landmarks),
            dash_frequency_hz=round(freq_hz, 2) if freq_hz is not None else None,
            frequency_speed_mps=round(speed_mps, 2) if speed_mps is not None else None,
            frequency_speed_kmh=round(speed_kmh, 1) if speed_kmh is not None else None,
            confidence=round(conf, 3),
        )
