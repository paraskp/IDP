"""
Comprehensive reliability and confidence evaluation engine (Section 18).
Evaluates optical flow consistency, tracker continuity, depth reliability,
and metric scale provenance.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ReliabilityReport:
    """Detailed reliability breakdown for speed estimates."""
    overall_confidence: float
    feature_tracking_score: float
    tracking_continuity_score: float
    depth_reliability_score: float
    scale_source_score: float
    is_statistically_sound: bool
    quality_grade: str  # 'HIGH', 'MEDIUM', 'LOW', 'UNRELIABLE'
    diagnostic_summary: str


class ConfidenceEngine:
    """
    Evaluates multi-factor confidence across computer vision and geometric stages.
    """

    @staticmethod
    def evaluate(
        optical_flow_confidence: float,
        detection_confidence: float,
        tracker_hits: int,
        depth_confidence: float,
        scale_source: str,
        is_calibrated: bool,
    ) -> ReliabilityReport:
        # 1. Feature tracking score
        flow_score = min(1.0, max(0.0, optical_flow_confidence))

        # 2. Tracking continuity score
        track_score = min(1.0, (tracker_hits / 8.0) * (detection_confidence or 0.5))

        # 3. Depth reliability score
        depth_score = min(1.0, max(0.0, depth_confidence))

        # 4. Scale source weighting
        if "User" in scale_source or "Known" in scale_source:
            scale_score = 0.95
        elif "Road Plane" in scale_source:
            scale_score = 0.75 if is_calibrated else 0.55
        elif "Dimensional" in scale_source:
            scale_score = 0.50
        else:
            scale_score = 0.15

        # Weighted aggregate
        overall = (
            0.25 * flow_score
            + 0.25 * track_score
            + 0.25 * depth_score
            + 0.25 * scale_score
        )
        overall = round(min(1.0, max(0.0, overall)), 2)

        if overall >= 0.75:
            grade = "HIGH"
        elif overall >= 0.50:
            grade = "MEDIUM"
        elif overall >= 0.30:
            grade = "LOW"
        else:
            grade = "UNRELIABLE"

        is_sound = (overall >= 0.40) and (scale_score >= 0.4)

        summary = (
            f"Quality: {grade} ({overall * 100:.0f}%) | "
            f"Flow: {flow_score:.2f}, Track: {track_score:.2f}, "
            f"Depth: {depth_score:.2f}, Scale: {scale_score:.2f} [{scale_source}]"
        )

        return ReliabilityReport(
            overall_confidence=overall,
            feature_tracking_score=round(flow_score, 2),
            tracking_continuity_score=round(track_score, 2),
            depth_reliability_score=round(depth_score, 2),
            scale_source_score=round(scale_score, 2),
            is_statistically_sound=is_sound,
            quality_grade=grade,
            diagnostic_summary=summary,
        )
