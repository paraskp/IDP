"""
Motion estimation, visual odometry, and ego speed subsystem (Phases 3, 4, 5).
"""

from core.motion.feature_detection import (
    FeaturePoint,
    StationaryFeatureDetector,
)
from core.motion.optical_flow import (
    FlowMatch,
    SparseOpticalFlowTracker,
    TrackingFrameResult,
)
from core.motion.visual_odometry import (
    PoseEstimate,
    VisualOdometry,
)
from core.motion.ego_speed import (
    EgoSpeedEstimator,
    EgoSpeedKalmanFilter,
    EgoSpeedResult,
)

__all__ = [
    "FeaturePoint",
    "StationaryFeatureDetector",
    "FlowMatch",
    "SparseOpticalFlowTracker",
    "TrackingFrameResult",
    "PoseEstimate",
    "VisualOdometry",
    "EgoSpeedEstimator",
    "EgoSpeedKalmanFilter",
    "EgoSpeedResult",
]
