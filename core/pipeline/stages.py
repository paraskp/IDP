"""
Pipeline stage definitions for Dashcam Speed Analyzer.
Enumerates pipeline phases across the analysis lifecycle.
"""

from enum import Enum, auto


class PipelineStage(Enum):
    """Enumeration of analysis pipeline stages."""
    VIDEO_INPUT = auto()          # Phase 1: Input & Metadata (Active)
    CALIBRATION = auto()          # Phase 2: Automatic Camera & Scene Calibration (TODO)
    FEATURE_TRACKING = auto()     # Phase 3: Stationary Feature Detection (TODO)
    VISUAL_ODOMETRY = auto()      # Phase 4: Visual Odometry (TODO)
    EGO_SPEED = auto()            # Phase 5: Ego Speed Estimation (TODO)
    VEHICLE_DETECTION = auto()    # Phase 6: Vehicle Detection (TODO)
    VEHICLE_TRACKING = auto()     # Phase 7: Vehicle Tracking (TODO)
    DEPTH_ESTIMATION = auto()     # Phase 8: Depth & Road Geometry (TODO)
    VEHICLE_SPEED = auto()        # Phase 9: Other Vehicle Speed Estimation (TODO)
    OUTPUT_GENERATION = auto()    # Phase 10: Video Annotation & Reports (TODO)
