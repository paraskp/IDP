"""
Optical flow tracking and motion consistency analysis module.
Implements bidirectional Pyramidal Lucas-Kanade with forward-backward error check,
RANSAC epipolar outlier rejection, and stationary feature motion tracking.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class FlowMatch:
    """Represents a tracked point match between consecutive frames."""
    pt_prev: Tuple[float, float]
    pt_curr: Tuple[float, float]
    displacement: Tuple[float, float]  # (dx, dy)
    magnitude: float
    is_stationary_inlier: bool = True
    fb_error: float = 0.0


@dataclass
class TrackingFrameResult:
    """Contains results of feature tracking between two consecutive frames."""
    matches: List[FlowMatch]
    inlier_matches: List[FlowMatch]
    rejected_matches: List[FlowMatch]
    mean_displacement: Tuple[float, float]
    median_speed_px: float
    tracking_ratio: float
    confidence: float


class SparseOpticalFlowTracker:
    """
    Bidirectional Lucas-Kanade optical flow tracker with RANSAC
    stationary feature verification.
    """

    def __init__(
        self,
        win_size: Tuple[int, int] = (21, 21),
        max_level: int = 3,
        max_fb_error: float = 1.5,
    ):
        self.win_size = win_size
        self.max_level = max_level
        self.max_fb_error = max_fb_error
        self.lk_params = dict(
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def track(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: np.ndarray,
    ) -> TrackingFrameResult:
        """
        Tracks points from prev_gray to curr_gray with bidirectional validation.

        Args:
            prev_gray: Previous grayscale frame.
            curr_gray: Current grayscale frame.
            prev_pts: Points of shape (N, 1, 2) in previous frame.

        Returns:
            TrackingFrameResult with verified inliers and diagnostics.
        """
        if prev_pts is None or len(prev_pts) == 0:
            return TrackingFrameResult(
                matches=[],
                inlier_matches=[],
                rejected_matches=[],
                mean_displacement=(0.0, 0.0),
                median_speed_px=0.0,
                tracking_ratio=0.0,
                confidence=0.0,
            )

        # 1. Forward Optical Flow: prev -> curr
        curr_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )

        # 2. Backward Optical Flow: curr -> prev
        back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, curr_pts, None, **self.lk_params
        )

        # 3. Bidirectional Consistency Check (Forward-Backward Error)
        fb_err = np.linalg.norm(prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)
        valid_mask = (status_fwd.flatten() == 1) & (status_bwd.flatten() == 1) & (fb_err <= self.max_fb_error)

        if not np.any(valid_mask):
            return TrackingFrameResult(
                matches=[],
                inlier_matches=[],
                rejected_matches=[],
                mean_displacement=(0.0, 0.0),
                median_speed_px=0.0,
                tracking_ratio=0.0,
                confidence=0.0,
            )

        p_prev = prev_pts.reshape(-1, 2)[valid_mask]
        p_curr = curr_pts.reshape(-1, 2)[valid_mask]
        errs = fb_err[valid_mask]

        # 4. RANSAC Epipolar / Homography Outlier Rejection
        # Points from moving vehicles will violate the background stationary transformation
        is_inlier = np.zeros(len(p_prev), dtype=bool)
        if len(p_prev) >= 8:
            _, inlier_mask = cv2.findFundamentalMat(
                p_prev, p_curr, cv2.FM_RANSAC, 1.5, 0.99
            )
            if inlier_mask is not None:
                is_inlier = inlier_mask.flatten().astype(bool)
            else:
                is_inlier = np.ones(len(p_prev), dtype=bool)
        else:
            is_inlier = np.ones(len(p_prev), dtype=bool)

        all_matches: List[FlowMatch] = []
        inliers: List[FlowMatch] = []
        rejected: List[FlowMatch] = []

        dxs, dys, speeds = [], [], []

        for i in range(len(p_prev)):
            p0 = (float(p_prev[i, 0]), float(p_prev[i, 1]))
            p1 = (float(p_curr[i, 0]), float(p_curr[i, 1]))
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            mag = float(np.hypot(dx, dy))
            inlier_flag = bool(is_inlier[i])

            match = FlowMatch(
                pt_prev=p0,
                pt_curr=p1,
                displacement=(dx, dy),
                magnitude=mag,
                is_stationary_inlier=inlier_flag,
                fb_error=float(errs[i]),
            )
            all_matches.append(match)

            if inlier_flag:
                inliers.append(match)
                dxs.append(dx)
                dys.append(dy)
                speeds.append(mag)
            else:
                rejected.append(match)

        mean_dx = float(np.mean(dxs)) if dxs else 0.0
        mean_dy = float(np.mean(dys)) if dys else 0.0
        med_speed = float(np.median(speeds)) if speeds else 0.0

        track_ratio = len(inliers) / len(prev_pts) if len(prev_pts) > 0 else 0.0
        conf = min(1.0, (track_ratio * 0.7) + min(0.3, len(inliers) / 50.0))

        return TrackingFrameResult(
            matches=all_matches,
            inlier_matches=inliers,
            rejected_matches=rejected,
            mean_displacement=(mean_dx, mean_dy),
            median_speed_px=med_speed,
            tracking_ratio=track_ratio,
            confidence=round(conf, 4),
        )

    def draw_debug_flow(
        self,
        frame: np.ndarray,
        result: TrackingFrameResult,
    ) -> np.ndarray:
        """
        Draws optical flow vectors onto the frame for visual verification.
        Green: Verified stationary inlier features.
        Red: Rejected outliers (moving objects / tracking anomalies).
        """
        vis = frame.copy()
        for m in result.rejected_matches:
            p0 = (int(round(m.pt_prev[0])), int(round(m.pt_prev[1])))
            p1 = (int(round(m.pt_curr[0])), int(round(m.pt_curr[1])))
            cv2.arrowedLine(vis, p0, p1, (0, 0, 240), 1, tipLength=0.3)
            cv2.circle(vis, p1, 2, (0, 0, 240), -1)

        for m in result.inlier_matches:
            p0 = (int(round(m.pt_prev[0])), int(round(m.pt_prev[1])))
            p1 = (int(round(m.pt_curr[0])), int(round(m.pt_curr[1])))
            cv2.arrowedLine(vis, p0, p1, (0, 230, 0), 2, tipLength=0.3)
            cv2.circle(vis, p1, 2, (0, 255, 0), -1)

        return vis
