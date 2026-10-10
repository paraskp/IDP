"""
Video annotation and visual telemetry overlay module (Phase 10).
Renders vehicle bounding boxes, persistent IDs, speed tags, confidence meters,
trajectories, and ego speed HUD onto output video frames.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from core.detection.tracker import VehicleTrack
from core.motion.ego_speed import EgoSpeedResult
from core.motion.road_grid import RoadGrid
from core.speed.vehicle_speed import VehicleDirection, VehicleSpeedResult


class VideoAnnotator:
    """
    Renders telemetry overlays, bounding boxes, road grids, and speed results onto video frames.
    """

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def annotate_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        ego_speed: Optional[EgoSpeedResult],
        tracks: List[VehicleTrack],
        speed_results: Dict[int, VehicleSpeedResult],
        show_trajectories: bool = True,
        road_grid: Optional[RoadGrid] = None,
        show_road_grid: bool = True,
    ) -> np.ndarray:
        """
        Draws comprehensive annotations on the given frame.

        Returns:
            Annotated frame (BGR).
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 0. Imaginary Road Grid Overlay
        if show_road_grid and road_grid is not None:
            self._draw_road_grid(vis, road_grid, ego_speed)

        # 1. Top HUD: Ego Vehicle Telemetry
        self._draw_ego_hud(vis, ego_speed, timestamp, frame_idx)

        # 2. Draw tracks and speed tags
        for track in tracks:
            tid = track.track_id
            speed_res = speed_results.get(tid)

            # Choose box color based on direction
            color = self._get_color_for_direction(
                speed_res.direction if speed_res else VehicleDirection.UNKNOWN
            )

            # Draw bounding box
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw trajectory trail
            if show_trajectories and len(track.trajectory) > 1:
                pts = [
                    (int(tp.center[0]), int(tp.center[1]))
                    for tp in track.trajectory[-15:]
                ]
                for p_idx in range(len(pts) - 1):
                    alpha_thick = max(1, int(3 * (p_idx + 1) / len(pts)))
                    cv2.line(vis, pts[p_idx], pts[p_idx + 1], color, alpha_thick)

            # Build label text
            lbl_line1 = f"#{tid} {track.class_name.upper()}"
            if speed_res and speed_res.speed_kmh is not None:
                lbl_line2 = f"{speed_res.speed_kmh:.0f} km/h ({speed_res.confidence * 100:.0f}%)"
            else:
                lbl_line2 = "Speed: calc..."

            # Background label box
            self._draw_label_badge(vis, (x1, max(20, y1 - 32)), [lbl_line1, lbl_line2], color)

        return vis

    def _draw_road_grid(
        self,
        vis: np.ndarray,
        road_grid: RoadGrid,
        ego_speed: Optional[EgoSpeedResult],
    ) -> None:
        """
        Renders the imaginary perspective road grid directly onto the road surface.
        """
        overlay = vis.copy()
        grid_data = road_grid.generate_grid_overlay(vis.shape[:2], max_depth_lines=18)

        grid_color = (220, 200, 0)      # BGR Cyan/Teal
        accent_color = (0, 255, 200)    # Greenish cyan

        # Draw subtle shaded grid tiles in checkerboard fashion
        for idx, cell in enumerate(grid_data.get("grid_cells", [])):
            if idx % 2 == 0:
                cv2.fillPoly(overlay, [cell], (80, 60, 20))

        # Draw transversal lines
        for (u0, v0), (u1, v1) in grid_data.get("transversal_lines", []):
            cv2.line(overlay, (u0, v0), (u1, v1), grid_color, 1, cv2.LINE_AA)

        # Draw longitudinal lines
        for (u0, v0), (u1, v1) in grid_data.get("longitudinal_lines", []):
            cv2.line(overlay, (u0, v0), (u1, v1), accent_color, 1, cv2.LINE_AA)

        # Blend grid onto original frame with transparency
        cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

    def _draw_ego_hud(
        self,
        vis: np.ndarray,
        ego_speed: Optional[EgoSpeedResult],
        timestamp: float,
        frame_idx: int,
    ) -> None:
        """Draws top HUD banner with Ego speed and timestamp."""
        h, w = vis.shape[:2]
        banner_h = 55

        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        cv2.line(vis, (0, banner_h), (w, banner_h), (80, 80, 80), 1)

        # Timecode
        mins = int(timestamp // 60)
        secs = timestamp % 60
        time_str = f"{mins:02d}:{secs:05.2f} (F#{frame_idx:04d})"
        cv2.putText(vis, time_str, (20, 35), self.font, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

        # Ego Speed & Grid Telemetry
        if ego_speed and ego_speed.speed_kmh is not None:
            ego_text = f"EGO: {ego_speed.speed_kmh:.1f} km/h"
            cv2.putText(vis, ego_text, (w - 480, 35), self.font, 0.75, (50, 255, 50), 2, cv2.LINE_AA)

            if ego_speed.squares_passed_per_second is not None:
                grid_info = f"Grid: {ego_speed.squares_passed_per_second:.1f} sq/s ({ego_speed.grid_square_size_m:.1f}m)"
                cv2.putText(vis, grid_info, (w - 280, 35), self.font, 0.52, (0, 220, 255), 1, cv2.LINE_AA)

            conf_text = f"{ego_speed.confidence * 100:.0f}%"
            cv2.putText(vis, conf_text, (w - 70, 35), self.font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, "EGO SPEED: Estimating via Road Grid...", (w - 460, 35), self.font, 0.60, (100, 200, 255), 1, cv2.LINE_AA)

    def _draw_label_badge(
        self,
        vis: np.ndarray,
        top_left: Tuple[int, int],
        lines: List[str],
        border_color: Tuple[int, int, int],
    ) -> None:
        """Draws a semi-transparent badge with multiple text lines."""
        x, y = top_left
        pad = 4
        line_h = 16
        total_h = len(lines) * line_h + pad * 2
        max_w = max(int(len(l) * 9.5) for l in lines) + pad * 2

        x2 = min(vis.shape[1] - 1, x + max_w)
        y2 = min(vis.shape[0] - 1, y + total_h)

        overlay = vis.copy()
        cv2.rectangle(overlay, (x, y), (x2, y2), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
        cv2.rectangle(vis, (x, y), (x2, y2), border_color, 1)

        for i, text in enumerate(lines):
            ty = y + pad + (i + 1) * line_h - 3
            col = (255, 255, 255) if i == 0 else (0, 255, 255)
            cv2.putText(vis, text, (x + pad + 2, ty), self.font, 0.45, col, 1, cv2.LINE_AA)

    @staticmethod
    def _get_color_for_direction(direction: VehicleDirection) -> Tuple[int, int, int]:
        """Color codes: Same direction = Cyan, Opposite = Orange, Crossing = Purple, Stationary = Yellow."""
        if direction == VehicleDirection.SAME_DIRECTION:
            return (255, 200, 0)   # Cyan / Light Blue
        elif direction == VehicleDirection.OPPOSITE_DIRECTION:
            return (0, 140, 255)   # Orange
        elif direction == VehicleDirection.CROSSING:
            return (200, 50, 200)  # Magenta
        elif direction == VehicleDirection.STATIONARY:
            return (0, 220, 220)   # Yellow
        return (180, 180, 180)     # Gray default
