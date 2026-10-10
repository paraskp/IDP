"""
Multi-object vehicle tracking module (Phase 7).
Implements ByteTrack association logic with persistent track IDs,
trajectory history, and occlusion handling.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.detection.vehicle_detector import Detection


class TrackState(Enum):
    NEW = auto()
    TRACKED = auto()
    LOST = auto()
    REMOVED = auto()


def compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


@dataclass
class TrajectoryPoint:
    timestamp: float
    frame_idx: int
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    bottom_center: Tuple[float, float]


class VehicleTrack:
    """Represents a single tracked vehicle with persistent identity and trajectory."""

    def __init__(self, track_id: int, initial_detection: Detection, timestamp: float, frame_idx: int):
        self.track_id = track_id
        self.class_name = initial_detection.class_name
        self.bbox = initial_detection.bbox
        self.confidence = initial_detection.confidence
        self.state = TrackState.NEW
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        # Velocity in pixels/sec: (vx, vy)
        self.velocity_px: Tuple[float, float] = (0.0, 0.0)
        self.trajectory: List[TrajectoryPoint] = [
            TrajectoryPoint(
                timestamp=timestamp,
                frame_idx=frame_idx,
                bbox=self.bbox,
                center=initial_detection.center,
                bottom_center=initial_detection.bottom_center,
            )
        ]

    def predict(self, dt: float) -> Tuple[float, float, float, float]:
        """Predicts expected bounding box based on linear velocity."""
        vx, vy = self.velocity_px
        dx = vx * dt
        dy = vy * dt
        x1, y1, x2, y2 = self.bbox
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def update(self, detection: Detection, timestamp: float, frame_idx: int) -> None:
        """Updates track with a matched detection."""
        prev_center = self.trajectory[-1].center if self.trajectory else detection.center
        prev_time = self.trajectory[-1].timestamp if self.trajectory else timestamp
        dt = max(1e-4, timestamp - prev_time)

        curr_center = detection.center
        vx = (curr_center[0] - prev_center[0]) / dt
        vy = (curr_center[1] - prev_center[1]) / dt

        # Exponential moving average for velocity
        self.velocity_px = (
            0.6 * self.velocity_px[0] + 0.4 * vx,
            0.6 * self.velocity_px[1] + 0.4 * vy,
        )

        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.state = TrackState.TRACKED

        self.trajectory.append(
            TrajectoryPoint(
                timestamp=timestamp,
                frame_idx=frame_idx,
                bbox=self.bbox,
                center=detection.center,
                bottom_center=detection.bottom_center,
            )
        )

    def mark_missed(self) -> None:
        """Marks track as missed in the current frame."""
        self.time_since_update += 1
        self.age += 1
        if self.time_since_update > 2:
            self.state = TrackState.LOST


class ByteVehicleTracker:
    """
    ByteTrack-inspired multi-object tracker for dashcam vehicle tracking.
    """

    def __init__(
        self,
        high_score_thresh: float = 0.5,
        match_iou_thresh: float = 0.25,
        max_age: int = 15,
        min_hits_to_confirm: int = 2,
    ):
        self.high_score_thresh = high_score_thresh
        self.match_iou_thresh = match_iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits_to_confirm

        self._next_id = 1
        self.tracks: List[VehicleTrack] = []

    def update(
        self,
        detections: List[Detection],
        timestamp: float,
        frame_idx: int,
        dt: float = 1.0 / 30.0,
    ) -> List[VehicleTrack]:
        """
        Updates tracker with detections from current frame.

        Returns:
            List of currently active, confirmed tracks.
        """
        # Separate detections into high-confidence and low-confidence
        det_high = [d for d in detections if d.confidence >= self.high_score_thresh]
        det_low = [d for d in detections if d.confidence < self.high_score_thresh]

        # Step 1: Associate high-score detections with existing active/lost tracks
        unmatched_tracks: List[VehicleTrack] = []
        unmatched_dets_high: List[Detection] = []

        active_pool = [t for t in self.tracks if t.state != TrackState.REMOVED]
        matched_tracks, unmatched_tracks, unmatched_dets_high = self._associate(
            active_pool, det_high, dt
        )

        for track, det in matched_tracks:
            track.update(det, timestamp, frame_idx)

        # Step 2: Associate remaining tracks with low-score detections (recovers occluded vehicles)
        matched_tracks_low, remaining_unmatched_tracks, _ = self._associate(
            unmatched_tracks, det_low, dt
        )

        for track, det in matched_tracks_low:
            track.update(det, timestamp, frame_idx)

        # Step 3: Initialize new tracks for unmatched high-confidence detections
        for det in unmatched_dets_high:
            new_track = VehicleTrack(self._next_id, det, timestamp, frame_idx)
            self._next_id += 1
            self.tracks.append(new_track)

        # Step 4: Mark remaining unmatched tracks as missed and remove stale tracks
        for track in remaining_unmatched_tracks:
            track.mark_missed()
            if track.time_since_update > self.max_age:
                track.state = TrackState.REMOVED

        # Filter confirmed tracks (active and enough hits)
        confirmed_tracks = [
            t for t in self.tracks
            if t.state == TrackState.TRACKED and t.hits >= self.min_hits and t.time_since_update == 0
        ]

        return confirmed_tracks

    def _associate(
        self,
        tracks: List[VehicleTrack],
        detections: List[Detection],
        dt: float,
    ) -> Tuple[List[Tuple[VehicleTrack, Detection]], List[VehicleTrack], List[Detection]]:
        """Greedy IoU bipartite association between tracks and detections."""
        if not tracks or not detections:
            return [], list(tracks), list(detections)

        # Cost matrix: IoU between predicted track box and detection box
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            pred_box = track.predict(dt)
            for j, det in enumerate(detections):
                iou_matrix[i, j] = compute_iou(pred_box, det.bbox)

        matched_pairs: List[Tuple[VehicleTrack, Detection]] = []
        unmatched_tracks: List[VehicleTrack] = []
        unmatched_dets: List[Detection] = []

        matched_t = set()
        matched_d = set()

        # Greedy matching by highest IoU
        while True:
            max_val = -1.0
            best_t, best_d = -1, -1
            for i in range(len(tracks)):
                if i in matched_t:
                    continue
                for j in range(len(detections)):
                    if j in matched_d:
                        continue
                    if iou_matrix[i, j] > max_val:
                        max_val = iou_matrix[i, j]
                        best_t, best_d = i, j

            if max_val < self.match_iou_thresh or best_t == -1:
                break

            matched_pairs.append((tracks[best_t], detections[best_d]))
            matched_t.add(best_t)
            matched_d.add(best_d)

        for i, t in enumerate(tracks):
            if i not in matched_t:
                unmatched_tracks.append(t)

        for j, d in enumerate(detections):
            if j not in matched_d:
                unmatched_dets.append(d)

        return matched_pairs, unmatched_tracks, unmatched_dets
