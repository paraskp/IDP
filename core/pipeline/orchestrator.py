"""
Pipeline orchestrator coordinating all 10 computer vision and geometric stages.
Executes end-to-end analysis, generates annotated video, and exports structured CSV/JSON results.
Completely decoupled from the PySide6 desktop GUI.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Union
import cv2
import numpy as np

from core.calibration import CalibrationResult, calibrate_scene
from core.depth import MonocularDepthEstimator
from core.detection import (
    BaseVehicleDetector,
    ByteVehicleTracker,
    SyntheticVehicleDetector,
    YOLOVehicleDetector,
)
from core.detection.landmark_detector import HybridLandmarkDetector
from core.motion import (
    EgoSpeedEstimator,
    EgoSpeedResult,
    SparseOpticalFlowTracker,
    StationaryFeatureDetector,
    VisualOdometry,
)
from core.motion.landmark_speed import LandmarkSpeedTracker
from core.motion.optical_flow import TrackingFrameResult
from core.pipeline.stages import PipelineStage
from core.results import (
    VideoAnnotator,
    compute_summary_statistics,
    export_to_csv,
    export_to_json,
    generate_speed_plots,
)
from core.speed import (
    VehicleDirection,
    VehicleSpeedEstimator,
    VehicleSpeedResult,
)
from core.video import (
    FrameData,
    VideoMetadata,
    VideoReader,
    VideoWriter,
    get_video_metadata,
    read_frames,
    validate_video_file,
)


@dataclass
class AnalysisSessionResult:
    """Consolidated results of an end-to-end video analysis session."""
    video_path: Path
    metadata: VideoMetadata
    calibration: CalibrationResult
    ego_speed_records: List[Dict[str, Any]]
    vehicle_records: List[Dict[str, Any]]
    summary: Dict[str, Any]
    output_video_path: Optional[Path] = None
    output_csv_path: Optional[Path] = None
    output_json_path: Optional[Path] = None
    output_plot_path: Optional[Path] = None
    elapsed_time_sec: float = 0.0


class PipelineOrchestrator:
    """
    Coordinates end-to-end computer-vision analysis pipeline across all phases.
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        detector: Optional[BaseVehicleDetector] = None,
        user_fov_deg: Optional[float] = None,
        camera_height_meters: float = 1.35,
        square_size_meters: float = 1.0,
        is_user_calibrated_scale: bool = False,
        ego_only: bool = True,
    ):
        self.video_path = Path(video_path).resolve()
        self.user_fov = user_fov_deg
        self.camera_height = camera_height_meters
        self.square_size = square_size_meters
        self.is_user_calibrated = is_user_calibrated_scale
        self.ego_only = ego_only

        # Default to YOLO detector if full mode is active, otherwise dormant
        self.fallback_detector = SyntheticVehicleDetector() if not ego_only else None
        if not ego_only:
            if detector is not None:
                self.detector = detector
            else:
                yolo = YOLOVehicleDetector()
                self.detector = yolo if yolo._model is not None else self.fallback_detector
        else:
            self.detector = None

        self.current_stage: PipelineStage = PipelineStage.VIDEO_INPUT
        self.validation = None
        self.metadata: Optional[VideoMetadata] = None

    def run_stage_1_video_input(
        self, progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> VideoMetadata:
        """Executes Phase 1: Validates video container and extracts metadata."""
        if progress_callback:
            progress_callback(0.1, "Validating video file...")
        self.validation = validate_video_file(self.video_path, raise_on_error=True)

        if progress_callback:
            progress_callback(0.5, "Extracting video metadata...")
        self.metadata = get_video_metadata(self.video_path)

        if progress_callback:
            progress_callback(1.0, "Video input validation complete.")
        return self.metadata

    def run_full_analysis(
        self,
        output_dir: Union[str, Path] = "outputs",
        generate_video: bool = True,
        sample_step: int = 1,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> AnalysisSessionResult:
        """
        Executes end-to-end video analysis across all 10 core stages.

        Args:
            output_dir: Directory to save outputs (video, csv, json, plots).
            generate_video: Whether to render and save the annotated video.
            sample_step: Frame step interval (1 = all frames, 2 = every 2nd frame).
            progress_callback: Callable(fraction: float, status_text: str).

        Returns:
            AnalysisSessionResult object.
        """
        start_time = time.perf_counter()
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = self.video_path.stem
        out_video_path = out_dir / f"{stem}_annotated.mp4" if generate_video else None
        out_csv_path = out_dir / f"{stem}_results.csv"
        out_json_path = out_dir / f"{stem}_results.json"
        out_plot_path = out_dir / f"{stem}_speed_graph.png"

        # -------------------------------------------------------------
        # Stage 1: Video Input & Metadata
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback(0.02, "Validating video & extracting metadata...")

        validate_video_file(self.video_path, raise_on_error=True)
        meta = get_video_metadata(self.video_path)

        # -------------------------------------------------------------
        # Stage 2: Camera & Scene Calibration
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback(0.08, "Calibrating camera & scene geometry...")

        sample_frames = []
        with VideoReader(self.video_path) as reader:
            for _ in range(min(15, meta.frame_count)):
                fd = reader.read_frame()
                if fd is not None:
                    sample_frames.append(fd.frame)

        if not sample_frames:
            raise RuntimeError("Failed to read sample frames for calibration.")

        calib = calibrate_scene(
            sample_frames=sample_frames,
            user_fov_deg=self.user_fov,
            user_camera_height=self.camera_height,
        )

        # -------------------------------------------------------------
        # Initialize Core Engines
        # -------------------------------------------------------------
        feature_detector = StationaryFeatureDetector(max_corners=250)
        flow_tracker = SparseOpticalFlowTracker()
        vo = VisualOdometry(calib.intrinsics)
        ego_speed_est = EgoSpeedEstimator(
            camera_model=calib.camera_model,
            road_geometry=calib.road_geometry,
            camera_height_meters=self.camera_height,
            square_size_meters=self.square_size,
            is_user_calibrated_scale=self.is_user_calibrated,
        )
        vehicle_tracker = ByteVehicleTracker(high_score_thresh=0.4, min_hits_to_confirm=2)
        depth_est = MonocularDepthEstimator(
            camera_model=calib.camera_model,
            road_geometry=calib.road_geometry,
            camera_height_meters=self.camera_height,
        )
        speed_est = VehicleSpeedEstimator(depth_est)
        landmark_detector = HybridLandmarkDetector()
        landmark_tracker = LandmarkSpeedTracker()
        annotator = VideoAnnotator()

        writer = None
        if generate_video and out_video_path is not None:
            writer = VideoWriter(
                output_path=out_video_path,
                fps=meta.fps / sample_step,
                frame_size=(meta.width, meta.height),
                codec="mp4v",
            )

        ego_records: List[Dict[str, Any]] = []
        vehicle_records: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # Stream Processing: Stages 3 through 9
        # -------------------------------------------------------------
        prev_gray: Optional[np.ndarray] = None
        total_frames = meta.frame_count or 1
        processed_count = 0

        try:
            for frame_data in read_frames(self.video_path, step=sample_step):
                idx = frame_data.index
                t_sec = frame_data.timestamp
                curr_bgr = frame_data.frame
                curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

                # Progress callback
                frac = 0.10 + 0.75 * (min(idx + 1, total_frames) / total_frames)
                if progress_callback and (processed_count % 10 == 0):
                    progress_callback(
                        round(frac, 3),
                        f"Analyzing frame #{idx:04d} ({frac * 100:.0f}%)...",
                    )

                # 1 & 2. Surrounding Vehicle Detection & Tracking (Dormant when ego_only=True)
                active_tracks = []
                dynamic_boxes = []
                if not self.ego_only:
                    detections = self.detector.detect(curr_bgr) if self.detector else []
                    if not detections and self.fallback_detector is not None:
                        detections = self.fallback_detector.detect(curr_bgr)
                    dynamic_boxes = [(int(d.x1), int(d.y1), int(d.x2), int(d.y2)) for d in detections]
                    active_tracks = vehicle_tracker.update(detections, t_sec, idx)

                # 2.5 Detect Stationary Road Landmarks & Markings
                road_mask = ego_speed_est.road_grid.create_road_grid_mask((meta.height, meta.width))
                landmarks = landmark_detector.detect(curr_bgr, road_roi_mask=road_mask)
                _ = landmark_tracker.update_dash_frequency(landmarks, t_sec, 1.0 / meta.fps)

                # 3. Optical Flow & Stationary Feature Motion (Stages 3 & 4)
                ego_speed_res = None
                if prev_gray is not None:
                    # Detect features avoiding detected vehicles
                    pts_prev = feature_detector.detect(prev_gray, dynamic_boxes=dynamic_boxes)
                    tracking_res = flow_tracker.track(prev_gray, curr_gray, pts_prev)

                    # Filter optical flow matches strictly against detected stationary landmarks if available
                    anchored_matches = landmark_tracker.filter_features_to_landmarks(
                        tracking_res.inlier_matches, landmarks
                    )
                    if len(anchored_matches) >= 4:
                        tracking_res = TrackingFrameResult(
                            matches=anchored_matches,
                            inlier_matches=anchored_matches,
                            rejected_matches=tracking_res.rejected_matches,
                            mean_displacement=tracking_res.mean_displacement,
                            median_speed_px=tracking_res.median_speed_px,
                            tracking_ratio=tracking_res.tracking_ratio,
                            confidence=min(0.95, tracking_res.confidence + 0.1),
                        )

                    pose_res = vo.estimate_motion(tracking_res)

                    # 4. Ego Speed Estimation (Stage 5)
                    ego_speed_res = ego_speed_est.estimate_speed(
                        idx, t_sec, tracking_res, pose_res
                    )

                    ego_records.append({
                        "timestamp": t_sec,
                        "frame_index": idx,
                        "speed_kmh": ego_speed_res.speed_kmh,
                        "speed_mps": ego_speed_res.speed_mps,
                        "acceleration_mps2": ego_speed_res.acceleration_mps2,
                        "confidence": ego_speed_res.confidence,
                        "scale_source": ego_speed_res.scale_source,
                        "squares_passed_per_second": ego_speed_res.squares_passed_per_second,
                        "squares_passed_per_frame": ego_speed_res.squares_passed_per_frame,
                        "grid_square_size_m": ego_speed_res.grid_square_size_m,
                    })

                # 5. Surrounding Vehicle Speed Estimation (Dormant when ego_only=True)
                speed_results: Dict[int, VehicleSpeedResult] = {}
                if not self.ego_only:
                    for track in active_tracks:
                        vres = speed_est.estimate_vehicle_speed(track, t_sec, ego_speed_res)
                        speed_results[track.track_id] = vres

                        vehicle_records.append({
                            "timestamp": t_sec,
                            "frame_index": idx,
                            "ego_speed": ego_speed_res.speed_kmh if ego_speed_res else None,
                            "vehicle_id": track.track_id,
                            "vehicle_class": track.class_name,
                            "vehicle_speed": vres.speed_kmh,
                            "relative_speed": vres.relative_speed_kmh,
                            "direction": vres.direction.value,
                            "depth_meters": vres.depth_meters,
                            "lateral_meters": vres.lateral_meters,
                            "confidence": vres.confidence,
                            "scale_source": vres.scale_source,
                        })

                # 6. Render annotations and write frame (Stage 10)
                if writer is not None:
                    annotated = annotator.annotate_frame(
                        curr_bgr,
                        idx,
                        t_sec,
                        ego_speed_res,
                        active_tracks,
                        speed_results,
                        road_grid=ego_speed_est.road_grid,
                        show_road_grid=True,
                        landmarks=landmarks,
                        show_landmarks=True,
                        ego_only=self.ego_only,
                    )
                    writer.write_frame(annotated)

                prev_gray = curr_gray
                processed_count += 1

        finally:
            if writer is not None:
                writer.close()

        # -------------------------------------------------------------
        # Stage 10 Exports: CSV, JSON, Plots, Summary
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback(0.92, "Exporting results, CSV, and summary reports...")

        # Export CSV
        export_to_csv(vehicle_records, out_csv_path)

        # Summary statistics
        summary = compute_summary_statistics(meta.duration, ego_records, vehicle_records)

        # Export JSON
        full_json_data = {
            "file": {
                "input_video": str(self.video_path),
                "duration_seconds": meta.duration,
                "fps": meta.fps,
                "resolution": f"{meta.width}x{meta.height}",
            },
            "calibration": calib.to_dict(),
            "summary_statistics": summary,
            "ego_speed_history": ego_records,
            "vehicle_tracks_history": vehicle_records,
        }
        export_to_json(full_json_data, out_json_path)

        # Generate Speed Graphs
        generate_speed_plots(ego_records, vehicle_records, out_plot_path)

        elapsed = time.perf_counter() - start_time

        if progress_callback:
            progress_callback(1.0, f"Analysis complete in {elapsed:.1f}s.")

        return AnalysisSessionResult(
            video_path=self.video_path,
            metadata=meta,
            calibration=calib,
            ego_speed_records=ego_records,
            vehicle_records=vehicle_records,
            summary=summary,
            output_video_path=out_video_path,
            output_csv_path=out_csv_path,
            output_json_path=out_json_path,
            output_plot_path=out_plot_path,
            elapsed_time_sec=round(elapsed, 2),
        )
