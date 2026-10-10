"""
Tests for Phase 13: End-to-End Pipeline Integration.
Runs the complete 10-stage computer-vision and geometric analysis workflow on synthetic footage.
"""

from pathlib import Path
import pytest

from core.detection.vehicle_detector import SyntheticVehicleDetector
from core.pipeline.orchestrator import PipelineOrchestrator
from core.video.reader import VideoReader


def test_end_to_end_pipeline_execution(sample_video_30fps: Path, tmp_path: Path):
    """
    Executes full pipeline across all phases and validates all generated artifacts.
    """
    orchestrator = PipelineOrchestrator(
        video_path=sample_video_30fps,
        detector=SyntheticVehicleDetector(),  # Fast deterministic detector for test isolation
        camera_height_meters=1.35,
    )

    progress_ticks = []

    def on_progress(frac: float, msg: str):
        progress_ticks.append(frac)

    result = orchestrator.run_full_analysis(
        output_dir=tmp_path / "run_output",
        generate_video=True,
        sample_step=2,
        progress_callback=on_progress,
    )

    assert result is not None
    assert len(progress_ticks) > 5

    # 1. Video Output Check
    assert result.output_video_path is not None
    assert result.output_video_path.exists()
    assert result.output_video_path.stat().st_size > 1000

    # Verify output video is readable
    with VideoReader(result.output_video_path) as v_read:
        assert v_read.metadata.frame_count > 0
        frame_0 = v_read.read_frame()
        assert frame_0 is not None

    # 2. CSV Output Check
    assert result.output_csv_path is not None
    assert result.output_csv_path.exists()
    csv_text = result.output_csv_path.read_text(encoding="utf-8")
    assert "timestamp,ego_speed,vehicle_id,vehicle_class" in csv_text

    # 3. JSON Output Check
    assert result.output_json_path is not None
    assert result.output_json_path.exists()

    # 4. Speed Plot Check
    assert result.output_plot_path is not None
    assert result.output_plot_path.exists()
    assert result.output_plot_path.stat().st_size > 1000

    # 5. Summary Metrics Check
    assert result.summary["duration_seconds"] > 0
    assert result.elapsed_time_sec > 0
