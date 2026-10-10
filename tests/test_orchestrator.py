"""
Tests for Phase 1 pipeline orchestration.
"""

from pathlib import Path

from core.pipeline.orchestrator import PipelineOrchestrator
from core.pipeline.stages import PipelineStage


def test_orchestrator_stage_1(sample_video_30fps: Path):
    """Verifies that the orchestrator executes Stage 1 video input correctly."""
    orchestrator = PipelineOrchestrator(sample_video_30fps)
    assert orchestrator.current_stage == PipelineStage.VIDEO_INPUT

    progress_events = []

    def on_progress(fraction: float, msg: str):
        progress_events.append((fraction, msg))

    meta = orchestrator.run_stage_1_video_input(progress_callback=on_progress)

    assert meta is not None
    assert meta.width == 1280
    assert meta.height == 720
    assert len(progress_events) == 3
    assert progress_events[-1][0] == 1.0
    assert orchestrator.validation is not None
    assert orchestrator.validation.is_valid is True
