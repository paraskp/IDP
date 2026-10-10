"""
Tests for synthetic dashcam video generation.
"""

from pathlib import Path
import pytest

from core.video.metadata import get_video_metadata
from core.video.reader import VideoReader
from sample_data.generate_synthetic import generate_synthetic_dashcam_video


def test_synthetic_generator_custom_params(tmp_path: Path):
    """Verifies that synthetic generator honours custom duration, fps, and resolution."""
    custom_video = tmp_path / "custom_synthetic.mp4"
    generate_synthetic_dashcam_video(
        output_path=custom_video,
        duration_sec=1.0,
        fps=20.0,
        resolution=(800, 450),
    )

    assert custom_video.exists()
    meta = get_video_metadata(custom_video)
    assert meta.width == 800
    assert meta.height == 450
    assert pytest.approx(meta.fps, rel=1e-2) == 20.0
    assert meta.frame_count == 20

    with VideoReader(custom_video) as reader:
        first_frame = reader.read_frame()
        assert first_frame is not None
        assert first_frame.frame.shape == (450, 800, 3)
