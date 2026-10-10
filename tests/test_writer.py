"""
Tests for video writer module.
"""

from pathlib import Path
import numpy as np
import pytest

from core.video.exceptions import VideoWriteError
from core.video.reader import VideoReader
from core.video.writer import VideoWriter


def test_video_writer_roundtrip(tmp_path: Path):
    """Verifies creating a video, writing frames, and reading them back."""
    out_path = tmp_path / "roundtrip.mp4"
    w, h = 320, 240
    fps = 25.0
    num_frames = 20

    with VideoWriter(out_path, fps=fps, frame_size=(w, h), codec="mp4v") as writer:
        for i in range(num_frames):
            frame = np.full((h, w, 3), i * 10, dtype=np.uint8)
            writer.write_frame(frame)
        assert writer.frames_written == num_frames

    assert out_path.exists()

    with VideoReader(out_path) as reader:
        assert reader.metadata.width == w
        assert reader.metadata.height == h
        assert pytest.approx(reader.metadata.fps, rel=1e-2) == fps
        assert reader.metadata.frame_count == num_frames


def test_video_writer_mismatched_dimensions(tmp_path: Path):
    """Verifies that writing a frame with incorrect shape raises VideoWriteError."""
    out_path = tmp_path / "mismatch.mp4"
    with VideoWriter(out_path, fps=30.0, frame_size=(640, 480)) as writer:
        bad_frame = np.zeros((300, 300, 3), dtype=np.uint8)
        with pytest.raises(VideoWriteError) as exc_info:
            writer.write_frame(bad_frame)
        assert "does not match" in str(exc_info.value).lower()
