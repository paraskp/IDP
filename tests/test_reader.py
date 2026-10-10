"""
Tests for video reader and frame streaming module.
"""

from pathlib import Path
import numpy as np
import pytest

from core.video.exceptions import SeekError
from core.video.reader import FrameData, VideoReader, read_frames


def test_video_reader_sequential(sample_video_30fps: Path):
    """Verifies that VideoReader reads all frames sequentially."""
    with VideoReader(sample_video_30fps) as reader:
        frames = list(reader)
        assert len(frames) == 60
        for i, f in enumerate(frames):
            assert isinstance(f, FrameData)
            assert f.index == i
            assert f.frame.shape == (720, 1280, 3)
            assert pytest.approx(f.timestamp, abs=1e-3) == (i / 30.0)


def test_video_reader_context_closing(sample_video_30fps: Path):
    """Verifies resource cleanup upon exiting context manager."""
    reader = VideoReader(sample_video_30fps)
    assert reader.is_open is True
    reader.close()
    assert reader.is_open is False


def test_video_reader_seek(sample_video_30fps: Path):
    """Verifies seeking to specific frames."""
    with VideoReader(sample_video_30fps) as reader:
        frame_10 = reader.get_frame(10)
        assert frame_10.index == 10

        frame_45 = reader.get_frame(45)
        assert frame_45.index == 45

        # Seeking backward
        frame_0 = reader.get_frame(0)
        assert frame_0.index == 0


def test_video_reader_seek_out_of_bounds(sample_video_30fps: Path):
    """Verifies that seeking out of bounds raises SeekError."""
    with VideoReader(sample_video_30fps) as reader:
        with pytest.raises(SeekError):
            reader.seek_to_frame(-1)

        with pytest.raises(SeekError):
            reader.seek_to_frame(9999)


def test_video_reader_get_at_timestamp(sample_video_30fps: Path):
    """Verifies extraction at a specific timestamp."""
    with VideoReader(sample_video_30fps) as reader:
        f = reader.get_frame_at_timestamp(1.0)
        assert f.index == 30


def test_read_frames_generator_step_and_limit(sample_video_30fps: Path):
    """Verifies read_frames generator with step intervals and max_frames limit."""
    # Step = 2 on 60 frames -> 30 frames
    frames_step2 = list(read_frames(sample_video_30fps, step=2))
    assert len(frames_step2) == 30
    assert frames_step2[0].index == 0
    assert frames_step2[1].index == 2

    # max_frames limit
    frames_max5 = list(read_frames(sample_video_30fps, max_frames=5))
    assert len(frames_max5) == 5

    # start_frame offset
    frames_start10 = list(read_frames(sample_video_30fps, start_frame=10, max_frames=5))
    assert len(frames_start10) == 5
    assert frames_start10[0].index == 10
