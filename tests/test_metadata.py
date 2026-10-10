"""
Tests for video metadata extraction module.
"""

from pathlib import Path
import pytest

from core.video.metadata import (
    VideoMetadata,
    format_seconds_to_timestamp,
    get_video_metadata,
)


def test_metadata_extraction_30fps(sample_video_30fps: Path):
    """Verifies metadata accuracy against 30 FPS, 720p ground truth."""
    meta = get_video_metadata(sample_video_30fps)
    assert isinstance(meta, VideoMetadata)
    assert meta.width == 1280
    assert meta.height == 720
    assert pytest.approx(meta.fps, rel=1e-2) == 30.0
    assert meta.frame_count == 60  # 2.0s * 30fps
    assert pytest.approx(meta.duration, rel=1e-2) == 2.0
    assert meta.aspect_ratio == round(1280 / 720, 4)
    assert meta.file_size_bytes > 0
    assert len(meta.codec) > 0


def test_metadata_extraction_60fps(sample_video_60fps: Path):
    """Verifies metadata accuracy against 60 FPS, 360p ground truth."""
    meta = get_video_metadata(sample_video_60fps)
    assert meta.width == 640
    assert meta.height == 360
    assert pytest.approx(meta.fps, rel=1e-2) == 60.0
    assert meta.frame_count == 90  # 1.5s * 60fps
    assert pytest.approx(meta.duration, rel=1e-2) == 1.5


def test_timestamp_for_frame(sample_video_30fps: Path):
    """Verifies timestamp calculations for frame indices."""
    meta = get_video_metadata(sample_video_30fps)
    assert meta.timestamp_for_frame(0) == 0.0
    assert pytest.approx(meta.timestamp_for_frame(30), rel=1e-3) == 1.0
    assert pytest.approx(meta.timestamp_for_frame(60), rel=1e-3) == 2.0


def test_frame_for_timestamp(sample_video_30fps: Path):
    """Verifies frame index calculation from timestamps."""
    meta = get_video_metadata(sample_video_30fps)
    assert meta.frame_for_timestamp(0.0) == 0
    assert meta.frame_for_timestamp(1.0) == 30
    assert meta.frame_for_timestamp(2.0) == 59  # Clamped to frame_count - 1


def test_format_seconds_to_timestamp():
    """Verifies timecode formatting utility."""
    assert format_seconds_to_timestamp(0.0) == "00:00:00.000"
    assert format_seconds_to_timestamp(1.5) == "00:00:01.500"
    assert format_seconds_to_timestamp(65.123) == "00:01:05.123"
    assert format_seconds_to_timestamp(3661.050) == "01:01:01.050"


def test_metadata_to_dict(sample_video_30fps: Path):
    """Verifies dictionary serialization."""
    meta = get_video_metadata(sample_video_30fps)
    d = meta.to_dict()
    assert isinstance(d, dict)
    assert d["width"] == 1280
    assert d["height"] == 720
    assert "duration" in d
    assert "fps" in d
