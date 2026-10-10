"""
Pytest fixtures for Dashcam Speed Analyzer test suite.
"""

from pathlib import Path
import shutil
import tempfile
from typing import Generator
import pytest

from sample_data.generate_synthetic import generate_synthetic_dashcam_video


@pytest.fixture(scope="session")
def sample_video_30fps(tmp_path_factory) -> Path:
    """Session-scoped synthetic 30 FPS, 720p test video."""
    temp_dir = tmp_path_factory.mktemp("videos")
    video_path = temp_dir / "test_30fps.mp4"
    generate_synthetic_dashcam_video(
        output_path=video_path,
        duration_sec=2.0,
        fps=30.0,
        resolution=(1280, 720),
    )
    return video_path


@pytest.fixture(scope="session")
def sample_video_60fps(tmp_path_factory) -> Path:
    """Session-scoped synthetic 60 FPS, 360p test video."""
    temp_dir = tmp_path_factory.mktemp("videos")
    video_path = temp_dir / "test_60fps.mp4"
    generate_synthetic_dashcam_video(
        output_path=video_path,
        duration_sec=1.5,
        fps=60.0,
        resolution=(640, 360),
    )
    return video_path


@pytest.fixture
def empty_video_file(tmp_path) -> Path:
    """Creates a 0-byte video file."""
    f = tmp_path / "empty_video.mp4"
    f.touch()
    return f


@pytest.fixture
def corrupted_video_file(tmp_path) -> Path:
    """Creates an MP4 file with random binary garbage."""
    f = tmp_path / "corrupted_video.mp4"
    f.write_bytes(b"\x00\x00\x00\x20ftypmp42" + b"\xFF" * 1024)
    return f


@pytest.fixture
def unsupported_format_file(tmp_path) -> Path:
    """Creates an unsupported file format."""
    f = tmp_path / "invalid_format.txt"
    f.write_text("not a video")
    return f
