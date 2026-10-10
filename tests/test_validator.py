"""
Tests for video file validation module.
"""

from pathlib import Path
import pytest

from core.video.exceptions import (
    CorruptedVideoError,
    InvalidVideoFormatError,
    VideoNotFoundError,
)
from core.video.validator import validate_video_file


def test_validate_existing_valid_video(sample_video_30fps: Path):
    """Verifies that a valid synthetic MP4 passes validation."""
    result = validate_video_file(sample_video_30fps, raise_on_error=True)
    assert result.is_valid is True
    assert result.error_message is None
    assert result.extension == ".mp4"
    assert result.file_size_bytes > 0


def test_validate_nonexistent_file():
    """Verifies that a missing file raises VideoNotFoundError."""
    fake_path = Path("does_not_exist_xyz.mp4")
    with pytest.raises(VideoNotFoundError) as exc_info:
        validate_video_file(fake_path, raise_on_error=True)
    assert "not found" in str(exc_info.value).lower()


def test_validate_nonexistent_file_no_raise():
    """Verifies that a missing file returns is_valid=False when raise_on_error=False."""
    fake_path = Path("does_not_exist_xyz.mp4")
    result = validate_video_file(fake_path, raise_on_error=False)
    assert result.is_valid is False
    assert result.error_message is not None
    assert "not found" in result.error_message.lower()


def test_validate_directory_path(tmp_path: Path):
    """Verifies that passing a directory raises InvalidVideoFormatError."""
    with pytest.raises(InvalidVideoFormatError) as exc_info:
        validate_video_file(tmp_path, raise_on_error=True)
    assert "not a regular file" in str(exc_info.value).lower()


def test_validate_unsupported_extension(unsupported_format_file: Path):
    """Verifies that unsupported extensions (.txt) are rejected."""
    with pytest.raises(InvalidVideoFormatError) as exc_info:
        validate_video_file(unsupported_format_file, raise_on_error=True)
    assert "unsupported video format" in str(exc_info.value).lower()


def test_validate_empty_file(empty_video_file: Path):
    """Verifies that a 0-byte video file raises CorruptedVideoError."""
    with pytest.raises(CorruptedVideoError) as exc_info:
        validate_video_file(empty_video_file, raise_on_error=True)
    assert "empty" in str(exc_info.value).lower()


def test_validate_corrupted_file(corrupted_video_file: Path):
    """Verifies that a video file containing invalid stream bytes raises CorruptedVideoError."""
    with pytest.raises(CorruptedVideoError) as exc_info:
        validate_video_file(corrupted_video_file, raise_on_error=True)
    assert "initial frame" in str(exc_info.value).lower() or "container" in str(exc_info.value).lower()
