"""
Core video ingestion, validation, metadata extraction, reading, and writing subsystem.
"""

from core.video.exceptions import (
    CorruptedVideoError,
    FrameReadError,
    InvalidVideoFormatError,
    SeekError,
    VideoError,
    VideoNotFoundError,
    VideoWriteError,
)
from core.video.metadata import (
    VideoMetadata,
    format_seconds_to_timestamp,
    get_video_metadata,
)
from core.video.reader import (
    FrameData,
    VideoReader,
    read_frames,
)
from core.video.validator import (
    SUPPORTED_EXTENSIONS,
    ValidationResult,
    validate_video_file,
)
from core.video.writer import VideoWriter

__all__ = [
    "VideoError",
    "VideoNotFoundError",
    "InvalidVideoFormatError",
    "CorruptedVideoError",
    "FrameReadError",
    "SeekError",
    "VideoWriteError",
    "VideoMetadata",
    "format_seconds_to_timestamp",
    "get_video_metadata",
    "FrameData",
    "VideoReader",
    "read_frames",
    "SUPPORTED_EXTENSIONS",
    "ValidationResult",
    "validate_video_file",
    "VideoWriter",
]
