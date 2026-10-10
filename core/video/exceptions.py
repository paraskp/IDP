"""
Domain-specific exceptions for the Dashcam Speed Analyzer video subsystem.
"""

from typing import Optional


class VideoError(Exception):
    """Base exception for all video processing errors."""

    def __init__(self, message: str, video_path: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.video_path = video_path

    def __str__(self) -> str:
        if self.video_path:
            return f"{self.message} [Path: {self.video_path}]"
        return self.message


class VideoNotFoundError(VideoError, FileNotFoundError):
    """Raised when the specified video file does not exist."""
    pass


class InvalidVideoFormatError(VideoError, ValueError):
    """Raised when the video file format/extension is unsupported or the container is invalid."""
    pass


class CorruptedVideoError(VideoError, RuntimeError):
    """Raised when the video file is corrupted or contains unreadable headers/streams."""
    pass


class FrameReadError(VideoError, RuntimeError):
    """Raised when a specific frame cannot be read or decoded from the video stream."""

    def __init__(self, message: str, frame_index: Optional[int] = None, video_path: Optional[str] = None):
        super().__init__(message, video_path)
        self.frame_index = frame_index

    def __str__(self) -> str:
        base = super().__str__()
        if self.frame_index is not None:
            return f"{base} (Frame index: {self.frame_index})"
        return base


class SeekError(VideoError, RuntimeError):
    """Raised when seeking to a specific frame or timestamp fails."""

    def __init__(self, message: str, target: Optional[float] = None, video_path: Optional[str] = None):
        super().__init__(message, video_path)
        self.target = target

    def __str__(self) -> str:
        base = super().__str__()
        if self.target is not None:
            return f"{base} (Target: {self.target})"
        return base


class VideoWriteError(VideoError, RuntimeError):
    """Raised when an error occurs while writing an output video."""
    pass
