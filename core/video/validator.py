"""
Video file validation module.
Verifies file existence, extension, container integrity, and decoder readability.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple, Union
import cv2

from core.video.exceptions import (
    CorruptedVideoError,
    InvalidVideoFormatError,
    VideoNotFoundError,
)

SUPPORTED_EXTENSIONS: Set[str] = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class ValidationResult:
    """Represents the result of a video validation check."""
    is_valid: bool
    path: Path
    extension: str
    file_size_bytes: int
    error_message: Optional[str] = None


def validate_video_file(
    video_path: Union[str, Path],
    raise_on_error: bool = True
) -> ValidationResult:
    """
    Validates that the file exists, has a supported format, non-zero size,
    and can be opened and decoded by OpenCV.

    Args:
        video_path: Path to the video file.
        raise_on_error: If True, raises appropriate VideoError subclasses on failure.
                        If False, returns a ValidationResult with is_valid=False.

    Returns:
        ValidationResult with validation status and diagnostics.

    Raises:
        VideoNotFoundError: If the file does not exist.
        InvalidVideoFormatError: If the extension is unsupported or file is empty.
        CorruptedVideoError: If OpenCV fails to open or read the stream.
    """
    path = Path(video_path).resolve()

    if not path.exists():
        err = f"Video file not found at '{path}'"
        if raise_on_error:
            raise VideoNotFoundError(err, video_path=str(path))
        return ValidationResult(
            is_valid=False,
            path=path,
            extension=path.suffix.lower(),
            file_size_bytes=0,
            error_message=err,
        )

    if not path.is_file():
        err = f"Path '{path}' is not a regular file."
        if raise_on_error:
            raise InvalidVideoFormatError(err, video_path=str(path))
        return ValidationResult(
            is_valid=False,
            path=path,
            extension=path.suffix.lower(),
            file_size_bytes=0,
            error_message=err,
        )

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        err = (
            f"Unsupported video format '{ext}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        if raise_on_error:
            raise InvalidVideoFormatError(err, video_path=str(path))
        return ValidationResult(
            is_valid=False,
            path=path,
            extension=ext,
            file_size_bytes=path.stat().st_size,
            error_message=err,
        )

    file_size = path.stat().st_size
    if file_size == 0:
        err = f"Video file '{path}' is empty (0 bytes)."
        if raise_on_error:
            raise CorruptedVideoError(err, video_path=str(path))
        return ValidationResult(
            is_valid=False,
            path=path,
            extension=ext,
            file_size_bytes=0,
            error_message=err,
        )

    # Verify OpenCV decoding capability
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        err = f"OpenCV failed to open video container at '{path}'."
        if raise_on_error:
            raise CorruptedVideoError(err, video_path=str(path))
        return ValidationResult(
            is_valid=False,
            path=path,
            extension=ext,
            file_size_bytes=file_size,
            error_message=err,
        )

    # Probe first frame to verify stream is not corrupt
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None or frame.size == 0:
        err = f"Unable to read initial frame from '{path}'. Video stream may be corrupt."
        if raise_on_error:
            raise CorruptedVideoError(err, video_path=str(path))
        return ValidationResult(
            is_valid=False,
            path=path,
            extension=ext,
            file_size_bytes=file_size,
            error_message=err,
        )

    return ValidationResult(
        is_valid=True,
        path=path,
        extension=ext,
        file_size_bytes=file_size,
        error_message=None,
    )
