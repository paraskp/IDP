"""
Video metadata extraction module.
Provides structured information about video resolution, frame rate,
frame count, duration, and codec.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Union
import cv2

from core.video.exceptions import CorruptedVideoError
from core.video.validator import validate_video_file


def decode_fourcc(fourcc_val: float) -> str:
    """Decodes OpenCV numeric FOURCC property into a 4-character string."""
    try:
        val = int(fourcc_val)
        chars = [chr((val >> (8 * i)) & 0xFF) for i in range(4)]
        codec_str = "".join(chars).strip()
        return codec_str if codec_str else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def format_seconds_to_timestamp(seconds: float) -> str:
    """Formats seconds into HH:MM:SS.mmm string."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3600000
    remainder = total_ms % 3600000
    minutes = remainder // 60000
    remainder = remainder % 60000
    secs = remainder // 1000
    ms = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


@dataclass(frozen=True)
class VideoMetadata:
    """Structured metadata representing video attributes."""
    video_path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float
    codec: str
    aspect_ratio: float
    file_size_bytes: int

    def timestamp_for_frame(self, frame_index: int) -> float:
        """Calculates the time in seconds corresponding to a frame index."""
        if self.fps <= 0:
            return 0.0
        return max(0.0, frame_index / self.fps)

    def frame_for_timestamp(self, timestamp_sec: float) -> int:
        """Calculates the nearest frame index for a given timestamp in seconds."""
        if self.fps <= 0 or timestamp_sec <= 0:
            return 0
        return min(int(round(timestamp_sec * self.fps)), max(0, self.frame_count - 1))

    def format_timestamp(self, seconds: float) -> str:
        """Returns HH:MM:SS.mmm formatted timestamp."""
        return format_seconds_to_timestamp(seconds)

    def to_dict(self) -> Dict[str, Any]:
        """Converts metadata to a standard serializable dictionary."""
        return asdict(self)


def get_video_metadata(video_path: Union[str, Path]) -> VideoMetadata:
    """
    Extracts comprehensive metadata from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        VideoMetadata instance containing video properties.

    Raises:
        VideoNotFoundError, InvalidVideoFormatError, CorruptedVideoError
    """
    validation = validate_video_file(video_path, raise_on_error=True)
    path_str = str(validation.path)

    cap = cv2.VideoCapture(path_str)
    try:
        if not cap.isOpened():
            raise CorruptedVideoError(
                f"Failed to open video capture for metadata inspection: '{path_str}'",
                video_path=path_str
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_val = cap.get(cv2.CAP_PROP_FOURCC)
        codec = decode_fourcc(fourcc_val)

        if width <= 0 or height <= 0:
            raise CorruptedVideoError(
                f"Invalid video dimensions {width}x{height} for '{path_str}'",
                video_path=path_str
            )

        aspect_ratio = round(width / height, 4) if height > 0 else 0.0

        # Handle abnormal frame count or fps
        if fps <= 0.0:
            fps = 30.0  # Fallback default if unspecified by container

        duration = round(frame_count / fps, 4) if (frame_count > 0 and fps > 0) else 0.0

        return VideoMetadata(
            video_path=path_str,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            duration=duration,
            codec=codec,
            aspect_ratio=aspect_ratio,
            file_size_bytes=validation.file_size_bytes,
        )
    finally:
        cap.release()
