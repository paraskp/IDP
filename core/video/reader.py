"""
Video frame reader module.
Provides streaming frame iteration, timestamp calculation, random access seeking,
and graceful handling of missing or corrupted frames.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Generator, Optional, Union
import cv2
import numpy as np

from core.video.exceptions import CorruptedVideoError, FrameReadError, SeekError
from core.video.metadata import VideoMetadata, format_seconds_to_timestamp, get_video_metadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameData:
    """Represents a decoded video frame with corresponding temporal information."""
    index: int
    timestamp: float
    timestamp_formatted: str
    frame: np.ndarray

    def __repr__(self) -> str:
        h, w, c = self.frame.shape if self.frame is not None else (0, 0, 0)
        return (
            f"FrameData(index={self.index}, timestamp={self.timestamp:.3f}s, "
            f"time='{self.timestamp_formatted}', shape=({w}x{h}x{c}))"
        )


class VideoReader:
    """
    Reader for video streams supporting sequential iteration and random-access seeking.
    Ensures safe resource management via context protocol.
    """

    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path).resolve()
        self.metadata: VideoMetadata = get_video_metadata(self.video_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._current_index: int = 0
        self._open_capture()

    def _open_capture(self) -> None:
        """Initializes or resets the internal cv2.VideoCapture instance."""
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise CorruptedVideoError(
                f"Failed to open video capture stream for '{self.video_path}'",
                video_path=str(self.video_path),
            )
        self._current_index = 0

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Releases the underlying video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        """Checks if the video capture is active and open."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def current_index(self) -> int:
        """Returns the current 0-based frame index."""
        return self._current_index

    def read_frame(self) -> Optional[FrameData]:
        """
        Reads the next frame sequentially.

        Returns:
            FrameData instance, or None if end-of-stream is reached.
        """
        if not self.is_open:
            raise CorruptedVideoError("VideoReader is closed.", video_path=str(self.video_path))

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None

        idx = self._current_index
        timestamp = self.metadata.timestamp_for_frame(idx)
        formatted = format_seconds_to_timestamp(timestamp)
        self._current_index += 1

        return FrameData(
            index=idx,
            timestamp=timestamp,
            timestamp_formatted=formatted,
            frame=frame,
        )

    def seek_to_frame(self, frame_index: int) -> None:
        """
        Positions the capture stream at the specified frame index.

        Args:
            frame_index: Target 0-based frame index.

        Raises:
            SeekError: If index is out of bounds or seek fails.
        """
        if not self.is_open:
            raise CorruptedVideoError("VideoReader is closed.", video_path=str(self.video_path))

        if frame_index < 0 or (self.metadata.frame_count > 0 and frame_index >= self.metadata.frame_count):
            raise SeekError(
                f"Frame index {frame_index} out of bounds [0, {self.metadata.frame_count - 1}]",
                target=float(frame_index),
                video_path=str(self.video_path),
            )

        success = self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        if not success:
            raise SeekError(
                f"Failed to seek to frame {frame_index}",
                target=float(frame_index),
                video_path=str(self.video_path),
            )
        self._current_index = frame_index

    def get_frame(self, frame_index: int) -> FrameData:
        """
        Extracts a specific frame by index using seeking.

        Args:
            frame_index: Target 0-based frame index.

        Returns:
            FrameData for the requested frame.

        Raises:
            FrameReadError: If frame cannot be read after seeking.
        """
        self.seek_to_frame(frame_index)
        frame_data = self.read_frame()
        if frame_data is None:
            raise FrameReadError(
                f"Failed to read frame at index {frame_index}",
                frame_index=frame_index,
                video_path=str(self.video_path),
            )
        return frame_data

    def get_frame_at_timestamp(self, timestamp_seconds: float) -> FrameData:
        """
        Extracts the frame nearest to the given timestamp.

        Args:
            timestamp_seconds: Timestamp in seconds from video start.

        Returns:
            FrameData for the nearest frame.
        """
        target_frame = self.metadata.frame_for_timestamp(timestamp_seconds)
        return self.get_frame(target_frame)

    def __iter__(self) -> Generator[FrameData, None, None]:
        """Iterates through all remaining frames until end-of-stream."""
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame


def read_frames(
    video_path: Union[str, Path],
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    step: int = 1,
) -> Generator[FrameData, None, None]:
    """
    High-level generator yielding frames from a video file with automatic cleanup.

    Args:
        video_path: Path to video file.
        start_frame: Index to start reading from (default 0).
        max_frames: Maximum number of frames to yield (default None for all).
        step: Frame sampling interval (default 1 for every frame).

    Yields:
        FrameData objects.
    """
    if step < 1:
        raise ValueError(f"Step must be >= 1, got {step}")

    with VideoReader(video_path) as reader:
        if start_frame > 0:
            reader.seek_to_frame(start_frame)

        yielded_count = 0
        current_frame_idx = start_frame

        while True:
            if max_frames is not None and yielded_count >= max_frames:
                break

            frame_data = reader.read_frame()
            if frame_data is None:
                break

            if (frame_data.index - start_frame) % step == 0:
                yield frame_data
                yielded_count += 1
                if step > 1:
                    next_target = frame_data.index + step
                    if reader.metadata.frame_count > 0 and next_target >= reader.metadata.frame_count:
                        break
                    reader.seek_to_frame(next_target)
