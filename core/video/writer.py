"""
Safe video writer utility module.
Provides context-managed writing of video streams and annotated video outputs.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np

from core.video.exceptions import VideoWriteError


class VideoWriter:
    """
    Context-managed video writer wrapping OpenCV's cv2.VideoWriter.
    Guarantees clean finalization and resource release on exit.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = "mp4v",
        is_color: bool = True,
    ):
        """
        Initializes VideoWriter.

        Args:
            output_path: Path where output video should be written.
            fps: Frame rate of the output video.
            frame_size: (width, height) tuple in pixels.
            codec: 4-character codec string (default 'mp4v').
            is_color: Whether frames are color (BGR).
        """
        self.output_path = Path(output_path).resolve()
        self.fps = float(fps)
        self.width, self.height = frame_size
        self.codec = codec
        self.is_color = is_color
        self._writer: Optional[cv2.VideoWriter] = None
        self._frames_written = 0

        self._init_writer()

    def _init_writer(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.width, self.height),
            self.is_color,
        )
        if not self._writer.isOpened():
            raise VideoWriteError(
                f"Failed to initialize cv2.VideoWriter for '{self.output_path}' "
                f"with codec='{self.codec}', fps={self.fps}, size=({self.width}x{self.height})",
                video_path=str(self.output_path),
            )

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Writes a single frame to the video stream.

        Args:
            frame: Numpy array representing the frame.

        Raises:
            VideoWriteError: If frame shape does not match or writer is closed.
        """
        if self._writer is None or not self._writer.isOpened():
            raise VideoWriteError("VideoWriter is closed.", video_path=str(self.output_path))

        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            raise VideoWriteError(
                f"Frame size ({w}x{h}) does not match configured size ({self.width}x{self.height})",
                video_path=str(self.output_path),
            )

        self._writer.write(frame)
        self._frames_written += 1

    @property
    def frames_written(self) -> int:
        """Returns total number of frames successfully written."""
        return self._frames_written

    def close(self) -> None:
        """Flushes and releases the video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
