"""
PySide6 asynchronous worker thread for video analysis.
Ensures CPU/GPU-intensive computer vision operations do not freeze the Qt GUI thread.
"""

from pathlib import Path
from typing import Optional

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:
    # Minimal fallback for headless test environments
    class QThread:
        pass
    def Signal(*args, **kwargs):
        class DummySignal:
            def emit(self, *a, **k): pass
            def connect(self, fn): pass
        return DummySignal()

from core.pipeline.orchestrator import AnalysisSessionResult, PipelineOrchestrator


class AnalysisWorker(QThread):
    """Worker thread running the 10-stage analysis pipeline asynchronously."""

    sig_progress = Signal(float, str)
    sig_finished = Signal(object)
    sig_error = Signal(str)

    def __init__(
        self,
        video_path: Path,
        output_dir: Path,
        user_fov: Optional[float] = None,
        camera_height: float = 1.35,
        square_size_meters: float = 1.0,
        generate_video: bool = True,
        sample_step: int = 1,
    ):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.user_fov = user_fov
        self.camera_height = camera_height
        self.square_size = square_size_meters
        self.generate_video = generate_video
        self.sample_step = sample_step
        self._is_cancelled = False

    def cancel(self) -> None:
        """Flags the worker thread for cancellation."""
        self._is_cancelled = True

    def run(self) -> None:
        """Thread entrypoint."""
        try:
            orchestrator = PipelineOrchestrator(
                video_path=self.video_path,
                user_fov_deg=self.user_fov,
                camera_height_meters=self.camera_height,
                square_size_meters=self.square_size,
                is_user_calibrated_scale=(self.user_fov is not None),
            )

            def on_progress(fraction: float, message: str):
                if not self._is_cancelled:
                    self.sig_progress.emit(fraction, message)

            result = orchestrator.run_full_analysis(
                output_dir=self.output_dir,
                generate_video=self.generate_video,
                sample_step=self.sample_step,
                progress_callback=on_progress,
            )

            if not self._is_cancelled:
                self.sig_finished.emit(result)
        except Exception as e:
            if not self._is_cancelled:
                self.sig_error.emit(str(e))
