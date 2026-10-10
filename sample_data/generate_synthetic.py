"""
Synthetic dashcam video generator for testing and validation.
Creates deterministic video sequences with known frame rates, resolutions,
temporal timestamps, and synthetic road scenes with moving markings.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from core.video.metadata import format_seconds_to_timestamp
from core.video.writer import VideoWriter


def generate_synthetic_dashcam_video(
    output_path: Path | str,
    duration_sec: float = 3.0,
    fps: float = 30.0,
    resolution: Tuple[int, int] = (1280, 720),
    codec: str = "mp4v",
    simulated_speed_kmh: float = 60.0,
) -> Path:
    """
    Generates a synthetic dashcam video with moving road markings, simulated
    other vehicles, and burnt-in diagnostic telemetry.

    Args:
        output_path: Target path for the synthetic video.
        duration_sec: Duration in seconds.
        fps: Target frames per second.
        resolution: (width, height) tuple in pixels.
        codec: OpenCV 4-character video codec.
        simulated_speed_kmh: Burn-in simulated ego speed.

    Returns:
        Path to the generated video file.
    """
    out_file = Path(output_path).resolve()
    w, h = resolution
    total_frames = int(round(duration_sec * fps))

    # Road geometry constants in image plane
    horizon_y = int(h * 0.45)
    vp_x = int(w * 0.5)
    vp_y = horizon_y
    road_bottom_left = int(w * 0.1)
    road_bottom_right = int(w * 0.9)

    with VideoWriter(out_file, fps=fps, frame_size=(w, h), codec=codec) as writer:
        for i in range(total_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)

            # Sky (top gradient)
            frame[0:horizon_y, :] = (220, 180, 140)  # Light dusty sky (BGR)

            # Ground / shoulders
            frame[horizon_y:h, :] = (45, 90, 45)  # Dark green grass/shoulder

            # Asphalt road polygon
            road_pts = np.array(
                [
                    [vp_x - 30, vp_y],
                    [vp_x + 30, vp_y],
                    [road_bottom_right, h],
                    [road_bottom_left, h],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(frame, [road_pts], (60, 60, 60))

            # Outer road boundaries (solid white lines)
            cv2.line(frame, (vp_x - 30, vp_y), (road_bottom_left, h), (255, 255, 255), 3)
            cv2.line(frame, (vp_x + 30, vp_y), (road_bottom_right, h), (255, 255, 255), 3)

            # Animated dashed center lane markings moving downwards
            cycle_phase = (i * 0.12) % 1.0
            num_dashes = 6
            for d in range(num_dashes):
                t1 = ((d + cycle_phase) / num_dashes) ** 1.8
                t2 = ((d + 0.5 + cycle_phase) / num_dashes) ** 1.8

                if 0.05 <= t1 <= 1.0:
                    y1 = int(vp_y + t1 * (h - vp_y))
                    y2 = int(vp_y + min(1.0, t2) * (h - vp_y))
                    x1 = vp_x
                    x2 = vp_x
                    thick = max(2, int(6 * t1))
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 220, 255), thick)  # Yellow center dashes

            # Simulated other vehicle in right lane moving ahead
            lead_car_scale = 0.5 + 0.15 * np.sin(i / 15.0)
            car_w = int(140 * lead_car_scale)
            car_h = int(90 * lead_car_scale)
            car_center_x = int(w * 0.65)
            car_center_y = int(horizon_y + (h - horizon_y) * (0.35 + 0.05 * lead_car_scale))

            car_x1 = car_center_x - car_w // 2
            car_y1 = car_center_y - car_h // 2
            car_x2 = car_center_x + car_w // 2
            car_y2 = car_center_y + car_h // 2

            # Car body (red car)
            cv2.rectangle(frame, (car_x1, car_y1), (car_x2, car_y2), (30, 30, 180), -1)
            # Rear window
            win_m = max(3, int(10 * lead_car_scale))
            cv2.rectangle(
                frame,
                (car_x1 + win_m, car_y1 + win_m),
                (car_x2 - win_m, car_y1 + car_h // 2),
                (40, 40, 40),
                -1,
            )
            # Taillights
            light_w = max(4, int(18 * lead_car_scale))
            light_h = max(3, int(10 * lead_car_scale))
            cv2.rectangle(
                frame,
                (car_x1 + 4, car_y2 - light_h - 4),
                (car_x1 + 4 + light_w, car_y2 - 4),
                (0, 0, 230),
                -1,
            )
            cv2.rectangle(
                frame,
                (car_x2 - 4 - light_w, car_y2 - light_h - 4),
                (car_x2 - 4, car_y2 - 4),
                (0, 0, 230),
                -1,
            )

            # Burnt-in Telemetry HUD (top left)
            timestamp_sec = i / fps
            time_str = format_seconds_to_timestamp(timestamp_sec)

            overlay = frame.copy()
            cv2.rectangle(overlay, (15, 15), (420, 120), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (15, 15), (420, 120), (100, 100, 100), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                frame,
                f"FRAME: {i:04d} / {total_frames:04d} ({fps:.1f} FPS)",
                (25, 45),
                font,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"TIMESTAMP: {time_str}",
                (25, 75),
                font,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"SIM EGO SPEED: {simulated_speed_kmh:.1f} km/h",
                (25, 105),
                font,
                0.55,
                (100, 255, 100),
                1,
                cv2.LINE_AA,
            )

            writer.write_frame(frame)

    return out_file


if __name__ == "__main__":
    sample_dir = Path(__file__).parent
    video_1 = sample_dir / "synthetic_dashcam_30fps.mp4"
    video_2 = sample_dir / "synthetic_dashcam_60fps.mp4"

    print(f"Generating synthetic video: {video_1} (720p, 30fps, 3.0s)...")
    generate_synthetic_dashcam_video(video_1, duration_sec=3.0, fps=30.0, resolution=(1280, 720))
    print(f"Created {video_1}")

    print(f"Generating synthetic video: {video_2} (360p, 60fps, 2.0s)...")
    generate_synthetic_dashcam_video(video_2, duration_sec=2.0, fps=60.0, resolution=(640, 360))
    print(f"Created {video_2}")
