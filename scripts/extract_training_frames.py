"""
Extract Training Frames Script.
Extracts diverse, non-blurred frames from dashcam videos to prepare datasets
for training custom YOLO road landmark & marking detectors on Roboflow or Colab.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def is_frame_sharp(frame: np.ndarray, blur_threshold: float = 80.0) -> bool:
    """Computes Laplacian variance to filter out motion-blurred frames."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= blur_threshold


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_stride: int = 15,
    max_frames: int = 300,
    blur_threshold: float = 75.0,
) -> int:
    """
    Extracts frames from video and writes sharp images to output_dir.

    Returns:
        Number of extracted frames.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return 0

    saved_count = 0
    frame_idx = 0
    video_stem = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret or saved_count >= max_frames:
            break

        if frame_idx % frame_stride == 0:
            if is_frame_sharp(frame, blur_threshold):
                out_name = f"{video_stem}_f{frame_idx:06d}.jpg"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1
                if saved_count % 25 == 0:
                    print(f"Extracted {saved_count} sharp frames...")

        frame_idx += 1

    cap.release()
    print(f"Done! Saved {saved_count} frames to {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract sharp training frames from dashcam videos.")
    parser.add_argument("video", type=str, help="Path to input dashcam video (.mp4)")
    parser.add_argument("--output", "-o", type=str, default="dataset/raw_frames", help="Output directory")
    parser.add_argument("--stride", "-s", type=int, default=15, help="Frame sample stride (default: 15)")
    parser.add_argument("--max", "-m", type=int, default=250, help="Max frames to extract (default: 250)")
    parser.add_argument("--blur-thresh", type=float, default=70.0, help="Laplacian blur threshold")

    args = parser.parse_args()
    extract_frames(
        video_path=Path(args.video),
        output_dir=Path(args.output),
        frame_stride=args.stride,
        max_frames=args.max,
        blur_threshold=args.blur_thresh,
    )


if __name__ == "__main__":
    main()
