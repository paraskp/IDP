"""
Full-featured CLI tool for Dashcam Speed Analyzer.
Supports video inspection (Phase 1) and end-to-end speed estimation (Phases 1-10).
"""

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.pipeline.orchestrator import PipelineOrchestrator
from core.video.exceptions import VideoError
from core.video.metadata import get_video_metadata
from core.video.reader import read_frames
from core.video.validator import validate_video_file


def format_bytes(num_bytes: int) -> str:
    """Formats raw bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def run_inspection(video_path: Path, sample_step: int = 1, as_json: bool = False) -> int:
    """Runs fast Phase 1 container and metadata inspection."""
    validation = validate_video_file(video_path, raise_on_error=True)
    meta = get_video_metadata(video_path)

    start_time = time.perf_counter()
    decoded_count = 0
    first_frame_shape = None
    for fd in read_frames(video_path, step=sample_step):
        if decoded_count == 0:
            first_frame_shape = fd.frame.shape
        decoded_count += 1

    elapsed = time.perf_counter() - start_time
    fps_read = decoded_count / elapsed if elapsed > 0 else 0.0

    if as_json:
        res = {
            "file": {"path": str(meta.video_path), "size_bytes": meta.file_size_bytes},
            "metadata": meta.to_dict(),
            "stream_check": {
                "decoded_frames": decoded_count,
                "decoding_fps": round(fps_read, 2),
                "frame_shape": list(first_frame_shape) if first_frame_shape else None,
            },
        }
        print(json.dumps(res, indent=2))
        return 0

    print("\n" + "=" * 65)
    print("          DASHCAM SPEED ANALYZER - VIDEO INSPECTION          ")
    print("=" * 65)
    print(f"File Path:        {meta.video_path}")
    print(f"File Size:        {format_bytes(meta.file_size_bytes)}")
    print(f"Format/Codec:     {validation.extension.upper()} / {meta.codec}")
    print("-" * 65)
    print(f"Resolution:       {meta.width} x {meta.height} (Aspect: {meta.aspect_ratio:.2f})")
    print(f"Frame Rate:       {meta.fps:.2f} FPS")
    print(f"Total Frames:     {meta.frame_count:,}")
    print(f"Duration:         {meta.duration:.3f} s ({meta.format_timestamp(meta.duration)})")
    print("-" * 65)
    print("Stream Integrity Check:")
    print(f"  Decoded Frames: {decoded_count} (step: {sample_step})")
    print(f"  Decoding Speed: {fps_read:.1f} FPS ({elapsed:.3f}s total)")
    print(f"  Frame Matrix:   {list(first_frame_shape) if first_frame_shape else None}")
    print(f"  Health Status:  {'PASSED [Healthy Stream]' if decoded_count > 0 else 'FAILED'}")
    print("=" * 65 + "\n")
    return 0


def run_full_pipeline(
    video_path: Path,
    output_dir: Path,
    generate_video: bool = True,
    sample_step: int = 1,
    user_fov: float = None,
    camera_height: float = 1.35,
    as_json: bool = False,
) -> int:
    """Runs complete 10-stage end-to-end analysis pipeline."""
    orchestrator = PipelineOrchestrator(
        video_path=video_path,
        user_fov_deg=user_fov,
        camera_height_meters=camera_height,
        is_user_calibrated_scale=(user_fov is not None),
    )

    def print_progress(fraction: float, message: str):
        if not as_json:
            bar_len = 30
            filled = int(round(fraction * bar_len))
            bar = "#" * filled + "-" * (bar_len - filled)
            sys.stdout.write(f"\r[{bar}] {fraction * 100:5.1f}% | {message:<45}")
            sys.stdout.flush()

    if not as_json:
        print("\n" + "=" * 65)
        print("          DASHCAM SPEED ANALYZER - FULL PIPELINE RUN          ")
        print("=" * 65)
        print(f"Input:         {video_path}")
        print(f"Output Dir:    {output_dir}")
        print(f"Sample Step:   {sample_step}")
        print(f"Generate Video:{generate_video}")
        print("-" * 65)

    result = orchestrator.run_full_analysis(
        output_dir=output_dir,
        generate_video=generate_video,
        sample_step=sample_step,
        progress_callback=print_progress if not as_json else None,
    )

    if not as_json:
        print("\n" + "-" * 65)
        print("ANALYSIS SUMMARY:")
        print(f"  Total Duration:     {result.summary['duration_seconds']}s")
        print(f"  Avg Ego Speed:      {result.summary['ego_vehicle']['average_speed_kmh']} km/h")
        print(f"  Max Ego Speed:      {result.summary['ego_vehicle']['max_speed_kmh']} km/h")
        print(f"  Vehicles Tracked:   {result.summary['total_unique_vehicles']}")
        print(f"  Avg Vehicle Speed:  {result.summary['surrounding_vehicles']['average_speed_kmh']} km/h")
        print(f"  Mean Confidence:    {result.summary['confidence']['mean_confidence'] * 100:.1f}%")
        print(f"  Processing Time:    {result.elapsed_time_sec}s")
        print("-" * 65)
        print("GENERATED ARTIFACTS:")
        if result.output_video_path:
            print(f"  Annotated Video:    {result.output_video_path}")
        if result.output_csv_path:
            print(f"  CSV Results:        {result.output_csv_path}")
        if result.output_json_path:
            print(f"  JSON Results:       {result.output_json_path}")
        if result.output_plot_path:
            print(f"  Speed Plots:        {result.output_plot_path}")
        print("=" * 65 + "\n")
    else:
        print(json.dumps(result.summary, indent=2))

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dashcam Speed Analyzer - Monocular Speed Estimation CLI"
    )
    parser.add_argument("video_path", type=str, help="Path to input dashcam video")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Run only container and metadata inspection (Phase 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs/)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip generating annotated video for faster processing",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame step interval for analysis (default: 1)",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="Optional manual camera horizontal FOV in degrees",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=1.35,
        help="Camera mount height in meters (default: 1.35)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )

    args = parser.parse_args()
    v_path = Path(args.video_path)

    try:
        if args.inspect_only:
            code = run_inspection(v_path, sample_step=args.step, as_json=args.json)
        else:
            code = run_full_pipeline(
                video_path=v_path,
                output_dir=Path(args.output_dir),
                generate_video=not args.no_video,
                sample_step=args.step,
                user_fov=args.fov,
                camera_height=args.camera_height,
                as_json=args.json,
            )
        sys.exit(code)
    except VideoError as e:
        print(f"\n[ERROR] Video Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Analysis Failed: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
