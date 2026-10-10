"""
Results export, video annotation, and statistics reporting subsystem (Phase 10).
"""

from core.results.video_annotator import VideoAnnotator
from core.results.exporter import export_to_csv, export_to_json
from core.results.statistics import compute_summary_statistics, generate_speed_plots
from core.results.benchmark import BenchmarkMetrics, compute_benchmark_metrics

__all__ = [
    "VideoAnnotator",
    "export_to_csv",
    "export_to_json",
    "compute_summary_statistics",
    "generate_speed_plots",
    "BenchmarkMetrics",
    "compute_benchmark_metrics",
]
