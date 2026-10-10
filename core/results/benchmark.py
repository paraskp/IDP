"""
Validation and benchmarking framework (Phase 14).
Evaluates speed estimation accuracy against ground-truth measurements (GPS / OBD-II).
Computes MAE, RMSE, Mean Percentage Error, and Median Absolute Error.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Statistical error metrics comparing estimated speeds against ground truth."""
    mae_kmh: float                 # Mean Absolute Error
    rmse_kmh: float                # Root Mean Square Error
    median_ae_kmh: float           # Median Absolute Error
    mean_percentage_error: float   # Mean Relative % Error
    sample_count: int
    ground_truth_mean_kmh: float
    estimated_mean_kmh: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mae_kmh": self.mae_kmh,
            "rmse_kmh": self.rmse_kmh,
            "median_ae_kmh": self.median_ae_kmh,
            "mean_percentage_error": self.mean_percentage_error,
            "sample_count": self.sample_count,
            "ground_truth_mean_kmh": self.ground_truth_mean_kmh,
            "estimated_mean_kmh": self.estimated_mean_kmh,
        }

    def summary(self) -> str:
        return (
            f"Benchmark Results (N={self.sample_count}):\n"
            f"  MAE:       {self.mae_kmh:.2f} km/h\n"
            f"  RMSE:      {self.rmse_kmh:.2f} km/h\n"
            f"  Median AE: {self.median_ae_kmh:.2f} km/h\n"
            f"  Error %:   {self.mean_percentage_error * 100:.1f}%\n"
            f"  GT Mean:   {self.ground_truth_mean_kmh:.1f} km/h | Est Mean: {self.estimated_mean_kmh:.1f} km/h"
        )


def compute_benchmark_metrics(
    ground_truth_speeds: List[float],
    estimated_speeds: List[Optional[float]],
) -> BenchmarkMetrics:
    """
    Computes standard statistical error metrics between ground truth and estimated speeds.

    Args:
        ground_truth_speeds: List of reference speeds in km/h.
        estimated_speeds: Corresponding estimated speeds in km/h (None for dropped estimates).

    Returns:
        BenchmarkMetrics dataclass.
    """
    paired = [
        (gt, est) for gt, est in zip(ground_truth_speeds, estimated_speeds)
        if est is not None and not math.isnan(est)
    ]

    if not paired:
        return BenchmarkMetrics(
            mae_kmh=0.0,
            rmse_kmh=0.0,
            median_ae_kmh=0.0,
            mean_percentage_error=0.0,
            sample_count=0,
            ground_truth_mean_kmh=0.0,
            estimated_mean_kmh=0.0,
        )

    gts = np.array([p[0] for p in paired], dtype=np.float64)
    ests = np.array([p[1] for p in paired], dtype=np.float64)

    errors = np.abs(ests - gts)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((ests - gts) ** 2)))
    med_ae = float(np.median(errors))

    # Avoid division by zero for stationary reference
    valid_pct_mask = gts > 1.0
    if np.any(valid_pct_mask):
        pct_errors = errors[valid_pct_mask] / gts[valid_pct_mask]
        mpe = float(np.mean(pct_errors))
    else:
        mpe = 0.0

    return BenchmarkMetrics(
        mae_kmh=round(mae, 2),
        rmse_kmh=round(rmse, 2),
        median_ae_kmh=round(med_ae, 2),
        mean_percentage_error=round(mpe, 3),
        sample_count=len(paired),
        ground_truth_mean_kmh=round(float(np.mean(gts)), 2),
        estimated_mean_kmh=round(float(np.mean(ests)), 2),
    )
