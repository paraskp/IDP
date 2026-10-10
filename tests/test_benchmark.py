"""
Tests for Phase 14: Validation and Benchmarking.
"""

import pytest
from core.results.benchmark import BenchmarkMetrics, compute_benchmark_metrics


def test_compute_benchmark_metrics():
    """Verifies MAE, RMSE, MedAE, and MPE against known mathematical values."""
    # Ground truth: 60, 60, 60, 60
    # Estimated:    62, 58, 61, 59
    # Absolute errors: 2, 2, 1, 1
    # MAE = (2 + 2 + 1 + 1) / 4 = 1.50
    # RMSE = sqrt((4 + 4 + 1 + 1)/4) = sqrt(2.5) = 1.581
    # MedAE = 1.5
    gt = [60.0, 60.0, 60.0, 60.0]
    est = [62.0, 58.0, 61.0, 59.0]

    metrics = compute_benchmark_metrics(gt, est)

    assert metrics.sample_count == 4
    assert pytest.approx(metrics.mae_kmh, abs=0.01) == 1.50
    assert pytest.approx(metrics.rmse_kmh, abs=0.01) == 1.58
    assert pytest.approx(metrics.median_ae_kmh, abs=0.01) == 1.50
    assert pytest.approx(metrics.mean_percentage_error, abs=0.01) == 0.025
    assert len(metrics.summary()) > 0
