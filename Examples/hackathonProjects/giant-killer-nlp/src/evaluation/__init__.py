"""
Evaluation module for Giant-Killer NLP project.
Benchmarking and comparison utilities.
"""

from .benchmark import (
    BenchmarkResults,
    benchmark_latency,
    compute_metrics,
    compare_models,
    run_full_evaluation,
)

__all__ = [
    "BenchmarkResults",
    "benchmark_latency",
    "compute_metrics",
    "compare_models",
    "run_full_evaluation",
]
