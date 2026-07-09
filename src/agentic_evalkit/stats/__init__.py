"""Aggregation, reliability, and run-compatibility statistics."""

from agentic_evalkit.stats.aggregate import (
    AggregateStats,
    RateEstimate,
    ResourceDistribution,
    aggregate_run,
    build_report_aggregates,
    pass_at_k_by_sample,
    wilson_interval,
)
from agentic_evalkit.stats.compare import ComparisonResult, compare_runs
from agentic_evalkit.stats.reliability import consistency_at_k, pass_at_k

__all__ = [
    "AggregateStats",
    "ComparisonResult",
    "RateEstimate",
    "ResourceDistribution",
    "aggregate_run",
    "build_report_aggregates",
    "compare_runs",
    "consistency_at_k",
    "pass_at_k",
    "pass_at_k_by_sample",
    "wilson_interval",
]
