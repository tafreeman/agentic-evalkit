"""Aggregation, reliability, and run-compatibility statistics."""

from agentic_evalkit.stats.aggregate import (
    AggregateStats,
    ContinuousEstimate,
    IntervalMethod,
    RateEstimate,
    ResourceDistribution,
    aggregate_run,
    build_report_aggregates,
    clustered_interval,
    pass_at_k_by_sample,
    wilson_interval,
)
from agentic_evalkit.stats.compare import ComparisonResult, compare_runs
from agentic_evalkit.stats.power import required_sample_size
from agentic_evalkit.stats.reliability import consistency_at_k, pass_at_k

__all__ = [
    "AggregateStats",
    "ComparisonResult",
    "ContinuousEstimate",
    "IntervalMethod",
    "RateEstimate",
    "ResourceDistribution",
    "aggregate_run",
    "build_report_aggregates",
    "clustered_interval",
    "compare_runs",
    "consistency_at_k",
    "pass_at_k",
    "pass_at_k_by_sample",
    "required_sample_size",
    "wilson_interval",
]
