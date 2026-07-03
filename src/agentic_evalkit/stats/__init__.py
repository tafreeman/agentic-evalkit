"""Aggregation, reliability, and run-compatibility statistics."""

from agentic_evalkit.stats.aggregate import (
    AggregateStats,
    RateEstimate,
    ResourceDistribution,
    aggregate_run,
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
    "compare_runs",
    "consistency_at_k",
    "pass_at_k",
    "wilson_interval",
]
