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
from agentic_evalkit.stats.compare import (
    DATASET_IDENTITY_FIELDS_CHECKED,
    PROVENANCE_FIELDS_CHECKED,
    ComparisonResult,
    comparability_snapshot,
    compare_runs,
)
from agentic_evalkit.stats.power import required_sample_size
from agentic_evalkit.stats.reliability import consistency_at_k, pass_at_k

__all__ = [
    "DATASET_IDENTITY_FIELDS_CHECKED",
    "PROVENANCE_FIELDS_CHECKED",
    "AggregateStats",
    "ComparisonResult",
    "ContinuousEstimate",
    "IntervalMethod",
    "RateEstimate",
    "ResourceDistribution",
    "aggregate_run",
    "build_report_aggregates",
    "clustered_interval",
    "comparability_snapshot",
    "compare_runs",
    "consistency_at_k",
    "pass_at_k",
    "pass_at_k_by_sample",
    "required_sample_size",
    "wilson_interval",
]
