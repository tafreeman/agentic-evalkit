"""Property-based tests for the interval and comparison helpers.

Derandomized with modest example counts so the suite stays hermetic and
reproducible. The ``compare_runs`` symmetry property uses single-attempt runs,
so it holds both before and after the cluster-bootstrap change for repeated
attempts (the clustered path negates identically, but is not needed here).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from _stats_fixtures import _execution, _grade, _sample
from hypothesis import given, settings
from hypothesis import strategies as st

from agentic_evalkit.models import (
    DatasetRef,
    EvalRunManifest,
    EvalRunResult,
    ExecutionStatus,
    GradeStatus,
    ResolvedDataset,
    RunSummary,
    SampleResult,
    SamplingPolicy,
)
from agentic_evalkit.stats.aggregate import clustered_interval, wilson_interval
from agentic_evalkit.stats.compare import compare_runs

_TOLERANCE = 1e-9
_STARTED_AT = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)
_FINISHED_AT = datetime(2026, 7, 2, 12, 5, 0, tzinfo=UTC)

_counts = st.integers(min_value=1, max_value=2000).flatmap(
    lambda total: st.tuples(st.integers(min_value=0, max_value=total), st.just(total))
)


@settings(derandomize=True, max_examples=100, deadline=None)
@given(_counts)
def test_wilson_bounds_lie_in_unit_interval_and_ordered(counts: tuple[int, int]) -> None:
    successes, total = counts
    lower, upper = wilson_interval(successes=successes, total=total)
    assert lower is not None
    assert upper is not None
    assert 0.0 <= lower <= upper <= 1.0


@settings(derandomize=True, max_examples=100, deadline=None)
@given(_counts)
def test_wilson_interval_contains_the_observed_proportion(counts: tuple[int, int]) -> None:
    successes, total = counts
    lower, upper = wilson_interval(successes=successes, total=total)
    assert lower is not None
    assert upper is not None
    p = successes / total
    assert lower - _TOLERANCE <= p <= upper + _TOLERANCE


@settings(derandomize=True, max_examples=100, deadline=None)
@given(st.integers(min_value=1, max_value=500), st.data())
def test_wilson_bounds_are_monotone_in_successes(total: int, data: st.DataObject) -> None:
    successes = data.draw(st.integers(min_value=0, max_value=total - 1))
    low_a, high_a = wilson_interval(successes=successes, total=total)
    low_b, high_b = wilson_interval(successes=successes + 1, total=total)
    assert low_a is not None and high_a is not None
    assert low_b is not None and high_b is not None
    assert low_a <= low_b + _TOLERANCE
    assert high_a <= high_b + _TOLERANCE


@settings(derandomize=True, max_examples=100, deadline=None)
@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=2,
        max_size=40,
    )
)
def test_clustered_interval_is_clamped_ordered_and_brackets_the_mean(
    cluster_means: list[float],
) -> None:
    lower, upper = clustered_interval(cluster_means=cluster_means)
    assert lower is not None
    assert upper is not None
    mean = math.fsum(cluster_means) / len(cluster_means)
    assert 0.0 <= lower <= upper <= 1.0
    assert lower - _TOLERANCE <= mean <= upper + _TOLERANCE


@settings(derandomize=True, max_examples=25, deadline=None)
@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), st.integers(2, 30))
def test_clustered_interval_is_zero_width_for_identical_clusters(value: float, m: int) -> None:
    lower, upper = clustered_interval(cluster_means=[value] * m)
    assert lower is not None
    assert upper is not None
    assert math.isclose(lower, upper, abs_tol=1e-6)


@settings(derandomize=True, max_examples=25, deadline=None)
@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_clustered_interval_needs_two_clusters(value: float) -> None:
    assert clustered_interval(cluster_means=[value]) == (None, None)
    assert clustered_interval(cluster_means=[]) == (None, None)


def _result(sample_id: str, *, passed: bool) -> SampleResult:
    return SampleResult(
        sample=_sample(sample_id),
        execution=_execution(sample_id, attempt=1, status=ExecutionStatus.COMPLETED),
        grade=_grade(
            sample_id,
            status=GradeStatus.PASS if passed else GradeStatus.FAIL,
            score=1.0 if passed else 0.0,
        ),
    )


def _run(run_id: str, outcomes: list[bool]) -> EvalRunResult:
    samples = tuple(_result(f"s{i:03d}", passed=ok) for i, ok in enumerate(outcomes))
    return EvalRunResult(
        run_id=run_id,
        manifest=EvalRunManifest(
            run_name="property-fixture",
            dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
            adapter="gsm8k@1",
            grader="normalized-exact@1",
            target_name="echo-target",
            sampling=SamplingPolicy(seed=7, temperature=0.0, attempts=1),
            attempts=1,
        ),
        resolved_dataset=ResolvedDataset(
            dataset_id="openai/gsm8k", revision="abc", config="main", split="test"
        ),
        samples=samples,
        summary=RunSummary(total=len(samples), passed=sum(outcomes)),
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


_paired_outcomes = st.lists(st.tuples(st.booleans(), st.booleans()), min_size=1, max_size=15)


@settings(derandomize=True, max_examples=30, deadline=None)
@given(_paired_outcomes, st.integers(min_value=0, max_value=1000))
def test_compare_runs_swap_negates_estimate_and_interval(
    pairs: list[tuple[bool, bool]], seed: int
) -> None:
    left = _run("left", [a for a, _ in pairs])
    right = _run("right", [b for _, b in pairs])
    forward = compare_runs(left, right, bootstrap_samples=100, seed=seed)
    backward = compare_runs(right, left, bootstrap_samples=100, seed=seed)
    assert math.isclose(forward.estimate, -backward.estimate, abs_tol=_TOLERANCE)
    assert math.isclose(forward.lower_percentile, -backward.upper_percentile, abs_tol=_TOLERANCE)
    assert math.isclose(forward.upper_percentile, -backward.lower_percentile, abs_tol=_TOLERANCE)
    assert forward.paired_count == backward.paired_count


@settings(derandomize=True, max_examples=30, deadline=None)
@given(_paired_outcomes, st.integers(min_value=0, max_value=1000))
def test_compare_runs_interval_is_ordered_and_within_unit_delta(
    pairs: list[tuple[bool, bool]], seed: int
) -> None:
    left = _run("left", [a for a, _ in pairs])
    right = _run("right", [b for _, b in pairs])
    result = compare_runs(left, right, bootstrap_samples=100, seed=seed)
    assert -1.0 <= result.lower_percentile <= result.upper_percentile <= 1.0
