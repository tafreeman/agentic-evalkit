"""Tests for run aggregation and Wilson confidence intervals (design §10, ADR-0008).

``aggregate_run`` recounts outcomes directly from ``run.samples`` (never
trusts a caller-supplied summary) using both grade status and execution
status, so operational failures (errors, timeouts, cancellations,
unavailable capabilities) are never confused with a definitive task
failure. Wilson bounds use ``statistics.NormalDist().inv_cdf(0.975)``
(Task 12 Step 4) and return ``None`` bounds on an empty denominator rather
than a misleading zero-width interval at 0% or 100%.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from agentic_evalkit.models import (
    DatasetRef,
    EvalRunManifest,
    EvalRunResult,
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
    ResolvedDataset,
    RunSummary,
    SampleResult,
)
from agentic_evalkit.stats.aggregate import aggregate_run, wilson_interval

_STARTED_AT = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)
_FINISHED_AT = datetime(2026, 7, 2, 12, 5, 0, tzinfo=UTC)


def _sample(sample_id: str) -> EvalSample:
    return EvalSample(
        sample_id=sample_id,
        input={"question": f"question for {sample_id}"},
        reference="42",
        source_digest=f"sha256:{sample_id}",
        adapter="gsm8k@1",
    )


def _execution(
    sample_id: str,
    *,
    attempt: int = 1,
    status: ExecutionStatus,
    latency_ms: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
) -> NormalizedExecutionResult:
    return NormalizedExecutionResult(
        sample_id=sample_id,
        attempt=attempt,
        status=status,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


def _grade(sample_id: str, *, status: GradeStatus, score: float | None = None) -> GradeResult:
    return GradeResult(
        sample_id=sample_id,
        grader="normalized-exact@1",
        status=status,
        score=score,
        hard_gate=False,
        created_at=_FINISHED_AT,
    )


def _manifest(**overrides: object) -> EvalRunManifest:
    defaults: dict[str, object] = dict(
        run_name="gsm8k-smoke",
        dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="echo-target",
    )
    defaults.update(overrides)
    return EvalRunManifest(**defaults)  # type: ignore[arg-type]


def _resolved_dataset(**overrides: object) -> ResolvedDataset:
    defaults: dict[str, object] = dict(dataset_id="openai/gsm8k", revision="abc")
    defaults.update(overrides)
    return ResolvedDataset(**defaults)  # type: ignore[arg-type]


def _run(samples: tuple[SampleResult, ...]) -> EvalRunResult:
    return EvalRunResult(
        run_id="run-001",
        manifest=_manifest(),
        resolved_dataset=_resolved_dataset(),
        samples=samples,
        # Deliberately wrong/stale summary: aggregate_run must recount from
        # samples rather than trusting this value.
        summary=RunSummary(total=999, passed=999),
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


# --- wilson_interval: known values ------------------------------------------


def test_wilson_interval_uses_normaldist_975_quantile() -> None:
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.975)
    lower, upper = wilson_interval(successes=1, total=4)
    assert lower is not None
    assert upper is not None
    # Manually recompute the Wilson bound with the same z to confirm the
    # implementation uses the 97.5th-percentile z-score (95% two-sided).
    p, n = 1 / 4, 4
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    expected_lower = (center - spread) / denom
    expected_upper = (center + spread) / denom
    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)


def test_wilson_interval_bounds_are_within_unit_range() -> None:
    lower, upper = wilson_interval(successes=0, total=10)
    assert lower is not None
    assert upper is not None
    assert 0.0 <= lower <= upper <= 1.0


def test_wilson_interval_all_successes_lower_bound_above_zero() -> None:
    # With 100% observed successes at finite n, Wilson's lower bound stays
    # strictly below the point estimate's naive certainty: it is > 0.0,
    # acknowledging that a small finite sample cannot prove a 100% true
    # rate. (The upper bound is exactly 1.0 in this case -- that is a
    # correct closed-form property of the Wilson interval when p=1, not a
    # clamping artifact.)
    lower, upper = wilson_interval(successes=10, total=10)
    assert lower is not None
    assert upper is not None
    assert upper == pytest.approx(1.0)
    assert 0.0 < lower < 1.0


def test_wilson_interval_empty_denominator_returns_none_bounds() -> None:
    assert wilson_interval(successes=0, total=0) == (None, None)


def test_wilson_interval_rejects_successes_greater_than_total() -> None:
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(successes=5, total=4)


def test_wilson_interval_rejects_negative_successes() -> None:
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(successes=-1, total=4)


# --- aggregate_run: exact recount from samples ------------------------------


def test_aggregate_run_recounts_outcomes_from_samples_not_summary() -> None:
    samples = (
        SampleResult(
            sample=_sample("s0"),
            execution=_execution("s0", status=ExecutionStatus.COMPLETED),
            grade=_grade("s0", status=GradeStatus.PASS, score=1.0),
        ),
        SampleResult(
            sample=_sample("s1"),
            execution=_execution("s1", status=ExecutionStatus.COMPLETED),
            grade=_grade("s1", status=GradeStatus.FAIL, score=0.0),
        ),
        SampleResult(
            sample=_sample("s2"),
            execution=_execution("s2", status=ExecutionStatus.COMPLETED),
            grade=_grade("s2", status=GradeStatus.PARTIAL, score=0.5),
        ),
        SampleResult(
            sample=_sample("s3"),
            execution=_execution("s3", status=ExecutionStatus.ERROR),
            grade=None,
        ),
        SampleResult(
            sample=_sample("s4"),
            execution=_execution("s4", status=ExecutionStatus.TIMEOUT),
            grade=None,
        ),
        SampleResult(
            sample=_sample("s5"),
            execution=_execution("s5", status=ExecutionStatus.CANCELLED),
            grade=None,
        ),
        SampleResult(
            sample=_sample("s6"),
            execution=_execution("s6", status=ExecutionStatus.COMPLETED),
            grade=_grade("s6", status=GradeStatus.ABSTAIN),
        ),
        SampleResult(
            sample=_sample("s7"),
            execution=_execution("s7", status=ExecutionStatus.COMPLETED),
            grade=_grade("s7", status=GradeStatus.UNAVAILABLE),
        ),
    )
    stats = aggregate_run(_run(samples))

    assert stats.total == 8
    assert stats.passed == 1
    assert stats.failed == 1
    assert stats.partial == 1
    assert stats.errors == 1
    assert stats.timeouts == 1
    assert stats.cancelled == 1
    assert stats.abstained == 1
    assert stats.unavailable == 1

    # Exact integer numerator/denominator, not a pre-rounded float.
    assert stats.pass_rate.numerator == 1
    assert stats.pass_rate.denominator == 8
    assert stats.pass_rate.value == pytest.approx(1 / 8)
    assert stats.pass_rate.lower_bound is not None
    assert stats.pass_rate.upper_bound is not None


def test_aggregate_run_error_execution_without_grade_counts_as_error_not_fail() -> None:
    # An execution ERROR with no grade must appear in stats.errors, and must
    # NOT be silently counted as a definitive grading FAIL -- operational
    # failures cannot masquerade as task failures (design §10).
    samples = (
        SampleResult(
            sample=_sample("s0"),
            execution=_execution("s0", status=ExecutionStatus.ERROR),
            grade=None,
        ),
    )
    stats = aggregate_run(_run(samples))
    assert stats.errors == 1
    assert stats.failed == 0
    assert stats.total == 1


def test_aggregate_run_empty_samples_returns_none_bounds_not_zero() -> None:
    stats = aggregate_run(_run(()))
    assert stats.total == 0
    assert stats.pass_rate.numerator == 0
    assert stats.pass_rate.denominator == 0
    assert stats.pass_rate.value is None
    assert stats.pass_rate.lower_bound is None
    assert stats.pass_rate.upper_bound is None


def test_aggregate_run_score_mean_only_over_defined_numeric_scores() -> None:
    samples = (
        SampleResult(
            sample=_sample("s0"),
            execution=_execution("s0", status=ExecutionStatus.COMPLETED),
            grade=_grade("s0", status=GradeStatus.PASS, score=1.0),
        ),
        SampleResult(
            sample=_sample("s1"),
            execution=_execution("s1", status=ExecutionStatus.COMPLETED),
            grade=_grade("s1", status=GradeStatus.PARTIAL, score=0.5),
        ),
        # No numeric score (grade is None due to operational error): must
        # be excluded from the score mean's denominator entirely, not
        # treated as a 0.0.
        SampleResult(
            sample=_sample("s2"),
            execution=_execution("s2", status=ExecutionStatus.ERROR),
            grade=None,
        ),
        # Grade present but score undefined (e.g. abstain): also excluded.
        SampleResult(
            sample=_sample("s3"),
            execution=_execution("s3", status=ExecutionStatus.COMPLETED),
            grade=_grade("s3", status=GradeStatus.ABSTAIN, score=None),
        ),
    )
    stats = aggregate_run(_run(samples))
    assert stats.score_mean is not None
    assert stats.score_mean == pytest.approx((1.0 + 0.5) / 2)
    assert stats.score_count == 2


def test_aggregate_run_score_mean_is_none_with_no_defined_scores() -> None:
    samples = (
        SampleResult(
            sample=_sample("s0"),
            execution=_execution("s0", status=ExecutionStatus.ERROR),
            grade=None,
        ),
    )
    stats = aggregate_run(_run(samples))
    assert stats.score_mean is None
    assert stats.score_count == 0


# --- aggregate_run: resource distributions ----------------------------------


def test_aggregate_run_latency_distribution_count_mean_p50_p95() -> None:
    samples = tuple(
        SampleResult(
            sample=_sample(f"s{i}"),
            execution=_execution(
                f"s{i}", status=ExecutionStatus.COMPLETED, latency_ms=float(value)
            ),
            grade=_grade(f"s{i}", status=GradeStatus.PASS, score=1.0),
        )
        for i, value in enumerate([100.0, 200.0, 300.0, 400.0, 500.0])
    )
    stats = aggregate_run(_run(samples))
    assert stats.latency_ms is not None
    assert stats.latency_ms.count == 5
    assert stats.latency_ms.mean == pytest.approx(300.0)
    assert stats.latency_ms.p50 is not None
    assert stats.latency_ms.p95 is not None
    # p95 must be at or above every value below the 95th percentile of the
    # sample and no larger than the maximum observed value.
    assert 400.0 <= stats.latency_ms.p95 <= 500.0


def test_aggregate_run_resource_distribution_none_when_no_data_reported() -> None:
    samples = (
        SampleResult(
            sample=_sample("s0"),
            execution=_execution("s0", status=ExecutionStatus.COMPLETED),
            grade=_grade("s0", status=GradeStatus.PASS, score=1.0),
        ),
    )
    stats = aggregate_run(_run(samples))
    assert stats.latency_ms is None
    assert stats.input_tokens is None
    assert stats.output_tokens is None
    assert stats.cost_usd is None


def test_aggregate_run_token_and_cost_distributions_ignore_missing_values() -> None:
    samples = (
        SampleResult(
            sample=_sample("s0"),
            execution=_execution(
                "s0",
                status=ExecutionStatus.COMPLETED,
                input_tokens=10,
                output_tokens=20,
                cost_usd=0.01,
            ),
            grade=_grade("s0", status=GradeStatus.PASS, score=1.0),
        ),
        SampleResult(
            sample=_sample("s1"),
            # No token/cost data reported by this attempt.
            execution=_execution("s1", status=ExecutionStatus.COMPLETED),
            grade=_grade("s1", status=GradeStatus.PASS, score=1.0),
        ),
    )
    stats = aggregate_run(_run(samples))
    assert stats.input_tokens is not None
    assert stats.input_tokens.count == 1
    assert stats.input_tokens.mean == pytest.approx(10.0)
    assert stats.output_tokens is not None
    assert stats.output_tokens.count == 1
    assert stats.cost_usd is not None
    assert stats.cost_usd.count == 1
    assert stats.cost_usd.mean == pytest.approx(0.01)
