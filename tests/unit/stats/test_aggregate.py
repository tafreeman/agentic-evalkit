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
from _stats_fixtures import _execution, _grade, _sample

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
from agentic_evalkit.stats.aggregate import (
    IntervalMethod,
    aggregate_run,
    build_report_aggregates,
    clustered_interval,
    pass_at_k_by_sample,
    wilson_interval,
)
from agentic_evalkit.stats.reliability import pass_at_k

_STARTED_AT = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)
_FINISHED_AT = datetime(2026, 7, 2, 12, 5, 0, tzinfo=UTC)


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


def _run_with_attempts(samples: tuple[SampleResult, ...], *, attempts: int) -> EvalRunResult:
    """Like ``_run`` but with a manifest whose ``attempts`` (and the
    validator-mirrored ``sampling.attempts``) is caller-controlled, for
    ``pass_at_k_by_sample``/``build_report_aggregates`` coverage that needs
    ``attempts > 1``."""
    manifest = _manifest(attempts=attempts, sampling=SamplingPolicy(attempts=attempts))
    return EvalRunResult(
        run_id="run-002",
        manifest=manifest,
        resolved_dataset=_resolved_dataset(),
        samples=samples,
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


# --- pass_at_k_by_sample: per-sample grouping and forwarding to pass_at_k ----


def _repeated_attempts(
    sample_id: str, *, statuses: tuple[GradeStatus, ...]
) -> tuple[SampleResult, ...]:
    return tuple(
        SampleResult(
            sample=_sample(sample_id),
            execution=_execution(sample_id, attempt=attempt, status=ExecutionStatus.COMPLETED),
            grade=_grade(
                sample_id, status=status, score=1.0 if status is GradeStatus.PASS else 0.0
            ),
        )
        for attempt, status in enumerate(statuses, start=1)
    )


def test_pass_at_k_by_sample_matches_directly_computed_pass_at_k_per_group() -> None:
    # s0: 2 of 3 attempts pass; s1: 0 of 3 attempts pass. Independently
    # counted here (not copied from the implementation) so the assertion
    # exercises pass_at_k_by_sample's own grouping/counting, cross-checked
    # against a direct pass_at_k call over the same (n, c) rather than a
    # hardcoded numeric literal.
    s0_statuses = (GradeStatus.PASS, GradeStatus.PASS, GradeStatus.FAIL)
    s1_statuses = (GradeStatus.FAIL, GradeStatus.FAIL, GradeStatus.FAIL)
    samples = _repeated_attempts("s0", statuses=s0_statuses) + _repeated_attempts(
        "s1", statuses=s1_statuses
    )
    run = _run_with_attempts(samples, attempts=3)

    estimates = pass_at_k_by_sample(run, k=3)

    expected_s0 = pass_at_k(
        total_attempts=len(s0_statuses),
        successful_attempts=sum(1 for status in s0_statuses if status is GradeStatus.PASS),
        k=3,
    )
    expected_s1 = pass_at_k(
        total_attempts=len(s1_statuses),
        successful_attempts=sum(1 for status in s1_statuses if status is GradeStatus.PASS),
        k=3,
    )
    assert estimates == {"s0": pytest.approx(expected_s0), "s1": pytest.approx(expected_s1)}


def test_pass_at_k_by_sample_omits_samples_with_fewer_than_k_attempts() -> None:
    # Only 2 attempts actually ran for this sample; pass_at_k requires
    # 1 <= k <= total_attempts, so k=3 must be omitted rather than raising
    # or fabricating an estimate.
    samples = _repeated_attempts("s0", statuses=(GradeStatus.PASS, GradeStatus.FAIL))
    run = _run_with_attempts(samples, attempts=2)
    estimates = pass_at_k_by_sample(run, k=3)
    assert estimates == {}


def test_pass_at_k_by_sample_empty_run_returns_empty_mapping() -> None:
    run = _run_with_attempts((), attempts=2)
    assert pass_at_k_by_sample(run, k=2) == {}


# --- build_report_aggregates: the CLI-facing envelope ------------------------


def test_build_report_aggregates_carries_aggregate_run_fields() -> None:
    samples = (
        SampleResult(
            sample=_sample("s0"),
            execution=_execution("s0", status=ExecutionStatus.COMPLETED),
            grade=_grade("s0", status=GradeStatus.PASS, score=1.0),
        ),
    )
    run = _run(samples)
    aggregates = build_report_aggregates(run)
    expected = aggregate_run(run).model_dump(mode="json")
    for key, value in expected.items():
        assert aggregates[key] == value


def test_build_report_aggregates_omits_pass_at_k_with_a_single_attempt() -> None:
    # _run() builds a manifest with the default attempts=1; pass@1 over one
    # attempt is exactly the pass/fail bit aggregate_run already reports, so
    # reporting it a second time would be redundant, not informative.
    samples = (
        SampleResult(
            sample=_sample("s0"),
            execution=_execution("s0", status=ExecutionStatus.COMPLETED),
            grade=_grade("s0", status=GradeStatus.PASS, score=1.0),
        ),
    )
    aggregates = build_report_aggregates(_run(samples))
    assert "pass_at_k" not in aggregates


def test_build_report_aggregates_includes_pass_at_k_with_repeated_attempts() -> None:
    samples = _repeated_attempts("s0", statuses=(GradeStatus.PASS, GradeStatus.FAIL))
    run = _run_with_attempts(samples, attempts=2)
    aggregates = build_report_aggregates(run)
    assert "pass_at_k" in aggregates
    pass_at_k_payload = aggregates["pass_at_k"]
    assert isinstance(pass_at_k_payload, dict)
    assert pass_at_k_payload["k"] == 2
    assert "s0" in pass_at_k_payload["by_sample_id"]
    expected_estimates = pass_at_k_by_sample(run, k=2)
    expected_mean = sum(expected_estimates.values()) / len(expected_estimates)
    assert pass_at_k_payload["mean"] == pytest.approx(expected_mean)


def test_build_report_aggregates_result_is_json_serializable() -> None:
    import json

    samples = _repeated_attempts("s0", statuses=(GradeStatus.PASS, GradeStatus.FAIL))
    run = _run_with_attempts(samples, attempts=2)
    # Must not raise: every value in the mapping is JSON-compatible, which
    # matters because this is exactly what a Reporter.write(aggregates=...)
    # call serializes to disk.
    json.dumps(build_report_aggregates(run))


# --- clustered_interval: cluster-robust bounds for repeated attempts (ADR-0016)


def test_clustered_interval_matches_inline_closed_form() -> None:
    from statistics import NormalDist, fmean, stdev

    # In-range cluster means so the bounds are not clamped: this exercises the
    # actual mean +/- z * stdev / sqrt(m) formula, recomputed inline from the
    # statistics module (never a hardcoded decimal).
    cluster_means = [0.4, 0.5, 0.6]
    lower, upper = clustered_interval(cluster_means=cluster_means)
    assert lower is not None
    assert upper is not None
    z = NormalDist().inv_cdf(0.975)
    center = fmean(cluster_means)
    spread = z * (stdev(cluster_means) / math.sqrt(len(cluster_means)))
    assert lower == pytest.approx(center - spread)
    assert upper == pytest.approx(center + spread)
    assert 0.0 <= lower <= upper <= 1.0


def test_clustered_interval_single_cluster_returns_none_bounds() -> None:
    # A single cluster has undefined between-cluster variance -- (None, None),
    # mirroring test_wilson_interval_empty_denominator_returns_none_bounds.
    assert clustered_interval(cluster_means=[0.5]) == (None, None)


def test_clustered_interval_clamps_bounds_to_unit_range() -> None:
    from statistics import NormalDist, fmean, stdev

    cluster_means = [0.0, 0.5, 1.0]
    z = NormalDist().inv_cdf(0.975)
    center = fmean(cluster_means)
    spread = z * (stdev(cluster_means) / math.sqrt(len(cluster_means)))
    # Sanity: the raw Wald bounds really do fall outside [0, 1] here, so the
    # clamp is doing real work rather than being a no-op.
    assert center - spread < 0.0
    assert center + spread > 1.0
    assert clustered_interval(cluster_means=cluster_means) == (0.0, 1.0)


# --- aggregate_run: interval-method selection and score_estimate (ADR-0016) --


def test_aggregate_run_single_attempt_pass_rate_interval_method_is_wilson() -> None:
    # attempts == 1 (the default _run): the pass rate is the unchanged Wilson
    # interval over the flat per-observation count, now labeled WILSON. The
    # bounds must be byte-identical to a direct wilson_interval call.
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
    )
    stats = aggregate_run(_run(samples))
    assert stats.pass_rate.interval_method is IntervalMethod.WILSON
    expected_lower, expected_upper = wilson_interval(successes=1, total=2)
    assert stats.pass_rate.lower_bound == expected_lower
    assert stats.pass_rate.upper_bound == expected_upper


def test_aggregate_run_repeated_attempts_uses_clustered_interval() -> None:
    from statistics import NormalDist, fmean, stdev

    # Three sample_ids, 2 attempts each: 0/2, 1/2, 2/2 passes -> per-sample_id
    # pass proportions [0.0, 0.5, 1.0]. The pooled numerator/denominator/value
    # stay exact (3 of 6), but the bounds come from clustered_interval and the
    # method is stamped CLUSTER_ROBUST.
    samples = (
        _repeated_attempts("s0", statuses=(GradeStatus.FAIL, GradeStatus.FAIL))
        + _repeated_attempts("s1", statuses=(GradeStatus.PASS, GradeStatus.FAIL))
        + _repeated_attempts("s2", statuses=(GradeStatus.PASS, GradeStatus.PASS))
    )
    stats = aggregate_run(_run_with_attempts(samples, attempts=2))

    assert stats.pass_rate.interval_method is IntervalMethod.CLUSTER_ROBUST
    assert stats.pass_rate.numerator == 3
    assert stats.pass_rate.denominator == 6
    assert stats.pass_rate.value == pytest.approx(3 / 6)

    proportions = [0.0, 0.5, 1.0]
    z = NormalDist().inv_cdf(0.975)
    center = fmean(proportions)
    spread = z * (stdev(proportions) / math.sqrt(len(proportions)))
    assert stats.pass_rate.lower_bound == pytest.approx(max(0.0, center - spread))
    assert stats.pass_rate.upper_bound == pytest.approx(min(1.0, center + spread))


def test_aggregate_run_single_distinct_sample_id_multi_attempt_returns_none_bounds() -> None:
    # One distinct sample_id but attempts > 1 -> a single cluster -> undefined
    # between-cluster variance -> (None, None) bounds, while the exact pooled
    # counts remain reportable.
    samples = _repeated_attempts("s0", statuses=(GradeStatus.PASS, GradeStatus.FAIL))
    stats = aggregate_run(_run_with_attempts(samples, attempts=2))
    assert stats.pass_rate.interval_method is IntervalMethod.CLUSTER_ROBUST
    assert stats.pass_rate.numerator == 1
    assert stats.pass_rate.denominator == 2
    assert stats.pass_rate.value == pytest.approx(1 / 2)
    assert stats.pass_rate.lower_bound is None
    assert stats.pass_rate.upper_bound is None


def test_aggregate_run_score_estimate_is_none_with_fewer_than_two_scores() -> None:
    # A single defined score has a mean but no spread: score_estimate is None
    # even though score_mean itself is defined (companion to
    # test_aggregate_run_score_mean_is_none_with_no_defined_scores).
    samples = (
        SampleResult(
            sample=_sample("s0"),
            execution=_execution("s0", status=ExecutionStatus.COMPLETED),
            grade=_grade("s0", status=GradeStatus.PASS, score=1.0),
        ),
    )
    stats = aggregate_run(_run(samples))
    assert stats.score_mean == pytest.approx(1.0)
    assert stats.score_count == 1
    assert stats.score_estimate is None


def test_aggregate_run_score_estimate_flat_mean_matches_score_mean_and_is_unclamped() -> None:
    from statistics import NormalDist, stdev

    # Flat (attempts == 1) regime over three defined scores. The estimate's mean
    # must equal score_mean exactly (no drift); because a score mean is not a
    # probability, its normal-approx bounds are NOT clamped to [0, 1] (here they
    # genuinely fall below 0 and above 1); the flat SEM carries no named
    # binary-rate method, so interval_method is None.
    scores = (1.0, 0.5, 0.0)
    samples = tuple(
        SampleResult(
            sample=_sample(f"s{i}"),
            execution=_execution(f"s{i}", status=ExecutionStatus.COMPLETED),
            grade=_grade(
                f"s{i}",
                status=GradeStatus.PASS if value == 1.0 else GradeStatus.PARTIAL,
                score=value,
            ),
        )
        for i, value in enumerate(scores)
    )
    stats = aggregate_run(_run(samples))
    estimate = stats.score_estimate
    assert estimate is not None
    assert stats.score_mean is not None
    assert estimate.mean == stats.score_mean
    assert estimate.n == 3
    assert estimate.interval_method is None

    z = NormalDist().inv_cdf(0.975)
    sem = stdev(scores) / math.sqrt(len(scores))
    assert estimate.sem == pytest.approx(sem)
    assert estimate.lower_bound == pytest.approx(stats.score_mean - z * sem)
    assert estimate.upper_bound == pytest.approx(stats.score_mean + z * sem)
    assert estimate.lower_bound < 0.0
    assert estimate.upper_bound > 1.0


def test_aggregate_run_score_estimate_cluster_robust_with_repeated_attempts() -> None:
    from statistics import NormalDist, stdev

    # attempts > 1 with >= 2 distinct sample_ids carrying scores: score_estimate
    # is cluster-robust (SEM over per-sample_id mean scores), its mean still the
    # exact pooled score_mean.
    samples = (
        _repeated_attempts("s0", statuses=(GradeStatus.FAIL, GradeStatus.FAIL))
        + _repeated_attempts("s1", statuses=(GradeStatus.PASS, GradeStatus.FAIL))
        + _repeated_attempts("s2", statuses=(GradeStatus.PASS, GradeStatus.PASS))
    )
    stats = aggregate_run(_run_with_attempts(samples, attempts=2))
    estimate = stats.score_estimate
    assert estimate is not None
    assert stats.score_mean is not None
    assert estimate.interval_method is IntervalMethod.CLUSTER_ROBUST
    assert estimate.mean == stats.score_mean
    assert estimate.n == 6  # six defined scores across the three clusters

    # _repeated_attempts scores PASS as 1.0 and everything else 0.0, so the
    # per-sample_id mean scores are [0.0, 0.5, 1.0].
    cluster_mean_scores = [0.0, 0.5, 1.0]
    z = NormalDist().inv_cdf(0.975)
    sem = stdev(cluster_mean_scores) / math.sqrt(len(cluster_mean_scores))
    assert estimate.sem == pytest.approx(sem)
    assert estimate.lower_bound == pytest.approx(stats.score_mean - z * sem)
    assert estimate.upper_bound == pytest.approx(stats.score_mean + z * sem)


def test_aggregate_run_score_estimate_single_cluster_multi_attempt_has_none_spread() -> None:
    # attempts > 1 but a single distinct sample_id: two defined scores exist (so
    # score_estimate has a defined mean), but there is only one cluster mean, so
    # the cluster-robust spread is undefined -> sem/bounds None.
    samples = _repeated_attempts("s0", statuses=(GradeStatus.PASS, GradeStatus.FAIL))
    stats = aggregate_run(_run_with_attempts(samples, attempts=2))
    estimate = stats.score_estimate
    assert estimate is not None
    assert estimate.mean == stats.score_mean
    assert estimate.n == 2
    assert estimate.interval_method is IntervalMethod.CLUSTER_ROBUST
    assert estimate.sem is None
    assert estimate.lower_bound is None
    assert estimate.upper_bound is None


def test_aggregate_run_cluster_mean_scores_skips_missing_scores_and_empty_clusters() -> None:
    from statistics import NormalDist, stdev

    # s0: one scored PASS + one errored attempt (no grade) -> cluster mean over
    # [1.0] only. s1: both attempts errored, no grades -> contributes no cluster
    # mean (never a fabricated 0.0). s2: one scored FAIL (0.0) + one errored ->
    # cluster mean over [0.0]. Exercises the "attempt without a score" and
    # "cluster without any score" skip paths of the cluster-robust regime.
    samples = (
        SampleResult(
            sample=_sample("s0"),
            execution=_execution("s0", attempt=1, status=ExecutionStatus.COMPLETED),
            grade=_grade("s0", status=GradeStatus.PASS, score=1.0),
        ),
        SampleResult(
            sample=_sample("s0"),
            execution=_execution("s0", attempt=2, status=ExecutionStatus.ERROR),
            grade=None,
        ),
        SampleResult(
            sample=_sample("s1"),
            execution=_execution("s1", attempt=1, status=ExecutionStatus.ERROR),
            grade=None,
        ),
        SampleResult(
            sample=_sample("s1"),
            execution=_execution("s1", attempt=2, status=ExecutionStatus.ERROR),
            grade=None,
        ),
        SampleResult(
            sample=_sample("s2"),
            execution=_execution("s2", attempt=1, status=ExecutionStatus.COMPLETED),
            grade=_grade("s2", status=GradeStatus.FAIL, score=0.0),
        ),
        SampleResult(
            sample=_sample("s2"),
            execution=_execution("s2", attempt=2, status=ExecutionStatus.ERROR),
            grade=None,
        ),
    )
    stats = aggregate_run(_run_with_attempts(samples, attempts=2))
    estimate = stats.score_estimate
    assert estimate is not None
    assert stats.score_mean is not None
    assert estimate.interval_method is IntervalMethod.CLUSTER_ROBUST
    assert estimate.n == 2  # only s0's 1.0 and s2's 0.0 are defined scores

    cluster_mean_scores = [1.0, 0.0]  # s0 -> 1.0, s2 -> 0.0; s1 contributes none
    z = NormalDist().inv_cdf(0.975)
    sem = stdev(cluster_mean_scores) / math.sqrt(len(cluster_mean_scores))
    assert estimate.sem == pytest.approx(sem)
    assert estimate.lower_bound == pytest.approx(stats.score_mean - z * sem)
    assert estimate.upper_bound == pytest.approx(stats.score_mean + z * sem)
