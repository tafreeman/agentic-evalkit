"""Tests for counting up a run's results and building confidence intervals
around the pass rate (design section 10, ADR-0008).

``aggregate_run`` is the function under test here: it recounts every
outcome directly from ``run.samples`` (it never just trusts a summary
object handed to it, in case that summary turns out to be stale or wrong),
and it looks at both the grade status and the execution status so that
"operational" failures -- errors, timeouts, cancellations, an unavailable
capability -- are never confused with a "definitive" task failure (the
system actually tried and got it wrong). See ``aggregate.py``'s module
docstring for the full explanation of that distinction.

These tests also check the "Wilson interval" -- a confidence interval (a
range that's likely to contain the true pass rate, given how few samples
we tested) computed with ``statistics.NormalDist().inv_cdf(0.975)``
(Task 12 Step 4). When there's no data at all (a zero denominator), the
bounds must come back as ``None`` rather than a fake, zero-width range
sitting at exactly 0% or 100% -- returning real numbers there would look
far more confident than the data actually justifies.
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
    defaults: dict[str, object] = {
        "run_name": "gsm8k-smoke",
        "dataset_ref": DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
        "adapter": "gsm8k@1",
        "grader": "normalized-exact@1",
        "target_name": "echo-target",
    }
    defaults.update(overrides)
    return EvalRunManifest(**defaults)  # type: ignore[arg-type]


def _resolved_dataset(**overrides: object) -> ResolvedDataset:
    defaults: dict[str, object] = {"dataset_id": "openai/gsm8k", "revision": "abc"}
    defaults.update(overrides)
    return ResolvedDataset(**defaults)  # type: ignore[arg-type]


def _run(samples: tuple[SampleResult, ...]) -> EvalRunResult:
    return EvalRunResult(
        run_id="run-001",
        manifest=_manifest(),
        resolved_dataset=_resolved_dataset(),
        samples=samples,
        # Deliberately wrong (stale) summary: aggregate_run must recount
        # from the samples themselves rather than trusting this number.
        summary=RunSummary(total=999, passed=999),
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


def _run_with_attempts(samples: tuple[SampleResult, ...], *, attempts: int) -> EvalRunResult:
    """Like ``_run``, but lets the caller control ``attempts`` (how many
    times each question was attempted) -- and keeps the manifest's
    ``sampling.attempts`` in sync with it, since a validator requires the
    two to match. Used by the ``pass_at_k_by_sample``/
    ``build_report_aggregates`` tests below, which need ``attempts`` to be
    greater than 1 to exercise their repeated-attempts behavior."""
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
    # This test recomputes the Wilson formula's bounds by hand, right here,
    # using that same z value -- so it's not just trusting that
    # wilson_interval calls the right function somewhere internally, it's
    # proving the result matches the formula exactly. The 97.5th percentile
    # is the right cutoff for a 95% interval because the interval is
    # two-sided: 2.5% of the bell curve is excluded on each end, leaving
    # the middle 95%.
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
    # Every one of the 10 samples passed, but the Wilson lower bound still
    # comes back strictly greater than 0.0 rather than jumping straight to
    # "100% certain." That is the whole point of using Wilson instead of a
    # naive calculation: a sample of only 10 can never fully prove the
    # true, real-world rate is exactly 100%, so the bottom edge of our
    # confidence range should stay below 1.0 to reflect that uncertainty.
    # (The upper bound being exactly 1.0 here is a genuine, correct result
    # of the Wilson formula's math when every sample passes -- it is not a
    # bug, and it is not the safety clamp mentioned in wilson_interval's
    # own docstring kicking in.)
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

    # The pass rate keeps its numerator/denominator as exact whole numbers
    # (1 out of 8), not a value that has already been rounded into a
    # single decimal.
    assert stats.pass_rate.numerator == 1
    assert stats.pass_rate.denominator == 8
    assert stats.pass_rate.value == pytest.approx(1 / 8)
    assert stats.pass_rate.lower_bound is not None
    assert stats.pass_rate.upper_bound is not None


def test_aggregate_run_error_execution_without_grade_counts_as_error_not_fail() -> None:
    # A run that errored out with no grade attached must be counted in
    # stats.errors, and must NOT quietly get counted as a FAIL instead. An
    # error means something broke before the system's answer could even be
    # judged -- it must never be reported as if the system tried and got
    # the wrong answer (design section 10).
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
        # This sample has no numeric score at all, because it errored out
        # before grading (grade is None). It must be left out of the
        # score-average calculation completely -- not counted as if it
        # had scored 0.0, which would unfairly drag the average down.
        SampleResult(
            sample=_sample("s2"),
            execution=_execution("s2", status=ExecutionStatus.ERROR),
            grade=None,
        ),
        # This one did get graded, but with no numeric score attached
        # (e.g. the grader abstained) -- it must also be left out of the
        # average, for the same reason.
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
    # p95 (the 95th percentile) means "the value below which 95% of the
    # observations fall." With these 5 evenly-spaced values, that must
    # land somewhere between the 4th value (400.0) and the largest one
    # (500.0).
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
    # s0: 2 of its 3 attempts passed; s1: 0 of its 3 attempts passed. These
    # counts are worked out independently right here in the test (not
    # copied from the implementation's own code), so the assertion below
    # actually tests that pass_at_k_by_sample groups and counts attempts
    # correctly -- checked against a direct call to pass_at_k with those
    # same counts, rather than against some hardcoded expected number that
    # might just happen to match a bug.
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
    # Only 2 attempts actually ran for this sample, but we are asking for
    # k=3. pass_at_k requires the number of attempts to be at least k, so
    # this sample must be left out of the result entirely -- not raise an
    # error, and not return a made-up estimate either.
    samples = _repeated_attempts("s0", statuses=(GradeStatus.PASS, GradeStatus.FAIL))
    run = _run_with_attempts(samples, attempts=2)
    estimates = pass_at_k_by_sample(run, k=3)
    assert estimates == {}


def test_pass_at_k_by_sample_empty_run_returns_empty_mapping() -> None:
    run = _run_with_attempts((), attempts=2)
    assert pass_at_k_by_sample(run, k=2) == {}


# --- build_report_aggregates: the full aggregates payload a report can use --


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
    # _run() builds a manifest where attempts defaults to 1. "pass@1" (the
    # odds that 1 out of 1 attempts passes) is just the same pass/fail bit
    # that aggregate_run already reports directly -- reporting it again
    # under a different name would not tell us anything new.
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
    # This call must not raise an error: every value in the returned
    # mapping needs to be something json.dumps can handle, because this is
    # exactly the data a Reporter.write(aggregates=...) call will later
    # write out to disk as a report file.
    json.dumps(build_report_aggregates(run))


# --- clustered_interval: cluster-robust bounds for repeated attempts (ADR-0016)


def test_clustered_interval_matches_inline_closed_form() -> None:
    from statistics import NormalDist, fmean, stdev

    # These cluster means are chosen to stay safely inside [0, 1], so the
    # safety clamp described in clustered_interval's own docstring never
    # kicks in here -- this test is purely checking the real formula, mean
    # +/- z * stdev / sqrt(m), recomputed by hand from Python's statistics
    # module (never copied in as a pre-computed decimal number).
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
    # With only one cluster, there is nothing to compare it against to
    # measure how much clusters vary from each other, so the spread is
    # undefined and the bounds must come back as (None, None) -- the same
    # "say we don't know rather than fake an answer" rule tested for
    # wilson_interval in
    # test_wilson_interval_empty_denominator_returns_none_bounds.
    assert clustered_interval(cluster_means=[0.5]) == (None, None)


def test_clustered_interval_clamps_bounds_to_unit_range() -> None:
    from statistics import NormalDist, fmean, stdev

    cluster_means = [0.0, 0.5, 1.0]
    z = NormalDist().inv_cdf(0.975)
    center = fmean(cluster_means)
    spread = z * (stdev(cluster_means) / math.sqrt(len(cluster_means)))
    # Sanity check: confirm the raw, un-clamped bounds really do fall
    # outside the valid [0, 1] range here, so we know the clamping below is
    # actually doing something -- not just checking a case where clamping
    # was never needed anyway.
    assert center - spread < 0.0
    assert center + spread > 1.0
    assert clustered_interval(cluster_means=cluster_means) == (0.0, 1.0)


# --- aggregate_run: interval-method selection and score_estimate (ADR-0016) --


def test_aggregate_run_single_attempt_pass_rate_interval_method_is_wilson() -> None:
    # With attempts == 1 (the default from _run()), each sample was only
    # attempted once, so the pass rate's confidence interval is just the
    # plain Wilson interval computed over the raw pass count -- labeled
    # with IntervalMethod.WILSON. Its bounds must match a direct call to
    # wilson_interval exactly, not just approximately.
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

    # Three different questions (sample_ids), each attempted twice: 0/2,
    # 1/2, and 2/2 of their attempts passed, respectively. That gives
    # per-sample_id pass fractions of [0.0, 0.5, 1.0]. The overall
    # numerator/denominator/value stay exact (3 passes out of 6 total
    # attempts either way), but because the same questions were attempted
    # more than once, the confidence bounds must come from
    # clustered_interval instead of a plain Wilson interval, and the
    # recorded method must say so (CLUSTER_ROBUST).
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
    # Only one distinct question (sample_id) here, but it was attempted
    # more than once. That means there is only a single cluster to work
    # with, so there is no way to measure how much clusters vary from each
    # other -- the bounds must come back as (None, None), even though the
    # exact pass counts themselves are still perfectly well-defined and
    # reportable.
    samples = _repeated_attempts("s0", statuses=(GradeStatus.PASS, GradeStatus.FAIL))
    stats = aggregate_run(_run_with_attempts(samples, attempts=2))
    assert stats.pass_rate.interval_method is IntervalMethod.CLUSTER_ROBUST
    assert stats.pass_rate.numerator == 1
    assert stats.pass_rate.denominator == 2
    assert stats.pass_rate.value == pytest.approx(1 / 2)
    assert stats.pass_rate.lower_bound is None
    assert stats.pass_rate.upper_bound is None


def test_aggregate_run_score_estimate_is_none_with_fewer_than_two_scores() -> None:
    # A single numeric score has an average (itself, trivially) but no
    # spread to build a confidence interval from -- so score_estimate must
    # be None even though score_mean is a real, defined number. This is the
    # companion case to
    # test_aggregate_run_score_mean_is_none_with_no_defined_scores, which
    # covers having zero scores instead of exactly one.
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

    # The plain case (attempts == 1) over three defined scores. The
    # estimate's mean must equal score_mean exactly, with no rounding drift
    # between the two. Unlike a pass rate, a score average is not a
    # probability, so its confidence bounds are NOT forced into [0, 1] here
    # -- and indeed they genuinely fall below 0 and above 1 in this
    # example, which is fine. Because this is the plain,
    # one-attempt-per-sample case (not the cluster-robust one),
    # interval_method must be None rather than naming either of the two
    # rate-interval methods.
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

    # attempts > 1, and at least 2 distinct sample_ids have numeric scores:
    # score_estimate must use the cluster-robust method (its standard error
    # computed over each sample_id's average score, not over every
    # individual score), while its mean stays exactly equal to the overall
    # pooled score_mean.
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
    assert estimate.n == 6  # six defined scores in total, spread across the three clusters

    # _repeated_attempts (defined above) scores a PASS as 1.0 and anything
    # else as 0.0, so each sample_id's average score across its two
    # attempts works out to [0.0, 0.5, 1.0] here.
    cluster_mean_scores = [0.0, 0.5, 1.0]
    z = NormalDist().inv_cdf(0.975)
    sem = stdev(cluster_mean_scores) / math.sqrt(len(cluster_mean_scores))
    assert estimate.sem == pytest.approx(sem)
    assert estimate.lower_bound == pytest.approx(stats.score_mean - z * sem)
    assert estimate.upper_bound == pytest.approx(stats.score_mean + z * sem)


def test_aggregate_run_score_estimate_single_cluster_multi_attempt_has_none_spread() -> None:
    # attempts > 1, but only one distinct sample_id: there are two defined
    # scores in total (so score_estimate does have a real mean to report),
    # but only one cluster mean to work with -- so there is nothing to
    # measure cluster-to-cluster spread from, and sem/lower_bound/upper_bound
    # must all come back None.
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

    # s0: one attempt scored PASS (1.0), the other attempt errored out with
    # no grade at all -> its cluster mean is computed over [1.0] only,
    # ignoring the errored attempt. s1: both attempts errored, so it has no
    # score at all -> it contributes no cluster mean whatsoever (never a
    # made-up 0.0 standing in for "no data"). s2: one attempt scored FAIL
    # (0.0), the other errored -> its cluster mean is computed over [0.0]
    # only. Together these three cases exercise both "an attempt with no
    # score gets skipped" and "a sample_id with no scored attempts at all
    # contributes nothing," inside the cluster-robust code path.
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
    assert estimate.n == 2  # only s0's 1.0 and s2's 0.0 count as defined scores; s1 has none

    cluster_mean_scores = [1.0, 0.0]  # s0's cluster mean is 1.0, s2's is 0.0; s1 contributes none
    z = NormalDist().inv_cdf(0.975)
    sem = stdev(cluster_mean_scores) / math.sqrt(len(cluster_mean_scores))
    assert estimate.sem == pytest.approx(sem)
    assert estimate.lower_bound == pytest.approx(stats.score_mean - z * sem)
    assert estimate.upper_bound == pytest.approx(stats.score_mean + z * sem)
