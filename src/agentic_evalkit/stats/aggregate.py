"""Run aggregation, exact outcome counting, and Wilson confidence intervals.

Design §10 and ADR-0008 require every report to retain sample-level
outcomes and separate operational failures (errors, timeouts,
cancellations, unavailable capabilities) from definitive grading outcomes
(pass/fail/partial/abstain), so a system that crashed cannot be reported
as if it had failed the task.

``aggregate_run`` recounts every outcome directly from ``run.samples``
rather than trusting any caller-supplied :class:`~agentic_evalkit.models.RunSummary`
so aggregation is always correct even if the summary attached to a run is
stale or wrong. It classifies each :class:`~agentic_evalkit.models.SampleResult`
using both its execution status and its grade status: an execution failure
(error/timeout/cancelled) is counted as that operational outcome even when
no grade was ever produced, and only a completed execution with a
definitive grade status contributes to the pass/fail/partial/abstain/
unavailable counts.

``wilson_interval`` computes a 95% Wilson score interval for a binary rate
using :class:`statistics.NormalDist` -- no numpy/scipy dependency.

``build_report_aggregates`` and ``pass_at_k_by_sample`` are the CLI-facing
seam that closes the gap between this module existing and a report actually
carrying its numbers: every reporter's ``write()`` accepts an optional
``aggregates: dict[str, JsonValue] | None`` (``agentic_evalkit.reporters.base.Reporter``),
documented there as "supplied by a caller that already ran
``agentic_evalkit.stats``" -- ``build_report_aggregates`` is exactly that
call, so ``agentic_evalkit.cli.runs.write_canonical_report`` and
``agentic_evalkit.cli.reports.report`` can both produce it with one line
rather than each re-deriving the same shape.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from statistics import NormalDist, fmean, stdev
from typing import TYPE_CHECKING, cast

from pydantic import JsonValue

from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.models.execution import ExecutionStatus
from agentic_evalkit.models.grades import GradeStatus
from agentic_evalkit.stats.reliability import pass_at_k

if TYPE_CHECKING:
    from agentic_evalkit.models.runs import EvalRunResult, SampleResult

__all__ = [
    "AggregateStats",
    "ContinuousEstimate",
    "IntervalMethod",
    "RateEstimate",
    "ResourceDistribution",
    "aggregate_run",
    "build_report_aggregates",
    "clustered_interval",
    "pass_at_k_by_sample",
    "wilson_interval",
]

# The 97.5th-percentile z-score of the standard normal distribution, i.e.
# the two-sided 95% critical value (Task 12 Step 4: "Wilson bounds using
# statistics.NormalDist().inv_cdf(0.975)"). Computed once at import time
# since it never depends on run data.
_Z_95: float = NormalDist().inv_cdf(0.975)


class IntervalMethod(StrEnum):
    """Which construction produced a rate's or score mean's interval bounds.

    ``WILSON`` is the per-observation Wilson score interval used when every
    observation is an independent trial (``attempts == 1``). ``CLUSTER_ROBUST``
    is the per-``sample_id`` cluster-robust Wald interval used when
    ``attempts > 1`` groups correlated repeated attempts, so the interval is
    not narrowed by pseudo-replication (ADR-0016). Stamped on the estimate so a
    consumer can tell, machine-readably, which method produced the bounds --
    a ``StrEnum`` rather than a free string, per the fixed-vocabulary rule
    ADR-0002 applies to every wire status.
    """

    WILSON = "wilson"
    CLUSTER_ROBUST = "cluster_robust"


def wilson_interval(*, successes: int, total: int) -> tuple[float | None, float | None]:
    """Return the 95% Wilson score interval for ``successes / total``.

    The Wilson interval is preferred over the naive normal approximation
    because it stays within ``[0, 1]`` and remains well-behaved at the
    extremes (``successes == 0`` or ``successes == total``), where a naive
    interval would incorrectly claim zero-width certainty.

    Args:
        successes: Exact count of successes. Must satisfy
            ``0 <= successes <= total``.
        total: Exact count of trials (the denominator).

    Returns:
        A ``(lower_bound, upper_bound)`` tuple. Both are ``None`` when
        ``total == 0`` -- an empty denominator has no defined confidence
        interval, and must never be reported as a misleadingly certain
        ``(0.0, 0.0)`` or similar.

    Raises:
        ValueError: If ``successes`` is negative or exceeds ``total``.
    """
    if successes < 0:
        raise ValueError(f"successes must be >= 0 (got {successes})")
    if successes > total:
        raise ValueError(
            f"successes must satisfy successes <= total (got successes={successes}, total={total})"
        )
    if total == 0:
        return (None, None)

    z = _Z_95
    p = successes / total
    n = total
    denominator = 1 + (z**2) / n
    center = p + (z**2) / (2 * n)
    spread = z * ((p * (1 - p) / n) + (z**2) / (4 * n**2)) ** 0.5
    lower = (center - spread) / denominator
    upper = (center + spread) / denominator
    # Clamp for floating-point safety only; the closed-form Wilson bound is
    # mathematically guaranteed to lie within [0, 1] already.
    return (max(0.0, lower), min(1.0, upper))


def clustered_interval(*, cluster_means: Sequence[float]) -> tuple[float | None, float | None]:
    """Return a 95% cluster-robust Wald interval over per-cluster means.

    When ``attempts > 1``, repeated attempts sharing a ``sample_id`` are
    correlated, not independent Bernoulli trials, so a Wilson interval over the
    pooled per-attempt count treats each attempt as independent and reports a
    narrower, more-certain interval than the data supports (pseudo-replication).
    This instead treats each ``sample_id`` cluster as one observation -- its
    mean -- and returns ``mean(cluster_means) +/- z * stdev(cluster_means) /
    sqrt(m)``, where ``m`` is the number of clusters and ``z`` is the same
    two-sided 95% critical value :func:`wilson_interval` uses. Cluster-robust
    SE over few distinct ``sample_id``\\ s is still only approximate and should
    not be over-trusted -- the same "never claim more certainty than the data
    supports" caution that motivates Wilson over the naive normal approximation.

    Args:
        cluster_means: One value per ``sample_id`` cluster (for a rate, that
            cluster's pass proportion, each in ``[0, 1]``).

    Returns:
        A ``(lower_bound, upper_bound)`` tuple, clamped to ``[0, 1]`` for the
        same floating-point safety :func:`wilson_interval` applies. Both are
        ``None`` when ``m < 2`` -- the between-cluster variance is undefined
        with a single cluster, and must never be reported as a misleadingly
        certain zero-width interval, mirroring :func:`wilson_interval`'s
        empty-denominator ``(None, None)`` case.
    """
    m = len(cluster_means)
    if m < 2:
        return (None, None)
    center = fmean(cluster_means)
    spread = _Z_95 * (stdev(cluster_means) / math.sqrt(m))
    return (max(0.0, center - spread), min(1.0, center + spread))


class RateEstimate(FrozenModel):
    """An exact binary rate with its 95% confidence interval.

    ``numerator``/``denominator`` are exact integers so a report can always
    show precisely how many of how many, never a pre-rounded float alone.
    ``interval_method`` records which construction produced ``lower_bound``/
    ``upper_bound`` -- :attr:`IntervalMethod.WILSON` for the independent-trial
    (``attempts == 1``) case, :attr:`IntervalMethod.CLUSTER_ROBUST` for the
    repeated-attempt case (ADR-0016). Additive, optional, and default-``None``
    so it stays within ``schema_version = "1"`` (ADR-0002).
    """

    numerator: int
    denominator: int
    value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    interval_method: IntervalMethod | None = None


class ContinuousEstimate(FrozenModel):
    """A continuous mean with its standard error and 95% confidence interval.

    Carries the uncertainty design section 10 requires for a "mean" exactly as
    :class:`RateEstimate` does for a rate. ``mean`` is always defined (the
    estimate is only built when at least two scores exist); ``sem`` and the
    bounds are ``None`` when the spread is undefined (a single cluster under the
    cluster-robust regime), never a fabricated zero-width interval. Unlike a
    rate, a score mean is not a probability, so the bounds are not clamped to
    ``[0, 1]``. ``interval_method`` is :attr:`IntervalMethod.CLUSTER_ROBUST`
    under repeated attempts and ``None`` for the flat, one-attempt SEM interval,
    which corresponds to no named binary-rate method (ADR-0016). ``n`` is the
    raw count of defined scores (``score_mean``'s denominator), not necessarily
    ``sem``'s divisor: under :attr:`IntervalMethod.CLUSTER_ROBUST` the standard
    error is computed over the distinct ``sample_id`` clusters, which may
    number fewer than ``n``, so ``sem != stdev(scores) / sqrt(n)`` in general.
    """

    mean: float
    n: int
    sem: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    interval_method: IntervalMethod | None = None


class ResourceDistribution(FrozenModel):
    """Count/mean/p50/p95 summary for a resource metric (latency/tokens/cost).

    Built only from samples that actually reported a value for this metric;
    a target that never reports latency contributes nothing here rather
    than an implicit zero.
    """

    count: int
    mean: float
    p50: float
    p95: float


class AggregateStats(FrozenModel):
    """Recounted, provenance-independent statistics for one run (design §10)."""

    total: int
    passed: int
    failed: int
    partial: int
    errors: int
    timeouts: int
    cancelled: int
    abstained: int
    unavailable: int
    pass_rate: RateEstimate
    score_mean: float | None = None
    score_count: int = 0
    score_estimate: ContinuousEstimate | None = None
    latency_ms: ResourceDistribution | None = None
    input_tokens: ResourceDistribution | None = None
    output_tokens: ResourceDistribution | None = None
    cost_usd: ResourceDistribution | None = None


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile over an already-sorted, nonempty list.

    Uses the common "nearest rank" method (ceil(fraction * n), 1-indexed)
    so p50/p95 always resolve to an actually-observed value rather than an
    interpolated one, matching the exact/no-fabricated-values spirit of
    Task 12 without requiring numpy.
    """
    n = len(sorted_values)
    rank = max(1, math.ceil(fraction * n))
    return sorted_values[min(rank, n) - 1]


def _distribution(values: list[float]) -> ResourceDistribution | None:
    if not values:
        return None
    ordered = sorted(values)
    return ResourceDistribution(
        count=len(ordered),
        mean=sum(ordered) / len(ordered),
        p50=_percentile(ordered, 0.50),
        p95=_percentile(ordered, 0.95),
    )


def _classify(sample: SampleResult) -> str:
    """Return the single outcome bucket one sample contributes to.

    Execution-level operational failures (error/timeout/cancelled) always
    win regardless of whether a grade happens to be attached, since those
    statuses mean the pipeline never reached a trustworthy grading
    decision. Only a completed execution is classified by its grade
    status; a completed execution with no grade at all (e.g. grading was
    skipped) is treated as an error rather than silently dropped.
    """
    execution = sample.execution
    if execution.status is ExecutionStatus.ERROR:
        return "errors"
    if execution.status is ExecutionStatus.TIMEOUT:
        return "timeouts"
    if execution.status is ExecutionStatus.CANCELLED:
        return "cancelled"
    if execution.status is ExecutionStatus.FAILED:
        return "failed"

    grade = sample.grade
    if grade is None:
        # Completed execution but no grade was ever produced: an
        # operational gap, not a definitive task failure.
        return "errors"
    if grade.status is GradeStatus.PASS:
        return "passed"
    if grade.status is GradeStatus.FAIL:
        return "failed"
    if grade.status is GradeStatus.PARTIAL:
        return "partial"
    if grade.status is GradeStatus.ABSTAIN:
        return "abstained"
    if grade.status is GradeStatus.UNAVAILABLE:
        return "unavailable"
    if grade.status is GradeStatus.ERROR:
        return "errors"
    return "errors"  # pragma: no cover - exhaustive GradeStatus enum above


def _attempts_by_sample_id(samples: Sequence[SampleResult]) -> dict[str, list[SampleResult]]:
    """Group samples by ``sample.sample.sample_id`` (design §10, ADR-0016).

    A manifest with ``attempts > 1`` produces one
    :class:`~agentic_evalkit.models.SampleResult` per attempt, all sharing the
    same sample ID. Both :func:`pass_at_k_by_sample` (per-sample ``pass@k``) and
    the cluster-robust interval path (:func:`aggregate_run` under repeated
    attempts) group by that ID, so the grouping lives here once rather than
    being duplicated in each caller.
    """
    attempts_by_sample: dict[str, list[SampleResult]] = {}
    for sample in samples:
        attempts_by_sample.setdefault(sample.sample.sample_id, []).append(sample)
    return attempts_by_sample


def _cluster_pass_proportions(samples: Sequence[SampleResult]) -> list[float]:
    """Per-``sample_id`` fraction of attempts that graded PASS.

    Each cluster is one observation for the cluster-robust interval; its value
    is that cluster's PASS count over its attempt count, counted with the same
    :func:`_classify` rule as the pooled ``passed`` total so the cluster means
    stay consistent with the exact pooled numerator.
    """
    return [
        sum(1 for attempt in attempts if _classify(attempt) == "passed") / len(attempts)
        for attempts in _attempts_by_sample_id(samples).values()
    ]


def _cluster_mean_scores(samples: Sequence[SampleResult]) -> list[float]:
    """Per-``sample_id`` mean of that cluster's defined grade scores.

    Only clusters with at least one defined numeric score contribute; a
    ``sample_id`` whose every attempt lacked a score adds nothing (never a
    fabricated ``0.0``), mirroring :func:`aggregate_run`'s score-mean denominator.
    """
    means: list[float] = []
    for attempts in _attempts_by_sample_id(samples).values():
        cluster_scores: list[float] = []
        for attempt in attempts:
            grade = attempt.grade
            if grade is not None and grade.score is not None:
                cluster_scores.append(grade.score)
        if cluster_scores:
            means.append(sum(cluster_scores) / len(cluster_scores))
    return means


def _sem_interval(
    *, center: float, units: Sequence[float]
) -> tuple[float | None, float | None, float | None]:
    """Return ``(sem, lower, upper)`` for a normal-approximation mean interval.

    ``units`` are the observations whose sample standard deviation drives the
    standard error: raw per-observation scores in the flat regime, or per-cluster
    mean scores in the cluster-robust regime. Fewer than two units leaves the
    spread undefined, so all three are ``None`` -- never a fabricated zero-width
    interval, the same discipline :func:`wilson_interval` and
    :func:`clustered_interval` keep for an undefined denominator/variance.
    """
    count = len(units)
    if count < 2:
        return (None, None, None)
    sem = stdev(units) / math.sqrt(count)
    return (sem, center - _Z_95 * sem, center + _Z_95 * sem)


def _score_estimate(
    *,
    scores: Sequence[float],
    score_mean: float | None,
    cluster_mean_scores: Sequence[float] | None,
) -> ContinuousEstimate | None:
    """Build the SEM/CI estimate for the mean grade score, or ``None``.

    ``scores`` is every defined grade score (the exact list ``score_mean`` is
    computed from); ``cluster_mean_scores`` is the per-``sample_id`` mean score
    when ``attempts > 1`` (cluster-robust regime) or ``None`` for the flat,
    one-attempt regime. Returns ``None`` when fewer than two scores are defined:
    a single score has a mean but no spread, so no honest interval exists
    (mirroring ``score_mean``'s own None-on-undefined discipline). The estimate's
    ``mean`` is the exact pooled ``score_mean`` -- like ``RateEstimate.value``,
    only the interval's derivation changes between regimes, never the point
    estimate.
    """
    n = len(scores)
    if n < 2 or score_mean is None:
        return None
    if cluster_mean_scores is None:
        sem, lower, upper = _sem_interval(center=score_mean, units=scores)
        method: IntervalMethod | None = None
    else:
        sem, lower, upper = _sem_interval(center=score_mean, units=cluster_mean_scores)
        method = IntervalMethod.CLUSTER_ROBUST
    return ContinuousEstimate(
        mean=score_mean,
        n=n,
        sem=sem,
        lower_bound=lower,
        upper_bound=upper,
        interval_method=method,
    )


def aggregate_run(run: EvalRunResult) -> AggregateStats:
    """Recount every outcome and resource metric from ``run.samples``.

    Never trusts ``run.summary``; every count here is derived solely from
    ``run.samples`` so aggregation is correct even for a run whose attached
    summary is stale.

    When ``run.manifest.attempts > 1``, repeated attempts at one ``sample_id``
    are correlated, so ``pass_rate``'s bounds (and ``score_estimate``'s) are
    computed cluster-robustly over per-``sample_id`` clusters
    (:func:`clustered_interval`) rather than over the pooled per-attempt count,
    and ``pass_rate.interval_method`` records which construction was used.
    ``numerator``/``denominator``/``value`` stay the exact pooled counts either
    way -- only the interval's derivation changes. Cluster-robust SE over few
    distinct ``sample_id``\\ s remains approximate and should not be over-trusted.

    Args:
        run: The complete run to aggregate.

    Returns:
        Exact outcome counts, a Wilson- or cluster-robust-bounded pass rate, a
        score mean (and matching ``score_estimate`` SEM/CI) over only the samples
        with a defined numeric grade score, and count/mean/p50/p95 resource
        distributions for whichever of latency/input-tokens/output-tokens/cost
        were actually reported.
    """
    counts = {
        "passed": 0,
        "failed": 0,
        "partial": 0,
        "errors": 0,
        "timeouts": 0,
        "cancelled": 0,
        "abstained": 0,
        "unavailable": 0,
    }
    scores: list[float] = []
    latencies: list[float] = []
    input_tokens: list[float] = []
    output_tokens: list[float] = []
    costs: list[float] = []

    for sample in run.samples:
        counts[_classify(sample)] += 1

        if sample.grade is not None and sample.grade.score is not None:
            scores.append(sample.grade.score)

        execution = sample.execution
        if execution.latency_ms is not None:
            latencies.append(execution.latency_ms)
        if execution.input_tokens is not None:
            input_tokens.append(float(execution.input_tokens))
        if execution.output_tokens is not None:
            output_tokens.append(float(execution.output_tokens))
        if execution.cost_usd is not None:
            costs.append(execution.cost_usd)

    total = len(run.samples)
    passed = counts["passed"]
    attempts = run.manifest.attempts
    if attempts > 1:
        # Repeated attempts at one sample_id are correlated: cluster-robust
        # bounds over per-sample_id pass proportions, not pooled Wilson.
        lower, upper = clustered_interval(cluster_means=_cluster_pass_proportions(run.samples))
        interval_method = IntervalMethod.CLUSTER_ROBUST
    else:
        lower, upper = wilson_interval(successes=passed, total=total)
        interval_method = IntervalMethod.WILSON
    pass_rate = RateEstimate(
        numerator=passed,
        denominator=total,
        value=(passed / total) if total > 0 else None,
        lower_bound=lower,
        upper_bound=upper,
        interval_method=interval_method,
    )

    score_mean = (sum(scores) / len(scores)) if scores else None
    cluster_mean_scores = _cluster_mean_scores(run.samples) if attempts > 1 else None
    score_estimate = _score_estimate(
        scores=scores,
        score_mean=score_mean,
        cluster_mean_scores=cluster_mean_scores,
    )

    return AggregateStats(
        total=total,
        passed=passed,
        failed=counts["failed"],
        partial=counts["partial"],
        errors=counts["errors"],
        timeouts=counts["timeouts"],
        cancelled=counts["cancelled"],
        abstained=counts["abstained"],
        unavailable=counts["unavailable"],
        pass_rate=pass_rate,
        score_mean=score_mean,
        score_count=len(scores),
        score_estimate=score_estimate,
        latency_ms=_distribution(latencies),
        input_tokens=_distribution(input_tokens),
        output_tokens=_distribution(output_tokens),
        cost_usd=_distribution(costs),
    )


def pass_at_k_by_sample(run: EvalRunResult, *, k: int) -> dict[str, float]:
    """Return each sample's ``pass@k`` estimate over its actually-run attempts.

    Groups ``run.samples`` by ``sample.sample.sample_id`` (a manifest with
    ``attempts > 1`` produces one :class:`~agentic_evalkit.models.SampleResult`
    per attempt, all sharing the same sample ID) and calls
    :func:`~agentic_evalkit.stats.reliability.pass_at_k` once per group with
    ``total_attempts`` set to that group's actual attempt count and
    ``successful_attempts`` set to how many of them graded
    :attr:`~agentic_evalkit.models.GradeStatus.PASS`.

    A sample whose group has fewer than ``k`` attempts is silently omitted
    (never fabricated as ``0.0`` or ``1.0``): ``pass_at_k`` requires
    ``1 <= k <= total_attempts``, and a sample that was not actually
    attempted ``k`` times has no defined ``pass@k`` estimate.

    Args:
        run: The complete run to compute per-sample ``pass@k`` for.
        k: Number of attempts hypothetically sampled per sample; typically
            ``run.manifest.attempts`` (every sample's full attempt budget).

    Returns:
        A mapping from ``sample_id`` to its ``pass@k`` estimate, covering
        only sample IDs whose attempt count is at least ``k``.
    """
    attempts_by_sample = _attempts_by_sample_id(run.samples)

    estimates: dict[str, float] = {}
    for sample_id, attempts in attempts_by_sample.items():
        total_attempts = len(attempts)
        if total_attempts < k:
            continue
        successful_attempts = sum(
            1
            for attempt in attempts
            if attempt.grade is not None and attempt.grade.status is GradeStatus.PASS
        )
        estimates[sample_id] = pass_at_k(
            total_attempts=total_attempts, successful_attempts=successful_attempts, k=k
        )
    return estimates


def build_report_aggregates(run: EvalRunResult) -> dict[str, JsonValue]:
    """Compute the full ``aggregates`` payload a report should carry for ``run``.

    Combines :func:`aggregate_run` (exact outcome counts, the Wilson-bounded
    pass rate, and latency/token/cost distributions) with
    :func:`pass_at_k_by_sample` (only when ``run.manifest.attempts > 1`` --
    with a single attempt per sample, ``pass@1`` over one attempt is exactly
    the pass/fail bit already in ``aggregate_run``'s counts, so reporting it
    a second time would be redundant, not informative) into the one
    JSON-compatible mapping every :class:`~agentic_evalkit.reporters.base.Reporter`
    accepts as its optional ``aggregates`` argument.

    Never fabricates a ``pass_at_k`` entry when no sample actually ran ``k``
    attempts: if :func:`pass_at_k_by_sample` returns an empty mapping (e.g.
    ``manifest.attempts > 1`` was configured but every sample errored before
    any attempt count could reach ``k``), the ``"pass_at_k"`` key is omitted
    entirely rather than reporting a mean of zero samples.
    """
    aggregates = cast("dict[str, JsonValue]", aggregate_run(run).model_dump(mode="json"))

    attempts = run.manifest.attempts
    if attempts > 1:
        per_sample = pass_at_k_by_sample(run, k=attempts)
        if per_sample:
            values = list(per_sample.values())
            aggregates["pass_at_k"] = {
                "k": attempts,
                "mean": sum(values) / len(values),
                "by_sample_id": cast("JsonValue", per_sample),
            }
    return aggregates
