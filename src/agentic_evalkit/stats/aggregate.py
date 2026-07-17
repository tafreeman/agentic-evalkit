"""Counting up results for a run, and putting confidence intervals around them.

Design §10 and ADR-0008 require every report to keep every individual
sample's outcome, and to keep two kinds of failure clearly separate:
"operational" failures where something broke before we could judge the
task at all (errors, timeouts, cancellations, unavailable capabilities),
versus "definitive" grading outcomes (pass/fail/partial/abstain) where the
task actually ran and got judged. This split matters because a system that
crashed should never be reported as if it had tried the task and failed
it.

``aggregate_run`` re-counts every outcome by walking ``run.samples``
itself, rather than trusting whatever count is already attached to the
run's summary (a :class:`~agentic_evalkit.models.RunSummary`, which might
be stale or wrong) -- so the totals in a report are always correct, even
if something else miscounted earlier. For each
:class:`~agentic_evalkit.models.SampleResult` it looks at both the
execution status and the grade status: if the run itself failed (error,
timeout, cancelled), that counts as an operational failure regardless of
whether a grade also happened to be attached. Only a sample that finished
running *and* has a definitive grade gets counted into the
pass/fail/partial/abstain/unavailable buckets.

``wilson_interval`` computes a "confidence interval" for a pass rate -- a
range that's likely to contain the true underlying rate, given that we
only tested a limited number of samples. A narrower range means we're more
sure of the number; a wider range means we should be more cautious about
trusting it. It uses a specific, well-established method called the
"Wilson score interval" (see that function's own docstring for why), built
from :class:`statistics.NormalDist` in Python's standard library -- so
this doesn't need numpy or scipy as a dependency.

``build_report_aggregates`` and ``pass_at_k_by_sample`` exist so that a
report can actually carry these numbers end-to-end: every reporter's
``write()`` method accepts an optional
``aggregates: dict[str, JsonValue] | None`` argument
(``agentic_evalkit.reporters.base.Reporter``), documented there as
"supplied by a caller that already ran ``agentic_evalkit.stats``".
``build_report_aggregates`` is exactly that caller, so
``agentic_evalkit.cli.runs.write_canonical_report`` and
``agentic_evalkit.cli.reports.report`` can each produce the full
aggregates payload in one line instead of each having to re-derive the
same shape themselves.
"""

from __future__ import annotations

import math
from enum import StrEnum
from statistics import NormalDist, fmean, stdev
from typing import TYPE_CHECKING, cast

from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.models.execution import ExecutionStatus
from agentic_evalkit.models.grades import GradeStatus
from agentic_evalkit.stats.reliability import pass_at_k

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import JsonValue

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

# A fixed number used throughout this module for building 95% confidence
# intervals: how many "standard deviations" out from the center of a
# normal (bell-curve) distribution you need to go to capture the middle
# 95% of it (leaving 2.5% in each of the two tails). Statisticians call
# this a "z-score" or "critical value" for a two-sided 95% interval. (Task
# 12 Step 4: "Wilson bounds using statistics.NormalDist().inv_cdf(0.975)".)
# Computed once here, at import time, since it's a fixed constant that
# never depends on any run's data.
_Z_95: float = NormalDist().inv_cdf(0.975)


class IntervalMethod(StrEnum):
    """Which method was used to build a confidence interval around a rate or a mean score.

    Which one applies depends on whether each sample was attempted once or
    multiple times:

    ``WILSON`` is used when every observation is an independent trial --
    i.e. each question in the dataset was only attempted once
    (``attempts == 1``). It's the "Wilson score interval" computed by
    :func:`wilson_interval` (see that function for what it is and why).

    ``CLUSTER_ROBUST`` is used when ``attempts > 1``, meaning the same
    question was attempted multiple times. Treating every attempt as a
    fully independent data point in that case would be a statistical
    mistake called "pseudo-replication": repeated attempts at the *same*
    question tend to succeed or fail together more than attempts at
    different questions do, so counting them as independent would make our
    confidence interval look falsely narrow (i.e. falsely confident). To
    avoid that, this groups all the attempts for a given ``sample_id`` into
    one cluster, takes each cluster's own average, and builds the interval
    from those per-cluster averages instead -- a "cluster-robust" interval,
    of the simple mean-plus-or-minus-margin form statisticians call a
    "Wald interval" (ADR-0016 -- see :func:`clustered_interval` for the
    exact formula).

    This choice is recorded directly on the estimate (as a ``StrEnum`` -- a
    fixed set of named string values, rather than any arbitrary free-form
    string) so that anything reading the result later can tell, without
    guessing, which of the two methods produced the bounds. Using a fixed,
    named set of values here, rather than any string that happens to be
    convenient, follows the same "every status comes from a known, fixed
    list" rule that ADR-0002 requires for every value written out to a
    report.
    """

    WILSON = "wilson"
    CLUSTER_ROBUST = "cluster_robust"


def wilson_interval(*, successes: int, total: int) -> tuple[float | None, float | None]:
    """Return a 95% confidence interval for a pass rate (``successes / total``).

    A confidence interval is a range that's likely to contain the true
    underlying rate, given that we only observed a limited number of
    trials -- e.g. if 8 out of 10 samples passed, the true long-run pass
    rate probably isn't exactly 80%, but a range like "50% to 95%" might be
    a good bet for where it really falls. This function computes that
    range using a specific, well-established formula called the "Wilson
    score interval."

    We use the Wilson formula instead of a simpler, more naive approach
    (assuming the pass/fail rate follows a plain bell curve) because the
    naive approach breaks down at the edges: if you got 0 out of 10, or 10
    out of 10, the naive method would claim a range of exactly zero width
    -- i.e. "we are 100% certain the true rate is exactly 0% (or 100%)" --
    which is clearly overconfident given how little data we have. The
    Wilson formula avoids that: it always stays within the valid
    ``[0, 1]`` range and gives a sensible, non-zero-width answer even at
    these extremes.

    Args:
        successes: Exact count of successes. Must satisfy
            ``0 <= successes <= total``.
        total: Exact count of trials (the denominator).

    Returns:
        A ``(lower_bound, upper_bound)`` tuple. Both are ``None`` when
        ``total == 0`` -- with zero trials there's no data to build any
        interval from, and returning something like ``(0.0, 0.0)`` here
        would misleadingly look like a confident, certain answer instead
        of "we don't know."

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
    # This clamp is only a safety net for tiny floating-point rounding
    # errors; the Wilson formula above is mathematically guaranteed to
    # already produce a value within [0, 1] on its own.
    return (max(0.0, lower), min(1.0, upper))


def clustered_interval(*, cluster_means: Sequence[float]) -> tuple[float | None, float | None]:
    """Return a 95% confidence interval built from per-cluster averages,
    for data where observations come in correlated groups rather than
    being fully independent.

    When ``attempts > 1``, the same question (``sample_id``) was attempted
    more than once, and those repeated attempts tend to succeed or fail
    together more than attempts at unrelated questions would -- they are
    "correlated," not independent yes/no coin-flip trials (what
    statisticians call "Bernoulli trials"). If we ignored that and fed the
    pooled count of all attempts straight into :func:`wilson_interval`, it
    would treat every attempt as if it were an independent data point, and
    report an interval that's narrower (i.e. looks more certain) than the
    data actually supports. Statisticians call this mistake
    "pseudo-replication."

    To avoid it, this function treats each ``sample_id``'s whole cluster
    of attempts as a single observation -- specifically, that cluster's
    average -- and builds the interval from those per-cluster averages
    instead: ``mean(cluster_means) +/- z * stdev(cluster_means) /
    sqrt(m)``, where ``m`` is the number of clusters and ``z`` is the same
    fixed 95%-confidence constant :func:`wilson_interval` uses. This kind
    of "average plus-or-minus a margin" interval is called a "Wald
    interval." Note that when there are only a few distinct ``sample_id``\\ s
    to average over, this estimate of how much clusters typically vary
    (their "standard error") is still only approximate and shouldn't be
    over-trusted -- the same "never claim more certainty than the data
    supports" caution that favors the Wilson interval over a naive one
    elsewhere in this module.

    Args:
        cluster_means: One value per ``sample_id`` cluster (for a rate, that
            cluster's pass proportion, each in ``[0, 1]``).

    Returns:
        A ``(lower_bound, upper_bound)`` tuple, clamped to ``[0, 1]`` for
        the same floating-point-rounding safety :func:`wilson_interval`
        applies. Both are ``None`` when ``m < 2`` -- with only one
        cluster, there's no way to measure how much clusters vary from
        each other, so there's nothing to build a spread from. Returning
        ``(None, None)`` here avoids reporting a misleadingly confident
        zero-width interval, mirroring how :func:`wilson_interval` returns
        ``(None, None)`` when there are zero trials to work with.
    """
    m = len(cluster_means)
    if m < 2:
        return (None, None)
    center = fmean(cluster_means)
    spread = _Z_95 * (stdev(cluster_means) / math.sqrt(m))
    return (max(0.0, center - spread), min(1.0, center + spread))


class RateEstimate(FrozenModel):
    """A pass/fail rate (e.g. "12 out of 20 passed") together with its 95%
    confidence interval.

    ``numerator``/``denominator`` are kept as exact integers, so a report
    can always show precisely how many passed out of how many were run,
    rather than only a rounded-off percentage. ``interval_method`` records
    which of the two methods in :class:`IntervalMethod` produced
    ``lower_bound``/``upper_bound`` -- :attr:`IntervalMethod.WILSON` for
    the case where each sample was attempted once,
    :attr:`IntervalMethod.CLUSTER_ROBUST` for the case where samples were
    attempted multiple times (ADR-0016). This field is additive, optional,
    and defaults to ``None`` so that adding it doesn't break the frozen
    ``schema_version = "1"`` wire format (ADR-0002).
    """

    numerator: int
    denominator: int
    value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    interval_method: IntervalMethod | None = None


class ContinuousEstimate(FrozenModel):
    """The average of a numeric score (not just pass/fail), together with
    its standard error and 95% confidence interval.

    This is the score-mean equivalent of what :class:`RateEstimate` is for
    a pass/fail rate -- both carry the same kind of "how uncertain is this
    number" information that design section 10 requires. ``mean`` is
    always present (this estimate is only built at all when there are at
    least two scores to average). ``sem`` -- short for "standard error of
    the mean," a measure of how much the average would likely wobble if
    we'd tested a different sample of the same size -- and the confidence
    bounds are ``None`` when that spread can't be measured (e.g. only a
    single cluster under the cluster-robust method below); we return
    ``None`` rather than fabricate a fake zero-width interval that would
    look more certain than it is. Unlike a pass/fail rate, a score average
    is not a probability, so its bounds are not forced into the ``[0, 1]``
    range.

    ``interval_method`` is :attr:`IntervalMethod.CLUSTER_ROBUST` when
    samples were attempted more than once (see that enum for why), and
    ``None`` for the plain, one-attempt-per-sample case -- there's no
    separately-named method for that flat case the way ``WILSON`` names
    one for rates (ADR-0016).

    ``n`` is simply the count of scores that went into ``mean`` -- it is
    *not* necessarily the same number used to divide when computing
    ``sem``. Under :attr:`IntervalMethod.CLUSTER_ROBUST`, the standard
    error is computed over the distinct ``sample_id`` clusters, and there
    can be fewer clusters than there are individual scores (``n``) -- so
    in general, ``sem`` is not simply ``stdev(scores) / sqrt(n)``.
    """

    mean: float
    n: int
    sem: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    interval_method: IntervalMethod | None = None


class ResourceDistribution(FrozenModel):
    """Count/mean/p50 (median)/p95 (95th-percentile) summary for a resource
    metric (latency/tokens/cost).

    Built only from samples that actually reported a value for this metric;
    a target that never reports latency contributes nothing here rather
    than an implicit zero.
    """

    count: int
    mean: float
    p50: float
    p95: float


class AggregateStats(FrozenModel):
    """Freshly recounted statistics for one run, computed directly from the
    run's own sample data rather than trusted from any attached summary
    (design §10)."""

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
    """Return the requested percentile (e.g. p50, p95) from an
    already-sorted, non-empty list, using the "nearest rank" method.

    "Nearest rank" means: rather than interpolating between two data
    points to compute a percentile, we always pick an actual value that
    was really observed in the data (computed as ``ceil(fraction * n)``,
    counting positions starting from 1). This keeps p50/p95 as real,
    observed measurements rather than made-up numbers in between two
    measurements, matching this codebase's general preference for exact,
    non-fabricated values (Task 12) -- and it does this without needing
    numpy.
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

    When a manifest specifies ``attempts > 1`` (the same question gets
    attempted more than once), the run produces one
    :class:`~agentic_evalkit.models.SampleResult` per attempt, all sharing
    the same sample ID. Both :func:`pass_at_k_by_sample` (which computes,
    per sample, the odds that at least one of ``k`` attempts would have
    passed -- see :mod:`agentic_evalkit.stats.reliability` for what
    "pass@k" means) and the cluster-robust interval path (used by
    :func:`aggregate_run` under repeated attempts) need to group attempts
    by sample ID, so that grouping logic lives here once instead of being
    duplicated in each caller.
    """
    attempts_by_sample: dict[str, list[SampleResult]] = {}
    for sample in samples:
        attempts_by_sample.setdefault(sample.sample.sample_id, []).append(sample)
    return attempts_by_sample


def _cluster_pass_proportions(samples: Sequence[SampleResult]) -> list[float]:
    """For each ``sample_id``, the fraction of its attempts that graded PASS.

    Each ``sample_id``'s fraction becomes one "cluster" observation used by
    the cluster-robust interval described above: it's that cluster's PASS
    count divided by its attempt count, using the exact same
    :func:`_classify` rule used to compute the overall ``passed`` total --
    so these per-cluster fractions are always consistent with that exact
    total count.
    """
    return [
        sum(1 for attempt in attempts if _classify(attempt) == "passed") / len(attempts)
        for attempts in _attempts_by_sample_id(samples).values()
    ]


def _cluster_mean_scores(samples: Sequence[SampleResult]) -> list[float]:
    """For each ``sample_id``, the average of that cluster's defined (i.e.
    actually present) grade scores.

    Only clusters that have at least one attempt with a real numeric score
    count here; if every attempt for a given ``sample_id`` had no score at
    all, that sample contributes nothing to the list (never a made-up
    ``0.0``) -- the same rule :func:`aggregate_run` follows when deciding
    what counts toward its own overall score-mean calculation.
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
    """Return ``(sem, lower, upper)``: the standard error and confidence
    interval bounds for a mean, using the normal-distribution
    approximation (mean plus-or-minus a margin based on the bell curve).

    ``units`` are the individual observations used to measure how spread
    out the data is (technically: their sample standard deviation is what
    the standard error is computed from). Depending on which case we're
    in, these are either the raw per-observation scores (when every sample
    was attempted only once) or the per-cluster mean scores (in the
    cluster-robust, repeated-attempts case -- see :class:`IntervalMethod`).
    With fewer than two units, there's no way to measure spread at all, so
    all three return values are ``None`` -- never a fabricated zero-width
    interval. This is the same "say we don't know rather than fake a
    confident answer" rule :func:`wilson_interval` and
    :func:`clustered_interval` follow when their own denominator or
    variance is undefined.
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
    """Build the full standard-error/confidence-interval estimate for the
    mean grade score, or return ``None`` if there isn't enough data to
    build one.

    ``scores`` is every grade score that was actually recorded (exactly
    the list that ``score_mean`` itself is computed from).
    ``cluster_mean_scores`` is the per-``sample_id`` average score, used
    when ``attempts > 1`` (the cluster-robust case), or ``None`` in the
    plain one-attempt-per-sample case. This returns ``None`` when fewer
    than two scores exist at all: a single score has an average but
    nothing to measure spread from, so there's no honest interval to
    report (the same "don't report undefined things" rule ``score_mean``
    itself follows). The ``mean`` on the returned estimate is always the
    exact overall ``score_mean`` -- exactly like ``RateEstimate.value``,
    only *how the interval around it is built* changes between the two
    cases, never the headline number itself.
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
    """Recount every outcome and every resource metric (latency, tokens,
    cost) directly from ``run.samples`` -- the individual sample results
    -- rather than from any pre-computed summary.

    This never trusts ``run.summary``: every count returned here is
    derived solely by walking ``run.samples`` one by one, so the result is
    correct even for a run whose attached summary turns out to be stale or
    wrong.

    When ``run.manifest.attempts > 1`` (each question was attempted more
    than once), repeated attempts at the same ``sample_id`` tend to
    succeed or fail together, so they aren't fully independent data
    points. In that case, ``pass_rate``'s confidence interval (and
    ``score_estimate``'s) is built using the cluster-robust method over
    per-``sample_id`` groups (:func:`clustered_interval`) instead of
    treating every individual attempt as independent, and
    ``pass_rate.interval_method`` records which of the two methods was
    actually used. Either way, the exact counts
    (``numerator``/``denominator``/``value``) stay the same -- only how
    the surrounding confidence interval is calculated changes. Note that
    when there are only a few distinct ``sample_id``\\ s to work with, this
    cluster-robust interval is still only an approximation and shouldn't
    be over-trusted.

    Args:
        run: The complete run to aggregate.

    Returns:
        Exact outcome counts; a pass rate with a confidence interval built
        either the Wilson way or the cluster-robust way (see
        :class:`IntervalMethod`); a mean grade score (and a matching
        ``score_estimate`` with its own standard error and confidence
        interval), computed only over the samples that actually have a
        numeric grade score; and count/mean/p50/p95 resource-usage
        summaries for whichever of latency, input tokens, output tokens,
        or cost were actually reported.
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
        # Repeated attempts at the same sample_id are correlated with each
        # other, so we use cluster-robust bounds over per-sample_id pass
        # proportions here, instead of a pooled Wilson interval that would
        # (wrongly) treat every attempt as independent.
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
    """Return each sample's ``pass@k`` estimate, computed from its actual attempts.

    "pass@k" answers: out of ``k`` attempts at the same question, what's
    the probability that at least one of them succeeds? It's a standard
    way of scoring a system that's allowed multiple tries at the same task
    (see :mod:`agentic_evalkit.stats.reliability` for the full explanation
    and formula).

    This groups ``run.samples`` by ``sample.sample.sample_id`` (a manifest
    with ``attempts > 1`` produces one
    :class:`~agentic_evalkit.models.SampleResult` per attempt, all sharing
    the same sample ID), then calls
    :func:`~agentic_evalkit.stats.reliability.pass_at_k` once per group,
    using that group's actual attempt count as ``total_attempts`` and how
    many of those attempts graded
    :attr:`~agentic_evalkit.models.GradeStatus.PASS` as
    ``successful_attempts``.

    A sample whose group ran fewer than ``k`` attempts is silently left
    out of the result (never given a made-up value like ``0.0`` or
    ``1.0``): ``pass_at_k`` requires ``1 <= k <= total_attempts``, and a
    sample that wasn't actually attempted ``k`` times simply has no
    defined ``pass@k`` value to report.

    Args:
        run: The complete run to compute per-sample ``pass@k`` for.
        k: Number of attempts to hypothetically sample per question;
            typically set to ``run.manifest.attempts`` (i.e. every
            sample's full attempt budget).

    Returns:
        A mapping from ``sample_id`` to its ``pass@k`` estimate, covering
        only the sample IDs that were attempted at least ``k`` times.
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
    """Compute the full ``aggregates`` payload that a report should carry for ``run``.

    This combines :func:`aggregate_run` (exact outcome counts, the pass
    rate with its confidence interval, and latency/token/cost
    distributions) with :func:`pass_at_k_by_sample` -- but only when
    ``run.manifest.attempts > 1``. With just a single attempt per sample,
    "pass@1" (succeeding in your one and only attempt) is exactly the same
    as the plain pass/fail count ``aggregate_run`` already produces, so
    reporting it a second time would be redundant rather than genuinely
    new information. The two are merged into the one JSON-compatible
    mapping that every :class:`~agentic_evalkit.reporters.base.Reporter`
    accepts as its optional ``aggregates`` argument.

    This never invents a ``pass_at_k`` entry when no sample actually
    completed ``k`` attempts: if :func:`pass_at_k_by_sample` comes back
    empty (for example, ``manifest.attempts > 1`` was configured, but
    every sample errored out before reaching ``k`` completed attempts),
    the ``"pass_at_k"`` key is left out of the result entirely, rather
    than reporting a meaningless "average of zero samples."
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
