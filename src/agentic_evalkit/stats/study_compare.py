"""Paired multi-arm comparison for bounded acceptance/rejection studies.

``summarize_study_run`` (``stats/study.py``) describes ONE arm. This module
compares N >= 2 arms of the *same* study: every non-baseline arm is paired
against the baseline arm instance-by-instance, so a difference is measured on
the very same tasks rather than on two marginal distributions.

Pairing and eligibility follow the established rules:

* Instances pair on ``sample_id``; repeated attempts aggregate with the same
  all-attempts rule as the per-arm summary (a task is legitimate only when
  every labeled attempt had a legitimate outcome), so correlated attempts are
  never counted as independent evidence (design §10).
* An attempt is *pairable* only when it completed, earned a definitive
  (PASS/FAIL) grade, and carries a valid ``grade.evidence["study"]`` record.
  Malformed study labels raise — they never hide behind an exclusion count.
* Instances missing an attempt, or with an unpairable attempt, on either side
  are excluded and counted — never imputed, never zero-filled.
* Resource metrics (latency, tokens, cost) pair only where BOTH sides report
  the metric for every attempt of the instance; missing coverage is reported
  per metric, not silently dropped from a "total".

Provenance: arms may differ only in ``run_name`` and target identity
(``target_name`` / ``target_fingerprint``) — everything else on the manifest,
plus the resolved dataset, must match exactly, or the comparison fails loudly
with :class:`~agentic_evalkit.errors.IncompatibleRuns` instead of reporting a
misleading delta.

Inference mirrors ``compare_experiment_arms``: an exact McNemar p-value on
discordant instance outcomes plus a seeded, deterministic paired bootstrap
interval for the delta (ADR-0008). The same machinery produces paired
intervals for per-instance resource deltas.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, cast

from agentic_evalkit.errors import IncompatibleRuns, ManifestValidationError
from agentic_evalkit.models.execution import ExecutionStatus
from agentic_evalkit.models.grades import GradeStatus
from agentic_evalkit.stats.study import _study_record

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agentic_evalkit.models.runs import EvalRunResult, SampleResult

__all__ = ["STUDY_RESOURCE_METRICS", "compare_study_runs"]

#: Per-instance resource metrics paired across arms (summed over attempts).
STUDY_RESOURCE_METRICS: tuple[str, ...] = (
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "cost_usd",
)

#: Manifest fields arms are allowed to differ in (everything else must match).
_ARM_IDENTITY_FIELDS: set[str] = {"run_name", "target_name", "target_fingerprint"}


def _incompatible(message: str) -> IncompatibleRuns:
    return IncompatibleRuns(message=message)


def _legitimate(record: Mapping[str, object]) -> bool:
    """Whether one validated study record is a legitimate outcome.

    Mirrors the category rule in ``stats/study.py``: a correct acceptance
    (expected accept, accepted, answer correct) or a correct rejection
    (expected reject, rejected).
    """
    expected = record["expected_action"]
    actual = record["actual_action"]
    if actual == "accepted":
        return expected == "accept" and record["answer_correct"] is True
    return expected == "reject"


def _metric_value(row: SampleResult, name: str) -> float | None:
    """Read one resource metric off an execution; ``None`` stays ``None``.

    Falls back to the timestamp duration for ``latency_ms`` exactly as the
    per-arm summary does. Non-finite values are malformed provenance and
    raise rather than being clamped.
    """
    value = cast("int | float | None", getattr(row.execution, name))
    if value is None and name == "latency_ms":
        execution = row.execution
        value = (execution.finished_at - execution.started_at).total_seconds() * 1000
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ManifestValidationError(message=f"execution.{name} must be finite when present")
    return numeric


def _bootstrap_interval(deltas: list[float], *, seed: int, iterations: int) -> list[float]:
    """Seeded paired-bootstrap 95% interval for the mean of ``deltas``."""
    rng = random.Random(seed)  # noqa: S311 - reproducible statistical resampling
    n = len(deltas)
    boot = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(iterations))
    return [boot[max(0, int(0.025 * iterations) - 1)], boot[int(0.975 * iterations) - 1]]


def _mcnemar_exact_p(baseline_only: int, arm_only: int) -> float:
    """Exact two-sided McNemar p-value on discordant paired outcomes."""
    discordant = baseline_only + arm_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(baseline_only, arm_only) + 1))
    # ``2.0 **`` (float) rather than ``2 **``: typeshed types int**int as Any
    # (negative exponents can yield float), which would trip warn_return_any.
    return min(1.0, 2 * tail / 2.0**discordant)


def _index_run(run: EvalRunResult) -> dict[tuple[str, int], SampleResult]:
    """Index a run by (sample_id, attempt); duplicates are never comparable."""
    index = {(row.sample.sample_id, row.execution.attempt): row for row in run.samples}
    if len(index) != len(run.samples):
        raise _incompatible("duplicate sample-attempt pairs within one arm")
    return index


def _check_arm_compatibility(baseline_name: str, runs: Mapping[str, EvalRunResult]) -> None:
    """Fail loudly unless every arm shares dataset and non-target provenance."""
    baseline = runs[baseline_name]
    baseline_manifest = baseline.manifest.model_dump(exclude=_ARM_IDENTITY_FIELDS)
    for name, run in runs.items():
        if name == baseline_name:
            continue
        if run.manifest.model_dump(exclude=_ARM_IDENTITY_FIELDS) != baseline_manifest:
            raise _incompatible(
                f"arm {name!r} differs from baseline in dataset, sampling, grader, "
                "or other non-target configuration"
            )
        if run.resolved_dataset != baseline.resolved_dataset:
            raise _incompatible(f"arm {name!r} used a different resolved dataset")


def _pairable_attempt(row: SampleResult | None) -> tuple[bool, bool]:
    """Return ``(pairable, legitimate)`` for one attempt row.

    ``pairable`` requires a completed execution, a definitive grade, and a
    present (valid) study label. Label *validation* happens unconditionally —
    a malformed label raises from ``_study_record`` even on an attempt that
    would otherwise be excluded.
    """
    if row is None:
        return False, False
    record = _study_record(row)
    grade = row.grade
    pairable = (
        row.execution.status is ExecutionStatus.COMPLETED
        and grade is not None
        and grade.status in {GradeStatus.PASS, GradeStatus.FAIL}
        and record is not None
    )
    return pairable, pairable and _legitimate(cast("Mapping[str, object]", record))


def _compare_pair(
    baseline: EvalRunResult,
    arm: EvalRunResult,
    *,
    baseline_name: str,
    arm_name: str,
    seed: int,
    iterations: int,
) -> dict[str, object]:
    """Compare one arm against the baseline on paired study instances."""
    baseline_index = _index_run(baseline)
    arm_index = _index_run(arm)
    if set(baseline_index) != set(arm_index):
        raise _incompatible(
            f"arm {arm_name!r} does not contain the same sample-attempt identities as the baseline"
        )
    attempts_per_instance = baseline.manifest.attempts
    identifiers = sorted({key[0] for key in baseline_index})

    outcomes: list[tuple[bool, bool]] = []
    excluded_instances = 0
    # Per-metric paired deltas; an instance joins a metric's pair set only
    # when every attempt on BOTH sides reports that metric.
    metric_deltas: dict[str, list[float]] = {name: [] for name in STUDY_RESOURCE_METRICS}
    metric_totals: dict[str, list[tuple[float, float]]] = {
        name: [] for name in STUDY_RESOURCE_METRICS
    }
    metric_excluded: dict[str, int] = dict.fromkeys(STUDY_RESOURCE_METRICS, 0)

    for identifier in identifiers:
        pairs = [
            (
                baseline_index.get((identifier, attempt)),
                arm_index.get((identifier, attempt)),
            )
            for attempt in range(1, attempts_per_instance + 1)
        ]
        pairable = True
        legitimate_baseline: list[bool] = []
        legitimate_arm: list[bool] = []
        for row_baseline, row_arm in pairs:
            ok_baseline, legit_baseline = _pairable_attempt(row_baseline)
            ok_arm, legit_arm = _pairable_attempt(row_arm)
            if not (ok_baseline and ok_arm):
                pairable = False
            legitimate_baseline.append(legit_baseline)
            legitimate_arm.append(legit_arm)
        if pairable:
            outcomes.append((all(legitimate_baseline), all(legitimate_arm)))
        else:
            excluded_instances += 1

        for name in STUDY_RESOURCE_METRICS:
            baseline_values: list[float] = []
            arm_values: list[float] = []
            covered = True
            for row_baseline, row_arm in pairs:
                if row_baseline is None or row_arm is None:
                    covered = False
                    break
                value_baseline = _metric_value(row_baseline, name)
                value_arm = _metric_value(row_arm, name)
                if value_baseline is None or value_arm is None:
                    covered = False
                    break
                baseline_values.append(value_baseline)
                arm_values.append(value_arm)
            if covered:
                total_baseline = sum(baseline_values)
                total_arm = sum(arm_values)
                metric_totals[name].append((total_baseline, total_arm))
                metric_deltas[name].append(total_arm - total_baseline)
            else:
                metric_excluded[name] += 1

    n = len(outcomes)
    if n == 0:
        raise _incompatible(
            f"no paired instances with labeled definitive outcomes in both "
            f"{baseline_name!r} and {arm_name!r}"
        )
    baseline_only = sum(b_ok and not a_ok for b_ok, a_ok in outcomes)
    arm_only = sum(a_ok and not b_ok for b_ok, a_ok in outcomes)
    deltas = [float(a_ok) - float(b_ok) for b_ok, a_ok in outcomes]

    resources: dict[str, object] = {}
    for name in STUDY_RESOURCE_METRICS:
        values = metric_deltas[name]
        if not values:
            resources[name] = {
                "paired_instances": 0,
                "excluded_instances": metric_excluded[name],
                "note": "no instance reported this metric on both arms",
            }
            continue
        totals = metric_totals[name]
        resources[name] = {
            "paired_instances": len(values),
            "excluded_instances": metric_excluded[name],
            "baseline_mean": sum(b for b, _ in totals) / len(totals),
            "arm_mean": sum(a for _, a in totals) / len(totals),
            "mean_delta_arm_minus_baseline": sum(values) / len(values),
            "bootstrap_95_interval": _bootstrap_interval(
                values, seed=seed + 1, iterations=iterations
            ),
        }

    return {
        "arms": [baseline_name, arm_name],
        "metric": "all-attempts legitimate outcome per study instance",
        "paired_instances": n,
        "excluded_instances": excluded_instances,
        "attempts_per_instance": attempts_per_instance,
        "baseline_legitimate_rate": sum(b for b, _ in outcomes) / n,
        "arm_legitimate_rate": sum(a for _, a in outcomes) / n,
        "delta_arm_minus_baseline": sum(deltas) / n,
        "bootstrap_95_interval": _bootstrap_interval(deltas, seed=seed, iterations=iterations),
        "baseline_only": baseline_only,
        "arm_only": arm_only,
        "mcnemar_exact_p": _mcnemar_exact_p(baseline_only, arm_only),
        "resources": resources,
    }


def compare_study_runs(
    runs: Mapping[str, EvalRunResult],
    *,
    baseline: str | None = None,
    bootstrap_seed: int = 20260908,
    bootstrap_iterations: int = 2000,
) -> dict[str, object]:
    """Compare N study arms pairwise against a baseline arm.

    Args:
        runs: Arm name -> run result. Two or more arms.
        baseline: Arm every other arm is compared against. Defaults to the
            first arm in mapping order.
        bootstrap_seed: Seed for the deterministic paired bootstrap; each arm
            pair derives its own seed from this one (ADR-0008).
        bootstrap_iterations: Resamples per interval.

    Returns:
        A JSON-safe dict with the baseline name, the pairing rule, one
        outcome comparison per non-baseline arm (paired legitimate-outcome
        rates, delta, bootstrap interval, exact McNemar p-value, discordant
        counts), and per-arm paired resource comparisons with explicit
        coverage counts.

    Raises:
        IncompatibleRuns: If fewer than two arms are given, the baseline is
            unknown, arms differ in non-target provenance or dataset, any arm
            has duplicate sample-attempt pairs, sample identity sets differ,
            or no instance is pairable for an arm.
        ManifestValidationError: If any study label is malformed — even on an
            instance that would otherwise be excluded.
    """
    if len(runs) < 2:
        raise _incompatible("study comparison requires at least two arms")
    baseline_name = baseline if baseline is not None else next(iter(runs))
    if baseline_name not in runs:
        raise _incompatible(f"unknown baseline arm {baseline_name!r}")
    _check_arm_compatibility(baseline_name, runs)

    comparisons: dict[str, object] = {}
    for index, (name, run) in enumerate(runs.items()):
        if name == baseline_name:
            continue
        comparisons[name] = _compare_pair(
            runs[baseline_name],
            run,
            baseline_name=baseline_name,
            arm_name=name,
            seed=bootstrap_seed + 1000 * index,
            iterations=bootstrap_iterations,
        )
    return {
        "schema": "agentic-evalkit.study-compare.v1",
        "baseline": baseline_name,
        "arms": list(runs),
        "pairing": "sample_id with all-attempts legitimate-outcome aggregation",
        "outcomes": comparisons,
        "qualification": (
            "conditional on paired, labeled, definitively graded instances; "
            "resource deltas are per-instance sums over attempts and cover "
            "only instances where both arms reported the metric"
        ),
    }
