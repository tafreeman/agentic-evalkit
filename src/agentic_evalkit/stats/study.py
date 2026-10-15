"""Deterministic summaries for bounded acceptance/rejection studies.

This module deliberately derives study results only from the explicit
``grade.evidence["study"]`` contract.  A PASS is not an acceptance label and
is never used as one.
"""

from __future__ import annotations

import math
from statistics import fmean
from typing import TYPE_CHECKING, cast

from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.models.execution import ExecutionStatus
from agentic_evalkit.models.grades import GradeStatus
from agentic_evalkit.stats.aggregate import wilson_interval

if TYPE_CHECKING:
    from agentic_evalkit.models.runs import EvalRunResult, SampleResult

__all__ = ["summarize_study_run"]


def _invalid(message: str) -> ManifestValidationError:
    return ManifestValidationError(message=message)


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def _distribution(values: list[float]) -> dict[str, int | float] | None:
    if not values:
        return None
    ordered = sorted(values)
    count = len(ordered)
    median = (
        ordered[(count - 1) // 2]
        if count % 2
        else (ordered[count // 2 - 1] + ordered[count // 2]) / 2
    )
    return {
        "count": count,
        "sum": sum(ordered),
        "mean": fmean(ordered),
        "median": median,
        "p95": ordered[min(count - 1, math.ceil(0.95 * count) - 1)],
    }


def _study_record(sample: SampleResult) -> dict[str, object] | None:
    grade = sample.grade
    if grade is None or "study" not in grade.evidence:
        return None
    value = grade.evidence["study"]
    if not isinstance(value, dict):
        raise _invalid("grade.evidence['study'] must be a mapping")
    expected = value.get("expected_action")
    actual = value.get("actual_action")
    if expected not in {"accept", "reject"}:
        raise _invalid("study.expected_action must be 'accept' or 'reject'")
    if actual not in {"accepted", "rejected"}:
        raise _invalid("study.actual_action must be 'accepted' or 'rejected'")
    if actual == "accepted" and type(value.get("answer_correct")) is not bool:
        raise _invalid("study.answer_correct must be a bool when actual_action is 'accepted'")
    for key in ("numeric_absolute_error",):
        numeric = value.get(key)
        if numeric is not None and (
            type(numeric) not in {int, float}
            or not math.isfinite(float(cast("float", numeric)))
            or cast("float", numeric) < 0
        ):
            raise _invalid(f"study.{key} must be a finite number >= 0")
    for key in ("retry_count", "fallback_count"):
        numeric = value.get(key)
        if numeric is not None and (type(numeric) is not int or numeric < 0):
            raise _invalid(f"study.{key} must be an int >= 0")
    return cast("dict[str, object]", value)


def _resource_value(value: int | float | None, name: str) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise _invalid(f"execution.{name} must be finite when present")
    return numeric


def summarize_study_run(run: EvalRunResult) -> dict[str, object]:
    """Summarize explicit study labels without exposing samples or outputs.

    Domain labels are eligible only for completed executions with PASS or
    FAIL grades.  Present labels are always validated, including labels on
    ineligible records, so malformed provenance cannot hide behind an
    operational failure.
    """
    categories = {
        "correct_acceptance": 0,
        "wrong_acceptance": 0,
        "correct_rejection": 0,
        "unnecessary_rejection": 0,
    }
    operational = {
        status.value: 0 for status in ExecutionStatus if status is not ExecutionStatus.COMPLETED
    }
    grade_outcomes = {
        "missing": 0,
        "error": 0,
        "partial": 0,
        "abstain": 0,
        "unavailable": 0,
    }
    resource_values: dict[str, list[float]] = {
        "latency_ms": [],
        "input_tokens": [],
        "output_tokens": [],
        "cost_usd": [],
    }
    resource_missing = dict.fromkeys(resource_values, 0)
    error_values: list[float] = []
    retries: list[float] = []
    fallbacks: list[float] = []
    eligible = 0
    labeled = 0
    expected_accept = 0
    expected_reject = 0
    accepted = 0
    family_categories: dict[str, dict[str, int]] = {}
    task_outcomes: dict[str, list[bool]] = {}

    for result in run.samples:
        execution = result.execution
        record = _study_record(result)
        for name in resource_values:
            value = _resource_value(getattr(execution, name), name)
            if value is None and name == "latency_ms":
                duration = (execution.finished_at - execution.started_at).total_seconds() * 1000
                value = _resource_value(duration, "timestamp duration")
            if value is None:
                resource_missing[name] += 1
            else:
                resource_values[name].append(value)

        grade = result.grade
        is_eligible = (
            execution.status is ExecutionStatus.COMPLETED
            and grade is not None
            and grade.status in {GradeStatus.PASS, GradeStatus.FAIL}
        )
        if not is_eligible:
            if execution.status is not ExecutionStatus.COMPLETED:
                operational[execution.status.value] += 1
            elif grade is None:
                grade_outcomes["missing"] += 1
            else:
                grade_outcomes[grade.status.value] += 1
            continue

        eligible += 1
        if record is None:
            grade_outcomes["missing"] += 1
            continue
        labeled += 1
        expected = cast("str", record["expected_action"])
        actual = cast("str", record["actual_action"])
        if expected == "accept":
            expected_accept += 1
        else:
            expected_reject += 1
        if actual == "accepted":
            accepted += 1
        correct = actual == "accepted" and record["answer_correct"] is True
        if expected == "accept" and correct:
            category = "correct_acceptance"
        elif actual == "accepted":
            category = "wrong_acceptance"
        elif expected == "reject" and actual == "rejected":
            category = "correct_rejection"
        else:
            category = "unnecessary_rejection"
        categories[category] += 1
        task_outcomes.setdefault(result.sample.sample_id, []).append(
            category in {"correct_acceptance", "correct_rejection"}
        )
        family = result.sample.metadata.get("family")
        if isinstance(family, str):
            counts = family_categories.setdefault(family, dict.fromkeys(categories, 0))
            counts[category] += 1
        if actual == "accepted" and "numeric_absolute_error" in record:
            error_values.append(float(cast("float", record["numeric_absolute_error"])))
        if "retry_count" in record:
            retries.append(float(cast("int", record["retry_count"])))
        if "fallback_count" in record:
            fallbacks.append(float(cast("int", record["fallback_count"])))

    task_complete = [all(outcomes) for outcomes in task_outcomes.values()]
    task_successes = sum(task_complete)
    lower, upper = wilson_interval(successes=task_successes, total=len(task_complete))
    family_summary = {
        family: {
            "categories": counts,
            "legitimate_outcome_rate": _rate(
                counts["correct_acceptance"] + counts["correct_rejection"],
                sum(counts.values()),
            ),
        }
        for family, counts in sorted(family_categories.items())
    }
    resources = {
        name: {
            "summary": _distribution(values),
            "missing": resource_missing[name],
            "reported": len(values),
        }
        for name, values in resource_values.items()
    }
    return {
        "attempts": {
            "total": len(run.samples),
            "eligible_domain_labels": eligible,
            "labeled_domain_labels": labeled,
            "missing_domain_labels": eligible - labeled,
        },
        "operational_failures": operational,
        "grade_outcomes": grade_outcomes,
        "categories": categories,
        "rates": {
            "correct_acceptance": _rate(categories["correct_acceptance"], expected_accept),
            "wrong_acceptance": _rate(categories["wrong_acceptance"], accepted),
            "correct_rejection": _rate(categories["correct_rejection"], expected_reject),
            "unnecessary_rejection": _rate(categories["unnecessary_rejection"], expected_accept),
            "legitimate_completion": _rate(categories["correct_acceptance"], expected_accept),
            "legitimate_rejection": _rate(categories["correct_rejection"], expected_reject),
            "safety_blocked": _rate(categories["correct_rejection"], expected_reject),
        },
        "acceptance_conditioned_absolute_error": _distribution(error_values),
        "per_attempt": {
            "retry_count": _distribution(retries),
            "fallback_count": _distribution(fallbacks),
            "note": "Descriptive only; repeated attempts are not treated as independent.",
        },
        "task_inference": {
            "eligible_tasks": len(task_complete),
            "legitimate_tasks": task_successes,
            "rate": task_successes / len(task_complete) if task_complete else None,
            "wilson_95": {"lower": lower, "upper": upper},
            "rule": "A task is eligible only when every labeled attempt has a legitimate outcome.",
        },
        "resources": {
            **resources,
            "runtime_note": "Summed per-attempt runtime is not wall-clock elapsed run time.",
        },
        **({"by_family": family_summary} if family_summary else {}),
    }
