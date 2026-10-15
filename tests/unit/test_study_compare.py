"""Tests for ``stats/study_compare.compare_study_runs``.

Multi-arm paired comparison: arms pair on ``sample_id`` with the all-attempts
legitimate-outcome rule; exclusions and missing resource coverage are counted,
never imputed; provenance or identity drift fails loudly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from agentic_evalkit.errors import IncompatibleRuns, ManifestValidationError
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
from agentic_evalkit.stats.study_compare import compare_study_runs

NOW = datetime(2026, 1, 1, tzinfo=UTC)

LEGITIMATE_ACCEPT = {
    "expected_action": "accept",
    "actual_action": "accepted",
    "answer_correct": True,
}
LEGITIMATE_REJECT = {"expected_action": "reject", "actual_action": "rejected"}
WRONG_ACCEPTANCE = {
    "expected_action": "reject",
    "actual_action": "accepted",
    "answer_correct": True,
}
UNNECESSARY_REJECTION = {"expected_action": "accept", "actual_action": "rejected"}


def _result(
    sample_id: str,
    *,
    study: object | None = None,
    execution: ExecutionStatus = ExecutionStatus.COMPLETED,
    grade: GradeStatus | None = GradeStatus.PASS,
    attempt: int = 1,
    cost_usd: float | None = 0.1,
) -> SampleResult:
    sample = EvalSample(
        sample_id=sample_id,
        input={"secret": "never aggregated"},
        source_digest="sha256:x",
        adapter="test",
    )
    execution_result = NormalizedExecutionResult(
        sample_id=sample_id,
        attempt=attempt,
        status=execution,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=2),
        input_tokens=4,
        output_tokens=3,
        cost_usd=cost_usd,
    )
    evidence = {} if study is None else {"study": study}
    verdict = (
        None
        if grade is None
        else GradeResult(
            sample_id=sample_id,
            grader="test",
            status=grade,
            evidence=evidence,
            created_at=NOW,
        )
    )
    return SampleResult(sample=sample, execution=execution_result, grade=verdict)


def _run(name: str, *samples: SampleResult, grader: str = "test") -> EvalRunResult:
    manifest = EvalRunManifest(
        run_name=name,
        dataset_ref=DatasetRef(provider="test", dataset_id="test"),
        adapter="test",
        grader=grader,
        target_name=name,  # arms differ in target identity — allowed
    )
    return EvalRunResult(
        run_id=name,
        manifest=manifest,
        resolved_dataset=ResolvedDataset(dataset_id="test", revision="1"),
        samples=samples,
        summary=RunSummary(total=len(samples)),
        started_at=NOW,
    )


def test_two_arms_paired_outcome_and_resource_deltas() -> None:
    baseline = _run(
        "base",
        _result("a", study=LEGITIMATE_ACCEPT),
        _result("b", study=LEGITIMATE_REJECT),
    )
    arm = _run(
        "arm",
        _result("a", study=LEGITIMATE_ACCEPT),
        _result("b", study=WRONG_ACCEPTANCE),
    )

    report = compare_study_runs({"base": baseline, "arm": arm})

    assert report["schema"] == "agentic-evalkit.study-compare.v1"
    assert report["baseline"] == "base"
    outcome = report["outcomes"]["arm"]
    assert outcome["paired_instances"] == 2
    assert outcome["excluded_instances"] == 0
    assert outcome["baseline_legitimate_rate"] == 1.0
    assert outcome["arm_legitimate_rate"] == 0.5
    assert outcome["delta_arm_minus_baseline"] == -0.5
    assert outcome["baseline_only"] == 1
    assert outcome["arm_only"] == 0
    # One discordant pair: exact two-sided McNemar is 1.0.
    assert outcome["mcnemar_exact_p"] == 1.0
    # Identical scripted resources: zero deltas, full coverage.
    resources = outcome["resources"]
    assert resources["input_tokens"]["paired_instances"] == 2
    assert resources["input_tokens"]["mean_delta_arm_minus_baseline"] == 0.0
    assert resources["latency_ms"]["baseline_mean"] == 2000.0
    assert resources["cost_usd"]["paired_instances"] == 2
    json.dumps(report)


def test_missing_grade_and_missing_cost_are_counted_not_imputed() -> None:
    baseline = _run(
        "base",
        _result("a", study=LEGITIMATE_ACCEPT),
        _result("b", study=LEGITIMATE_REJECT),
    )
    arm = _run(
        "arm",
        _result("a", study=LEGITIMATE_ACCEPT, cost_usd=None),
        # Ungraded: the whole instance drops out of the outcome pairing.
        _result("b", study=LEGITIMATE_REJECT, grade=None),
    )

    outcome = compare_study_runs({"base": baseline, "arm": arm})["outcomes"]["arm"]

    assert outcome["paired_instances"] == 1
    assert outcome["excluded_instances"] == 1
    cost = outcome["resources"]["cost_usd"]
    # Instance "a" lacks cost on the arm side, "b" is missing a grade but
    # still has executions — cost pairing only needs the metric on both
    # sides, so "b" pairs and "a" is counted excluded.
    assert cost["paired_instances"] == 1
    assert cost["excluded_instances"] == 1
    assert cost["mean_delta_arm_minus_baseline"] == pytest.approx(0.0)


def test_three_arms_compare_against_the_baseline() -> None:
    baseline = _run("base", _result("a", study=LEGITIMATE_ACCEPT))
    middle = _run("middle", _result("a", study=LEGITIMATE_ACCEPT))
    worst = _run("worst", _result("a", study=UNNECESSARY_REJECTION))

    report = compare_study_runs({"base": baseline, "middle": middle, "worst": worst})

    assert set(report["outcomes"]) == {"middle", "worst"}
    assert report["outcomes"]["middle"]["delta_arm_minus_baseline"] == 0.0
    assert report["outcomes"]["worst"]["delta_arm_minus_baseline"] == -1.0


def test_explicit_baseline_is_used() -> None:
    baseline = _run("base", _result("a", study=LEGITIMATE_ACCEPT))
    arm = _run("arm", _result("a", study=LEGITIMATE_ACCEPT))
    report = compare_study_runs({"arm": arm, "base": baseline}, baseline="base")
    assert report["baseline"] == "base"
    assert set(report["outcomes"]) == {"arm"}


def test_non_target_provenance_drift_fails() -> None:
    baseline = _run("base", _result("a", study=LEGITIMATE_ACCEPT))
    arm = _run("arm", _result("a", study=LEGITIMATE_ACCEPT), grader="other")
    with pytest.raises(IncompatibleRuns, match="non-target"):
        compare_study_runs({"base": baseline, "arm": arm})


def test_sample_identity_mismatch_fails() -> None:
    baseline = _run("base", _result("a", study=LEGITIMATE_ACCEPT))
    arm = _run("arm", _result("b", study=LEGITIMATE_ACCEPT))
    with pytest.raises(IncompatibleRuns, match="same sample-attempt"):
        compare_study_runs({"base": baseline, "arm": arm})


def test_duplicate_attempts_within_one_arm_fail() -> None:
    baseline = _run("base", _result("a", study=LEGITIMATE_ACCEPT))
    arm = _run(
        "arm",
        _result("a", study=LEGITIMATE_ACCEPT),
        _result("a", study=LEGITIMATE_ACCEPT),
    )
    with pytest.raises(IncompatibleRuns, match="duplicate"):
        compare_study_runs({"base": baseline, "arm": arm})


def test_malformed_labels_raise_even_on_excluded_instances() -> None:
    baseline = _run("base", _result("a", study=LEGITIMATE_ACCEPT))
    arm = _run(
        "arm",
        # Malformed label (bad expected_action) on an otherwise pairable row.
        _result("a", study={"expected_action": "maybe", "actual_action": "accepted"}),
    )
    with pytest.raises(ManifestValidationError):
        compare_study_runs({"base": baseline, "arm": arm})


def test_too_few_arms_and_unknown_baseline_fail() -> None:
    only = _run("base", _result("a", study=LEGITIMATE_ACCEPT))
    with pytest.raises(IncompatibleRuns, match="at least two arms"):
        compare_study_runs({"base": only})
    other = _run("arm", _result("a", study=LEGITIMATE_ACCEPT))
    with pytest.raises(IncompatibleRuns, match="unknown baseline"):
        compare_study_runs({"base": only, "arm": other}, baseline="nope")


def test_no_pairable_instances_fail() -> None:
    baseline = _run("base", _result("a", execution=ExecutionStatus.TIMEOUT))
    arm = _run("arm", _result("a", study=LEGITIMATE_ACCEPT))
    with pytest.raises(IncompatibleRuns, match="no paired instances"):
        compare_study_runs({"base": baseline, "arm": arm})


def test_bootstrap_is_deterministic() -> None:
    def _arms() -> dict[str, EvalRunResult]:
        return {
            "base": _run(
                "base",
                _result("a", study=LEGITIMATE_ACCEPT),
                _result("b", study=LEGITIMATE_REJECT),
                _result("c", study=LEGITIMATE_ACCEPT),
            ),
            "arm": _run(
                "arm",
                _result("a", study=LEGITIMATE_ACCEPT),
                _result("b", study=WRONG_ACCEPTANCE),
                _result("c", study=UNNECESSARY_REJECTION),
            ),
        }

    first = compare_study_runs(_arms())["outcomes"]["arm"]
    second = compare_study_runs(_arms())["outcomes"]["arm"]
    assert first == second
