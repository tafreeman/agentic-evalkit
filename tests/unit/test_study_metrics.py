from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agentic_evalkit.errors import ManifestValidationError
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
from agentic_evalkit.stats.study import summarize_study_run

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _result(
    sample_id: str,
    *,
    execution: ExecutionStatus = ExecutionStatus.COMPLETED,
    grade: GradeStatus | None = GradeStatus.PASS,
    study: object | None = None,
    attempt: int = 1,
) -> SampleResult:
    sample = EvalSample(
        sample_id=sample_id,
        input={"secret": "never aggregate"},
        source_digest="sha256:x",
        adapter="test",
        metadata={"family": "math"},
    )
    run = NormalizedExecutionResult(
        sample_id=sample_id,
        attempt=attempt,
        status=execution,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=2),
        input_tokens=4,
        output_tokens=3,
        cost_usd=0.1,
    )
    evidence = {} if study is None else {"study": study}
    verdict = (
        None
        if grade is None
        else GradeResult(
            sample_id=sample_id, grader="test", status=grade, evidence=evidence, created_at=NOW
        )
    )
    return SampleResult(sample=sample, execution=run, grade=verdict)


def _run(*samples: SampleResult) -> EvalRunResult:
    manifest = EvalRunManifest(
        run_name="study",
        dataset_ref=DatasetRef(provider="test", dataset_id="test"),
        adapter="test",
        grader="test",
        target_name="test",
    )
    dataset = ResolvedDataset(dataset_id="test", revision="1")
    return EvalRunResult(
        run_id="run",
        manifest=manifest,
        resolved_dataset=dataset,
        samples=samples,
        summary=RunSummary(total=len(samples)),
        started_at=NOW,
    )


def test_study_categories_denominators_resources_and_no_raw_inputs() -> None:
    accept_correct = {
        "expected_action": "accept",
        "actual_action": "accepted",
        "answer_correct": True,
        "numeric_absolute_error": 0.5,
        "retry_count": 1,
    }
    reject_accepted = {
        "expected_action": "reject",
        "actual_action": "accepted",
        "answer_correct": True,
        "fallback_count": 2,
    }
    reject_rejected = {"expected_action": "reject", "actual_action": "rejected"}
    accept_rejected = {"expected_action": "accept", "actual_action": "rejected"}
    summary = summarize_study_run(
        _run(
            _result("a", study=accept_correct),
            _result("b", study=reject_accepted),
            _result("c", study=reject_rejected),
            _result("d", study=accept_rejected),
        )
    )
    assert summary["categories"] == {
        "correct_acceptance": 1,
        "wrong_acceptance": 1,
        "correct_rejection": 1,
        "unnecessary_rejection": 1,
    }
    assert summary["rates"]["correct_acceptance"] == {
        "numerator": 1,
        "denominator": 2,
        "value": 0.5,
    }
    assert summary["acceptance_conditioned_absolute_error"]["mean"] == 0.5
    assert summary["resources"]["latency_ms"]["summary"]["sum"] == 8000.0
    assert "secret" not in repr(summary)


def test_missing_labels_and_operational_and_grade_states_are_separate() -> None:
    summary = summarize_study_run(
        _run(
            _result("missing"),
            _result("timeout", execution=ExecutionStatus.TIMEOUT),
            _result("abstain", grade=GradeStatus.ABSTAIN),
            _result("unavailable", grade=GradeStatus.UNAVAILABLE),
        )
    )
    assert summary["attempts"] == {
        "total": 4,
        "eligible_domain_labels": 1,
        "labeled_domain_labels": 0,
        "missing_domain_labels": 1,
    }
    assert summary["operational_failures"]["timeout"] == 1
    assert summary["grade_outcomes"]["abstain"] == 1
    assert summary["grade_outcomes"]["unavailable"] == 1


def test_present_malformed_study_record_is_rejected_even_if_ineligible() -> None:
    with pytest.raises(ManifestValidationError, match="expected_action"):
        summarize_study_run(
            _run(
                _result(
                    "bad",
                    execution=ExecutionStatus.ERROR,
                    study={"expected_action": "maybe", "actual_action": "rejected"},
                )
            )
        )


def test_repeated_attempts_only_have_task_level_wilson_inference() -> None:
    good = {"expected_action": "accept", "actual_action": "accepted", "answer_correct": True}
    bad = {"expected_action": "accept", "actual_action": "rejected"}
    summary = summarize_study_run(
        _run(
            _result("same", study=good, attempt=1),
            _result("same", study=bad, attempt=2),
            _result("other", study=good),
        )
    )
    assert summary["task_inference"]["eligible_tasks"] == 2
    assert summary["task_inference"]["legitimate_tasks"] == 1
    assert "wilson_95" not in summary["per_attempt"]
