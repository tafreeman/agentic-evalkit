"""Experiment-arm comparisons stay paired and exclude operational failures."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentic_evalkit.errors import IncompatibleRuns
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
from agentic_evalkit.stats.experiment import compare_experiment_arms


def _row(sample_id: str, *, passed: bool = True, operational_error: bool = False) -> SampleResult:
    now = datetime.now(UTC)
    sample = EvalSample(
        sample_id=sample_id, input={}, source_digest=f"sha256:{sample_id}", adapter="fixture"
    )
    if operational_error:
        execution = NormalizedExecutionResult(
            sample_id=sample_id,
            attempt=1,
            status=ExecutionStatus.ERROR,
            error={"type": "target", "message": "unavailable"},
            started_at=now,
            finished_at=now,
        )
        return SampleResult(sample=sample, execution=execution)
    grade_status = GradeStatus.PASS if passed else GradeStatus.FAIL
    execution = NormalizedExecutionResult(
        sample_id=sample_id,
        attempt=1,
        status=ExecutionStatus.COMPLETED,
        output={"answer": "fixture"},
        started_at=now,
        finished_at=now,
    )
    grade = GradeResult(
        sample_id=sample_id,
        grader="fixture",
        status=grade_status,
        score=1.0 if passed else 0.0,
        created_at=now,
    )
    return SampleResult(sample=sample, execution=execution, grade=grade)


def _run(*, target: str, rows: tuple[SampleResult, ...]) -> EvalRunResult:
    now = datetime.now(UTC)
    manifest = EvalRunManifest(
        run_name="fixture",
        dataset_ref=DatasetRef(provider="local", dataset_id="fixture"),
        adapter="fixture",
        grader="fixture",
        target_name="target",
        target_fingerprint=target,
        environment_fingerprint="environment",
        code_fingerprint="code",
    )
    return EvalRunResult(
        run_id=target,
        manifest=manifest,
        resolved_dataset=ResolvedDataset(dataset_id="fixture", revision="sha256:fixture"),
        samples=rows,
        summary=RunSummary(total=len(rows)),
        started_at=now,
        finished_at=now,
    )


def test_comparison_excludes_operational_results_from_paired_success() -> None:
    left = _run(target="left", rows=(_row("one"), _row("two", operational_error=True)))
    right = _run(target="right", rows=(_row("one", passed=False), _row("two")))

    result = compare_experiment_arms(left, right, names=("left", "right"))

    assert result["paired_instances"] == 1
    assert result["excluded_instances"] == 1
    assert result["right_pass_rate"] == 0.0


def test_comparison_refuses_different_sample_attempt_sets() -> None:
    left = _run(target="left", rows=(_row("one"),))
    right = _run(target="right", rows=(_row("two"),))

    with pytest.raises(IncompatibleRuns, match="sample-attempt"):
        compare_experiment_arms(left, right, names=("left", "right"))
