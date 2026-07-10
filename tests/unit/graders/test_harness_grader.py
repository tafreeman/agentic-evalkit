"""Tests for :class:`agentic_evalkit.graders.harness.HarnessGrader` (ADR-0014).

Hermetic: the authoritative-verification boundary is the packaged, in-memory
``FakeHarnessExecutor`` (no Docker, no swebench). Proves the load-bearing
mapping -- only a real ``resolved`` verdict hard-gates, and an operational
failure (unavailable/error) never becomes a task ``FAIL`` (ADR-0005/0008).
"""

from datetime import UTC, datetime

import pytest
from pydantic import JsonValue

from agentic_evalkit.benchmarks.harness import FakeHarnessExecutor, HarnessResult, HarnessStatus
from agentic_evalkit.graders.harness import HarnessGrader
from agentic_evalkit.models import (
    EvalSample,
    ExecutionStatus,
    GradeStatus,
    NormalizedExecutionResult,
)

_SAMPLE_ID = "swebench-verified:org__repo-1"


def _sample() -> EvalSample:
    return EvalSample(
        sample_id=_SAMPLE_ID,
        input={"problem_statement": "fix the bug", "repo": "org/repo"},
        metadata={"instance_id": "org__repo-1"},
        source_digest="sha256:row",
        adapter="swebench-verified@1",
    )


def _execution(
    *, status: ExecutionStatus = ExecutionStatus.COMPLETED, output: dict[str, JsonValue] | None
) -> NormalizedExecutionResult:
    now = datetime.now(UTC)
    return NormalizedExecutionResult(
        sample_id=_SAMPLE_ID,
        attempt=1,
        output=output,
        status=status,
        started_at=now,
        finished_at=now,
    )


def _predictor(sample: EvalSample, execution: NormalizedExecutionResult) -> dict[str, JsonValue]:
    return {"instance_id": "org__repo-1", "model_name_or_path": "t", "model_patch": "diff"}


def _grader(result: HarnessResult) -> HarnessGrader:
    return HarnessGrader(
        executor=FakeHarnessExecutor(default_result=result),
        predictor=_predictor,
        benchmark="swebench-verified@1",
        name="swebench-harness@1",
    )


@pytest.mark.asyncio
async def test_resolved_true_is_a_hard_gated_pass() -> None:
    grader = _grader(HarnessResult(status=HarnessStatus.COMPLETED, resolved=True, message="ok"))
    result = await grader.grade(_sample(), _execution(output={"model_patch": "diff"}))
    assert result.status is GradeStatus.PASS
    assert result.score == pytest.approx(1.0)
    assert result.hard_gate is True
    assert result.grader == "swebench-harness@1"
    assert result.evidence["harness_status"] == "completed"


@pytest.mark.asyncio
async def test_resolved_false_is_a_hard_gated_fail() -> None:
    grader = _grader(HarnessResult(status=HarnessStatus.COMPLETED, resolved=False, message="no"))
    result = await grader.grade(_sample(), _execution(output={"model_patch": "diff"}))
    assert result.status is GradeStatus.FAIL
    assert result.score == pytest.approx(0.0)
    assert result.hard_gate is True


@pytest.mark.asyncio
async def test_unavailable_never_hard_gates_and_is_not_a_fail() -> None:
    grader = _grader(
        HarnessResult(status=HarnessStatus.UNAVAILABLE, resolved=None, message="extra missing")
    )
    result = await grader.grade(_sample(), _execution(output={"model_patch": "diff"}))
    assert result.status is GradeStatus.UNAVAILABLE
    assert result.score is None
    assert result.hard_gate is False


@pytest.mark.asyncio
async def test_infrastructure_error_never_hard_gates_and_is_not_a_fail() -> None:
    grader = _grader(
        HarnessResult(
            status=HarnessStatus.ERROR,
            resolved=None,
            message="image pull failed",
            error={"code": "image_pull_failed"},
        )
    )
    result = await grader.grade(_sample(), _execution(output={"model_patch": "diff"}))
    assert result.status is GradeStatus.ERROR
    assert result.score is None
    assert result.hard_gate is False
    assert result.evidence["harness_error"] == {"code": "image_pull_failed"}


@pytest.mark.asyncio
async def test_completed_without_a_verdict_is_unavailable_not_a_guess() -> None:
    """A COMPLETED result whose ``resolved`` is None carries no verdict, so it
    must not be coerced into a pass/fail."""
    grader = _grader(HarnessResult(status=HarnessStatus.COMPLETED, resolved=None, message="?"))
    result = await grader.grade(_sample(), _execution(output={"model_patch": "diff"}))
    assert result.status is GradeStatus.UNAVAILABLE
    assert result.hard_gate is False


@pytest.mark.asyncio
async def test_non_completed_execution_is_not_verifiable() -> None:
    grader = _grader(HarnessResult(status=HarnessStatus.COMPLETED, resolved=True, message="ok"))
    result = await grader.grade(_sample(), _execution(status=ExecutionStatus.ERROR, output=None))
    assert result.status is GradeStatus.UNAVAILABLE
    assert result.hard_gate is False


@pytest.mark.asyncio
async def test_completed_execution_with_no_output_is_not_verifiable() -> None:
    grader = _grader(HarnessResult(status=HarnessStatus.COMPLETED, resolved=True, message="ok"))
    result = await grader.grade(_sample(), _execution(output=None))
    assert result.status is GradeStatus.UNAVAILABLE


@pytest.mark.asyncio
async def test_spilled_output_is_a_diagnostic_error_not_a_silent_unavailable() -> None:
    """A large patch spilled by the runner (output=None + an output_ref
    artifact) must not be miscounted as capability-unavailable; it surfaces as
    an explicit ERROR naming the spill (Codex review, P2)."""
    grader = _grader(HarnessResult(status=HarnessStatus.COMPLETED, resolved=True, message="ok"))
    now = datetime.now(UTC)
    spilled = NormalizedExecutionResult(
        sample_id=_SAMPLE_ID,
        attempt=1,
        output=None,
        artifacts={"output_ref": "sha256:deadbeef"},
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )
    result = await grader.grade(_sample(), spilled)
    assert result.status is GradeStatus.ERROR
    assert result.hard_gate is False
    assert "spilled" in str(result.evidence["reason"])


@pytest.mark.asyncio
async def test_predictor_failure_is_an_error_not_a_fail() -> None:
    def _bad_predictor(
        sample: EvalSample, execution: NormalizedExecutionResult
    ) -> dict[str, JsonValue]:
        raise ValueError("no patch in output")

    grader = HarnessGrader(
        executor=FakeHarnessExecutor(
            default_result=HarnessResult(
                status=HarnessStatus.COMPLETED, resolved=True, message="ok"
            )
        ),
        predictor=_bad_predictor,
        benchmark="swebench-verified@1",
        name="swebench-harness@1",
    )
    result = await grader.grade(_sample(), _execution(output={"model_patch": "diff"}))
    assert result.status is GradeStatus.ERROR
    assert result.hard_gate is False
    assert "no patch in output" in str(result.evidence["reason"])


def test_grade_result_from_harness_grader_has_no_resolved_attribute() -> None:
    """Extends the harness-suite invariant to this grader: an authoritative
    resolution verdict can never be smuggled out on a plain GradeResult."""
    from agentic_evalkit.models import GradeResult

    grade = GradeResult(
        sample_id=_SAMPLE_ID,
        grader="swebench-harness@1",
        status=GradeStatus.PASS,
        score=1.0,
        hard_gate=True,
        created_at=datetime.now(UTC),
    )
    assert not hasattr(grade, "resolved")
