"""Tests for :class:`agentic_evalkit.graders.harness.HarnessGrader` (ADR-0014).

These tests never touch Docker or the real SWE-bench harness -- they use the
packaged, in-memory ``FakeHarnessExecutor`` test double instead, so they run
fast and produce the same result every time ("hermetic"). What they prove is
the one rule this grader most needs to get right: only a real ``resolved``
verdict from the harness is allowed to hard-gate (force a failure that other
good scores can't average away). An operational failure -- the harness
itself being unavailable or hitting an error, as opposed to a genuine
wrong-answer verdict -- must never turn into a task ``FAIL`` (ADR-0005/0008).
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
    """When the AI's answer is too large to keep inline, the runner writes it
    out to separate storage and leaves only a reference behind here
    (``output=None`` plus an ``output_ref`` entry in ``artifacts`` -- this is
    called "spilling" the output). That's a real, gradable answer that just
    happens to live somewhere else; it must not be mistaken for "there's
    nothing to check" (which would report UNAVAILABLE). Instead the grader
    should report an explicit ERROR that names the spill, so a human reading
    the result can tell the two situations apart (flagged by a code-review
    tool named Codex, priority P2)."""
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
async def test_failed_spill_is_a_diagnostic_error_not_a_silent_unavailable() -> None:
    """The other way a spill can leave ``output=None`` behind: the artifact
    store refused or failed to write the oversized answer, so there is an
    ``output_spill_error`` record and -- unlike the spilled-successfully case
    above -- no stored bytes to point at. The AI still produced a real answer;
    it just was never persisted anywhere. Reporting UNAVAILABLE ("execution
    produced no output") would be plainly false, so the grader reports an
    explicit ERROR naming the failed spill instead, telling the reader this
    result can't be re-graded and the sample has to be re-run."""
    grader = _grader(HarnessResult(status=HarnessStatus.COMPLETED, resolved=True, message="ok"))
    now = datetime.now(UTC)
    spill_failed = NormalizedExecutionResult(
        sample_id=_SAMPLE_ID,
        attempt=1,
        output=None,
        artifacts={
            "output_spill_error": {
                "type": "ArtifactStoreLimitExceeded",
                "code": "output_spill_failed",
                "message": "artifact payload exceeds the configured maximum",
            }
        },
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )
    result = await grader.grade(_sample(), spill_failed)
    assert result.status is GradeStatus.ERROR
    assert result.hard_gate is False
    reason = str(result.evidence["reason"])
    assert "spilled" in reason
    assert "output_spill_failed" in reason


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
    """Other tests in this project already check this same rule for other
    graders; this test checks it again for this one: the harness's real,
    official verdict (``resolved``) must never leak onto a plain
    ``GradeResult`` as an extra attribute. ``GradeResult`` only ever exposes
    the generic PASS/FAIL/etc. status -- never a harness-specific field."""
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "planted",
    [
        pytest.param({"note": "retrying"}, id="dict-without-a-code"),
        pytest.param("totally fine actually", id="not-a-dict-at-all"),
        pytest.param({"code": "something_else"}, id="some-other-code"),
    ],
)
async def test_a_target_cannot_hijack_the_spill_diagnosis_by_key_name(
    planted: JsonValue,
) -> None:
    """``artifacts`` is target-controlled -- it is documented as "any extra
    files or data the system produced" -- so the presence of an
    ``output_spill_error`` key proves nothing on its own. Only a record
    carrying the ``output_spill_failed`` taxonomy code is believed; anything
    else falls through to the ordinary handling below it.

    Without that check, a bare ``spill_error["code"]`` subscript raised
    ``KeyError`` straight out of ``grade()`` -- fatal on the re-grade path
    this branch exists to serve -- and a target-authored value of any length
    was interpolated verbatim into ``GradeResult.evidence``, which, unlike
    the runner's own recorded messages, has no length bound.
    """
    grader = _grader(HarnessResult(status=HarnessStatus.COMPLETED, resolved=True, message="ok"))
    now = datetime.now(UTC)
    execution = NormalizedExecutionResult(
        sample_id=_SAMPLE_ID,
        attempt=1,
        output=None,
        artifacts={"output_spill_error": planted},
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )

    result = await grader.grade(_sample(), execution)

    # Falls through to the "no output and no artifact reference" branch
    # rather than raising or reporting a spill failure that never happened.
    assert result.status is GradeStatus.UNAVAILABLE


@pytest.mark.asyncio
async def test_a_genuine_spill_record_still_wins_over_a_stale_output_ref() -> None:
    """The ordering rule this branch was built around, re-pinned now that the
    branch also validates the record: when a real spill-failure record sits
    beside an ``output_ref`` a target wrote itself, the failure is still the
    honest report -- those bytes were never written.
    """
    grader = _grader(HarnessResult(status=HarnessStatus.COMPLETED, resolved=True, message="ok"))
    now = datetime.now(UTC)
    execution = NormalizedExecutionResult(
        sample_id=_SAMPLE_ID,
        attempt=1,
        output=None,
        artifacts={
            "output_ref": "sha256:" + "ab" * 32,
            "output_spill_error": {
                "type": "OSError",
                "code": "output_spill_failed",
                "message": "no space left on device",
            },
        },
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )

    result = await grader.grade(_sample(), execution)

    assert result.status is GradeStatus.ERROR
    assert "output_spill_failed" in str(result.evidence["reason"])


@pytest.mark.asyncio
async def test_an_oversized_output_ref_is_truncated_before_it_reaches_the_evidence() -> None:
    """``output_ref`` is target-controlled -- the key is not reserved, and a
    target is free to put a megabyte of text under it. ``GradeResult.evidence``
    has no length bound of its own (unlike the runner's recorded messages,
    which go through ``_safe_error_message``), so without ``_bounded_ref``
    that value is copied verbatim into every report that renders the grade.

    A genuine reference is a 71-character ``sha256:`` digest, so this branch
    never fires on real data -- which is exactly why it needs a test: nothing
    else in the suite reaches it, and the truncation the CHANGELOG and
    ADR-0020 both cite as hardening would otherwise ship unexercised.
    """
    grader = _grader(HarnessResult(status=HarnessStatus.COMPLETED, resolved=True, message="ok"))
    now = datetime.now(UTC)
    execution = NormalizedExecutionResult(
        sample_id=_SAMPLE_ID,
        attempt=1,
        output=None,
        artifacts={"output_ref": "z" * 5000},
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )

    result = await grader.grade(_sample(), execution)

    reason = str(result.evidence["reason"])
    assert result.status is GradeStatus.ERROR
    assert "...[truncated]" in reason
    # Bounded to the cap plus the marker and the surrounding sentence, rather
    # than growing with whatever the target sent.
    assert len(reason) < 500
    assert "z" * 200 not in reason
