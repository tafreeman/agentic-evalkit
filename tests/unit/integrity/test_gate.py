"""Pins the integrity gate: a flagged run can never grade PASS (spec go/no-go item 1)."""

from datetime import UTC, datetime
from itertools import product

import pytest

from agentic_evalkit.graders.composite import CompositeGrader, WeightedGrader
from agentic_evalkit.integrity import integrity_gated
from agentic_evalkit.models import (
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
)

_SCORES = {GradeStatus.PASS: 1.0, GradeStatus.PARTIAL: 0.5, GradeStatus.FAIL: 0.0}


class FixedGrader:
    def __init__(self, status: GradeStatus) -> None:
        self.status = status

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        return GradeResult(
            sample_id=sample.sample_id,
            grader=f"fixed-{self.status.value}",
            status=self.status,
            score=_SCORES.get(self.status),
            created_at=datetime.now(UTC),
        )


def _case() -> tuple[EvalSample, NormalizedExecutionResult]:
    now = datetime.now(UTC)
    sample = EvalSample(sample_id="case", input={}, adapter="test", source_digest="sha256:test")
    execution = NormalizedExecutionResult(
        sample_id="case",
        attempt=1,
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
        output={"patch": "diff"},
    )
    return sample, execution


@pytest.mark.asyncio
@pytest.mark.parametrize(("integrity", "oracle"), list(product(GradeStatus, GradeStatus)))
async def test_flagged_run_never_passes(integrity: GradeStatus, oracle: GradeStatus) -> None:
    sample, execution = _case()
    grader = integrity_gated(
        oracle=FixedGrader(oracle), integrity=FixedGrader(integrity), name="gated"
    )
    result = await grader.grade(sample, execution)
    if integrity is not GradeStatus.PASS:
        assert result.status is not GradeStatus.PASS
    elif oracle in _SCORES:
        assert result.status is oracle
        assert result.score == _SCORES[oracle]
    else:
        assert result.status is GradeStatus.UNAVAILABLE


@pytest.mark.asyncio
async def test_exploit_flag_is_unavailable_not_fail() -> None:
    sample, execution = _case()
    grader = integrity_gated(
        oracle=FixedGrader(GradeStatus.PASS),
        integrity=FixedGrader(GradeStatus.UNAVAILABLE),
        name="gated",
    )
    result = await grader.grade(sample, execution)
    assert result.status is GradeStatus.UNAVAILABLE
    assert result.score is None


@pytest.mark.asyncio
async def test_partial_integrity_is_a_veto() -> None:
    sample, execution = _case()
    grader = integrity_gated(
        oracle=FixedGrader(GradeStatus.PASS),
        integrity=FixedGrader(GradeStatus.PARTIAL),
        name="gated",
    )
    result = await grader.grade(sample, execution)
    assert result.status is GradeStatus.UNAVAILABLE


@pytest.mark.asyncio
async def test_hand_wired_integrity_without_hard_gate_lets_exploit_pass() -> None:
    """The trap this module exists for. If this ever stops passing, revisit the gate."""
    sample, execution = _case()
    grader = CompositeGrader(
        name="hand-wired",
        graders=(
            WeightedGrader(FixedGrader(GradeStatus.UNAVAILABLE), weight=1.0, hard_gate=False),
            WeightedGrader(FixedGrader(GradeStatus.PASS), weight=1.0, hard_gate=False),
        ),
    )
    result = await grader.grade(sample, execution)
    assert result.status is GradeStatus.PASS


@pytest.mark.asyncio
async def test_hand_wired_partial_hard_gate_lets_exploit_pass() -> None:
    """The second trap: CompositeGrader's hard gate does not veto PARTIAL."""
    sample, execution = _case()
    grader = CompositeGrader(
        name="hand-wired",
        graders=(
            WeightedGrader(FixedGrader(GradeStatus.PARTIAL), weight=0.0, hard_gate=True),
            WeightedGrader(FixedGrader(GradeStatus.PASS), weight=1.0, hard_gate=False),
        ),
    )
    result = await grader.grade(sample, execution)
    assert result.status is GradeStatus.PASS
