"""Tests for :mod:`agentic_evalkit.graders.composite` (plan Task 10 Step 2/4).

The hard-gate test below is copied verbatim from the plan
(docs/plans/2026-07-02-agentic-evalkit-initial-release.md, Task 10 Step 2)
and must pass unmodified.
"""

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter

from agentic_evalkit.graders.composite import CompositeGrader, SchemaGrader, WeightedGrader
from agentic_evalkit.models import (
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
)


def _sample() -> EvalSample:
    return EvalSample(
        sample_id="s1",
        input={"question": "ping"},
        reference="pong",
        source_digest="sha256:row",
        adapter="identity@1",
    )


def _execution() -> NormalizedExecutionResult:
    now = datetime.now(UTC)
    return NormalizedExecutionResult(
        sample_id="s1",
        attempt=1,
        output={"answer": "pong"},
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )


class _StaticGrader:
    """A test double that always returns the same status/score."""

    def __init__(self, status: GradeStatus, score: float | None) -> None:
        self._status = status
        self._score = score

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        return GradeResult(
            sample_id=sample.sample_id,
            grader="static@1",
            status=self._status,
            score=self._score,
            hard_gate=False,
            created_at=datetime.now(UTC),
        )


@pytest.mark.asyncio
async def test_failed_hard_gate_cannot_be_averaged_away() -> None:
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(_StaticGrader(GradeStatus.FAIL, 0.0), weight=0.2, hard_gate=True),
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=0.8, hard_gate=False),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.FAIL
    assert result.hard_gate is True
    assert result.score == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_passing_hard_gate_with_partial_weighted_mean_is_partial_not_pass() -> None:
    """Both components individually PASS, but their weighted mean (0.75) is
    below 1.0, so the composite honestly reports PARTIAL rather than
    rounding up to PASS.
    """
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=1.0, hard_gate=True),
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 0.5), weight=1.0, hard_gate=False),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PARTIAL
    assert result.hard_gate is False
    assert result.score == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_full_weighted_mean_with_passing_hard_gate_is_pass() -> None:
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=1.0, hard_gate=True),
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=1.0, hard_gate=False),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS
    assert result.hard_gate is False
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_score_is_weighted_mean_over_available_numeric_subscores_only() -> None:
    """A sub-grader with score=None (e.g. abstain) is excluded from the mean,
    not treated as zero.
    """
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=1.0, hard_gate=False),
            WeightedGrader(_StaticGrader(GradeStatus.ABSTAIN, None), weight=3.0, hard_gate=False),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    # Only the first grader contributes a numeric score; its weight of 1.0
    # is the sole denominator, so the weighted mean is 1.0, not 0.25.
    assert result.score == pytest.approx(1.0)
    assert result.status is GradeStatus.PASS


@pytest.mark.asyncio
async def test_every_child_result_is_preserved_in_evidence() -> None:
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=1.0, hard_gate=True),
            WeightedGrader(_StaticGrader(GradeStatus.FAIL, 0.0), weight=1.0, hard_gate=False),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    children = result.evidence["children"]
    assert isinstance(children, tuple | list)
    assert len(children) == 2


@pytest.mark.asyncio
async def test_missing_grader_result_is_error_not_zero() -> None:
    """A component grader that itself errors must surface as ERROR/UNAVAILABLE,
    never be silently treated as a zero score.
    """

    class _RaisingGrader:
        async def grade(
            self, sample: EvalSample, execution: NormalizedExecutionResult
        ) -> GradeResult:
            raise RuntimeError("boom")

    grader = CompositeGrader(
        name="quality@1",
        graders=(WeightedGrader(_RaisingGrader(), weight=1.0, hard_gate=False),),
    )
    result = await grader.grade(_sample(), _execution())
    assert result.status in (GradeStatus.ERROR, GradeStatus.UNAVAILABLE)
    assert result.score is None


@pytest.mark.asyncio
async def test_all_scores_unavailable_yields_unavailable_not_zero() -> None:
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(
                _StaticGrader(GradeStatus.UNAVAILABLE, None), weight=1.0, hard_gate=False
            ),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.UNAVAILABLE
    assert result.score is None


# --- Story 1.2 (R-010): composite-grading edge integrity --------------------
#
# Guards FR11: a hard-gated FAIL can never be averaged away, a raising
# component never contributes a silent zero, and missing / abstained /
# unavailable / errored sub-scores are excluded from the weighted mean rather
# than counted as 0. The cases below close the gaps left by the tests above,
# which cover the single-component and abstain-only shapes but not the mixed
# shapes (a survivor alongside a non-definitive component) where a silent
# zero-contribution bug would actually hide.


class _AlwaysRaisesGrader:
    """A component grader that always raises during grading.

    Module-level (distinct from the function-local ``_RaisingGrader`` used by
    ``test_missing_grader_result_is_error_not_zero`` above) so the mixed-shape
    Story 1.2 tests below can share one raising component.
    """

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_hard_gate_fail_wins_over_a_near_perfect_majority() -> None:
    """One low-weight hard-gated FAIL forces the whole composite to FAIL even
    when every other (much higher-weighted) component passes with a perfect
    score, so a hard gate can never be diluted by out-voting weight.
    """
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(_StaticGrader(GradeStatus.FAIL, 0.0), weight=0.01, hard_gate=True),
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=100.0, hard_gate=False),
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=100.0, hard_gate=False),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.FAIL
    assert result.hard_gate is True


@pytest.mark.asyncio
async def test_raising_component_is_excluded_from_the_mean_not_scored_zero() -> None:
    """A component that raises alongside a passing one must not drag the
    weighted mean down toward zero: its heavy weight is dropped from the
    denominator entirely, so the surviving PASS keeps the mean at 1.0 rather
    than the 0.09 a silent zero-contribution would produce.
    """
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=1.0, hard_gate=False),
            WeightedGrader(_AlwaysRaisesGrader(), weight=10.0, hard_gate=False),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    # The raising component contributes neither its weight nor a zero score:
    # if it were treated as 0.0 with weight 10.0 the mean would be ~0.09.
    assert result.score == pytest.approx(1.0)
    assert result.status is GradeStatus.PASS


@pytest.mark.asyncio
async def test_raising_component_surfaces_as_error_child_result() -> None:
    """The raising component's own child result is an explicit ERROR with a
    ``None`` score recorded in evidence -- never a silently-fabricated 0.0 --
    so a report can attribute the failure to that component.
    """
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=1.0, hard_gate=False),
            WeightedGrader(_AlwaysRaisesGrader(), weight=1.0, hard_gate=False),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    children = result.evidence["children"]
    assert isinstance(children, tuple | list)
    errored = [child for child in children if child["status"] == GradeStatus.ERROR.value]
    assert len(errored) == 1
    assert errored[0]["score"] is None


@pytest.mark.asyncio
async def test_unavailable_and_error_subscores_are_excluded_from_the_mean() -> None:
    """UNAVAILABLE and ERROR sub-scores (not just ABSTAIN) are dropped from the
    weighted mean rather than counted as 0, so only the definitive PASS drives
    the composite score.
    """
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=1.0, hard_gate=False),
            WeightedGrader(
                _StaticGrader(GradeStatus.UNAVAILABLE, None), weight=5.0, hard_gate=False
            ),
            WeightedGrader(_StaticGrader(GradeStatus.ERROR, None), weight=5.0, hard_gate=False),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    # Only the first component's weight (1.0) is in the denominator; the other
    # two are excluded, so the mean is 1.0, not 1.0/11.0.
    assert result.score == pytest.approx(1.0)
    assert result.status is GradeStatus.PASS


@pytest.mark.asyncio
async def test_definitive_zero_scored_component_still_lowers_the_mean() -> None:
    """A component that returns a definitive numeric 0.0 (e.g. a FAIL with a
    real score) is legitimately included in the mean -- the exclusion rule
    applies only to non-definitive statuses / ``None`` scores, never to a
    genuine zero.
    """
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=1.0, hard_gate=False),
            WeightedGrader(_StaticGrader(GradeStatus.FAIL, 0.0), weight=1.0, hard_gate=False),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    # Both scores are definitive numbers, so the mean is (1.0 + 0.0) / 2.
    assert result.score == pytest.approx(0.5)
    assert result.status is GradeStatus.PARTIAL


class _Answer:
    """A trivial structured payload for SchemaGrader tests."""

    def __init__(self, value: int) -> None:
        self.value = value


_AnswerAdapter: TypeAdapter[dict[str, int]] = TypeAdapter(dict[str, int])


@pytest.mark.asyncio
async def test_schema_grader_passes_when_output_matches_type_adapter() -> None:
    grader = SchemaGrader(name="schema@1", adapter=_AnswerAdapter)
    now = datetime.now(UTC)
    execution = NormalizedExecutionResult(
        sample_id="s1",
        attempt=1,
        output={"value": 5},
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )
    result = await grader.grade(_sample(), execution)
    assert result.status is GradeStatus.PASS


@pytest.mark.asyncio
async def test_schema_grader_fails_on_type_mismatch() -> None:
    grader = SchemaGrader(name="schema@1", adapter=_AnswerAdapter)
    now = datetime.now(UTC)
    execution = NormalizedExecutionResult(
        sample_id="s1",
        attempt=1,
        output={"value": "not-an-int"},
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )
    result = await grader.grade(_sample(), execution)
    assert result.status is GradeStatus.FAIL
