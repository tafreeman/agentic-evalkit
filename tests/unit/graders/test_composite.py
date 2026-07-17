"""Tests for :mod:`agentic_evalkit.graders.composite` (plan Task 10 Step 2/4).

The very first test below involves a "hard gate": a component grader that
can force the entire combined result to fail no matter how well the other
components score, rather than just being averaged in like a normal score.
That first test is copied word-for-word from the project's implementation
plan (docs/plans/2026-07-02-agentic-evalkit-initial-release.md, Task 10
Step 2) and must keep passing exactly as written -- don't change it.
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
    """A minimal fake grader for tests: it always returns the same fixed
    status and score, no matter what sample or execution it's given."""

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
    """If a sub-grader's score is ``None`` (for example, because it
    abstained -- declined to give a verdict, often for lack of the data it
    needed) it is left out of the average entirely, not treated as if it
    had scored zero.
    """
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=1.0, hard_gate=False),
            WeightedGrader(_StaticGrader(GradeStatus.ABSTAIN, None), weight=3.0, hard_gate=False),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    # Only the first grader produced a usable numeric score, so its weight
    # of 1.0 is the only thing in the denominator: the weighted mean comes
    # out to 1.0, not 0.25 (which is what you'd get if the second grader's
    # missing score were wrongly counted as a 0).
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
    """If a component grader raises an exception while grading, that must
    show up as an explicit ERROR or UNAVAILABLE status -- never be silently
    treated as if it had returned a score of zero.
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
# These tests protect requirement FR11: if a sub-grader is marked as a
# "hard gate" and it FAILs, the whole composite result must FAIL too -- no
# amount of averaging with passing scores can undo that. Likewise, a
# sub-grader that raises an exception must never quietly count as a score
# of 0, and any sub-grader that comes back missing, abstained, unavailable,
# or errored should simply be left out of the weighted average rather than
# counted as a zero. The tests above already cover the simple cases (just
# one component, or an abstain on its own); the tests below add the
# trickier "mixed" cases, where one grader passes normally alongside
# another that gives a non-conclusive result -- exactly the situation
# where a bug that silently turns "no score" into "a score of zero" would
# otherwise slip through unnoticed.


class _AlwaysRaisesGrader:
    """A sub-grader that always raises an exception when asked to grade.

    This is defined once here at module level -- as opposed to the
    ``_RaisingGrader`` class defined earlier inside a single test function
    -- so that all of the "mixed" Story 1.2 tests below can share this same
    always-failing grader instead of each defining their own copy.
    """

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_hard_gate_fail_wins_over_a_near_perfect_majority() -> None:
    """A single hard-gated FAIL forces the whole composite result to FAIL,
    even though it carries a tiny weight and every other component -- each
    with far more weight -- passes with a perfect score. In other words, a
    hard gate can't be outvoted just by piling on enough high-scoring,
    heavily-weighted components.
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
    """When one component raises an exception while another passes, the
    error must not drag the weighted average down toward zero. The failing
    component's weight is dropped from the average entirely, so the
    passing component's score alone gives a mean of 1.0 -- instead of the
    roughly 0.09 you'd get if the erroring component had been wrongly
    counted as a score of 0 with its full weight of 10.0.
    """
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=1.0, hard_gate=False),
            WeightedGrader(_AlwaysRaisesGrader(), weight=10.0, hard_gate=False),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    # The component that raised an exception contributes nothing at all --
    # not its weight, and not a score of 0.0 either. (If it had wrongly been
    # scored as 0.0 with its weight of 10.0 still counted, the mean would
    # come out to about 0.09 instead of 1.0.)
    assert result.score == pytest.approx(1.0)
    assert result.status is GradeStatus.PASS


@pytest.mark.asyncio
async def test_raising_component_surfaces_as_error_child_result() -> None:
    """The component that raised an exception should still show up as its
    own child result in the evidence, with an explicit ERROR status and a
    ``None`` score -- never a made-up 0.0 -- so that anyone reading a report
    later can see exactly which component failed and why.
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
    """Sub-graders that come back UNAVAILABLE or ERROR -- not just ABSTAIN --
    are left out of the weighted average rather than counted as 0, so only
    the one sub-grader that actually reached a real PASS/FAIL verdict
    affects the final composite score.
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
    # Only the first component's weight (1.0) counts toward the average; the
    # other two are left out entirely, so the mean comes out to 1.0 -- not
    # 1.0/11.0, which is what it would be if their weights still counted.
    assert result.score == pytest.approx(1.0)
    assert result.status is GradeStatus.PASS


@pytest.mark.asyncio
async def test_definitive_zero_scored_component_still_lowers_the_mean() -> None:
    """A component that returns an actual, deliberate score of 0.0 (for
    example, a real FAIL with a real numeric score) IS included in the
    average like normal. The "leave it out" rule from the tests above only
    applies when a component has no real answer at all (a non-conclusive
    status, or a ``None`` score) -- it never applies to a genuine,
    intentional zero.
    """
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=1.0, hard_gate=False),
            WeightedGrader(_StaticGrader(GradeStatus.FAIL, 0.0), weight=1.0, hard_gate=False),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    # Both components here returned real numeric scores (not ``None``), so
    # the mean is simply (1.0 + 0.0) / 2.
    assert result.score == pytest.approx(0.5)
    assert result.status is GradeStatus.PARTIAL


class _Answer:
    """A simple object with a known shape, used by the SchemaGrader tests
    below."""

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
