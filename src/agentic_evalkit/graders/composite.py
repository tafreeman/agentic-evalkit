"""Graders that check output structure, and combine other graders into one verdict.

Design §9, plan Task 10 Steps 2/4.

``CompositeGrader`` is how several graders get combined into a single
overall result. It runs every component grader, keeps every one of their
individual results (nothing is discarded), and combines their numeric
scores into one weighted average -- but only over the components that
actually produced a usable number. A component that couldn't produce a
score (because it abstained, errored, or was unavailable) is left out of
that average entirely; it is never counted as a zero, which would unfairly
drag the average down. If any component marked as a "hard gate" (see
``WeightedGrader`` below) fails, the whole composite fails
(``hard_gate=True``) no matter how well the other components scored -- a
hard-gated failure can never be averaged away by good scores elsewhere.
And if a component grader raises an exception instead of returning a
result, that shows up as an explicit ``GradeStatus.ERROR`` for that
component, never as a silent zero score that could be mistaken for a real
failing grade.
"""

from datetime import UTC, datetime
from typing import Any, cast

from pydantic import TypeAdapter, ValidationError

from agentic_evalkit.graders.base import Grader
from agentic_evalkit.models import (
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
)

# Statuses meaning "this component didn't produce a definitive grade" --
# as opposed to a clear pass or fail. A component whose status is one of
# these has its score left out of the composite's weighted average
# entirely (see `_weighted_mean` below), rather than being counted as zero.
_NON_DEFINITIVE_STATUSES = frozenset(
    {GradeStatus.ABSTAIN, GradeStatus.ERROR, GradeStatus.UNAVAILABLE}
)


class SchemaGrader:
    """Checks ``NormalizedExecutionResult.output``'s fields/types via a Pydantic ``TypeAdapter``."""

    def __init__(self, *, name: str, adapter: TypeAdapter[Any]) -> None:
        self._name = name
        self._adapter = adapter

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        now = datetime.now(UTC)
        if execution.status is not ExecutionStatus.COMPLETED or execution.output is None:
            return GradeResult(
                sample_id=sample.sample_id,
                grader=self._name,
                grader_type="schema",
                status=GradeStatus.UNAVAILABLE,
                score=None,
                hard_gate=False,
                evidence={"reason": "execution did not complete"},
                created_at=now,
            )
        try:
            self._adapter.validate_python(execution.output)
        except ValidationError as error:
            return GradeResult(
                sample_id=sample.sample_id,
                grader=self._name,
                grader_type="schema",
                status=GradeStatus.FAIL,
                score=0.0,
                hard_gate=False,
                evidence={"validation_error": str(error)},
                created_at=now,
            )
        return GradeResult(
            sample_id=sample.sample_id,
            grader=self._name,
            grader_type="schema",
            status=GradeStatus.PASS,
            score=1.0,
            hard_gate=False,
            created_at=now,
        )


class WeightedGrader:
    """Wraps a component ``Grader`` with its weight and its ``hard_gate`` flag."""

    def __init__(self, grader: Grader, *, weight: float, hard_gate: bool) -> None:
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}")
        self.grader = grader
        self.weight = weight
        self.hard_gate = hard_gate


class CompositeGrader:
    """Combines graders into one verdict; a hard-gate failure can't be outweighed elsewhere.

    Args:
        name: A stable label for this grader, recorded on the composite's
            ``GradeResult`` so you can tell which grader produced it.
        graders: The component graders, each wrapped in a ``WeightedGrader``,
            in the order they should be evaluated. That order is preserved
            in the evidence so a report can show exactly how much each
            component contributed to the final result.
    """

    def __init__(self, *, name: str, graders: tuple[WeightedGrader, ...]) -> None:
        self._name = name
        self._graders = graders

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        now = datetime.now(UTC)
        child_results: list[GradeResult] = []
        for component in self._graders:
            child_results.append(await self._grade_component(component, sample, execution))

        any_hard_gate_failed = any(
            component.hard_gate and child.status is GradeStatus.FAIL
            for component, child in zip(self._graders, child_results, strict=True)
        )

        weighted_score, total_weight = self._weighted_mean(child_results)
        evidence: dict[str, Any] = {
            "children": [
                {
                    "grader": child.grader,
                    "status": child.status.value,
                    "score": child.score,
                    "weight": component.weight,
                    "hard_gate": component.hard_gate,
                    # Each component's own detailed evidence rides along
                    # here too, not just its status/score, so a report built
                    # from this composite result still has the full detail
                    # behind each component (for example, the
                    # grounded-citation grader's per-check breakdown --
                    # ADR-0012). This is what makes the class docstring's
                    # "preserves every child result" promise actually true:
                    # without this line, only the summary numbers would
                    # survive, not the reasoning behind them.
                    "evidence": child.evidence,
                }
                for component, child in zip(self._graders, child_results, strict=True)
            ]
        }

        if any_hard_gate_failed:
            return GradeResult(
                sample_id=sample.sample_id,
                grader=self._name,
                grader_type="composite",
                status=GradeStatus.FAIL,
                score=weighted_score,
                hard_gate=True,
                evidence=evidence,
                created_at=now,
            )

        if total_weight == 0.0:
            # None of the components produced a usable numeric score (they
            # all abstained, errored, or were unavailable). With nothing to
            # average, the composite has no basis for a pass/fail verdict,
            # so it reports "unavailable" instead of making up a fake
            # average out of zero real scores.
            return GradeResult(
                sample_id=sample.sample_id,
                grader=self._name,
                grader_type="composite",
                status=GradeStatus.UNAVAILABLE,
                score=None,
                hard_gate=False,
                evidence=evidence,
                created_at=now,
            )

        # At this point `weighted_score` can't actually be `None`:
        # `_weighted_mean` only returns `None` together with `total_weight ==
        # 0.0`, and that case already returned above. We could double-check
        # this with an `assert`, but Python silently removes `assert`
        # statements when run with the `-O` (optimize) flag, so an `assert`
        # here wouldn't reliably catch a bug if one ever crept in. `cast` is
        # different: it's purely a hint for the type checker (mypy) that has
        # no effect at runtime, so it just tells mypy "trust us, this is a
        # float" without depending on a check that might silently not run.
        weighted_score = cast("float", weighted_score)
        status = GradeStatus.PASS if weighted_score >= 1.0 else GradeStatus.PARTIAL
        # Even though no hard-gated component failed, a weighted average of
        # exactly 0 still means every component that counted toward the
        # score scored zero -- that's a clear failure, not "partial
        # credit," so it's reported as FAIL rather than PARTIAL.
        if weighted_score == 0.0:
            status = GradeStatus.FAIL

        return GradeResult(
            sample_id=sample.sample_id,
            grader=self._name,
            grader_type="composite",
            status=status,
            score=weighted_score,
            hard_gate=False,
            evidence=evidence,
            created_at=now,
        )

    @staticmethod
    async def _grade_component(
        component: WeightedGrader, sample: EvalSample, execution: NormalizedExecutionResult
    ) -> GradeResult:
        try:
            return await component.grader.grade(sample, execution)
        except Exception as error:
            # Catching `Exception` broadly here is deliberate: if any single
            # component grader blows up, that shouldn't crash the entire
            # composite grading run. Instead, it becomes one explicit ERROR
            # result for this component, and grading continues normally for
            # every other component and sample.
            return GradeResult(
                sample_id=sample.sample_id,
                grader="unknown",
                grader_type="composite_component",
                status=GradeStatus.ERROR,
                score=None,
                hard_gate=False,
                evidence={"error": repr(error)},
                created_at=datetime.now(UTC),
            )

    def _weighted_mean(self, child_results: list[GradeResult]) -> tuple[float | None, float]:
        numerator = 0.0
        denominator = 0.0
        for component, child in zip(self._graders, child_results, strict=True):
            if child.status in _NON_DEFINITIVE_STATUSES or child.score is None:
                continue
            numerator += component.weight * child.score
            denominator += component.weight
        if denominator == 0.0:
            return None, 0.0
        return numerator / denominator, denominator
