"""Schema and composite objective graders (design §9, plan Task 10 Steps 2/4).

``CompositeGrader`` runs its component graders, preserves every child
result, computes the weighted mean over the *available* numeric sub-scores
(missing/abstained/unavailable scores are excluded from the mean, never
treated as zero), and fails with ``hard_gate=True`` when any hard-gated
component fails. A component grader that itself raises surfaces as
``GradeStatus.ERROR``, not a silent zero.
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

# Statuses that represent "this component did not produce a definitive
# grade" rather than a hard pass/fail. Their scores are excluded from the
# composite's weighted mean.
_NON_DEFINITIVE_STATUSES = frozenset(
    {GradeStatus.ABSTAIN, GradeStatus.ERROR, GradeStatus.UNAVAILABLE}
)


class SchemaGrader:
    """Validates ``NormalizedExecutionResult.output`` against a Pydantic ``TypeAdapter``."""

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
    """Pairs a component ``Grader`` with a composite weight and hard-gate flag."""

    def __init__(self, grader: Grader, *, weight: float, hard_gate: bool) -> None:
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}")
        self.grader = grader
        self.weight = weight
        self.hard_gate = hard_gate


class CompositeGrader:
    """Combines multiple graders under a noncompensable hard-gate policy.

    Args:
        name: Stable grader identifier reported on the composite ``GradeResult``.
        graders: Ordered ``WeightedGrader`` components. Order is preserved in
            evidence so reports can show each component's contribution.
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
                    # The child's own evidence rides along so a composite
                    # report keeps each component's audit trail (e.g. the
                    # grounded-citation per-check breakdown, ADR-0012) --
                    # making the docstring's "preserves every child result"
                    # claim true for evidence, not just status/score.
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
            # No component produced a definitive numeric score: the composite
            # cannot claim a pass/fail verdict, so it reports unavailable
            # rather than fabricating a zero-weighted average.
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

        # `_weighted_mean` guarantees `weighted_score is not None` whenever
        # `total_weight != 0.0` (the branch above already returned otherwise);
        # `cast` documents that proven invariant without a runtime check that
        # `assert` would strip under `python -O`.
        weighted_score = cast("float", weighted_score)
        status = GradeStatus.PASS if weighted_score >= 1.0 else GradeStatus.PARTIAL
        # A composite whose weighted mean is exactly 0 with no failed hard
        # gate is still a definitive fail signal.
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
            # Deliberately broad: any component-grader failure must surface
            # as an explicit ERROR result rather than propagate and abort
            # the whole composite.
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
