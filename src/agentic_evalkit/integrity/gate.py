"""Wire an integrity grader in front of an oracle so a flagged run can never PASS.

Two ways ``CompositeGrader`` can let a flagged run through if wired by hand:

- A non-hard-gate child that returns ``UNAVAILABLE`` is left out of the weighted
  mean, so an exploit flag beside an oracle PASS still grades PASS.
- A hard-gate child only vetoes on ``FAIL`` or a non-verdict
  (``ABSTAIN``/``ERROR``/``UNAVAILABLE``). A hard-gate ``PARTIAL`` beside an
  oracle PASS still grades PASS.

A weighted integrity child is wrong the other way too: a clean integrity PASS
(score 1.0) averaged with an oracle FAIL (score 0.0) reads as PARTIAL.

So the integrity grader goes in as ``hard_gate=True`` with ``weight=0`` (it can
veto but never scores), and its result is narrowed to PASS, ERROR or
UNAVAILABLE first. Build the pair through :func:`integrity_gated` rather than
by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_evalkit.graders.composite import CompositeGrader, WeightedGrader
from agentic_evalkit.models import EvalSample, GradeResult, GradeStatus, NormalizedExecutionResult

if TYPE_CHECKING:
    from agentic_evalkit.graders.base import Grader

_KEPT = frozenset({GradeStatus.PASS, GradeStatus.ERROR})


class _IntegrityVeto:
    """Narrows an integrity verdict: anything but PASS or ERROR becomes UNAVAILABLE."""

    def __init__(self, inner: Grader) -> None:
        self._inner = inner

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        result = await self._inner.grade(sample, execution)
        if result.status in _KEPT:
            return result
        return GradeResult(
            sample_id=result.sample_id,
            grader=result.grader,
            grader_type=result.grader_type,
            status=GradeStatus.UNAVAILABLE,
            score=None,
            hard_gate=result.hard_gate,
            evidence={**result.evidence, "integrity_original_status": result.status.value},
            artifact_refs=result.artifact_refs,
            created_at=result.created_at,
        )


def integrity_gated(*, oracle: Grader, integrity: Grader, name: str) -> CompositeGrader:
    """Return a composite where ``integrity`` vetoes and ``oracle`` alone scores.

    Contract (pinned by ``tests/unit/integrity/test_gate.py``):

    - integrity not PASS -> the composite is never PASS;
    - integrity PASS and a definitive oracle (PASS, PARTIAL, FAIL) -> the
      oracle's status and score;
    - integrity PASS and a non-definitive oracle -> UNAVAILABLE.
    """
    return CompositeGrader(
        name=name,
        graders=(
            WeightedGrader(_IntegrityVeto(integrity), weight=0.0, hard_gate=True),
            WeightedGrader(oracle, weight=1.0, hard_gate=False),
        ),
    )
