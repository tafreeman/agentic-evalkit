"""Shared fixtures for stats tests (aggregate + compare).

``test_aggregate.py`` and ``test_compare.py`` both build the same per-sample
building blocks -- an :class:`~agentic_evalkit.models.EvalSample`, a
:class:`~agentic_evalkit.models.NormalizedExecutionResult`, and a
:class:`~agentic_evalkit.models.GradeResult` -- from the same field set
(``sample_id``, ``input``, ``reference='42'``, ``source_digest``,
``adapter='gsm8k@1'`` for the sample; ``sample_id``/``attempt``/``status``/
``started_at``/``finished_at`` for the execution; ``sample_id``/``grader``/
``status``/``score``/``hard_gate``/``created_at`` for the grade). Each test
file's ``_run()`` (or manifest-equivalent) stays file-local because they
genuinely differ: ``test_aggregate.py`` needs one simple, fixed-provenance
run, while ``test_compare.py`` parameterizes every provenance field to test
compatibility mismatches.
"""

from datetime import UTC, datetime

from agentic_evalkit.models import (
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
)

STARTED_AT = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)
FINISHED_AT = datetime(2026, 7, 2, 12, 5, 0, tzinfo=UTC)


def _sample(sample_id: str) -> EvalSample:
    return EvalSample(
        sample_id=sample_id,
        input={"question": f"question for {sample_id}"},
        reference="42",
        source_digest=f"sha256:{sample_id}",
        adapter="gsm8k@1",
    )


def _execution(
    sample_id: str,
    *,
    attempt: int = 1,
    status: ExecutionStatus,
    output: dict[str, object] | None = None,
    latency_ms: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
) -> NormalizedExecutionResult:
    return NormalizedExecutionResult(
        sample_id=sample_id,
        attempt=attempt,
        output=output,
        status=status,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
    )


def _grade(
    sample_id: str,
    *,
    grader: str = "normalized-exact@1",
    status: GradeStatus,
    score: float | None = None,
    hard_gate: bool = False,
    evidence: dict[str, object] | None = None,
) -> GradeResult:
    return GradeResult(
        sample_id=sample_id,
        grader=grader,
        status=status,
        score=score,
        hard_gate=hard_gate,
        evidence=evidence if evidence is not None else {},
        created_at=FINISHED_AT,
    )
