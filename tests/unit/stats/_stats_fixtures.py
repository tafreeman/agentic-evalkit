"""Shared building blocks for the ``test_aggregate.py`` and ``test_compare.py`` tests.

Both test files need to build the same three per-sample objects for every
test case: an :class:`~agentic_evalkit.models.EvalSample` (the question
being asked), a :class:`~agentic_evalkit.models.NormalizedExecutionResult`
(what happened when the system under test tried to answer it), and a
:class:`~agentic_evalkit.models.GradeResult` (whether that answer was
judged correct). Rather than have each test file construct these three
objects from scratch with every required field spelled out, the three
helper functions below (``_sample``, ``_execution``, ``_grade``) do that
once, here, so both test files only need to fill in the handful of fields
each specific test actually cares about.

Each test file still keeps its own ``_run()`` helper (the function that
assembles a full run out of these samples) rather than sharing one,
because the two files genuinely need different things from it:
``test_aggregate.py`` just needs one simple run with fixed values
everywhere, while ``test_compare.py`` needs to independently vary every
single "provenance" field -- every fact about exactly what was run and how
(see ``compare.py``'s module docstring for the full explanation of that
term) -- so it can test that a mismatch in each one gets caught.
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
