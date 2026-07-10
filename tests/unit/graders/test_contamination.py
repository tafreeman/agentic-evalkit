"""Tests for :mod:`agentic_evalkit.graders.contamination` (ADR-0013).

Hermetic, no network. Covers the helper's empty-input behavior, partial
leak subsets, normalization-insensitive matching (case- and
whitespace-mangled echoes), the fixed evidence payload shape, JSON
round-trips, agreement with the grounded-citation grader's own canary
check, and an end-to-end merge into a ``GradeResult``.
"""

import json
from datetime import UTC, datetime

import pytest

from agentic_evalkit.graders.contamination import (
    canary_leak_evidence,
    find_canary_leaks,
    normalize_for_containment,
)
from agentic_evalkit.graders.grounding import GroundedCitationGrader
from agentic_evalkit.models import (
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
)
from agentic_evalkit.models.grounding import GroundingCheck

_CANARY_A = "TRIPWIRE-ALPHA-001"
_CANARY_B = "TRIPWIRE-BETA-002"


def test_empty_text_or_empty_canaries_return_empty() -> None:
    assert find_canary_leaks("", (_CANARY_A,)) == ()
    assert find_canary_leaks("some text", ()) == ()
    assert find_canary_leaks("   ", (_CANARY_A,)) == ()


def test_partial_leak_returns_exactly_the_leaked_subset_in_canary_order() -> None:
    text = f"the log mentions {_CANARY_B} but nothing else"
    assert find_canary_leaks(text, (_CANARY_A, _CANARY_B)) == (_CANARY_B,)


def test_case_mangled_leak_is_detected() -> None:
    """Matching is normalization-insensitive (adversarial review finding,
    2026-07-09): a case-mangled canary echo must not evade the tripwire."""
    text = f"see {_CANARY_A.lower()} in the appendix"
    assert find_canary_leaks(text, (_CANARY_A,)) == (_CANARY_A,)


def test_whitespace_mangled_leak_is_detected() -> None:
    canary = "TRIPWIRE GAMMA 003"
    text = "the note reads tripwire\n   gamma 003 verbatim"
    assert find_canary_leaks(text, (canary,)) == (canary,)


def test_absent_canary_is_not_reported() -> None:
    assert find_canary_leaks("perfectly clean output", (_CANARY_A, _CANARY_B)) == ()


def test_duplicate_canary_ids_are_deduplicated() -> None:
    text = f"echoed: {_CANARY_A}"
    assert find_canary_leaks(text, (_CANARY_A, _CANARY_A)) == (_CANARY_A,)


def test_blank_canary_token_never_matches() -> None:
    """A whitespace-only canary normalizes to empty and must not vacuously
    match every haystack (guards Python's ``"" in text`` truth)."""
    assert find_canary_leaks("any text at all", ("   ",)) == ()


def test_normalizer_matches_the_documented_steps() -> None:
    assert normalize_for_containment("  Café   AU\tLait ") == "café au lait"


def test_evidence_shape_for_clean_and_leaked() -> None:
    assert canary_leak_evidence(()) == {"canary_check": "clean", "leaked_canary_ids": []}
    assert canary_leak_evidence((_CANARY_A,)) == {
        "canary_check": "leaked",
        "leaked_canary_ids": [_CANARY_A],
    }


def test_every_return_value_is_json_serializable() -> None:
    for payload in (canary_leak_evidence(()), canary_leak_evidence((_CANARY_A, _CANARY_B))):
        assert json.loads(json.dumps(payload)) == payload


# --- consistency with the grounded-citation grader (ADR-0012/ADR-0013) -----


def _grounding_sample(document_text: str) -> EvalSample:
    return EvalSample(
        sample_id="grounded-citation:t1",
        input={
            "question": "What does the ops log say?",
            "documents": [{"doc_id": "doc-a", "title": "Ops", "text": document_text}],
        },
        reference="the failover drill completed in ninety seconds",
        metadata={
            "required_evidence": ["doc-a"],
            "canary_tokens": [_CANARY_A],
            "gold_spans": [
                {"doc_id": "doc-a", "quote": "the failover drill completed in ninety seconds"}
            ],
        },
        source_digest="sha256:test-row",
        adapter="grounded-citation-tasks@1",
    )


@pytest.mark.asyncio
async def test_helper_and_grounding_grader_agree_on_a_case_mangled_leak() -> None:
    """The reusable helper and the grounded-citation grader share one
    tripwire semantics: both must flag the same case-mangled echo."""
    document_text = (
        f"the failover drill completed in ninety seconds. {_CANARY_A} "
        "Operators log every drill in the master ledger."
    )
    answer_text = f"Drill took ninety seconds, per marker {_CANARY_A.lower()}."
    payload = {
        "answer": answer_text,
        "citations": [
            {"doc_id": "doc-a", "quote": "the failover drill completed in ninety seconds"}
        ],
    }
    now = datetime.now(UTC)
    execution = NormalizedExecutionResult(
        sample_id="grounded-citation:t1",
        attempt=1,
        output=payload,
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )
    grader = GroundedCitationGrader(name="grounded-citation-deterministic@1")
    result = await grader.grade(_grounding_sample(document_text), execution)

    helper_leaks = find_canary_leaks(answer_text, (_CANARY_A,))
    assert helper_leaks == (_CANARY_A,)
    assert result.status is GradeStatus.FAIL
    checks = result.evidence["checks"]
    assert checks[GroundingCheck.CANARY_LEAK.value]["leaked_canary_tokens"] == list(helper_leaks)


# --- end-to-end reuse: evidence merges into a GradeResult -------------------


class _CanaryAwareGrader:
    """Minimal grader proving the documented reuse path: call the helper,
    merge its evidence payload, return a valid ``GradeResult``."""

    def __init__(self, *, canary_ids: tuple[str, ...]) -> None:
        self._canary_ids = canary_ids

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        text = str((execution.output or {}).get("answer", ""))
        leaked = find_canary_leaks(text, self._canary_ids)
        return GradeResult(
            sample_id=sample.sample_id,
            grader="canary-aware@1",
            status=GradeStatus.FAIL if leaked else GradeStatus.PASS,
            score=0.0 if leaked else 1.0,
            hard_gate=False,
            evidence={"note": "policy applied by this grader", **canary_leak_evidence(leaked)},
            created_at=datetime.now(UTC),
        )


@pytest.mark.asyncio
async def test_evidence_payload_merges_into_a_grade_result_and_round_trips() -> None:
    sample = EvalSample(
        sample_id="s1",
        input={"question": "?"},
        source_digest="sha256:row",
        adapter="identity@1",
    )
    now = datetime.now(UTC)
    execution = NormalizedExecutionResult(
        sample_id="s1",
        attempt=1,
        output={"answer": f"contains {_CANARY_A}"},
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )
    result = await _CanaryAwareGrader(canary_ids=(_CANARY_A,)).grade(sample, execution)
    assert result.status is GradeStatus.FAIL
    assert result.evidence["canary_check"] == "leaked"
    assert GradeResult.model_validate_json(result.model_dump_json()) == result
