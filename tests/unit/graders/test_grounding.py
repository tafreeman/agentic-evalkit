"""Tests for :mod:`agentic_evalkit.graders.grounding` (ADR-0012).

"Grounded citation" grading checks that an AI's answer is actually backed
up by real quotes from its source documents, rather than just sounding
plausible. This file covers three layers:

1. The deterministic tier -- rule-based checks, no AI judge involved --
   tested one at a time: things like "does this quote actually appear in
   the document it's attributed to" and "does the answer cite enough of
   the documents it was required to use." Includes a regression test for a
   loophole an adversarial review found, where a single trivial one-word
   quote was enough to slip past these checks.
2. The "rubric binding" (`RubricBoundJudgeClient`), which wraps an AI judge
   so it always scores against a fixed, written rubric (a checklist of
   criteria).
3. The factory function that assembles the deterministic checks and the AI
   judge into one combined grader, and the guarantees that assembly makes:
   the judge's score carries zero weight so it can never move the final
   numeric score by itself; a hard failure in the deterministic tier still
   forces the whole result to fail (see the "hard gate" concept explained
   in test_composite.py); and trying to attach saved calibration data
   without also supplying a judge client is rejected right at construction
   time.
"""

from datetime import UTC, datetime

import pytest
from pydantic import JsonValue

from agentic_evalkit.graders.grounding import (
    GRADING_SCOPE,
    GroundedCitationGrader,
    RubricBoundJudgeClient,
    build_grounded_citation_grader,
    build_grounding_rubric,
)
from agentic_evalkit.graders.judge import CalibrationArtifact, JudgeRequest, JudgeResponse
from agentic_evalkit.graders.rubric import Rubric, RubricCriterion
from agentic_evalkit.models import (
    EvalSample,
    ExecutionStatus,
    GradeStatus,
    NormalizedExecutionResult,
)
from agentic_evalkit.models.grounding import GroundingCheck

_CANARY_A = "TRIPWIRE-ALPHA-001"
_CANARY_B = "TRIPWIRE-BETA-002"
_DOC_A_TEXT = (
    "Alpha station's molten-salt loop runs at negative pressure. "
    f"{_CANARY_A} The loop is inspected every twelve hours by the night crew."
)
_DOC_B_TEXT = (
    "Beta station stores backup fuel cells in a shielded vault. "
    f"{_CANARY_B} Fuel cells are rotated monthly during scheduled maintenance windows."
)
_GOOD_QUOTE_A = "molten-salt loop runs at negative pressure"
_GOOD_QUOTE_B = "Fuel cells are rotated monthly"
_CLEAN_ANSWER = "Alpha runs its loop at negative pressure; Beta rotates fuel cells monthly."


def _grounding_sample() -> EvalSample:
    return EvalSample(
        sample_id="grounded-citation:t1",
        input={
            "question": "How do Alpha and Beta stations maintain their equipment?",
            "documents": [
                {"doc_id": "doc-a", "title": "Alpha ops", "text": _DOC_A_TEXT},
                {"doc_id": "doc-b", "title": "Beta ops", "text": _DOC_B_TEXT},
            ],
        },
        reference=_GOOD_QUOTE_A,
        metadata={
            "required_evidence": ["doc-a", "doc-b"],
            "canary_tokens": [_CANARY_A, _CANARY_B],
            "gold_spans": [{"doc_id": "doc-a", "quote": _GOOD_QUOTE_A}],
        },
        tags=("grounded-citation",),
        source_digest="sha256:test-row",
        adapter="grounded-citation-tasks@1",
    )


def _execution(payload: dict[str, JsonValue] | None) -> NormalizedExecutionResult:
    now = datetime.now(UTC)
    return NormalizedExecutionResult(
        sample_id="grounded-citation:t1",
        attempt=1,
        output=payload,
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )


def _payload(
    *,
    answer: str = _CLEAN_ANSWER,
    citations: list[dict[str, str]] | None = None,
) -> dict[str, JsonValue]:
    if citations is None:
        citations = [
            {"doc_id": "doc-a", "quote": _GOOD_QUOTE_A},
            {"doc_id": "doc-b", "quote": _GOOD_QUOTE_B},
        ]
    return {"answer": answer, "citations": citations}  # type: ignore[dict-item]


def _grader() -> GroundedCitationGrader:
    return GroundedCitationGrader(name="grounded-citation-deterministic@1")


def _checks(result_evidence: dict[str, object]) -> dict[str, dict[str, object]]:
    checks = result_evidence["checks"]
    assert isinstance(checks, dict)
    return checks


# --- deterministic tier: pass/fail branches --------------------------------


@pytest.mark.asyncio
async def test_full_grounding_discipline_passes() -> None:
    result = await _grader().grade(_grounding_sample(), _execution(_payload()))
    assert result.status is GradeStatus.PASS
    assert result.score == pytest.approx(1.0)
    assert result.hard_gate is False
    assert result.evidence["grading_scope"] == GRADING_SCOPE
    assert result.evidence["failed_checks"] == []


@pytest.mark.asyncio
async def test_degenerate_one_word_quotes_cannot_pass() -> None:
    """Regression test for a loophole an adversarial review found: an AI
    could try to game this grader by "citing" just one real, verbatim word
    per required document (e.g. quoting only the word "loop"), plus some
    other non-empty answer text. That single word genuinely does appear in
    the document, so the faithfulness check (is this quote really in the
    source?) passes -- but it's too trivial to count as a real citation, so
    the citation-present and evidence-coverage checks correctly still fail
    and block the grade."""
    degenerate = _payload(
        citations=[
            {"doc_id": "doc-a", "quote": "loop"},
            {"doc_id": "doc-b", "quote": "monthly"},
        ]
    )
    result = await _grader().grade(_grounding_sample(), _execution(degenerate))
    assert result.status is GradeStatus.FAIL
    assert result.score == pytest.approx(0.0)
    checks = _checks(result.evidence)
    assert checks[GroundingCheck.QUOTE_FAITHFULNESS.value]["passed"] is True
    assert checks[GroundingCheck.CITATION_PRESENT.value]["passed"] is False
    assert checks[GroundingCheck.EVIDENCE_COVERAGE.value]["passed"] is False


@pytest.mark.asyncio
async def test_hallucinated_doc_id_fails_resolution() -> None:
    payload = _payload(
        citations=[
            {"doc_id": "doc-a", "quote": _GOOD_QUOTE_A},
            {"doc_id": "doc-b", "quote": _GOOD_QUOTE_B},
            {"doc_id": "doc-zz", "quote": "a quote from a document that does not exist"},
        ]
    )
    result = await _grader().grade(_grounding_sample(), _execution(payload))
    assert result.status is GradeStatus.FAIL
    assert result.evidence["failed_checks"] == [GroundingCheck.CITATION_RESOLUTION.value]
    checks = _checks(result.evidence)
    assert checks[GroundingCheck.CITATION_RESOLUTION.value]["unknown_doc_ids"] == ["doc-zz"]


@pytest.mark.asyncio
async def test_fabricated_quote_fails_faithfulness_and_coverage() -> None:
    payload = _payload(
        citations=[
            {"doc_id": "doc-a", "quote": "the reactor melted down entirely yesterday"},
            {"doc_id": "doc-b", "quote": _GOOD_QUOTE_B},
        ]
    )
    result = await _grader().grade(_grounding_sample(), _execution(payload))
    assert result.status is GradeStatus.FAIL
    assert result.evidence["failed_checks"] == [
        GroundingCheck.EVIDENCE_COVERAGE.value,
        GroundingCheck.QUOTE_FAITHFULNESS.value,
    ]
    checks = _checks(result.evidence)
    unfaithful = checks[GroundingCheck.QUOTE_FAITHFULNESS.value]["unfaithful_citations"]
    assert unfaithful == [
        {"doc_id": "doc-a", "quote": "the reactor melted down entirely yesterday"}
    ]


@pytest.mark.asyncio
async def test_missing_required_evidence_fails_coverage() -> None:
    payload = _payload(citations=[{"doc_id": "doc-a", "quote": _GOOD_QUOTE_A}])
    result = await _grader().grade(_grounding_sample(), _execution(payload))
    assert result.status is GradeStatus.FAIL
    assert result.evidence["failed_checks"] == [GroundingCheck.EVIDENCE_COVERAGE.value]
    checks = _checks(result.evidence)
    coverage = checks[GroundingCheck.EVIDENCE_COVERAGE.value]
    assert coverage["uncovered_required_doc_ids"] == ["doc-b"]


@pytest.mark.asyncio
async def test_canary_leak_in_answer_fails() -> None:
    payload = _payload(answer=f"The inspection marker is {_CANARY_A}, cited for completeness.")
    result = await _grader().grade(_grounding_sample(), _execution(payload))
    assert result.status is GradeStatus.FAIL
    assert result.evidence["failed_checks"] == [GroundingCheck.CANARY_LEAK.value]
    checks = _checks(result.evidence)
    assert checks[GroundingCheck.CANARY_LEAK.value]["leaked_canary_tokens"] == [_CANARY_A]


@pytest.mark.asyncio
async def test_case_mangled_canary_leak_is_still_detected() -> None:
    """A "canary" is a unique, made-up marker planted in a source document
    to detect whether the AI is echoing content it shouldn't have direct
    access to. This test checks that a leaked canary is still caught even
    if its case has been scrambled along the way -- a gap found during
    review: matching normalizes case on both sides before comparing, so a
    case-scrambled echo can't dodge the check."""
    payload = _payload(answer=f"see {_CANARY_A.lower()} in the appendix")
    result = await _grader().grade(_grounding_sample(), _execution(payload))
    assert result.status is GradeStatus.FAIL
    checks = _checks(result.evidence)
    assert checks[GroundingCheck.CANARY_LEAK.value]["leaked_canary_tokens"] == [_CANARY_A]


@pytest.mark.asyncio
async def test_canary_leak_via_quote_fails_even_when_faithful() -> None:
    """Here the canary marker is quoted word-for-word from the source
    document, so it passes the faithfulness check (the quote really is
    genuine). But it's still flagged as a leak: the canary check isn't
    about whether a quote is accurate, it's about whether the answer
    exposes a marker that was deliberately planted to catch exactly this
    kind of leak -- faithful quoting included."""
    payload = _payload(
        citations=[
            {"doc_id": "doc-a", "quote": f"{_CANARY_A} The loop is inspected"},
            {"doc_id": "doc-b", "quote": _GOOD_QUOTE_B},
        ]
    )
    result = await _grader().grade(_grounding_sample(), _execution(payload))
    assert result.status is GradeStatus.FAIL
    assert result.evidence["failed_checks"] == [GroundingCheck.CANARY_LEAK.value]


@pytest.mark.asyncio
async def test_blank_answer_fails_nonempty() -> None:
    result = await _grader().grade(_grounding_sample(), _execution(_payload(answer="   ")))
    assert result.status is GradeStatus.FAIL
    assert result.evidence["failed_checks"] == [GroundingCheck.ANSWER_NONEMPTY.value]


@pytest.mark.asyncio
async def test_empty_quote_is_never_faithful() -> None:
    """In Python, checking whether an empty string appears inside any text
    always returns ``True`` (``"" in text`` is always true). This test
    makes sure that quirk doesn't let an empty quote count as faithful -- a
    citation that quotes nothing should fail the faithfulness check, not
    silently pass it."""
    payload = _payload(
        citations=[
            {"doc_id": "doc-a", "quote": ""},
            {"doc_id": "doc-b", "quote": _GOOD_QUOTE_B},
        ]
    )
    result = await _grader().grade(_grounding_sample(), _execution(payload))
    checks = _checks(result.evidence)
    assert checks[GroundingCheck.QUOTE_FAITHFULNESS.value]["passed"] is False


@pytest.mark.asyncio
async def test_malformed_payload_fails_structured_contract() -> None:
    result = await _grader().grade(_grounding_sample(), _execution({"answer": 7}))
    assert result.status is GradeStatus.FAIL
    assert result.score == pytest.approx(0.0)
    assert result.evidence["failed_checks"] == [GroundingCheck.STRUCTURED_CONTRACT.value]
    checks = _checks(result.evidence)
    assert "validation_error" in checks[GroundingCheck.STRUCTURED_CONTRACT.value]


@pytest.mark.asyncio
async def test_extra_payload_keys_fail_structured_contract() -> None:
    """The AI's output is only allowed to contain exactly two fields,
    ``answer`` and ``citations`` -- nothing else (enforced through
    Pydantic's ``extra="forbid"`` setting, which rejects any undeclared
    field). An extra, unexpected key in the payload is treated as a
    contract violation that fails the check, not as harmless noise to
    ignore."""
    payload = _payload()
    payload["debug"] = True
    result = await _grader().grade(_grounding_sample(), _execution(payload))
    assert result.evidence["failed_checks"] == [GroundingCheck.STRUCTURED_CONTRACT.value]


@pytest.mark.asyncio
async def test_citationless_answer_fails_presence_and_coverage() -> None:
    """This uses the exact output shape produced by the project's built-in
    demo target, ``zero_target`` (a stand-in for a real AI that always
    answers the literal string ``"0"`` with no citations at all -- used
    elsewhere in the codebase purely as a smoke test, not a genuine answer
    generator). That output, ``{"answer": "0"}``, is still valid enough to
    parse successfully (the ``citations`` field just defaults to an empty
    list), so it must fail because citations are missing, not because the
    payload itself is malformed."""
    result = await _grader().grade(_grounding_sample(), _execution({"answer": "0"}))
    assert result.status is GradeStatus.FAIL
    checks = _checks(result.evidence)
    assert checks[GroundingCheck.STRUCTURED_CONTRACT.value]["passed"] is True
    assert checks[GroundingCheck.CITATION_PRESENT.value]["passed"] is False
    assert checks[GroundingCheck.EVIDENCE_COVERAGE.value]["passed"] is False


@pytest.mark.asyncio
async def test_non_completed_execution_is_unavailable() -> None:
    now = datetime.now(UTC)
    failed = NormalizedExecutionResult(
        sample_id="grounded-citation:t1",
        attempt=1,
        output=None,
        status=ExecutionStatus.ERROR,
        started_at=now,
        finished_at=now,
    )
    result = await _grader().grade(_grounding_sample(), failed)
    assert result.status is GradeStatus.UNAVAILABLE
    assert result.score is None


@pytest.mark.asyncio
async def test_completed_execution_with_no_output_is_unavailable() -> None:
    result = await _grader().grade(_grounding_sample(), _execution(None))
    assert result.status is GradeStatus.UNAVAILABLE


@pytest.mark.asyncio
async def test_structured_output_is_preferred_over_output() -> None:
    now = datetime.now(UTC)
    execution = NormalizedExecutionResult(
        sample_id="grounded-citation:t1",
        attempt=1,
        output={"answer": 3},
        structured_output=_payload(),
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )
    result = await _grader().grade(_grounding_sample(), execution)
    assert result.status is GradeStatus.PASS


@pytest.mark.asyncio
async def test_sample_without_grounding_oracle_abstains() -> None:
    foreign = EvalSample(
        sample_id="gsm8k:0",
        input={"question": "1+1?"},
        reference="2",
        source_digest="sha256:row",
        adapter="gsm8k@1",
    )
    result = await _grader().grade(foreign, _execution(_payload()))
    assert result.status is GradeStatus.ABSTAIN
    assert result.score is None


def test_min_substantive_quote_tokens_must_be_positive() -> None:
    with pytest.raises(ValueError, match="min_substantive_quote_tokens"):
        GroundedCitationGrader(name="grounded@1", min_substantive_quote_tokens=0)


@pytest.mark.asyncio
async def test_substance_floor_is_configurable() -> None:
    lenient = GroundedCitationGrader(name="grounded@1", min_substantive_quote_tokens=1)
    degenerate = _payload(
        citations=[
            {"doc_id": "doc-a", "quote": "loop"},
            {"doc_id": "doc-b", "quote": "monthly"},
        ]
    )
    result = await lenient.grade(_grounding_sample(), _execution(degenerate))
    assert result.status is GradeStatus.PASS


# --- rubric binding ----------------------------------------------------------


class _ScriptedJudgeClient:
    """A fake judge client for tests: instead of actually calling an AI
    model, it always returns the same scripted verdict, and it records
    every request it receives so a test can inspect what was sent to it."""

    fingerprint = "sha256:scripted-judge"

    def __init__(self, *, verdict: str = "fail", score: float | None = 0.0) -> None:
        self.requests: list[JudgeRequest] = []
        self._verdict = verdict
        self._score = score

    async def judge(self, request: JudgeRequest) -> JudgeResponse:
        self.requests.append(request)
        return JudgeResponse(
            fingerprint=self.fingerprint,
            verdict=self._verdict,
            score=self._score,
            parse_ok=True,
            abstained=False,
        )


def _request(prompt: str = "question=How?", **metadata: object) -> JudgeRequest:
    return JudgeRequest(
        sample_id="s1",
        prompt=prompt,
        candidate_output="answer text",
        reference="reference text",
        metadata=dict(metadata),  # type: ignore[arg-type]
    )


def test_fingerprint_covers_inner_judge_and_rubric_content() -> None:
    inner = _ScriptedJudgeClient()
    rubric = build_grounding_rubric()
    bound = RubricBoundJudgeClient(inner, rubric=rubric)
    assert inner.fingerprint in bound.fingerprint
    assert rubric.rubric_id in bound.fingerprint

    other_rubric = Rubric(
        rubric_id="grounded-citation-rubric@1",
        criteria=(RubricCriterion(criterion_id="faithfulness", description="Different wording."),),
    )
    other = RubricBoundJudgeClient(_ScriptedJudgeClient(), rubric=other_rubric)
    assert other.fingerprint != bound.fingerprint


@pytest.mark.asyncio
async def test_rubric_is_rendered_into_the_prompt_with_concrete_reversal() -> None:
    inner = _ScriptedJudgeClient()
    bound = RubricBoundJudgeClient(inner, rubric=build_grounding_rubric())

    await bound.judge(_request())
    forward_prompt = inner.requests[0].prompt
    assert "Never return a numeric rating." in forward_prompt
    assert forward_prompt.endswith("question=How?")
    assert (
        forward_prompt.index("[faithfulness]")
        < forward_prompt.index("[completeness]")
        < forward_prompt.index("[sufficiency]")
    )

    await bound.judge(_request(reversed=True))
    reversed_prompt = inner.requests[1].prompt
    assert (
        reversed_prompt.index("[sufficiency]")
        < reversed_prompt.index("[completeness]")
        < reversed_prompt.index("[faithfulness]")
    )


@pytest.mark.asyncio
async def test_matching_inner_fingerprint_is_lifted_to_the_composite_identity() -> None:
    inner = _ScriptedJudgeClient()
    bound = RubricBoundJudgeClient(inner, rubric=build_grounding_rubric())
    response = await bound.judge(_request())
    assert response.fingerprint == bound.fingerprint


@pytest.mark.asyncio
async def test_inner_fingerprint_mismatch_passes_through_unlifted() -> None:
    class _MismatchedJudge(_ScriptedJudgeClient):
        async def judge(self, request: JudgeRequest) -> JudgeResponse:
            response = await super().judge(request)
            return response.model_copy(update={"fingerprint": "sha256:evil"})

    bound = RubricBoundJudgeClient(_MismatchedJudge(), rubric=build_grounding_rubric())
    response = await bound.judge(_request())
    assert response.fingerprint == "sha256:evil"


def test_grounding_rubric_expresses_the_three_axes() -> None:
    rubric = build_grounding_rubric()
    ids = tuple(criterion.criterion_id for criterion in rubric.criteria)
    assert ids == ("faithfulness", "completeness", "sufficiency")
    by_id = {criterion.criterion_id: criterion for criterion in rubric.criteria}
    assert by_id["faithfulness"].hard_gate is True
    assert by_id["completeness"].hard_gate is True
    assert by_id["sufficiency"].hard_gate is False
    assert all(criterion.requires_evidence for criterion in rubric.criteria)
    assert all(criterion.scale == "binary" for criterion in rubric.criteria)


# --- composite factory --------------------------------------------------------


def test_calibration_without_judge_client_is_rejected() -> None:
    calibration = CalibrationArtifact(
        calibration_id="cal-1",
        judge_fingerprint="sha256:scripted-judge",
        expires_at=datetime(2099, 1, 1, tzinfo=UTC),
        calibrated_at=datetime(2026, 7, 1, tzinfo=UTC),
        true_positive=40,
        true_negative=40,
        false_positive=1,
        false_negative=1,
        threshold=0.85,
    )
    with pytest.raises(ValueError, match="judge_client"):
        build_grounded_citation_grader(calibration=calibration)


@pytest.mark.asyncio
async def test_uncalibrated_judge_is_score_inert_and_never_gates() -> None:
    """With no calibration data supplied, the AI judge's weight is 0.0.
    This test checks that a FAIL verdict from that judge doesn't move the
    combined score at all, and doesn't force the whole result to fail
    either -- only the deterministic (rule-based) tier is allowed to do
    that."""
    grader = build_grounded_citation_grader(
        judge_client=_ScriptedJudgeClient(verdict="fail", score=0.0)
    )
    result = await grader.grade(_grounding_sample(), _execution(_payload()))
    assert result.status is GradeStatus.PASS
    assert result.score == pytest.approx(1.0)
    assert result.hard_gate is False

    children = {child["grader"]: child for child in result.evidence["children"]}
    judge_child = children["grounded-sufficiency-judge@1"]
    assert judge_child["status"] == GradeStatus.FAIL.value
    assert judge_child["hard_gate"] is False
    assert judge_child["weight"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_deterministic_failure_hard_gates_the_composite() -> None:
    grader = build_grounded_citation_grader(judge_client=_ScriptedJudgeClient())
    result = await grader.grade(_grounding_sample(), _execution({"answer": "0"}))
    assert result.status is GradeStatus.FAIL
    assert result.hard_gate is True

    children = {child["grader"]: child for child in result.evidence["children"]}
    deterministic_child = children["grounded-citation-deterministic@1"]
    assert deterministic_child["hard_gate"] is True
    assert deterministic_child["status"] == GradeStatus.FAIL.value
    # Each individual check's pass/fail detail is still available here,
    # even after being wrapped inside the combined grader's own evidence --
    # every child result carries its own full evidence along with it.
    failed_checks = deterministic_child["evidence"]["failed_checks"]
    assert GroundingCheck.CITATION_PRESENT.value in failed_checks


@pytest.mark.asyncio
async def test_factory_without_judge_is_deterministic_only() -> None:
    grader = build_grounded_citation_grader()
    result = await grader.grade(_grounding_sample(), _execution(_payload()))
    assert result.status is GradeStatus.PASS
    assert len(result.evidence["children"]) == 1
