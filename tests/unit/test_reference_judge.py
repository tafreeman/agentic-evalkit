"""Tests for the packaged, deterministic ``ReferenceJudgeClient`` (T2-A(c)).

Covers the client in isolation (fingerprint stability, containment
matching, abstention with no reference, indifference to the position-bias
probe's ``metadata``) and its integration with ``JudgeGrader`` proving the
one safety property that matters most: wired in permanently uncalibrated,
it can never hard-gate a release, regardless of the caller's ``gate``
argument.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentic_evalkit.examples.reference_judge import ReferenceJudgeClient
from agentic_evalkit.graders.judge import JudgeGrader, JudgeRequest
from agentic_evalkit.models import (
    EvalSample,
    ExecutionStatus,
    GradeStatus,
    NormalizedExecutionResult,
)

_STARTED_AT = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)
_FINISHED_AT = datetime(2026, 7, 2, 12, 5, 0, tzinfo=UTC)


def _sample(sample_id: str = "s0", *, reference: str | None = "42") -> EvalSample:
    return EvalSample(
        sample_id=sample_id,
        input={"question": "what is the answer?"},
        reference=reference,
        source_digest=f"sha256:{sample_id}",
        adapter="gsm8k@1",
    )


def _execution(sample_id: str = "s0", *, answer: str) -> NormalizedExecutionResult:
    return NormalizedExecutionResult(
        sample_id=sample_id,
        attempt=1,
        output={"answer": answer},
        status=ExecutionStatus.COMPLETED,
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


def _request(*, candidate_output: str, reference: str | None) -> JudgeRequest:
    return JudgeRequest(
        sample_id="s0", prompt="p", candidate_output=candidate_output, reference=reference
    )


# --- ReferenceJudgeClient in isolation ---------------------------------------


def test_fingerprint_is_deterministic_across_instances() -> None:
    assert ReferenceJudgeClient().fingerprint == ReferenceJudgeClient().fingerprint


def test_fingerprint_has_sha256_prefix() -> None:
    assert ReferenceJudgeClient().fingerprint.startswith("sha256:")


@pytest.mark.asyncio
async def test_pass_when_reference_is_a_substring_of_candidate() -> None:
    client = ReferenceJudgeClient()
    response = await client.judge(_request(candidate_output="the answer is 42", reference="42"))
    assert response.verdict == "pass"
    assert response.score == 1.0
    assert response.parse_ok is True
    assert response.abstained is False


@pytest.mark.asyncio
async def test_fail_when_reference_is_not_a_substring_of_candidate() -> None:
    client = ReferenceJudgeClient()
    response = await client.judge(_request(candidate_output="the answer is 7", reference="42"))
    assert response.verdict == "fail"
    assert response.score == 0.0


@pytest.mark.asyncio
async def test_matching_is_whitespace_and_case_insensitive() -> None:
    client = ReferenceJudgeClient()
    response = await client.judge(
        _request(candidate_output="THE   ANSWER   is   FoRtY-Two", reference="forty-two")
    )
    assert response.verdict == "pass"


@pytest.mark.asyncio
async def test_abstains_when_sample_has_no_reference() -> None:
    client = ReferenceJudgeClient()
    response = await client.judge(_request(candidate_output="anything", reference=None))
    assert response.abstained is True
    assert response.score is None
    assert response.parse_ok is True  # abstention is not a parse failure


@pytest.mark.asyncio
async def test_empty_reference_never_matches_by_vacuous_substring() -> None:
    """An empty (but non-None) reference must never "match" every candidate
    via the vacuous-substring case (``"" in anything`` is always True in
    Python) -- that would make every sample with a blank reference pass."""
    client = ReferenceJudgeClient()
    response = await client.judge(_request(candidate_output="anything at all", reference=""))
    assert response.verdict == "fail"


@pytest.mark.asyncio
async def test_verdict_is_unaffected_by_position_bias_probe_metadata() -> None:
    """JudgeGrader sends a second request with metadata={"reversed": True}
    to probe for position bias. A containment check has no "option order" to
    be biased by, so both calls must agree -- proven directly here, and
    exercised through JudgeGrader's own probe in the integration test below.
    """
    client = ReferenceJudgeClient()
    primary = await client.judge(_request(candidate_output="the answer is 42", reference="42"))
    reversed_request = _request(candidate_output="the answer is 42", reference="42").model_copy(
        update={"metadata": {"reversed": True}}
    )
    reversed_response = await client.judge(reversed_request)
    assert primary.verdict == reversed_response.verdict


# --- integration with JudgeGrader: the critical safety property -------------


@pytest.mark.asyncio
async def test_uncalibrated_judge_grader_never_hard_gates_even_when_asked_to() -> None:
    """The one property that matters most for shipping this in a manifest's
    default grader table: even with ``gate=True`` requested, an uncalibrated
    ReferenceJudgeClient-backed JudgeGrader can never set hard_gate=True
    (design §9 / JudgeGrader's own calibration=None contract)."""
    grader = JudgeGrader(
        ReferenceJudgeClient(), calibration=None, gate=True, name="judge-reference@1"
    )
    result = await grader.grade(_sample(reference="42"), _execution(answer="the answer is 42"))
    assert result.hard_gate is False
    assert result.status == GradeStatus.PASS
    assert result.judge_calibration_ref is None


@pytest.mark.asyncio
async def test_uncalibrated_judge_grader_reports_fail_without_gating() -> None:
    grader = JudgeGrader(
        ReferenceJudgeClient(), calibration=None, gate=True, name="judge-reference@1"
    )
    result = await grader.grade(_sample(reference="42"), _execution(answer="nope"))
    assert result.status == GradeStatus.FAIL
    assert result.hard_gate is False
