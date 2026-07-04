"""Tests for :mod:`agentic_evalkit.graders.judge` (plan Task 10 Steps 6-9).

The expiry test below is copied verbatim from the plan
(docs/plans/2026-07-02-agentic-evalkit-initial-release.md, Task 10 Step 6)
and must pass unmodified. The remaining tests cover Steps 7-8: minimum
held-out sample counts, TPR/TNR thresholds, fingerprint equality, position
bias / reversed-order checks, malformed structured output, bounded parse
retries, and explicit abstention -- each proven to NOT produce a
release-gating pass.
"""

from datetime import UTC, datetime, timedelta

import pytest

from agentic_evalkit.graders.judge import (
    CalibrationArtifact,
    JudgeGrader,
    JudgeRequest,
    JudgeResponse,
)
from agentic_evalkit.models import (
    EvalSample,
    ExecutionStatus,
    GradeStatus,
    NormalizedExecutionResult,
)


def _sample() -> EvalSample:
    return EvalSample(
        sample_id="s1",
        input={"question": "Is the sky blue?"},
        reference="yes",
        source_digest="sha256:row",
        adapter="identity@1",
    )


def _execution() -> NormalizedExecutionResult:
    now = datetime.now(UTC)
    return NormalizedExecutionResult(
        sample_id="s1",
        attempt=1,
        output={"answer": "yes, the sky is blue"},
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )


def _valid_calibration(**overrides: object) -> CalibrationArtifact:
    defaults: dict[str, object] = {
        "calibration_id": "cal-1",
        "judge_fingerprint": "judge:model:prompt",
        "expires_at": datetime.now(UTC) + timedelta(days=30),
        "calibrated_at": datetime.now(UTC),
        "true_positive": 95,
        "true_negative": 97,
        "false_positive": 3,
        "false_negative": 5,
        "threshold": 0.7,
    }
    defaults.update(overrides)
    return CalibrationArtifact.model_validate(defaults)


class _FakeJudge:
    """A deterministic ``JudgeClient`` test double returning a fixed verdict."""

    def __init__(
        self,
        score: float,
        *,
        fingerprint: str = "judge:model:prompt",
        verdict: str = "pass",
        abstain: bool = False,
        malformed_responses: int = 0,
    ) -> None:
        self._score = score
        self._fingerprint = fingerprint
        self._verdict = verdict
        self._abstain = abstain
        self._malformed_responses = malformed_responses
        self.calls: list[JudgeRequest] = []

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    async def judge(self, request: JudgeRequest) -> JudgeResponse:
        self.calls.append(request)
        if len(self.calls) <= self._malformed_responses:
            return JudgeResponse(
                fingerprint=self._fingerprint,
                verdict="",
                score=None,
                parse_ok=False,
                abstained=False,
            )
        if self._abstain:
            return JudgeResponse(
                fingerprint=self._fingerprint,
                verdict="",
                score=None,
                parse_ok=True,
                abstained=True,
            )
        return JudgeResponse(
            fingerprint=self._fingerprint,
            verdict=self._verdict,
            score=self._score,
            parse_ok=True,
            abstained=False,
        )


@pytest.mark.asyncio
async def test_expired_calibration_cannot_gate() -> None:
    calibration = CalibrationArtifact(
        calibration_id="cal-1",
        judge_fingerprint="judge:model:prompt",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
        true_positive=40,
        true_negative=40,
        false_positive=5,
        false_negative=5,
        threshold=0.7,
    )
    grader = JudgeGrader(_FakeJudge(0.9), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.UNAVAILABLE
    assert "expired" in result.evidence["reason"]


@pytest.mark.asyncio
async def test_insufficient_positive_sample_count_cannot_gate() -> None:
    """29 positives (TP+FN) is below the 30-minimum floor."""
    calibration = _valid_calibration(true_positive=20, false_negative=9)
    assert calibration.true_positive + calibration.false_negative == 29
    grader = JudgeGrader(_FakeJudge(0.9), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert not (result.status is GradeStatus.PASS and result.hard_gate is True)
    assert result.hard_gate is False
    assert "sample" in result.evidence["reason"] or "calibration" in result.evidence["reason"]


@pytest.mark.asyncio
async def test_insufficient_negative_sample_count_cannot_gate() -> None:
    """29 negatives (TN+FP) is below the 30-minimum floor.

    ``status`` may still faithfully report the judge's raw verdict (PASS);
    the load-bearing invariant is that a "gating pass" -- PASS *combined
    with* ``hard_gate=True`` -- can never occur here.
    """
    calibration = _valid_calibration(true_negative=20, false_positive=9)
    assert calibration.true_negative + calibration.false_positive == 29
    grader = JudgeGrader(_FakeJudge(0.9), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert not (result.status is GradeStatus.PASS and result.hard_gate is True)
    assert result.hard_gate is False


@pytest.mark.asyncio
async def test_tpr_below_threshold_cannot_gate() -> None:
    # TPR = TP / (TP+FN) = 20/40 = 0.5, below threshold 0.7.
    calibration = _valid_calibration(true_positive=20, false_negative=20, threshold=0.7)
    grader = JudgeGrader(_FakeJudge(0.9), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert not (result.status is GradeStatus.PASS and result.hard_gate is True)
    assert result.hard_gate is False


@pytest.mark.asyncio
async def test_tnr_below_threshold_cannot_gate() -> None:
    # TNR = TN / (TN+FP) = 20/40 = 0.5, below threshold 0.7.
    calibration = _valid_calibration(true_negative=20, false_positive=20, threshold=0.7)
    grader = JudgeGrader(_FakeJudge(0.9), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert not (result.status is GradeStatus.PASS and result.hard_gate is True)
    assert result.hard_gate is False


@pytest.mark.asyncio
async def test_fingerprint_mismatch_cannot_gate() -> None:
    """A judge whose live fingerprint differs from the calibration artifact's
    fingerprint is never trusted, no matter how good its verdict looks.
    """
    calibration = _valid_calibration(judge_fingerprint="judge:model-a:prompt-v1")
    mismatched_judge = _FakeJudge(0.95, fingerprint="judge:model-b:prompt-v2")
    grader = JudgeGrader(mismatched_judge, calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert not (result.status is GradeStatus.PASS and result.hard_gate is True)
    assert result.hard_gate is False
    assert "fingerprint" in result.evidence["reason"]


@pytest.mark.asyncio
async def test_well_calibrated_judge_can_pass_and_gate() -> None:
    """Sanity check: a judge that clears every bar (sufficient samples,
    TPR/TNR above threshold, matching fingerprint, unexpired, valid parse)
    can actually produce a gating PASS. Without this, the negative tests
    above would be vacuously true.
    """
    calibration = _valid_calibration()
    grader = JudgeGrader(_FakeJudge(0.9), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS
    assert result.hard_gate is True


@pytest.mark.asyncio
async def test_reversed_answer_order_position_bias_check() -> None:
    """A judge that reverses its verdict when option order is swapped fails
    the position-bias check and cannot gate, even with valid calibration.
    """

    class _PositionBiasedJudge:
        fingerprint = "judge:model:prompt"

        async def judge(self, request: JudgeRequest) -> JudgeResponse:
            # Deliberately flips the verdict based on ordering to simulate
            # position bias: the "reversed" pass sees a different answer.
            reversed_flag = bool(request.metadata.get("reversed"))
            return JudgeResponse(
                fingerprint=self.fingerprint,
                verdict="fail" if reversed_flag else "pass",
                score=0.1 if reversed_flag else 0.9,
                parse_ok=True,
                abstained=False,
            )

    calibration = _valid_calibration()
    grader = JudgeGrader(_PositionBiasedJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert not (result.status is GradeStatus.PASS and result.hard_gate is True)
    assert result.hard_gate is False
    assert "position" in result.evidence["reason"] or "bias" in result.evidence["reason"]


@pytest.mark.asyncio
async def test_malformed_structured_output_retries_then_reports_parse_error() -> None:
    """Parse failures retry at most twice (three attempts total); if every
    attempt is malformed, the grader reports an explicit parse error -- it
    never silently converts this into a task failure or a gating pass.
    """
    calibration = _valid_calibration()
    always_malformed = _FakeJudge(0.9, malformed_responses=99)
    grader = JudgeGrader(always_malformed, calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status in (GradeStatus.ERROR, GradeStatus.UNAVAILABLE)
    assert result.hard_gate is False
    assert len(always_malformed.calls) <= 3
    assert "parse" in result.evidence["reason"]


@pytest.mark.asyncio
async def test_retry_recovers_after_transient_malformed_response() -> None:
    """One malformed response followed by a valid one succeeds within the
    parse-retry budget (<= 2 retries, i.e. <= 3 total parse attempts). The
    grader also issues one further position-bias probe call once parsing
    succeeds, so total call count is parse attempts (2) + 1 probe = 3.
    """
    calibration = _valid_calibration()
    recovers_on_retry = _FakeJudge(0.9, malformed_responses=1)
    grader = JudgeGrader(recovers_on_retry, calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS
    assert result.evidence["parse_attempts"] == 2
    assert len(recovers_on_retry.calls) == 3


@pytest.mark.asyncio
async def test_explicit_abstention_cannot_gate() -> None:
    calibration = _valid_calibration()
    abstaining_judge = _FakeJudge(0.9, abstain=True)
    grader = JudgeGrader(abstaining_judge, calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.ABSTAIN
    assert result.hard_gate is False


@pytest.mark.asyncio
async def test_uncalibrated_judge_cannot_gate_even_with_gate_flag_false_by_default() -> None:
    """When ``gate=False``, the judge is explicitly advisory-only: even a
    perfectly calibrated judge with a passing verdict never sets
    ``hard_gate=True``.
    """
    calibration = _valid_calibration()
    grader = JudgeGrader(_FakeJudge(0.9), calibration=calibration, gate=False)
    result = await grader.grade(_sample(), _execution())
    assert result.hard_gate is False


@pytest.mark.asyncio
async def test_missing_calibration_cannot_gate() -> None:
    """No calibration artifact at all: the judge grades in advisory mode
    only, never claiming to gate a release, and never attaches a
    calibration reference it does not have.
    """
    grader = JudgeGrader(_FakeJudge(0.9), calibration=None, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.hard_gate is False
    assert result.judge_calibration_ref is None
