"""Ratified judge calibration floor (Story 1.1, decision D-1 as amended
2026-07-04 by the code-review adjudication).

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 1, Story 1.1),
``coverage  and quality reports/test-design-architecture.md`` (R-003, D-1),
and the 2026-07-04 review decisions D1/D3 recorded in
``_bmad-output/implementation-artifacts/code-review-2026-07-04-p0-p1-branch.md``.

The amended demotion matrix this file pins:

    affirmatively BAD evidence  -> GradeStatus.UNAVAILABLE outright
        - expired ``expires_at`` (pre-existing semantics)
        - TNR < 0.95 or TPR < 0.85 with statistically meaningful counts
          (the ratified project floor; a lax caller ``threshold`` can
          never gate below it)
    ABSENT evidence             -> advisory only, can never gate
        - no ``calibrated_at`` (age unprovable)
        - ``calibrated_at`` older than 90 days
    boundary values             -> exactly-at-floor still gates
        - TNR == 0.95, TPR == 0.85, age == 90 days
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

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


class _FakeJudge:
    """A deterministic passing ``JudgeClient`` with a stable fingerprint.

    Returns the same verdict for the primary and the reversed position-bias
    probe, so a demotion can only come from the calibration, never from parse
    failure, abstention, fingerprint mismatch, or position bias.
    """

    fingerprint = "judge:model:prompt"

    def __init__(self, score: float = 0.9) -> None:
        self._score = score

    async def judge(self, request: JudgeRequest) -> JudgeResponse:
        return JudgeResponse(
            fingerprint=self.fingerprint,
            verdict="pass",
            score=self._score,
            parse_ok=True,
            abstained=False,
        )


def _artifact(**overrides: object) -> CalibrationArtifact:
    """Build a calibration that clears its OWN ``threshold`` and the >=30/>=30
    held-out sample floor, so only the *project* floor is under test. Callers
    override the confusion-matrix counts to push TPR/TNR above or below the
    ratified minimums.
    """
    defaults: dict[str, object] = {
        "calibration_id": "cal-1",
        "judge_fingerprint": "judge:model:prompt",
        "expires_at": datetime.now(UTC) + timedelta(days=30),
        "calibrated_at": datetime.now(UTC),
        "true_positive": 99,
        "true_negative": 99,
        "false_positive": 1,
        "false_negative": 1,
        "threshold": 0.7,  # deliberately lax: the artifact clears its own bar
    }
    defaults.update(overrides)
    return CalibrationArtifact.model_validate(defaults)


# --- bad evidence: sub-floor TNR/TPR demotes to UNAVAILABLE (D1) -------------


async def test_below_project_tnr_floor_demotes_to_unavailable() -> None:
    # TNR = TN/(TN+FP) = 90/100 = 0.90, below the 0.95 project floor, even
    # though it clears the artifact's own threshold=0.7. TPR is high (0.99).
    # A sub-floor calibration is affirmatively bad evidence: the result is
    # UNAVAILABLE outright, never an advisory PASS (ratified D-1 letter).
    calibration = _artifact(true_negative=90, false_positive=10, true_positive=99, false_negative=1)
    assert calibration.true_negative_rate == pytest.approx(0.90)
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.UNAVAILABLE
    assert result.hard_gate is False
    assert result.score is None
    reason = result.evidence.get("reason")
    assert isinstance(reason, str) and "project minimum 0.95" in reason


async def test_below_project_tpr_floor_demotes_to_unavailable() -> None:
    # TPR = TP/(TP+FN) = 80/100 = 0.80, below the 0.85 project floor; TNR high.
    calibration = _artifact(true_positive=80, false_negative=20, true_negative=99, false_positive=1)
    assert calibration.true_positive_rate == pytest.approx(0.80)
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.UNAVAILABLE
    assert result.hard_gate is False
    reason = result.evidence.get("reason")
    assert isinstance(reason, str) and "project minimum 0.85" in reason


async def test_sub_floor_rates_on_insufficient_samples_stay_advisory() -> None:
    # With fewer than 30 held-out samples per class the rates are noise, not
    # affirmative evidence -- the floor defers to the insufficient-sample
    # advisory demotion instead of declaring the artifact UNAVAILABLE.
    calibration = _artifact(true_positive=4, false_negative=1, true_negative=4, false_positive=1)
    assert calibration.floor_failure_reason() is None
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS  # advisory verdict from the score
    assert result.hard_gate is False
    reason = result.evidence.get("reason")
    assert isinstance(reason, str) and "below the required minimum" in reason


# --- absent evidence: undated/stale blocks gating only (D3) ------------------


async def test_calibration_older_than_max_age_cannot_gate_but_grades_advisorily() -> None:
    # Unexpired (expires_at in the future) but calibrated 120 days ago, beyond
    # the ratified 90-day max age. Absent/stale age evidence blocks gating but
    # does NOT destroy advisory grading (D-1 as amended 2026-07-04): the
    # pre-branch advisory capability of existing artifacts is preserved.
    now = datetime.now(UTC)
    calibration = _artifact(
        calibration_id="cal-old",
        calibrated_at=now - timedelta(days=120),
        expires_at=now + timedelta(days=30),
    )
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS  # advisory verdict from the score
    assert result.hard_gate is False
    assert result.judge_calibration_ref is None
    reason = result.evidence.get("reason")
    assert isinstance(reason, str) and "age exceeds the maximum of 90 days" in reason


async def test_undated_calibration_cannot_gate_but_grades_advisorily() -> None:
    # No calibrated_at (every pre-branch artifact): age is unprovable, so the
    # artifact can never gate -- but it still grades advisorily.
    calibration = _artifact(calibrated_at=None)
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS
    assert result.hard_gate is False
    reason = result.evidence.get("reason")
    assert isinstance(reason, str) and "no calibrated_at" in reason


def test_usability_seam_itself_reports_age_failures() -> None:
    # The documented "can this calibration gate" seam must report age failures
    # directly, so no future caller can bypass the age floor by consulting
    # usability_failure_reason without also calling age_failure_reason.
    now = datetime.now(UTC)
    undated = _artifact(calibrated_at=None)
    stale = _artifact(calibrated_at=now - timedelta(days=120), expires_at=now + timedelta(days=30))
    fresh = _artifact()
    assert undated.usability_failure_reason(now=now) is not None
    assert stale.usability_failure_reason(now=now) is not None
    assert fresh.usability_failure_reason(now=now) is None


# --- boundary values: exactly-at-floor still gates ---------------------------


async def test_tnr_exactly_at_floor_still_gates() -> None:
    # TNR = 95/100 == 0.95 exactly: the floor is inclusive (>= 0.95 gates).
    calibration = _artifact(true_negative=95, false_positive=5, true_positive=99, false_negative=1)
    assert calibration.true_negative_rate == pytest.approx(0.95)
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS
    assert result.hard_gate is True


async def test_tpr_exactly_at_floor_still_gates() -> None:
    # TPR = 85/100 == 0.85 exactly: inclusive floor.
    calibration = _artifact(true_positive=85, false_negative=15, true_negative=99, false_positive=1)
    assert calibration.true_positive_rate == pytest.approx(0.85)
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS
    assert result.hard_gate is True


def test_age_exactly_at_max_still_permits_gating() -> None:
    # "age <= 90 days" is inclusive: exactly 90 days old is not a failure.
    # Pinned at the seam level with an injected ``now`` -- the grader's own
    # wall clock cannot represent "exactly 90 days" deterministically.
    now = datetime.now(UTC)
    at_limit = _artifact(calibrated_at=now - timedelta(days=90), expires_at=now + timedelta(days=1))
    just_over = _artifact(
        calibrated_at=now - timedelta(days=90, seconds=1), expires_at=now + timedelta(days=1)
    )
    assert at_limit.age_failure_reason(now=now) is None
    assert just_over.age_failure_reason(now=now) is not None


# --- construction-time rejection of unusable timestamps ----------------------


def test_naive_calibrated_at_is_rejected_at_construction() -> None:
    # A naive timestamp would make the age arithmetic raise at grade time,
    # and a crash is not a demotion (D-1 is fail-closed, never fail-crashed).
    naive_timestamp = datetime.now()  # deliberately naive: the rejection under test
    with pytest.raises(ValidationError, match="timezone-aware"):
        _artifact(calibrated_at=naive_timestamp)


def test_calibrated_at_after_expiry_is_rejected_at_construction() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="must not be after expires_at"):
        _artifact(calibrated_at=now + timedelta(days=40), expires_at=now + timedelta(days=30))


# --- constants and no-over-restriction guards --------------------------------


def test_project_floor_constants_match_ratified_values() -> None:
    # Pins the ratified D-1 numbers as named constants (no magic numbers).
    from agentic_evalkit.graders import judge

    assert judge.PROJECT_MIN_TNR == 0.95
    assert judge.PROJECT_MIN_TPR == 0.85
    assert judge.PROJECT_MAX_CALIBRATION_AGE_DAYS == 90


async def test_calibration_clearing_the_floor_still_gates() -> None:
    # Guard against over-restriction: a calibration above BOTH floors
    # (TNR=0.97, TPR=0.90) and fresh must still be allowed to gate, so the new
    # floor does not silently disable otherwise-valid calibrations.
    calibration = _artifact(true_positive=90, false_negative=10, true_negative=97, false_positive=3)
    assert calibration.true_negative_rate == pytest.approx(0.97)
    assert calibration.true_positive_rate == pytest.approx(0.90)
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS
    assert result.hard_gate is True
