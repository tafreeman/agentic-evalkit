"""ATDD red-phase scaffolds for Story 1.1 -- enforce the ratified judge
calibration floor (decision D-1, 2026-07-04).

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 1, Story 1.1) and
``coverage  and quality reports/test-design-architecture.md`` (R-003, D-1).

The ratified project floor a ``CalibrationArtifact`` must clear BEFORE it may
hard-gate, IN ADDITION to the existing 5-condition gate in
``agentic_evalkit.graders.judge``:

    TNR >= 0.95   (clamps the false-"pass" rate <= 5% -- the overclaim R-003
                   exists to prevent)
    TPR >= 0.85
    calibration age <= 90 days

Today ``CalibrationArtifact.usability_failure_reason`` only checks TPR/TNR
against the artifact's OWN caller-supplied ``threshold`` and ``expires_at`` --
there is no project floor and no calibration-age field. These scaffolds
therefore assert behavior that does not exist yet and are marked
``@pytest.mark.skip`` (TDD red phase); a developer activates each by removing
the skip once the floor is implemented.

Implementation notes for the dev (from the ratified decision):
  * Add named project-floor constants (no magic numbers), e.g.
    ``PROJECT_MIN_TNR = 0.95``, ``PROJECT_MIN_TPR = 0.85``,
    ``PROJECT_MAX_CALIBRATION_AGE_DAYS = 90`` in ``graders/judge.py``.
  * Add an additive ``calibrated_at: datetime`` field to
    ``CalibrationArtifact`` (schema_version stays "1"; additive per ADR-0002)
    and demote to ``UNAVAILABLE`` when ``now - calibrated_at`` exceeds the max
    age, even while ``expires_at`` is still in the future.
  * Enforce the TNR/TPR floor so a lax caller ``threshold`` cannot gate.
"""

from __future__ import annotations

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


async def test_below_project_tnr_floor_cannot_gate() -> None:
    # TNR = TN/(TN+FP) = 90/100 = 0.90, below the 0.95 project floor, even
    # though it clears the artifact's own threshold=0.7. TPR is high (0.99).
    calibration = _artifact(
        true_negative=90, false_positive=10, true_positive=99, false_negative=1
    )
    assert calibration.true_negative_rate == pytest.approx(0.90)
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.hard_gate is False
    assert not (result.status is GradeStatus.PASS and result.hard_gate is True)


async def test_below_project_tpr_floor_cannot_gate() -> None:
    # TPR = TP/(TP+FN) = 80/100 = 0.80, below the 0.85 project floor; TNR high.
    calibration = _artifact(
        true_positive=80, false_negative=20, true_negative=99, false_positive=1
    )
    assert calibration.true_positive_rate == pytest.approx(0.80)
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.hard_gate is False


async def test_calibration_older_than_max_age_cannot_gate() -> None:
    # Unexpired (expires_at in the future) but calibrated 120 days ago, beyond
    # the ratified 90-day max age. Requires a new additive ``calibrated_at``
    # field on CalibrationArtifact.
    now = datetime.now(UTC)
    calibration = CalibrationArtifact.model_validate(
        {
            "calibration_id": "cal-old",
            "judge_fingerprint": "judge:model:prompt",
            "calibrated_at": now - timedelta(days=120),
            "expires_at": now + timedelta(days=30),
            "true_positive": 99,
            "true_negative": 99,
            "false_positive": 1,
            "false_negative": 1,
            "threshold": 0.7,
        }
    )
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.UNAVAILABLE
    assert result.hard_gate is False


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
    calibration = _artifact(
        true_positive=90, false_negative=10, true_negative=97, false_positive=3
    )
    assert calibration.true_negative_rate == pytest.approx(0.97)
    assert calibration.true_positive_rate == pytest.approx(0.90)
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS
    assert result.hard_gate is True
