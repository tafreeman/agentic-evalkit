"""The measurement half of the calibration gate (ADR-0024).

``tests/unit/graders/test_judge_calibration_floor.py`` pins down what a
calibration must clear before a judge may gate a release. This module pins
down the other half: that :func:`~agentic_evalkit.graders.measure.measure_calibration`
actually produces such a calibration, from a judge and a set of answers a
human has already labeled.

Every test here uses a stub ``JudgeClient`` -- the protocol is small enough
that one is a few lines -- so the whole path is exercised with no provider,
no network, and no API key, and stays in the default ``-m "not live"`` suite.
The stubs are the point, not a shortcut: a judge that abstains on demand or
raises on the third sample is exactly what a real provider does occasionally
and what the counting rules exist to handle, and a real model could not be
made to do it reliably.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from agentic_evalkit.graders.calibration import (
    PROJECT_MAX_CALIBRATION_AGE_DAYS,
    AuthorityLevel,
    judge_authority,
)
from agentic_evalkit.graders.judge import (
    JudgeGrader,
    JudgeRequest,
    JudgeResponse,
    JudgeResponseStatus,
)
from agentic_evalkit.graders.measure import DEFAULT_PASS_SCORE_THRESHOLD, measure_calibration
from agentic_evalkit.models import (
    CalibrationLabel,
    EvalSample,
    ExecutionStatus,
    GradeStatus,
    LabeledJudgeSample,
    NormalizedExecutionResult,
)

_FINGERPRINT = "sha256:stub-judge-v1"
_NOW = datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)

#: Enough negatives that a perfect true-negative rate clears its 95% Wilson
#: lower bound against the 0.95 project floor. At a point estimate of 1.0 the
#: bound is n/(n + z^2) with z = 1.96, so it takes 73 samples to reach 0.95 --
#: the 30-sample class minimum is nowhere near sufficient on its own, which is
#: exactly ADR-0020's point. 80 leaves room above the boundary.
_GATING_NEGATIVES = 80
#: The TPR floor of 0.85 needs only 22 by the same formula, so the 30-sample
#: class minimum binds here instead; 40 clears both.
_GATING_POSITIVES = 40


class _StubJudge:
    """A ``JudgeClient`` whose answer to each sample is scripted by ``sample_id``.

    Defaults to scoring 1.0 (a "good" verdict) for any sample it has no
    scripted response for, so a test only has to describe the samples it
    cares about.
    """

    def __init__(
        self,
        responses: dict[str, JudgeResponse] | None = None,
        *,
        fingerprint: str = _FINGERPRINT,
        raises_on: frozenset[str] = frozenset(),
    ) -> None:
        self.fingerprint = fingerprint
        self._responses = responses or {}
        self._raises_on = raises_on
        self.calls: list[JudgeRequest] = []

    async def judge(self, request: JudgeRequest) -> JudgeResponse:
        self.calls.append(request)
        if request.sample_id in self._raises_on:
            raise RuntimeError(f"provider exploded on {request.sample_id}")
        scripted = self._responses.get(request.sample_id)
        if scripted is not None:
            return scripted
        return _response(1.0)


def _response(
    score: float | None,
    *,
    status: JudgeResponseStatus = JudgeResponseStatus.OK,
    parse_ok: bool = True,
    abstained: bool = False,
    fingerprint: str = _FINGERPRINT,
) -> JudgeResponse:
    return JudgeResponse(
        fingerprint=fingerprint,
        verdict="pass" if score is not None and score >= 0.5 else "fail",
        score=score,
        parse_ok=parse_ok,
        abstained=abstained,
        status=status,
    )


def _labeled(sample_id: str, label: CalibrationLabel) -> LabeledJudgeSample:
    return LabeledJudgeSample(
        sample_id=sample_id,
        prompt="what is the answer?",
        candidate_output="the answer is 42",
        label=label,
        reference="42",
    )


def _labeled_set(*, positives: int, negatives: int) -> tuple[LabeledJudgeSample, ...]:
    return tuple(
        [_labeled(f"pos-{i}", CalibrationLabel.GOOD) for i in range(positives)]
        + [_labeled(f"neg-{i}", CalibrationLabel.BAD) for i in range(negatives)]
    )


def _perfect_judge() -> _StubJudge:
    """A judge that is right about every sample in a ``_labeled_set``."""
    return _StubJudge(
        {
            f"neg-{i}": _response(0.0)
            for i in range(max(_GATING_NEGATIVES, _MISCALIBRATED_NEGATIVES))
        }
    )


#: Sample counts for a judge that is measurably bad rather than merely
#: unmeasured: enough negatives to clear the class minimum, so a sub-floor
#: TNR counts as proof rather than noise.
_MISCALIBRATED_NEGATIVES = 100


# --- what the artifact records ---------------------------------------------


async def test_calibrated_at_is_always_set() -> None:
    """The whole reason this function exists rather than a hand-written artifact.

    ``calibrated_at`` is optional on the model, and an artifact without it can
    never gate. A measurement that just happened knows exactly when it
    happened, so it always says so.
    """
    artifact = await measure_calibration(
        _StubJudge(), _labeled_set(positives=1, negatives=0), calibration_id="cal-1"
    )

    assert artifact.calibrated_at is not None
    assert artifact.calibrated_at.tzinfo is not None
    assert artifact.age_failure_reason() is None


async def test_expires_at_is_derived_from_the_project_maximum_age() -> None:
    artifact = await measure_calibration(
        _StubJudge(), _labeled_set(positives=1, negatives=0), calibration_id="cal-1", now=_NOW
    )

    assert artifact.calibrated_at == _NOW
    assert artifact.expires_at == _NOW + timedelta(days=PROJECT_MAX_CALIBRATION_AGE_DAYS)


async def test_judge_fingerprint_is_taken_from_the_judge_not_the_caller() -> None:
    """No parameter exists for it, so an artifact cannot describe another judge."""
    judge = _StubJudge(fingerprint="sha256:some-other-judge")

    artifact = await measure_calibration(
        judge, _labeled_set(positives=1, negatives=0), calibration_id="cal-1"
    )

    assert artifact.judge_fingerprint == judge.fingerprint


async def test_total_labeled_records_the_whole_input_including_non_verdicts() -> None:
    samples = _labeled_set(positives=2, negatives=2)
    judge = _StubJudge({"pos-1": _response(None, abstained=True)}, raises_on=frozenset({"neg-1"}))

    artifact = await measure_calibration(judge, samples, calibration_id="cal-1")

    assert artifact.total_labeled == len(samples)
    assert artifact.abstained_count == 1
    assert artifact.error_count == 1
    # The two counted classes plus the two non-verdicts account for everything.
    assert artifact.positive_count + artifact.negative_count == len(samples) - 2


async def test_the_judge_is_called_once_per_sample() -> None:
    judge = _StubJudge()
    samples = _labeled_set(positives=3, negatives=2)

    await measure_calibration(judge, samples, calibration_id="cal-1")

    assert [call.sample_id for call in judge.calls] == [s.sample_id for s in samples]


async def test_the_judge_receives_the_prompt_output_and_reference() -> None:
    judge = _StubJudge()

    await measure_calibration(
        judge, (_labeled("pos-0", CalibrationLabel.GOOD),), calibration_id="cal-1"
    )

    request = judge.calls[0]
    assert request.prompt == "what is the answer?"
    assert request.candidate_output == "the answer is 42"
    assert request.reference == "42"


# --- the confusion matrix ---------------------------------------------------


async def test_each_verdict_lands_in_the_right_confusion_matrix_cell() -> None:
    samples = (
        _labeled("tp", CalibrationLabel.GOOD),
        _labeled("fn", CalibrationLabel.GOOD),
        _labeled("tn", CalibrationLabel.BAD),
        _labeled("fp", CalibrationLabel.BAD),
    )
    judge = _StubJudge(
        {
            "tp": _response(1.0),  # said good about a good answer
            "fn": _response(0.0),  # said bad about a good answer
            "tn": _response(0.0),  # said bad about a bad answer
            "fp": _response(1.0),  # said good about a bad answer
        }
    )

    artifact = await measure_calibration(judge, samples, calibration_id="cal-1")

    assert (artifact.true_positive, artifact.false_negative) == (1, 1)
    assert (artifact.true_negative, artifact.false_positive) == (1, 1)


async def test_the_score_threshold_matches_the_graders_pass_bar() -> None:
    """A calibration must measure the decision ``JudgeGrader`` will actually make.

    If the two defaults ever diverged, an artifact would faithfully describe
    a pass/fail rule nobody applies, which is worse than no artifact at all.
    """
    assert JudgeGrader(_StubJudge(), calibration=None, gate=False)._pass_score_threshold == (
        DEFAULT_PASS_SCORE_THRESHOLD
    )


async def test_a_score_exactly_at_the_threshold_counts_as_good() -> None:
    judge = _StubJudge({"pos-0": _response(DEFAULT_PASS_SCORE_THRESHOLD)})

    artifact = await measure_calibration(
        judge, (_labeled("pos-0", CalibrationLabel.GOOD),), calibration_id="cal-1"
    )

    assert artifact.true_positive == 1


async def test_a_custom_score_threshold_is_honored() -> None:
    judge = _StubJudge({"pos-0": _response(0.6)})

    artifact = await measure_calibration(
        judge,
        (_labeled("pos-0", CalibrationLabel.GOOD),),
        calibration_id="cal-1",
        pass_score_threshold=0.9,
    )

    assert (artifact.true_positive, artifact.false_negative) == (0, 1)


# --- non-verdicts stay out of the classes -----------------------------------


@pytest.mark.parametrize(
    ("response", "expected_counter"),
    [
        pytest.param(_response(None, abstained=True), "abstained", id="explicit-abstention"),
        pytest.param(
            _response(None, status=JudgeResponseStatus.REFUSED), "abstained", id="refused"
        ),
        pytest.param(_response(None, status=JudgeResponseStatus.TIMEOUT), "errored", id="timeout"),
        pytest.param(
            _response(None, status=JudgeResponseStatus.RATE_LIMITED), "errored", id="rate-limited"
        ),
        pytest.param(
            _response(None, status=JudgeResponseStatus.ERROR), "errored", id="judge-error"
        ),
        pytest.param(_response(0.9, parse_ok=False), "errored", id="unparseable"),
        pytest.param(_response(None), "errored", id="verdict-with-no-score"),
        pytest.param(
            _response(1.0, fingerprint="sha256:a-different-judge"),
            "errored",
            id="answered-by-another-judge",
        ),
    ],
)
async def test_a_non_verdict_is_counted_on_its_own_and_never_as_a_class(
    response: JudgeResponse, expected_counter: str
) -> None:
    """A judge must not be able to improve its measured accuracy by not answering.

    Every case here produced no usable verdict. Folding any of them into a
    class would either credit the judge for a question it dodged or blame it
    for one it never saw.
    """
    samples = (_labeled("s0", CalibrationLabel.GOOD),)

    artifact = await measure_calibration(
        _StubJudge({"s0": response}), samples, calibration_id="cal-1"
    )

    assert (artifact.abstained_count, artifact.error_count) == (
        (1, 0) if expected_counter == "abstained" else (0, 1)
    )
    assert artifact.positive_count == 0
    assert artifact.negative_count == 0


async def test_one_erroring_sample_does_not_abort_the_measurement() -> None:
    """Per-sample isolation, the same rule the runner and JudgeGrader follow."""
    samples = _labeled_set(positives=3, negatives=0)
    judge = _StubJudge(raises_on=frozenset({"pos-1"}))

    artifact = await measure_calibration(judge, samples, calibration_id="cal-1")

    assert artifact.error_count == 1
    assert artifact.true_positive == 2
    assert len(judge.calls) == 3


async def test_cancellation_is_not_swallowed() -> None:
    """``CancelledError`` is not an ``Exception``, so cancelling really cancels."""

    class _CancellingJudge:
        fingerprint = _FINGERPRINT

        async def judge(self, request: JudgeRequest) -> JudgeResponse:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await measure_calibration(
            _CancellingJudge(),
            (_labeled("s0", CalibrationLabel.GOOD),),
            calibration_id="cal-1",
        )


async def test_an_empty_labeled_set_produces_an_artifact_rather_than_an_error() -> None:
    """Emit, do not refuse (ADR-0024) -- the artifact then reports its own emptiness."""
    artifact = await measure_calibration(_StubJudge(), (), calibration_id="cal-1")

    assert artifact.total_labeled == 0
    assert judge_authority(artifact).level is AuthorityLevel.ADVISORY


# --- what the resulting artifact entitles the judge to ----------------------


async def test_a_thin_labeled_set_is_advisory_and_names_the_shortfall() -> None:
    artifact = await measure_calibration(
        _perfect_judge(), _labeled_set(positives=12, negatives=12), calibration_id="cal-thin"
    )

    authority = judge_authority(artifact, judge_fingerprint=_FINGERPRINT)

    assert authority.level is AuthorityLevel.ADVISORY
    assert authority.reason is not None
    assert "12 held-out positive samples" in authority.reason
    assert "minimum of 30" in authority.reason


async def test_a_measurably_bad_judge_is_unavailable_not_advisory() -> None:
    """Proof of a bad judge and absence of proof are different facts (D-1).

    This judge answered plenty of questions -- well past the class minimum,
    so the numbers are not noise -- and got a lot of them wrong. That is
    evidence, and it says the judge is unreliable.
    """
    samples = _labeled_set(positives=40, negatives=_MISCALIBRATED_NEGATIVES)
    # Right about every good answer, but calls a third of the bad ones good:
    # a TNR near 0.67, far below the 0.95 floor.
    judge = _StubJudge(
        {f"neg-{i}": _response(0.0 if i % 3 else 1.0) for i in range(_MISCALIBRATED_NEGATIVES)}
    )

    artifact = await measure_calibration(judge, samples, calibration_id="cal-bad")
    authority = judge_authority(artifact, judge_fingerprint=_FINGERPRINT)

    assert authority.level is AuthorityLevel.UNAVAILABLE
    assert authority.reason is not None
    assert "below the project minimum" in authority.reason


async def test_a_fully_measured_judge_earns_gating_authority() -> None:
    artifact = await measure_calibration(
        _perfect_judge(),
        _labeled_set(positives=_GATING_POSITIVES, negatives=_GATING_NEGATIVES),
        calibration_id="cal-good",
    )

    authority = judge_authority(artifact, judge_fingerprint=_FINGERPRINT)

    assert authority.level is AuthorityLevel.GATING
    assert authority.reason is None
    assert artifact.usability_failure_reason() is None


async def test_a_measured_artifact_lets_a_judge_actually_hold_hard_gate() -> None:
    """The end of the whole chain, and the thing no input could produce before.

    Everything above tests the measurement; this tests that what it measured
    is accepted by the gate it was measured for -- the artifact goes straight
    into ``JudgeGrader`` and a passing sample comes back able to block a
    release.
    """
    artifact = await measure_calibration(
        _perfect_judge(),
        _labeled_set(positives=_GATING_POSITIVES, negatives=_GATING_NEGATIVES),
        calibration_id="cal-good",
    )
    grader = JudgeGrader(_StubJudge(), calibration=artifact, gate=True)
    sample = EvalSample(
        sample_id="s1",
        input={"question": "what is the answer?"},
        reference="42",
        source_digest="sha256:abc",
        adapter="manual@1",
    )
    execution = NormalizedExecutionResult(
        sample_id="s1",
        attempt=1,
        output={"answer": "the answer is 42"},
        status=ExecutionStatus.COMPLETED,
        started_at=_NOW,
        finished_at=_NOW,
    )

    result = await grader.grade(sample, execution)

    assert result.status is GradeStatus.PASS
    assert result.hard_gate is True
    assert result.judge_calibration_ref == "cal-good"


async def test_a_measured_artifact_cannot_gate_a_different_judge() -> None:
    """The fingerprint is recorded from the judge, so swapping judges is caught."""
    artifact = await measure_calibration(
        _perfect_judge(),
        _labeled_set(positives=_GATING_POSITIVES, negatives=_GATING_NEGATIVES),
        calibration_id="cal-good",
    )

    authority = judge_authority(artifact, judge_fingerprint="sha256:a-different-judge")

    assert authority.level is AuthorityLevel.ADVISORY
    assert authority.reason is not None
    assert "does not match live judge fingerprint" in authority.reason
