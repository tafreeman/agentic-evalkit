"""The project's official, ratified rule for when a judge's calibration is
good enough to trust (Story 1.1, decision D-1, as amended 2026-07-04 by the
code-review adjudication -- "adjudication" here just means the formal
decision that was written up after a code review raised questions about the
original rule).

Source documents recording where this rule came from:
``_bmad-output/planning-artifacts/epics.md`` (Epic 1, Story 1.1),
``coverage  and quality reports/test-design-architecture.md`` (requirement
R-003, decision D-1), and the 2026-07-04 review decisions D1/D3 recorded in
``_bmad-output/implementation-artifacts/code-review-2026-07-04-p0-p1-branch.md``.

This file pins down the amended rule for when a calibration gets demoted
(i.e. can no longer be used to gate/block a release), extended with
ADR-0020's "insufficient evidence" tier:

    affirmatively BAD evidence -> demoted all the way to
    GradeStatus.UNAVAILABLE (this is solid proof the judge isn't reliable)
        - the calibration's ``expires_at`` has passed (this rule already
          existed before this amendment)
        - the raw ("point") true-negative rate is below 0.95, or the raw
          true-positive rate is below 0.85, and there were enough held-out
          examples for that number to be meaningful (this is the ratified,
          project-wide floor; even if the caller configures a more lenient
          ``threshold``, it can never gate below this floor)
    ABSENT evidence -> advisory only, can never gate a release (but this
    isn't treated as proof the judge is bad)
        - there's no ``calibrated_at`` timestamp at all, so we can't prove
          how old the calibration is
        - ``calibrated_at`` is older than 90 days
    INSUFFICIENT evidence -> advisory only, can never gate a release
    (ADR-0020)
        - the raw true-negative/true-positive rate is at or above the
          floor, but its 95% "Wilson lower bound" (a conservative estimate
          of how low the true rate could plausibly be, given the sample
          size) is below the floor: there aren't yet enough held-out
          examples to actually prove the rate clears the floor
    boundary (edge-case) values
        - a raw rate sitting exactly at the floor is NOT treated as
          affirmatively bad (never UNAVAILABLE -- the floor is inclusive,
          meaning "at or above" counts as passing), but under ADR-0020 it
          still can't gate when the sample size is small, because the 95%
          Wilson lower bound for a rate sitting exactly at the floor always
          comes out somewhat below that same point estimate
        - a calibration that is exactly 90 days old is still within the
          allowed age and can still gate
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
    """Build a calibration that clears every bar by default: its own
    ``threshold``, the minimum of 30 held-out examples per class, AND the
    ADR-0020 "Wilson lower bound" check. That way, each test below only
    needs to override the one or two numbers it actually wants to test,
    while everything else stays comfortably passing.

    The defaults use 2000 held-out examples per class (TPR = 0.95 as a raw
    rate, with a Wilson lower bound of about 0.94, still above the 0.85
    floor; TNR = 0.97 as a raw rate, with a lower bound of about 0.96,
    above the 0.95 floor). This is the same large sample size used by
    ``test_judge.py``'s ``_valid_calibration``. We need a sample this big
    because, at a smaller sample of 100 examples, even a very high raw rate
    of 0.99 only produces a Wilson lower bound of about 0.945 -- still
    below the 0.95 TNR floor. At n=100, nothing could ever gate at all, no
    matter how good the raw numbers looked.
    """
    defaults: dict[str, object] = {
        "calibration_id": "cal-1",
        "judge_fingerprint": "judge:model:prompt",
        "expires_at": datetime.now(UTC) + timedelta(days=30),
        "calibrated_at": datetime.now(UTC),
        "true_positive": 1900,
        "true_negative": 1940,
        "false_positive": 60,
        "false_negative": 100,
        "threshold": 0.7,  # deliberately lax: the artifact clears its own bar
    }
    defaults.update(overrides)
    return CalibrationArtifact.model_validate(defaults)


# --- bad evidence: a TNR/TPR below the project floor demotes to
# UNAVAILABLE outright (decision D1) -------------


async def test_below_project_tnr_floor_demotes_to_unavailable() -> None:
    # TNR (true negative rate) = TN/(TN+FP) = 90/100 = 0.90, which is below
    # the project's 0.95 floor -- even though it clears this artifact's own,
    # more lenient threshold of 0.7. TPR is set high (0.99) so it isn't also
    # a factor here. A TNR below the project floor is solid, affirmative
    # proof the judge isn't accurate enough, so the result must be
    # UNAVAILABLE outright, never an advisory PASS (this is the literal rule
    # ratified by decision D-1).
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
    # TPR (true positive rate) = TP/(TP+FN) = 80/100 = 0.80, below the
    # project's 0.85 floor. TNR is set high so it isn't also a factor here.
    calibration = _artifact(true_positive=80, false_negative=20, true_negative=99, false_positive=1)
    assert calibration.true_positive_rate == pytest.approx(0.80)
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.UNAVAILABLE
    assert result.hard_gate is False
    reason = result.evidence.get("reason")
    assert isinstance(reason, str) and "project minimum 0.85" in reason


async def test_sub_floor_rates_on_insufficient_samples_stay_advisory() -> None:
    # With fewer than 30 held-out examples per class, a rate is just noise --
    # too small a sample to count as real evidence either way. So instead of
    # declaring the artifact UNAVAILABLE (which would mean "proven bad"),
    # this defers to the separate "not enough samples" advisory-only
    # demotion.
    calibration = _artifact(true_positive=4, false_negative=1, true_negative=4, false_positive=1)
    assert calibration.floor_failure_reason() is None
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS  # advisory verdict from the score
    assert result.hard_gate is False
    reason = result.evidence.get("reason")
    assert isinstance(reason, str) and "below the required minimum" in reason


# --- absent evidence: an undated or stale calibration blocks gating only,
# it doesn't mean the judge is bad (decision D3) ------------------


async def test_calibration_older_than_max_age_cannot_gate_but_grades_advisorily() -> None:
    # This calibration hasn't expired yet (expires_at is still in the
    # future), but it was measured 120 days ago -- past the ratified 90-day
    # maximum age. A stale age blocks gating, but it does NOT also disable
    # advisory grading (D-1, as amended 2026-07-04): a calibration that
    # already existed before this rule was introduced can still be used for
    # ordinary, advisory (non-gating) grades, just as it always could.
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
    # No calibrated_at timestamp at all (true of every calibration created
    # before this rule existed): there's no way to prove the calibration
    # isn't stale, so it can never gate a release -- but it can still
    # produce an ordinary, advisory grade.
    calibration = _artifact(calibrated_at=None)
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS
    assert result.hard_gate is False
    reason = result.evidence.get("reason")
    assert isinstance(reason, str) and "no calibrated_at" in reason


def test_usability_seam_itself_reports_age_failures() -> None:
    # `usability_failure_reason` is the one method callers are meant to use
    # to ask "can this calibration gate a release?" -- so it must report age
    # problems directly, on its own, rather than relying on every caller to
    # remember to *also* separately call `age_failure_reason`. This test
    # makes sure nobody can accidentally bypass the age check by only
    # calling `usability_failure_reason`.
    now = datetime.now(UTC)
    undated = _artifact(calibrated_at=None)
    stale = _artifact(calibrated_at=now - timedelta(days=120), expires_at=now + timedelta(days=30))
    # We deliberately pass calibrated_at=now here, instead of letting
    # `_artifact()` fall back to its own default of datetime.now(UTC)
    # (which would read the clock a few microseconds after the `now`
    # captured above). Otherwise, the "calibrated in the future" check
    # could trip on that tiny gap between the two clock reads on a fast
    # machine -- a flaky failure that has nothing to do with what this test
    # is actually checking.
    fresh = _artifact(calibrated_at=now)
    assert undated.usability_failure_reason(now=now) is not None
    assert stale.usability_failure_reason(now=now) is not None
    assert fresh.usability_failure_reason(now=now) is None


# --- boundary (edge-case) values: a rate sitting exactly at the floor
# counts as passing, but still can't gate when the sample size is small -


async def test_tnr_point_exactly_at_floor_blocks_gating_as_insufficient_evidence() -> None:
    # TNR = 95/100 = 0.95 exactly, landing right on the project floor. The
    # raw ("point") floor check is inclusive -- it only fails below 0.95,
    # not at or above it -- so this is never treated as affirmatively-bad,
    # UNAVAILABLE-worthy evidence. But under ADR-0020, the 95% Wilson lower
    # bound for a rate this size (about 0.888, out of 100 examples) sits
    # below that same 0.95 floor -- meaning there isn't yet enough evidence
    # to be confident the true rate clears it. So gating is blocked, but
    # advisory grading keeps working. The "good answer" class here (99
    # correct out of 100) is set up so its own Wilson lower bound (about
    # 0.945) comfortably clears its 0.85 floor, so this test isolates the
    # TNR condition specifically.
    calibration = _artifact(true_negative=95, false_positive=5, true_positive=99, false_negative=1)
    assert calibration.true_negative_rate == pytest.approx(0.95)
    assert calibration.floor_failure_reason() is None  # point floor: inclusive
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS  # advisory verdict, not UNAVAILABLE
    assert result.hard_gate is False
    reason = result.evidence.get("reason")
    assert isinstance(reason, str) and "TNR 95% Wilson lower bound" in reason
    assert "insufficient held-out evidence to gate" in reason


async def test_tpr_point_exactly_at_floor_blocks_gating_as_insufficient_evidence() -> None:
    # TPR = 85/100 = 0.85 exactly, right on the project floor -- the same
    # kind of edge case as the TNR test above: the raw point floor is
    # inclusive (not UNAVAILABLE), but the Wilson lower bound for a rate
    # this size (about 0.767, out of 100 examples) is below 0.85, so gating
    # is blocked as insufficient evidence. For the "bad answer" class in
    # this test, 100 examples isn't a big enough sample to clear both the
    # point floor and the Wilson floor at the same time (a 0.99 point rate
    # only produces a Wilson lower bound of about 0.945, still under the
    # 0.95 floor) -- so this test uses the larger, sufficient-evidence
    # sample size from the shared default instead, to isolate the TPR
    # condition specifically.
    calibration = _artifact(true_positive=85, false_negative=15)
    assert calibration.true_positive_rate == pytest.approx(0.85)
    assert calibration.floor_failure_reason() is None  # point floor: inclusive
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS  # advisory verdict, not UNAVAILABLE
    assert result.hard_gate is False
    reason = result.evidence.get("reason")
    assert isinstance(reason, str) and "TPR 95% Wilson lower bound" in reason
    assert "insufficient held-out evidence to gate" in reason


def test_age_exactly_at_max_still_permits_gating() -> None:
    # "age <= 90 days" is inclusive: a calibration that's exactly 90 days
    # old still passes the age check, rather than being treated as too old.
    # This test calls `age_failure_reason` directly with a fixed, injected
    # `now` value instead of relying on the real system clock -- the real
    # clock keeps advancing while the test runs, so there's no reliable way
    # to make it land on exactly 90 days using real time.
    now = datetime.now(UTC)
    at_limit = _artifact(calibrated_at=now - timedelta(days=90), expires_at=now + timedelta(days=1))
    just_over = _artifact(
        calibrated_at=now - timedelta(days=90, seconds=1), expires_at=now + timedelta(days=1)
    )
    assert at_limit.age_failure_reason(now=now) is None
    assert just_over.age_failure_reason(now=now) is not None


# --- timestamps that would cause a crash later are rejected immediately,
# at construction time ----------------------


def test_naive_calibrated_at_is_rejected_at_construction() -> None:
    # A "naive" timestamp (one with no timezone attached) would make the
    # date math this class does later raise an exception at grading time --
    # and crashing is not an acceptable way to demote a calibration. D-1's
    # whole point is to fail safely, falling back to a safe result, never to
    # fail by crashing.
    naive_timestamp = datetime.now()  # deliberately naive: this is exactly what should get rejected
    with pytest.raises(ValidationError, match="timezone-aware"):
        _artifact(calibrated_at=naive_timestamp)


def test_naive_expires_at_is_rejected_at_construction() -> None:
    # is_expired() compares expires_at against datetime.now(UTC), which
    # always has a timezone attached. In Python, comparing a timezone-aware
    # datetime to a naive one raises a TypeError -- so a naive expires_at
    # would crash at grading time instead of cleanly demoting the result.
    naive_timestamp = datetime.now()  # deliberately naive: this is exactly what should get rejected
    with pytest.raises(ValidationError, match="timezone-aware"):
        _artifact(expires_at=naive_timestamp)


def test_calibrated_at_after_expiry_is_rejected_at_construction() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="must not be after expires_at"):
        _artifact(calibrated_at=now + timedelta(days=40), expires_at=now + timedelta(days=30))


async def test_future_dated_calibrated_at_cannot_gate() -> None:
    # calibrated_at is set in the future here, but still before expires_at
    # (so the "calibrated_at must not be after expires_at" check made at
    # construction time doesn't catch it). A future-dated calibration must
    # not be mistaken for a perfectly fresh, trustworthy one: if we
    # subtracted calibrated_at from the current time without checking for
    # this, the result would come out negative, which would never trip the
    # "older than 90 days" check -- silently treating an impossible
    # timestamp as if it were safely within the 90-day limit.
    now = datetime.now(UTC)
    calibration = _artifact(
        calibrated_at=now + timedelta(days=10), expires_at=now + timedelta(days=30)
    )
    assert calibration.age_failure_reason(now=now) is not None
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS  # advisory verdict from the score
    assert result.hard_gate is False


# --- named constants, and a check that the rule isn't too strict --------


def test_project_floor_constants_match_ratified_values() -> None:
    # Locks in the official D-1 numbers as named constants, rather than
    # scattering the raw numbers 0.95/0.85/90 throughout the codebase where
    # they'd be easy to accidentally change (a "magic number").
    from agentic_evalkit.graders import judge

    assert judge.PROJECT_MIN_TNR == 0.95
    assert judge.PROJECT_MIN_TPR == 0.85
    assert judge.PROJECT_MAX_CALIBRATION_AGE_DAYS == 90


async def test_calibration_clearing_the_floor_still_gates() -> None:
    # This checks that the rule isn't accidentally too strict: a fresh
    # calibration that clears BOTH floors, backed by enough held-out
    # examples to prove it (2000 per class: TNR = 0.97 as a raw rate with a
    # Wilson lower bound of about 0.96; TPR = 0.90 as a raw rate with a
    # lower bound of about 0.886), must still actually be allowed to gate a
    # release. Without a positive test like this, a bug in the ADR-0020
    # evidence check could silently make it impossible for ANY calibration
    # to ever gate.
    calibration = _artifact(true_positive=1800, false_negative=200)
    assert calibration.true_negative_rate == pytest.approx(0.97)
    assert calibration.true_positive_rate == pytest.approx(0.90)
    grader = JudgeGrader(_FakeJudge(), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS
    assert result.hard_gate is True
