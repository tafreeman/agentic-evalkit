"""Tests for :mod:`agentic_evalkit.graders.judge` (plan Task 10 Steps 6-9).

The expiry test below is copied verbatim from the plan
(docs/plans/2026-07-02-agentic-evalkit-initial-release.md, Task 10 Step 6)
and must pass unmodified. The remaining tests cover Steps 7-8: the minimum
number of held-out examples a calibration needs, the TPR/TNR accuracy
thresholds, checking that the judge's fingerprint (a hash of its model +
prompt) matches the calibration's, the position-bias check (does the judge
change its answer when the two options being compared are swapped?),
malformed structured output, bounded parse retries, and explicit
abstention -- every one of these is proven to NOT let a result gate (block)
a release.
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import JsonValue, ValidationError

from agentic_evalkit.graders.judge import (
    _DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS,
    CalibrationArtifact,
    JudgeGrader,
    JudgeRequest,
    JudgeResponse,
    JudgeResponseStatus,
    _stringify_output,
)
from agentic_evalkit.models import (
    EvalSample,
    ExecutionStatus,
    GradeStatus,
    NormalizedExecutionResult,
)
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY

#: A secret-shaped substring matching ``DEFAULT_REDACTION_POLICY``'s
#: ``hf_[A-Za-z0-9]{16,}`` pattern (20 chars after the prefix).
_PLANTED_SECRET = "hf_" + "a1B2c3D4e5F6g7H8i9J0"


def _sample() -> EvalSample:
    return EvalSample(
        sample_id="s1",
        input={"question": "Is the sky blue?"},
        reference="yes",
        source_digest="sha256:row",
        adapter="identity@1",
    )


def _execution() -> NormalizedExecutionResult:
    return _execution_with_output({"answer": "yes, the sky is blue"})


def _execution_with_output(output: dict[str, JsonValue]) -> NormalizedExecutionResult:
    now = datetime.now(UTC)
    return NormalizedExecutionResult(
        sample_id="s1",
        attempt=1,
        output=output,
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )


def _valid_calibration(**overrides: object) -> CalibrationArtifact:
    # These numbers give TPR = 1900/2000 = 0.95 and TNR = 1940/2000 = 0.97 --
    # the same raw accuracy rates the original n=100 fixture used, just
    # measured on a much bigger sample (2000 held-out examples per class
    # instead of 100). The bigger sample matters because of the "Wilson
    # lower bound" check (explained in calibration.py): a rate based on only
    # 100 examples isn't trustworthy enough to prove it clears the project's
    # floors (TPR >= 0.85, TNR >= 0.95), even though the raw rate looks
    # fine. A rate based on 2000 examples is trustworthy enough -- its
    # Wilson lower bounds come out to about 0.940 for TPR and 0.962 for TNR,
    # both above the floors. In short: this fixture has to be big enough to
    # actually prove the judge is good, not just claim a good-looking number
    # from too small a sample.
    defaults: dict[str, object] = {
        "calibration_id": "cal-1",
        "judge_fingerprint": "judge:model:prompt",
        "expires_at": datetime.now(UTC) + timedelta(days=30),
        "calibrated_at": datetime.now(UTC),
        "true_positive": 1900,
        "true_negative": 1940,
        "false_positive": 60,
        "false_negative": 100,
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
        status: JudgeResponseStatus = JudgeResponseStatus.OK,
        rationale: str | None = None,
    ) -> None:
        self._score = score
        self._fingerprint = fingerprint
        self._verdict = verdict
        self._abstain = abstain
        self._malformed_responses = malformed_responses
        self._status = status
        self._rationale = rationale
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
            status=self._status,
            rationale=self._rationale,
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
    """There are only 29 real "good-answer" examples in this calibration
    (true_positive + false_negative), one short of the required minimum of
    30 per class."""
    calibration = _valid_calibration(true_positive=20, false_negative=9)
    assert calibration.true_positive + calibration.false_negative == 29
    grader = JudgeGrader(_FakeJudge(0.9), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert not (result.status is GradeStatus.PASS and result.hard_gate is True)
    assert result.hard_gate is False
    assert "sample" in result.evidence["reason"] or "calibration" in result.evidence["reason"]


@pytest.mark.asyncio
async def test_insufficient_negative_sample_count_cannot_gate() -> None:
    """There are only 29 real "bad-answer" examples in this calibration
    (true_negative + false_positive), one short of the required minimum of
    30 per class.

    ``status`` can still honestly come back as PASS -- we're not saying the
    judge's raw verdict was wrong. What actually matters, and what this test
    is really checking, is that PASS combined with ``hard_gate=True`` (a
    "gating pass" that could block a release) can never happen here.
    """
    calibration = _valid_calibration(true_negative=20, false_positive=9)
    assert calibration.true_negative + calibration.false_positive == 29
    grader = JudgeGrader(_FakeJudge(0.9), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert not (result.status is GradeStatus.PASS and result.hard_gate is True)
    assert result.hard_gate is False


@pytest.mark.asyncio
async def test_tpr_below_threshold_cannot_gate() -> None:
    # TPR (true positive rate) = TP / (TP+FN) = 20/40 = 0.5, below this
    # judge's own configured threshold of 0.7.
    calibration = _valid_calibration(true_positive=20, false_negative=20, threshold=0.7)
    grader = JudgeGrader(_FakeJudge(0.9), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert not (result.status is GradeStatus.PASS and result.hard_gate is True)
    assert result.hard_gate is False


@pytest.mark.asyncio
async def test_tnr_below_threshold_cannot_gate() -> None:
    # TNR (true negative rate) = TN / (TN+FP) = 20/40 = 0.5, below this
    # judge's own configured threshold of 0.7.
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
    """Sanity check: a judge that clears every bar (enough samples, TPR/TNR
    above threshold, matching fingerprint, an unexpired calibration, a
    response that parses) can actually produce a PASS that gates. This
    matters because, without a positive example like this one, all the
    "cannot gate" tests above would trivially keep passing even if the code
    were broken in a way that made it impossible for ANYTHING to ever gate --
    this test proves gating is still possible when every condition really is
    satisfied.
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


# --- ADR-0018: blank out secrets in candidate_output, and cut it short if
# it's too long, before it ever reaches the judge ---


@pytest.mark.asyncio
async def test_secret_shaped_candidate_output_is_redacted_before_reaching_the_judge() -> None:
    """A secret-shaped substring in execution.output never reaches JudgeClient.judge().

    Captures the real ``JudgeRequest`` the fake client was called with,
    not just the grade's final status.
    """
    execution = _execution_with_output({"answer": f"the token is {_PLANTED_SECRET}"})
    judge = _FakeJudge(0.9)
    grader = JudgeGrader(judge, calibration=None, gate=False)

    result = await grader.grade(_sample(), execution)

    assert judge.calls, "the fake judge was never called"
    received = judge.calls[0].candidate_output
    assert _PLANTED_SECRET not in received
    assert "[REDACTED]" in received
    assert result.evidence["candidate_output_redacted"] is True


@pytest.mark.asyncio
async def test_oversized_candidate_output_is_truncated_before_reaching_the_judge() -> None:
    """A candidate_output longer than the character limit gets cut down to
    that limit (plus a marker showing where it was cut) before it reaches
    the judge -- and the evidence records both that the cut happened and
    the original, pre-cut length.
    """
    long_answer = "x" * (_DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS + 500)
    execution = _execution_with_output({"answer": long_answer})
    stringified = _stringify_output(execution.output)  # no secrets here: redaction is a no-op
    judge = _FakeJudge(0.9)
    grader = JudgeGrader(judge, calibration=None, gate=False)

    result = await grader.grade(_sample(), execution)

    expected_omitted = len(stringified) - _DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS
    expected_received = (
        stringified[:_DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS]
        + f"...[truncated, {expected_omitted} chars omitted]"
    )
    assert judge.calls[0].candidate_output == expected_received
    assert result.evidence["candidate_output_truncated"] is True
    assert result.evidence["candidate_output_original_chars"] == len(stringified)


@pytest.mark.asyncio
async def test_redaction_policy_none_disables_redaction() -> None:
    """``redaction_policy=None`` opts out: a planted secret reaches the judge
    verbatim, and no ``candidate_output_redacted`` key is added at all.
    """
    execution = _execution_with_output({"answer": f"the token is {_PLANTED_SECRET}"})
    judge = _FakeJudge(0.9)
    grader = JudgeGrader(judge, calibration=None, gate=False, redaction_policy=None)

    result = await grader.grade(_sample(), execution)

    assert _PLANTED_SECRET in judge.calls[0].candidate_output
    assert "candidate_output_redacted" not in result.evidence


@pytest.mark.asyncio
async def test_max_candidate_output_chars_none_disables_truncation() -> None:
    """``max_candidate_output_chars=None`` opts out: an oversized output
    reaches the judge whole, and neither truncation evidence key is added.
    """
    long_answer = "x" * (_DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS + 500)
    execution = _execution_with_output({"answer": long_answer})
    stringified = _stringify_output(execution.output)
    judge = _FakeJudge(0.9)
    grader = JudgeGrader(judge, calibration=None, gate=False, max_candidate_output_chars=None)

    result = await grader.grade(_sample(), execution)

    assert judge.calls[0].candidate_output == stringified
    assert "candidate_output_truncated" not in result.evidence
    assert "candidate_output_original_chars" not in result.evidence


@pytest.mark.asyncio
async def test_default_construction_uses_the_named_default_policy_and_bound() -> None:
    """Omitting ``redaction_policy``/``max_candidate_output_chars`` entirely
    behaves identically to passing the named defaults explicitly -- proving
    this is the real default code path, not merely that explicit values work
    (as the tests above already show).
    """
    padding = "y" * _DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS
    output: dict[str, JsonValue] = {"answer": f"{padding} {_PLANTED_SECRET}"}

    implicit_judge = _FakeJudge(0.9)
    implicit_grader = JudgeGrader(implicit_judge, calibration=None, gate=False)
    implicit_result = await implicit_grader.grade(_sample(), _execution_with_output(output))

    explicit_judge = _FakeJudge(0.9)
    explicit_grader = JudgeGrader(
        explicit_judge,
        calibration=None,
        gate=False,
        redaction_policy=DEFAULT_REDACTION_POLICY,
        max_candidate_output_chars=_DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS,
    )
    explicit_result = await explicit_grader.grade(_sample(), _execution_with_output(output))

    assert implicit_judge.calls[0].candidate_output == explicit_judge.calls[0].candidate_output
    assert implicit_result.evidence == explicit_result.evidence
    # Sanity check: confirm the shared fixture actually triggers both
    # redaction and truncation, so the equality assertion above is really
    # comparing two meaningful results -- not just two calls that both
    # happened to do nothing.
    assert implicit_result.evidence["candidate_output_redacted"] is True
    assert implicit_result.evidence["candidate_output_truncated"] is True


@pytest.mark.asyncio
async def test_prompt_and_reference_are_never_redacted_or_truncated() -> None:
    """Only candidate_output goes through the redact-and-shorten pipeline.

    ``prompt`` (from ``sample.input``) and ``reference`` (from
    ``sample.reference``) come from the dataset/task definition, not from
    the AI actually being evaluated (sometimes called the "system under
    test") -- so planting the same secret-shaped pattern and an oversized
    length in both of them should leave them completely untouched. This is
    the test most likely to catch a bug where an over-eager implementation
    redacts the whole ``JudgeRequest`` instead of just ``candidate_output``.
    """
    long_reference = _PLANTED_SECRET + "z" * (_DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS + 500)
    sample = EvalSample(
        sample_id="s1",
        input={"question": f"token {_PLANTED_SECRET}"},
        reference=long_reference,
        source_digest="sha256:row",
        adapter="identity@1",
    )
    execution = _execution_with_output({"answer": "short and clean, nothing to redact"})
    judge = _FakeJudge(0.9)
    grader = JudgeGrader(judge, calibration=None, gate=False)

    await grader.grade(sample, execution)

    request = judge.calls[0]
    assert _PLANTED_SECRET in request.prompt
    assert request.reference == long_reference
    assert request.reference is not None
    assert len(request.reference) == len(long_reference)


@pytest.mark.asyncio
async def test_clean_short_output_adds_no_candidate_output_evidence_keys() -> None:
    """A clean run (no secrets, output well under the character limit) adds
    none of the candidate_output_* evidence keys. This checks the "only add
    an evidence key when something actually happened" rule from the other
    direction -- paired with the tests above that check the key IS added
    when it should be.
    """
    judge = _FakeJudge(0.9)
    grader = JudgeGrader(judge, calibration=None, gate=False)

    result = await grader.grade(_sample(), _execution())

    assert "candidate_output_redacted" not in result.evidence
    assert "candidate_output_truncated" not in result.evidence
    assert "candidate_output_original_chars" not in result.evidence


# --- ADR-0020: the Wilson lower-bound check, the response "status" field,
# mapping transport failures to a grade, and the judge's optional rationale
# text ---


@pytest.mark.asyncio
async def test_wilson_lower_bound_below_floor_blocks_gating_but_grades_advisorily() -> None:
    """This test's held-out "bad answer" class has 30 examples, and the
    judge got 29 of them right (TNR = 29/30 = 0.9667) -- comfortably above
    the project's 0.95 TNR floor, if you just look at the raw rate. But
    with only 30 examples, the "Wilson lower bound" (a conservative,
    statistics-based estimate of how low the true rate could plausibly be,
    explained in calibration.py) comes out to only about 0.833, which does
    NOT clear that same 0.95 floor. That's not proof the judge is bad -- it
    just means there aren't yet enough examples to be confident the judge
    really is that accurate. So gating is blocked, but the judge still
    produces an ordinary, advisory grade (ADR-0020). The "good answer"
    class is sized generously enough that its own bounds clear easily, so
    this test isolates the TNR-side Wilson failure by itself.
    """
    calibration = _valid_calibration(true_negative=29, false_positive=1)
    # The raw, point-in-time rates clear both floors here, so this is NOT
    # the "affirmatively bad, UNAVAILABLE" path -- only the more
    # conservative Wilson lower bound falls short.
    assert calibration.floor_failure_reason() is None
    judge = _FakeJudge(0.9)
    grader = JudgeGrader(judge, calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.PASS  # advisory verdict from the score
    assert result.hard_gate is False
    assert result.judge_calibration_ref is None
    reason = result.evidence["reason"]
    assert isinstance(reason, str) and "Wilson lower bound" in reason
    # This is the advisory path (a calibration failure is already present),
    # so the position-bias probe never runs -- only one call to the judge
    # happens in total.
    assert len(judge.calls) == 1


@pytest.mark.asyncio
async def test_point_estimate_below_project_floor_stays_unavailable() -> None:
    """This is the flip side of the test above (still ADR-0020, but this
    particular rule is unchanged from the original D-1 decision): here the
    raw, point-in-time accuracy number itself is below the project floor --
    not just its conservative Wilson lower bound. That is solid, affirmative
    proof the judge isn't accurate enough, so the result is demoted straight
    to UNAVAILABLE. This is a different failure mode than the
    Wilson-lower-bound test above, where the raw number looked fine but
    there weren't enough examples to trust it.
    """
    # The raw ("point") TNR here is 90/100 = 0.90, below the 0.95 floor. The
    # positive ("good answer") class is set up to clear its own floors
    # easily, so this test isolates just the TNR point-floor failure.
    calibration = _valid_calibration(true_negative=90, false_positive=10)
    grader = JudgeGrader(_FakeJudge(0.9), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.UNAVAILABLE
    assert result.hard_gate is False
    reason = result.evidence["reason"]
    assert isinstance(reason, str) and "project minimum" in reason


@pytest.mark.asyncio
async def test_raising_judge_client_yields_single_error_sample_with_transport_evidence() -> None:
    """If the JudgeClient itself raises an exception (as opposed to just
    returning a response that fails to parse), the grader stops immediately
    instead of retrying it several times, and turns the failure into
    exactly ONE graded ERROR result carrying ``judge_transport_error``
    evidence -- it never lets the exception bubble up and crash the whole
    evaluation run (ADR-0020). The exception's message gets secrets blanked
    out before being saved, because an exception message could accidentally
    contain text copied from the AI's own answer (ADR-0018).
    """

    class _RaisingJudge:
        fingerprint = "judge:model:prompt"

        def __init__(self) -> None:
            self.calls = 0

        async def judge(self, request: JudgeRequest) -> JudgeResponse:
            self.calls += 1
            raise RuntimeError(f"connection reset while leaking {_PLANTED_SECRET}")

    judge = _RaisingJudge()
    grader = JudgeGrader(judge, calibration=None, gate=False)
    result = await grader.grade(_sample(), _execution())

    assert result.status is GradeStatus.ERROR
    assert result.hard_gate is False
    # It failed on the very first call to the judge -- no parse-retries
    # were attempted, because this is a different kind of failure (the call
    # itself raised an exception) than a response that merely fails to parse.
    assert judge.calls == 1
    assert result.evidence["judge_transport_error"] == "RuntimeError"
    message = result.evidence["judge_transport_error_message"]
    assert isinstance(message, str)
    assert _PLANTED_SECRET not in message
    assert "[REDACTED]" in message


@pytest.mark.asyncio
async def test_refused_status_maps_to_abstain() -> None:
    """A REFUSED response means the judge declined to give a real verdict at
    all -- so it maps to ABSTAIN, never to a task FAIL, and it can never
    gate (ADR-0020). The recorded reason names the status that caused this.
    This check happens before the position-bias probe is even attempted: a
    non-OK status skips straight past the rest of the grading logic.
    """
    judge = _FakeJudge(0.9, status=JudgeResponseStatus.REFUSED)
    grader = JudgeGrader(judge, calibration=_valid_calibration(), gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.ABSTAIN
    assert result.hard_gate is False
    reason = result.evidence["reason"]
    assert isinstance(reason, str) and "refused" in reason
    assert len(judge.calls) == 1


@pytest.mark.asyncio
async def test_timeout_status_maps_to_error() -> None:
    """A TIMEOUT response means our infrastructure had a problem talking to
    the judge -- it isn't a verdict about the AI's answer at all. So it maps
    to ERROR (ADR-0008 keeps infrastructure problems separate from task
    failures) and can never gate a release (ADR-0020).
    """
    judge = _FakeJudge(0.9, status=JudgeResponseStatus.TIMEOUT)
    grader = JudgeGrader(judge, calibration=_valid_calibration(), gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.ERROR
    assert result.hard_gate is False
    reason = result.evidence["reason"]
    assert isinstance(reason, str) and "timeout" in reason


@pytest.mark.asyncio
async def test_uncalibrated_grade_makes_exactly_one_judge_call() -> None:
    """An advisory-only grade never triggers the extra position-bias probe
    call (ADR-0020): even when the caller passes ``gate=True``, a judge with
    no calibration is called exactly once per sample. The second (probe)
    call is reserved for the path that could actually gate a release.
    """
    judge = _FakeJudge(0.9)
    grader = JudgeGrader(judge, calibration=None, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.hard_gate is False
    assert len(judge.calls) == 1


@pytest.mark.asyncio
async def test_calibrated_fail_sample_still_runs_probe_and_records_reason() -> None:
    """The position-bias probe on the gating path does NOT only run when the
    verdict is PASS (ADR-0020): even when a calibrated judge's answer is
    FAIL, the probe still runs, so a position-bias problem still shows up
    in the evidence even on a sample that already failed. This guards
    against a future bug where someone adds a check that skips the probe
    whenever the main verdict isn't a pass.
    """

    class _FailWithPositionBiasJudge:
        fingerprint = "judge:model:prompt"

        def __init__(self) -> None:
            self.calls = 0

        async def judge(self, request: JudgeRequest) -> JudgeResponse:
            self.calls += 1
            reversed_flag = bool(request.metadata.get("reversed"))
            # The main verdict is FAIL, but the reversed (position-bias
            # probe) call flips to PASS -- so this test can check that the
            # disagreement is still caught even though the main verdict
            # wasn't a PASS.
            return JudgeResponse(
                fingerprint=self.fingerprint,
                verdict="pass" if reversed_flag else "fail",
                score=0.9 if reversed_flag else 0.1,
                parse_ok=True,
                abstained=False,
            )

    judge = _FailWithPositionBiasJudge()
    grader = JudgeGrader(judge, calibration=_valid_calibration(), gate=True)
    result = await grader.grade(_sample(), _execution())

    assert result.status is GradeStatus.FAIL  # primary score 0.1 < threshold
    assert result.hard_gate is False
    # The probe still ran (a second call to the judge) even though the main
    # verdict was FAIL.
    assert judge.calls == 2
    reason = result.evidence["reason"]
    assert isinstance(reason, str)
    assert "position" in reason or "bias" in reason


@pytest.mark.asyncio
async def test_rationale_is_redacted_and_truncated_in_evidence() -> None:
    """A judge's ``rationale`` (its free-text explanation of its own verdict)
    could accidentally repeat back text from the AI's own answer, so -- just
    like the AI's answer itself -- it gets secrets blanked out and then cut
    short if it's too long (ADR-0018) before being saved to
    ``evidence["judge_rationale"]``. This is purely a record for a human to
    read later; the grading logic itself never looks at it (ADR-0020).
    """
    long_tail = "z" * (_DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS + 500)
    rationale = f"reference matched, incidentally leaking {_PLANTED_SECRET} {long_tail}"
    judge = _FakeJudge(0.9, rationale=rationale)
    grader = JudgeGrader(judge, calibration=None, gate=False)
    result = await grader.grade(_sample(), _execution())

    recorded = result.evidence["judge_rationale"]
    assert isinstance(recorded, str)
    assert _PLANTED_SECRET not in recorded
    assert "[REDACTED]" in recorded
    assert "truncated" in recorded  # confirms the truncation marker text is present


@pytest.mark.asyncio
async def test_clean_ok_response_records_no_rationale_or_transport_evidence() -> None:
    """This project's "only add an evidence key when something actually
    happened" rule also holds for the newer ADR-0020 evidence keys: a clean
    OK response with no rationale adds neither ``judge_rationale`` nor any
    ``judge_transport_error`` key.
    """
    judge = _FakeJudge(0.9)
    grader = JudgeGrader(judge, calibration=None, gate=False)
    result = await grader.grade(_sample(), _execution())
    assert "judge_rationale" not in result.evidence
    assert "judge_transport_error" not in result.evidence
    assert "judge_transport_error_message" not in result.evidence


def test_calibration_coverage_fields_reject_negative_values() -> None:
    """The extra record-keeping fields added by ADR-0020 (``total_labeled``,
    ``abstained_count``, ``error_count``) must be zero or positive whenever
    they're actually supplied.

    ``None`` (the default) means "nobody recorded this" and is allowed; but a
    negative count is rejected at construction time. This reuses the same
    validator that already enforces the same non-negative rule on the
    required true_positive/true_negative/false_positive/false_negative
    counts (sometimes called a "confusion matrix": a table of how often the
    judge's answer did or didn't match the real answer).
    """
    with pytest.raises(ValidationError):
        _valid_calibration(error_count=-1)
    with pytest.raises(ValidationError):
        _valid_calibration(abstained_count=-5)
    # Non-negative values (and None) construct fine.
    assert _valid_calibration(total_labeled=0).total_labeled == 0
    assert _valid_calibration().error_count is None
