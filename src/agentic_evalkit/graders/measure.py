"""Measuring a judge against hand-labeled answers, to produce its calibration (ADR-0024).

``graders.calibration`` defines what a calibration *is* and what it has to
clear before a judge may block a release; ``integrations.base.judge_authority``
turns one into a verdict. Neither of them can make one. Until this module
existed, the only way to obtain a ``CalibrationArtifact`` was to hand-write
the four confusion-matrix counts and invent a fingerprint -- which is to say
the gate could reject a judge but nothing could let one through it honestly.
This module is the other half: run the judge over examples whose right
answers are already known, count what it got right, and record that.

Three properties are worth stating outright, because each is a decision
rather than an implementation detail.

* **The measurement applies the same decision function the grader applies.**
  :class:`~agentic_evalkit.graders.judge.JudgeGrader` turns a judge's score
  into pass/fail by comparing it against ``pass_score_threshold``. So does
  this module, using the same parameter with the same default. Candidate
  text is also redacted then truncated with the same helper and defaults
  the grader uses, so authority is never earned on a raw string the live
  path will rewrite. If either half ever diverged, the artifact would
  faithfully describe a decision nobody makes in production, which is worse
  than having no artifact at all -- an unmeasured judge at least announces
  itself as unmeasured.
* **``calibrated_at`` is always set.** The field is optional on the artifact
  so that records written before it existed still load, and an artifact
  without it can never gate (``age_failure_reason``). That is the right
  default for data arriving from elsewhere, and the wrong outcome for a
  measurement that just happened on this machine at a time we know exactly.
  Every artifact this module produces is dated, and ``expires_at`` is
  derived from it rather than supplied.
* **The artifact carries counts, not text.** No prompt, no candidate
  output, no judge rationale, and no exception message reaches it. A
  calibration is a small file that gets committed, attached to a release,
  and read by people who were never shown the labeled set -- so it is kept
  structurally incapable of leaking what it was measured on, rather than
  relying on a redaction pass to catch it later (ADR-0024).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

from agentic_evalkit.graders.calibration import (
    PROJECT_MAX_CALIBRATION_AGE_DAYS,
    PROJECT_MIN_TPR,
    CalibrationArtifact,
)
from agentic_evalkit.graders.judge import (
    DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS,
    JudgeRequest,
    JudgeResponseStatus,
    prepare_candidate_output_text,
)
from agentic_evalkit.models.calibration import CalibrationLabel
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY, RedactionPolicy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentic_evalkit.graders.judge import JudgeClient, JudgeResponse
    from agentic_evalkit.models.calibration import LabeledJudgeSample

__all__ = ["DEFAULT_PASS_SCORE_THRESHOLD", "measure_calibration"]

#: The score at or above which a judge's verdict counts as "this answer is
#: good", mirroring ``JudgeGrader``'s own ``pass_score_threshold`` default so
#: a calibration measures the decision the grader will actually make. Defined
#: here rather than imported because that one is a constructor default on a
#: class, not a module constant; the two are checked against each other in
#: ``tests/unit/graders/test_measure.py``.
DEFAULT_PASS_SCORE_THRESHOLD = 0.5

#: What one labeled sample contributed. ``GOOD``/``BAD`` are the judge's own
#: verdict on the answer (scored against the sample's label to land in one of
#: the four confusion-matrix cells); ``ABSTAINED`` and ``ERRORED`` are
#: non-verdicts, counted on their own and never folded into a class.
_JudgeOutcome = Literal["good", "bad", "abstained", "errored"]


def _classify_response(
    response: JudgeResponse,
    *,
    judge_fingerprint: str,
    pass_score_threshold: float,
) -> _JudgeOutcome:
    """Reduce one judge response to what it contributes to the counts.

    The order of checks matches
    :meth:`~agentic_evalkit.graders.judge.JudgeGrader.grade` deliberately,
    so a sample that would be an ``ERROR`` or ``ABSTAIN`` during a real run
    is also not counted as a verdict during calibration. Measuring by one
    rule and grading by another would produce an artifact describing a
    judge that does not exist.

    Four separate things all mean "this sample yielded no usable verdict"
    and are counted as errors rather than guessed at: an operational
    failure the judge reported itself (rate limit, timeout, its own
    error), a response whose fingerprint says some *other* judge answered
    it, output that could not be parsed, and a parsed non-abstention that
    still carried no score. The last is the subtle one -- a judge that
    returns ``score=None`` while claiming to have rendered a verdict has
    told us nothing we can score, and picking a side would invent evidence.
    """
    if response.status is JudgeResponseStatus.REFUSED:
        return "abstained"
    if response.status is not JudgeResponseStatus.OK:
        return "errored"
    if response.fingerprint != judge_fingerprint:
        return "errored"
    if not response.parse_ok:
        return "errored"
    if response.abstained:
        return "abstained"
    if response.score is None:
        return "errored"
    return "good" if response.score >= pass_score_threshold else "bad"


async def _judge_one(
    judge: JudgeClient,
    sample: LabeledJudgeSample,
    *,
    judge_fingerprint: str,
    pass_score_threshold: float,
    redaction_policy: RedactionPolicy,
    max_candidate_output_chars: int | None,
) -> _JudgeOutcome:
    """Ask the judge about one labeled sample; never let it take the run down.

    A judge that raises on one sample is the same class of problem the
    runner and ``JudgeGrader`` already isolate per sample (ADR-0008): it is
    our infrastructure failing, not evidence about the judge's accuracy, and
    it must not abort a measurement that may be most of the way through a
    labeled set. So the exception ends this sample and nothing else.

    The candidate text is redacted then truncated with the same helper and
    defaults ``JudgeGrader`` uses, so the artifact describes the inputs the
    live grader will actually forward (ADR-0018). Prompt and reference stay
    untouched for the same reason they do at grade time.

    The exception itself is deliberately not retained. It could quote the
    candidate output back at us, and the artifact this feeds is defined to
    hold no free text at all (see the module docstring) -- so there is
    nowhere safe to put it, and the honest record is that this sample
    produced no verdict. ``asyncio.CancelledError`` is not an ``Exception``
    subclass, so cancelling a calibration run still actually cancels it.
    """
    request = JudgeRequest(
        sample_id=sample.sample_id,
        prompt=sample.prompt,
        candidate_output=prepare_candidate_output_text(
            sample.candidate_output,
            redaction_policy=redaction_policy,
            max_candidate_output_chars=max_candidate_output_chars,
        ),
        reference=sample.reference,
    )
    try:
        response = await judge.judge(request)
    except Exception:
        # Deliberately swallowed without recording the message; see the
        # docstring above. The incremented error_count is the record.
        return "errored"
    return _classify_response(
        response,
        judge_fingerprint=judge_fingerprint,
        pass_score_threshold=pass_score_threshold,
    )


async def measure_calibration(
    judge: JudgeClient,
    samples: Sequence[LabeledJudgeSample],
    *,
    calibration_id: str,
    threshold: float = PROJECT_MIN_TPR,
    pass_score_threshold: float = DEFAULT_PASS_SCORE_THRESHOLD,
    redaction_policy: RedactionPolicy = DEFAULT_REDACTION_POLICY,
    max_candidate_output_chars: int | None = DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS,
    now: datetime | None = None,
) -> CalibrationArtifact:
    """Run ``judge`` over ``samples`` and record how often it was right.

    The judge is called exactly once per sample, in order. Each verdict is
    compared against that sample's human label and lands in one of the four
    confusion-matrix cells; a sample the judge declined or failed on lands in
    ``abstained_count``/``error_count`` instead, so a judge cannot improve its
    measured accuracy by refusing the questions it would have got wrong.
    Those non-verdict counts also feed the coverage floor on the artifact:
    a high abstain/error share blocks gating even when the answered rows look
    perfect (ADR-0024).

    Candidate text is redacted then truncated with the same defaults
    :class:`~agentic_evalkit.graders.judge.JudgeGrader` uses, so the
    measurement is of the inputs the grader will actually send.

    Whether the returned artifact is *good enough to gate* is not decided
    here and cannot be read off these counts by eye -- pass it to
    :func:`~agentic_evalkit.integrations.base.judge_authority`, which is the
    one place that decision is made.

    Args:
        judge: The judge being measured. Its ``fingerprint`` is read once, at
            the start, and recorded on the artifact -- a caller never types a
            fingerprint, so an artifact can never claim to describe a judge
            other than the one that produced it.
        samples: The labeled examples to measure against. May be any size,
            including empty; a set too thin to support gating still produces
            an artifact, which then reports its own insufficiency (ADR-0024).
        calibration_id: A stable name for this measurement, recorded on the
            artifact and quoted in any reason explaining why it cannot gate.
        threshold: The judge's own configured pass bar for the resulting
            TPR/TNR. Defaults to the project's TPR floor. Note this is a
            floor a caller may raise but never usefully lower: the
            project-wide minimums apply on top of it and are stricter for
            TNR (:mod:`agentic_evalkit.graders.calibration`).
        pass_score_threshold: The score at or above which the judge's verdict
            counts as "good". Must match the value the ``JudgeGrader`` that
            will use this artifact was built with, or the artifact describes
            a different decision than the one being gated on.
        redaction_policy: Secret patterns applied to ``candidate_output``
            before the judge sees it. Defaults to the same policy
            ``JudgeGrader`` defaults to; pass the same override the
            consuming grader uses.
        max_candidate_output_chars: Length bound applied after redaction.
            Defaults to the same bound ``JudgeGrader`` defaults to; ``None``
            disables truncation.
        now: The moment to record as ``calibrated_at``. Defaults to the
            current UTC time; tests pass a fixed value. Must be
            timezone-aware -- ``CalibrationArtifact`` rejects a naive
            timestamp rather than risk a comparison crash later.

    Returns:
        A dated :class:`~agentic_evalkit.graders.calibration.CalibrationArtifact`
        whose ``expires_at`` is ``calibrated_at`` plus the project's maximum
        calibration age.
    """
    judge_fingerprint = judge.fingerprint
    calibrated_at = now or datetime.now(UTC)

    true_positive = true_negative = false_positive = false_negative = 0
    abstained_count = error_count = 0

    for sample in samples:
        outcome = await _judge_one(
            judge,
            sample,
            judge_fingerprint=judge_fingerprint,
            pass_score_threshold=pass_score_threshold,
            redaction_policy=redaction_policy,
            max_candidate_output_chars=max_candidate_output_chars,
        )
        if outcome == "abstained":
            abstained_count += 1
        elif outcome == "errored":
            error_count += 1
        elif sample.label is CalibrationLabel.GOOD:
            # The judge answered about an answer that really is good, so it
            # was right exactly when it said so.
            if outcome == "good":
                true_positive += 1
            else:
                false_negative += 1
        elif outcome == "bad":
            true_negative += 1
        else:
            false_positive += 1

    return CalibrationArtifact(
        calibration_id=calibration_id,
        judge_fingerprint=judge_fingerprint,
        calibrated_at=calibrated_at,
        expires_at=calibrated_at + timedelta(days=PROJECT_MAX_CALIBRATION_AGE_DAYS),
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        threshold=threshold,
        total_labeled=len(samples),
        abstained_count=abstained_count,
        error_count=error_count,
    )
