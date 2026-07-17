"""Grading with an AI judge that's been checked against real human answers.

An AI judge is just another model call: ask it "is this answer good?" and
it gives you a verdict. The problem is that an ungated judge can rubber-stamp
a bad release just as confidently as a good one — there's no way to tell,
from the verdict alone, whether the judge actually knows what it's talking
about. So before ``JudgeGrader`` will let a judge's verdict block a release
(``hard_gate=True`` on the returned :class:`GradeResult`), it requires proof
that the judge is trustworthy, plus a series of runtime sanity checks. Every
condition below has to hold, or the result falls back to an informational
("advisory") grade that can't block anything:

- **The judge actually responded.** If the call to the judge raises an
  exception, that's treated as our infrastructure breaking, not the AI
  getting a wrong answer — it becomes ``GradeStatus.ERROR`` for just that
  one sample, and the rest of the run keeps going (ADR-0008 / ADR-0020).
- **The judge's response says "here's my verdict," not something else.**
  A judge can also say "I decline to answer" (mapped to
  ``GradeStatus.ABSTAIN`` — a non-verdict, not a failure) or report its own
  transport problem like a timeout or rate limit (mapped to
  ``GradeStatus.ERROR``, kept separate from an actual wrong-answer failure,
  ADR-0020).
- **The judge we just called is the exact judge the calibration data is
  about.** We check this by comparing fingerprints (a hash of the model +
  prompt) — if they don't match, the calibration data doesn't apply to this
  judge.
- **The calibration hasn't expired.** An expired calibration is treated as
  unusable outright: ``GradeStatus.UNAVAILABLE``.
- **The judge's measured accuracy clears our minimum bar.** Concretely: it
  has to correctly say "wrong" at least 95% of the time on real wrong
  answers, and correctly say "right" at least 85% of the time on real right
  answers (decision D-1). If the judge's own configured threshold is more
  lenient than that, our minimum still wins — nobody can quietly lower the
  bar. Falling short of this bar is solid proof the judge isn't reliable, so
  it also demotes the result to ``GradeStatus.UNAVAILABLE``.
- **That accuracy number is actually trustworthy, not just lucky.** We
  compute a conservative lower estimate (a "Wilson lower bound" — a standard
  statistics technique for asking "how bad could the true rate plausibly be,
  given how few samples we tested on?") for both accuracy numbers above, and
  require *that* to also clear the bar. A judge can look accurate on paper
  while having only been tested on a handful of examples; this catches that.
  Unlike an outright-bad accuracy number, this doesn't mean the judge is
  broken — it means we don't have enough evidence yet — so it blocks the
  release-gating verdict without marking the result ``UNAVAILABLE``; an
  advisory grade still comes through (ADR-0020, updating ADR-0007's
  original raw-accuracy-only check).
- **The calibration data is dated, and it's recent.** We need to know when
  the judge was checked (a timezone-aware ``calibrated_at`` timestamp), and
  that check needs to be no older than 90 days. Missing or old data means we
  simply can't gate — like the point above, it doesn't mean the judge is
  bad, so it doesn't produce ``UNAVAILABLE``; it just can't approve a
  release (D-1, as amended 2026-07-04).
- **The calibration was actually tested on enough examples.** At least 30
  real "wrong" examples and 30 real "right" examples, or the accuracy
  numbers are just noise.
- **The judge clears its own configured threshold too** — a caller-supplied
  bar the judge must ALSO meet, on top of everything above.
- **The judge doesn't flip its answer when we swap the order of the two
  options it's comparing.** This "does the order of the answers change the
  verdict" check only runs when the result could actually gate a release
  (this keeps the advisory-only path down to a single judge call per
  question, ADR-0020); it still runs for a calibrated ``FAIL`` result, not
  just a ``PASS``, so we always know whether order-bias was a factor. If the
  order-swapped call itself fails, that failure blocks gating too — it never
  crashes the whole grading run.
- **The judge gave a real, parseable answer** (not gibberish it couldn't
  finish forming). Unparseable responses get retried up to twice (three
  tries total); a hard transport failure, by contrast, doesn't get retried
  at all — it's reported immediately.

If any single one of these isn't satisfied, the result falls back to an
advisory grade rather than one that can block a release. Several of these
cases also get their own specific status (rather than a generic "fail") so
the actual reason survives in ``evidence["reason"]`` instead of being
flattened into a plain pass/fail. If the judge includes a free-text
explanation of its verdict, we keep it — redacted and length-capped the same
way the AI's actual answer is (ADR-0018) — in ``evidence["judge_rationale"]``
purely as a record for a human to read later; nothing in the grading logic
itself ever looks at it.
"""

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import Field, JsonValue

from agentic_evalkit.graders.calibration import (
    PROJECT_MAX_CALIBRATION_AGE_DAYS as PROJECT_MAX_CALIBRATION_AGE_DAYS,
)
from agentic_evalkit.graders.calibration import (
    PROJECT_MIN_TNR as PROJECT_MIN_TNR,
)
from agentic_evalkit.graders.calibration import (
    PROJECT_MIN_TPR as PROJECT_MIN_TPR,
)
from agentic_evalkit.graders.calibration import (
    CalibrationArtifact as CalibrationArtifact,
)
from agentic_evalkit.models import EvalSample, GradeResult, GradeStatus, NormalizedExecutionResult
from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY, RedactionPolicy

# If the judge's response doesn't parse, we try again -- but only up to
# this many extra times, so 3 judge calls total at most.
_MAXIMUM_PARSE_RETRIES = 2

# How long (in characters, not bytes -- close enough for a length cutoff,
# no need for byte-exact precision here) the AI's answer is allowed to be
# before we cut it short when sending it to the judge (ADR-0018). Matches
# `runner.py`'s own `_LARGE_OUTPUT_THRESHOLD_BYTES` (8192) -- same number,
# reused for consistency -- but defined again here rather than imported,
# since that constant is private to `runner.py`.
_DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS = 8192


class JudgeRequest(FrozenModel):
    """A single request sent to a :class:`JudgeClient`.

    ``metadata`` carries caller-defined context (e.g. a ``reversed`` flag
    for the position-bias probe); it is opaque to this module.
    """

    sample_id: str
    prompt: str
    candidate_output: str
    reference: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class JudgeResponseStatus(StrEnum):
    """What happened when we called the judge -- not the verdict itself (ADR-0020).

    This is a fixed list of named outcomes rather than a plain boolean or a
    free-form string (ADR-0002 requires that everywhere in this project), so
    a ``JudgeClient`` implementation can say *why* it isn't returning a real
    verdict, instead of that distinction getting lost. ``OK`` is the only
    status where :meth:`JudgeGrader.grade` actually looks at the verdict;
    every other status is handled before we even get there: ``REFUSED``
    becomes ``GradeStatus.ABSTAIN`` (the judge declined to answer, which
    isn't the same as the AI failing the task), and
    ``RATE_LIMITED``/``TIMEOUT``/``ERROR`` become ``GradeStatus.ERROR`` (our
    infrastructure had a problem, which is also not the same as the AI
    failing the task -- ADR-0008 keeps those two kinds of failure separate).
    """

    OK = "ok"
    REFUSED = "refused"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    ERROR = "error"


class JudgeResponse(FrozenModel):
    """A judge's answer to one :class:`JudgeRequest`.

    ``parse_ok=False`` means we couldn't make sense of the judge's raw
    output at all -- that's a problem with the judge's response, not a
    failing grade for the AI being evaluated (see the module docstring).
    ``abstained=True`` means the response parsed fine, but the judge
    explicitly said "I'm not going to render a verdict here."

    ``status`` records what actually happened on the call (see
    :class:`JudgeResponseStatus`, added in ADR-0020). It's optional and
    defaults to :attr:`JudgeResponseStatus.OK`, so any existing
    ``JudgeClient`` written before this field existed still behaves exactly
    as it did before (``schema_version`` stays ``"1"``). ``rationale`` is an
    optional free-text explanation the judge can attach to its verdict --
    it's kept purely as a record for a human to read (redacted and
    length-capped first), and nothing in the grading logic itself reads it.
    """

    fingerprint: str
    verdict: str
    score: float | None
    parse_ok: bool
    abstained: bool
    status: JudgeResponseStatus = JudgeResponseStatus.OK
    rationale: str | None = None


@runtime_checkable
class JudgeClient(Protocol):
    """Provider-neutral judge boundary (design §9)."""

    fingerprint: str

    async def judge(self, request: JudgeRequest) -> JudgeResponse: ...


def _passing_score_to_status(score: float | None, threshold: float) -> GradeStatus:
    if score is None:
        return GradeStatus.UNAVAILABLE
    return GradeStatus.PASS if score >= threshold else GradeStatus.FAIL


class JudgeGrader:
    """Grades one execution using an AI judge, gating only when it's proven reliable.

    Args:
        judge: The judge to call. Any object with an async ``judge()``
            method and a ``fingerprint`` attribute works (see
            :class:`JudgeClient`) -- this class doesn't care which AI
            provider it's talking to.
        calibration: The proof that this judge is reliable, or ``None`` if
            we don't have any. Without it, the judge can still be asked for
            an opinion, but that opinion can never block a release
            (``hard_gate`` is always ``False``, and the result won't
            reference any calibration).
        gate: Whether the caller *wants* this judge's verdict to be able to
            block a release, assuming every reliability check below passes.
            Even when this is ``True``, a single failed check is enough to
            fall back to an advisory-only grade.
        pass_score_threshold: The minimum score (once we trust the verdict)
            that counts as a pass.
        name: A stable label for this grader, recorded on the result so you
            can tell which grader produced it.
        redaction_policy: What counts as a "secret-shaped" string (an API
            key, a token, etc.) that should be blanked out of the AI's
            answer before it's shown to the judge. This is never applied to
            the original question or the reference answer -- see the note
            below for why. Defaults to
            :data:`~agentic_evalkit.reporters.base.DEFAULT_REDACTION_POLICY`,
            so secrets don't leak to a judge by accident. Pass ``None`` (or
            an empty policy) to turn this off.
        max_candidate_output_chars: The longest the AI's answer is allowed
            to be before we cut it short (with a marker showing it was cut)
            when sending it to the judge. Defaults to 8192 characters. Pass
            ``None`` to never cut it short.

    Since ADR-0018, what the judge actually sees for the AI's answer can
    differ from the AI's literal, original output: secret-shaped text gets
    blanked out and an overly long answer gets cut short before it's sent.
    A judge implementation shouldn't assume it's seeing the exact original
    text. The question itself and the reference answer, by contrast, are
    never touched -- they come from the dataset/task definition, which this
    framework controls, not from the AI being tested.

    On judge-call cost (ADR-0020): the extra "does the verdict flip if we
    swap answer order" check is a second call to the judge, and it only
    happens when the result could actually gate a release -- meaning
    ``gate=True`` and every reliability check already passed. So an
    advisory-only grade (``gate=False``, or any failed check) costs exactly
    one judge call, not two. If the judge call itself throws an exception,
    that becomes a single ``GradeStatus.ERROR`` for that one sample -- it
    never takes down the whole evaluation run (ADR-0008).
    """

    def __init__(
        self,
        judge: JudgeClient,
        *,
        calibration: CalibrationArtifact | None,
        gate: bool,
        pass_score_threshold: float = 0.5,
        name: str = "judge@1",
        redaction_policy: RedactionPolicy | None = DEFAULT_REDACTION_POLICY,
        max_candidate_output_chars: int | None = _DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS,
    ) -> None:
        self._judge = judge
        self._calibration = calibration
        self._gate = gate
        self._pass_score_threshold = pass_score_threshold
        self._name = name
        self._redaction_policy = redaction_policy
        self._max_candidate_output_chars = max_candidate_output_chars

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        now = datetime.now(UTC)
        candidate_output, candidate_output_evidence = self._prepare_candidate_output(
            execution.output
        )
        request = JudgeRequest(
            sample_id=sample.sample_id,
            # `prompt`/`reference` come from the dataset/task definition,
            # which this framework controls -- never from the AI being
            # tested. Only `candidate_output` (the AI's own answer) gets
            # redacted/truncated before being sent to the judge; see
            # `_prepare_candidate_output`.
            prompt=_stringify_input(sample.input),
            candidate_output=candidate_output,
            reference=sample.reference,
        )

        response, retry_evidence = await self._judge_with_bounded_retries(request)
        if response is None:
            return self._missing_response_result(
                sample, now, retry_evidence, candidate_output_evidence
            )

        # A response that parsed fine can still include the judge's own
        # explanation of its verdict; we keep that as a record (redacted and
        # length-capped like the AI's answer was, ADR-0018), but nothing
        # below reads it when deciding pass/fail.
        response_evidence = self._response_evidence(response)

        # Each check below either returns a non-gating GradeResult (meaning
        # "stop here, this is the answer") or None (meaning "this check
        # passed, keep going to the next one"). We compare with `is not
        # None` on purpose, rather than just `if result:` -- that way this
        # sequence can't be silently broken if GradeResult ever grows a
        # custom truthiness check down the line.
        non_ok_result = self._non_ok_envelope_result(
            response, sample, now, candidate_output_evidence, retry_evidence, response_evidence
        )
        if non_ok_result is not None:
            return non_ok_result

        fingerprint_result = self._fingerprint_mismatch_result(
            response, sample, now, candidate_output_evidence, response_evidence
        )
        if fingerprint_result is not None:
            return fingerprint_result

        abstained_result = self._abstained_result(
            response, sample, now, candidate_output_evidence, response_evidence
        )
        if abstained_result is not None:
            return abstained_result

        unusable_result = self._unusable_calibration_result(
            sample, now, candidate_output_evidence, response_evidence
        )
        if unusable_result is not None:
            return unusable_result

        calibration_failure = self._calibration_failure_reason(now=now)
        # We only run the answer-order check when this result could actually
        # gate a release: `gate=True` and every reliability check above
        # passed (ADR-0020). We still run it even if the verdict itself is a
        # FAIL, not just a PASS, so we always know whether order-bias was a
        # factor. Any other case (gate=False, or a failed reliability check)
        # costs exactly one judge call for this sample -- never a second one.
        position_bias_reason: str | None = None
        if self._gate and calibration_failure is None:
            position_bias_reason = await self._position_bias_failure_reason(request, response)

        status = _passing_score_to_status(response.score, self._pass_score_threshold)
        can_gate = (
            self._gate
            and calibration_failure is None
            and position_bias_reason is None
            and status is GradeStatus.PASS
        )

        reason = calibration_failure or position_bias_reason
        return self._result(
            sample,
            now,
            status=status,
            score=response.score,
            hard_gate=can_gate,
            reason=reason,
            calibration_ref=(
                self._calibration.calibration_id
                if can_gate and self._calibration is not None
                else None
            ),
            extra_evidence={**candidate_output_evidence, **retry_evidence, **response_evidence},
        )

    def _missing_response_result(
        self,
        sample: EvalSample,
        now: datetime,
        retry_evidence: dict[str, JsonValue],
        candidate_output_evidence: dict[str, JsonValue],
    ) -> GradeResult:
        """Build the ERROR result for when the judge call failed or gave garbage.

        Two different things land here, but both are our infrastructure
        having a problem, not the AI failing the task (ADR-0020): either the
        call to the judge itself raised an exception, or the judge kept
        returning unparseable output even after we retried it. The retry
        helper tells us which one happened via the evidence it records
        (`judge_transport_error` for the former). Either way, this produces
        one ERROR result for this one sample -- it doesn't abort the run.
        """
        if "judge_transport_error" in retry_evidence:
            reason = "judge transport call raised before a verdict was returned"
        else:
            reason = "judge response could not be parsed after bounded retries"
        return self._result(
            sample,
            now,
            status=GradeStatus.ERROR,
            score=None,
            hard_gate=False,
            reason=reason,
            extra_evidence={**candidate_output_evidence, **retry_evidence},
        )

    def _non_ok_envelope_result(
        self,
        response: JudgeResponse,
        sample: EvalSample,
        now: datetime,
        candidate_output_evidence: dict[str, JsonValue],
        retry_evidence: dict[str, JsonValue],
        response_evidence: dict[str, JsonValue],
    ) -> GradeResult | None:
        """Handle a call that didn't come back with a real verdict, or return ``None`` if it did.

        Checked before we even look at the fingerprint or whether the judge
        abstained (ADR-0020): a refusal means the judge chose not to answer
        (ABSTAIN -- that's not the same as the AI failing the task), and a
        rate-limit/timeout/error means our infrastructure hit a snag (ERROR,
        kept separate from an actual task failure per ADR-0008). Neither can
        gate a release; the returned reason says exactly which one happened.
        """
        if response.status is JudgeResponseStatus.OK:
            return None
        if response.status is JudgeResponseStatus.REFUSED:
            status = GradeStatus.ABSTAIN
            reason = f"judge declined to render a verdict (status {response.status.value!r})"
        else:
            status = GradeStatus.ERROR
            reason = f"judge reported an operational failure (status {response.status.value!r})"
        return self._result(
            sample,
            now,
            status=status,
            score=None,
            hard_gate=False,
            reason=reason,
            extra_evidence={
                **candidate_output_evidence,
                **retry_evidence,
                **response_evidence,
            },
        )

    def _fingerprint_mismatch_result(
        self,
        response: JudgeResponse,
        sample: EvalSample,
        now: datetime,
        candidate_output_evidence: dict[str, JsonValue],
        response_evidence: dict[str, JsonValue],
    ) -> GradeResult | None:
        """Return an ERROR if the response's judge doesn't match this calibration."""
        if response.fingerprint == self._judge_fingerprint():
            return None
        return self._result(
            sample,
            now,
            status=GradeStatus.ERROR,
            score=None,
            hard_gate=False,
            reason=(
                f"judge fingerprint {response.fingerprint!r} does not match "
                f"live judge fingerprint {self._judge_fingerprint()!r}"
            ),
            extra_evidence={**candidate_output_evidence, **response_evidence},
        )

    def _abstained_result(
        self,
        response: JudgeResponse,
        sample: EvalSample,
        now: datetime,
        candidate_output_evidence: dict[str, JsonValue],
        response_evidence: dict[str, JsonValue],
    ) -> GradeResult | None:
        """Return an ABSTAIN result if the judge chose not to answer, else ``None``."""
        if not response.abstained:
            return None
        return self._result(
            sample,
            now,
            status=GradeStatus.ABSTAIN,
            score=None,
            hard_gate=False,
            reason="judge explicitly abstained from rendering a verdict",
            extra_evidence={**candidate_output_evidence, **response_evidence},
        )

    def _unusable_calibration_result(
        self,
        sample: EvalSample,
        now: datetime,
        candidate_output_evidence: dict[str, JsonValue],
        response_evidence: dict[str, JsonValue],
    ) -> GradeResult | None:
        """Return UNAVAILABLE if we have solid proof the judge is unreliable, else ``None``.

        This only covers the "we have solid proof this judge can't be
        trusted" case: its calibration expired, or its measured accuracy is
        genuinely below our minimum bar (decision D-1). This check runs
        first, before anything else, so a genuinely bad judge can never
        sneak through with an advisory PASS. The other case -- we simply
        don't have *enough* proof either way, because the calibration record
        is missing/old or the sample size was too small -- is different: it
        blocks gating too, but through `usability_failure_reason` elsewhere,
        and it doesn't mark the result UNAVAILABLE (D-1, as amended
        2026-07-04; ADR-0020).
        """
        if self._calibration is None:
            return None
        if self._calibration.is_expired(now=now):
            unusable_reason: str | None = (
                f"calibration {self._calibration.calibration_id!r} expired at "
                f"{self._calibration.expires_at.isoformat()}"
            )
        else:
            unusable_reason = self._calibration.floor_failure_reason()
        if unusable_reason is None:
            return None
        return self._result(
            sample,
            now,
            status=GradeStatus.UNAVAILABLE,
            score=None,
            hard_gate=False,
            reason=unusable_reason,
            extra_evidence={**candidate_output_evidence, **response_evidence},
        )

    def _judge_fingerprint(self) -> str:
        return self._judge.fingerprint

    def _prepare_candidate_output(
        self, output: dict[str, JsonValue] | None
    ) -> tuple[str, dict[str, JsonValue]]:
        """Turn the AI's answer into text, blank out secrets, then shorten it if needed.

        This only touches ``candidate_output`` -- the actual words the AI
        being tested produced. It's deliberately never applied to the
        question (``prompt``) or the reference answer (``reference``),
        because both of those come from the dataset/task definition, which
        this framework controls; they aren't something the AI wrote, so
        there's nothing in them to blank out or worry about leaking
        (``reporters.base._redact_execution``'s docstring explains the same
        reasoning).

        We blank out secrets before shortening the text, never the other way
        around: shortening first could cut a secret-shaped string in half at
        the cutoff point and let the un-blanked remainder slip through.

        Returns the final text plus a small evidence dict -- it only
        includes a key for something that actually happened (e.g. it won't
        say "redacted" if nothing was found to redact), matching how
        ``HarnessGrader`` handles its own evidence keys.
        """
        stringified = _stringify_output(output)
        evidence: dict[str, JsonValue] = {}

        redacted = self._redact_candidate_output(stringified)
        if redacted != stringified:
            evidence["candidate_output_redacted"] = True

        truncated = self._truncate_candidate_output(redacted)
        if truncated != redacted:
            evidence["candidate_output_truncated"] = True
            evidence["candidate_output_original_chars"] = len(redacted)

        return truncated, evidence

    def _response_evidence(self, response: JudgeResponse) -> dict[str, JsonValue]:
        """Pull out anything worth recording from a parsed response, for humans only.

        Right now that's just the optional ``rationale`` (ADR-0020) -- the
        judge's own free-text explanation of its verdict. Since that text
        could echo back parts of the AI's answer, we blank out secrets and
        shorten it exactly the way we do for the AI's answer itself
        (ADR-0018) before saving it to ``evidence["judge_rationale"]``. We
        only add that key when there actually is a rationale to record, the
        same "don't add a key for nothing" rule the AI's-answer evidence
        keys follow. This is purely a record for a human to read later --
        the grading logic never looks at a rationale (or anything else that
        reads like the judge's confidence) when deciding pass/fail, by
        design (§9 requires objective checks to come before anything
        subjective).
        """
        if response.rationale is None:
            return {}
        redacted_rationale = self._truncate_candidate_output(
            self._redact_candidate_output(response.rationale)
        )
        return {"judge_rationale": redacted_rationale}

    def _redact_candidate_output(self, value: str) -> str:
        """Replace anything that looks like a secret in ``value`` with ``"[REDACTED]"``.

        Just a plain text-in, text-out function. Two other places in this
        codebase do the same thing (``reporters.base._redact_string`` and
        ``EvalRunner._redact`` in ``runner.py``), but this module can't
        import either of those private helpers directly (``runner.py``'s
        own docstring explains why, and the same reasoning applies here),
        so this is the same logic written again locally against the same
        shared :class:`RedactionPolicy` settings.
        """
        redacted = value
        for pattern in self._compiled_secret_patterns():
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted

    def _truncate_candidate_output(self, value: str) -> str:
        """Cut ``value`` down to ``self._max_candidate_output_chars`` and note that we did.

        Pass ``None`` to never cut it short (the actual default,
        ``_DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS``, does have a limit). The
        marker we add at the end makes it obvious -- to a human or to the
        judge model reading this -- that this isn't the complete original
        text.
        """
        limit = self._max_candidate_output_chars
        if limit is None or len(value) <= limit:
            return value
        omitted = len(value) - limit
        return f"{value[:limit]}...[truncated, {omitted} chars omitted]"

    def _compiled_secret_patterns(self) -> tuple[re.Pattern[str], ...]:
        """Turn the configured secret patterns into ready-to-use regexes, or none.

        Works the same way ``EvalRunner._compiled_secret_patterns`` does in
        ``runner.py``: you get nothing back both when redaction was
        explicitly turned off (``None``) and when a policy was given that
        just happens to list no patterns. Normally, though, the default
        policy (:data:`~agentic_evalkit.reporters.base.DEFAULT_REDACTION_POLICY`)
        does list real patterns, so the usual case compiles those.
        """
        if self._redaction_policy is None:
            return ()
        return tuple(re.compile(pattern) for pattern in self._redaction_policy.secret_patterns)

    def _calibration_failure_reason(self, *, now: datetime | None = None) -> str | None:
        if self._calibration is None:
            return "no calibration artifact was supplied; judge is advisory-only"
        if self._calibration.judge_fingerprint != self._judge_fingerprint():
            return (
                f"calibration fingerprint {self._calibration.judge_fingerprint!r} does not "
                f"match live judge fingerprint {self._judge_fingerprint()!r}"
            )
        return self._calibration.usability_failure_reason(now=now)

    async def _position_bias_failure_reason(
        self, request: JudgeRequest, primary: JudgeResponse
    ) -> str | None:
        """Ask the judge again with the two answers swapped, and check it still agrees with itself.

        If the judge changes its mind just because we changed which answer
        it saw first, that's a sign it's biased by order rather than by
        which answer is actually better -- and a judge like that can't be
        trusted to gate a release, even if everything else about its
        calibration checks out. If this second call itself throws an
        exception, that's treated as a reason we can't gate (not a crash
        that takes down the run, ADR-0020) -- and since an exception message
        could contain text from the AI's answer, we blank out secrets and
        shorten it (ADR-0018) before including it in that reason.
        """
        reversed_request = request.model_copy(update={"metadata": {"reversed": True}})
        try:
            reversed_response = await self._judge.judge(reversed_request)
        except Exception as error:
            redacted_message = self._truncate_candidate_output(
                self._redact_candidate_output(str(error))
            )
            return (
                f"position-bias probe raised {type(error).__name__} "
                f"({redacted_message}); cannot confirm order-invariance, so cannot gate"
            )
        if (
            reversed_response.parse_ok
            and not reversed_response.abstained
            and reversed_response.verdict != primary.verdict
        ):
            return (
                "position-bias check failed: verdict changed from "
                f"{primary.verdict!r} to {reversed_response.verdict!r} under "
                "reversed option order"
            )
        return None

    async def _judge_with_bounded_retries(
        self, request: JudgeRequest
    ) -> tuple[JudgeResponse | None, dict[str, JsonValue]]:
        """Call the judge, and only retry if its response didn't parse (at most twice).

        A call that raises an exception is a different problem from a call
        that parses badly, and we don't retry it -- it fails immediately
        instead of hammering the judge with retries, and it doesn't use up
        any of our two allowed parse-retries (ADR-0020). When that happens,
        this returns ``(None, evidence)``, where ``evidence`` records the
        exception's type (``judge_transport_error``) and a shortened,
        secret-blanked version of its message (``judge_transport_error_message``)
        -- an exception message could contain text from the AI's answer, so
        it gets the same treatment as ``candidate_output`` (ADR-0018) before
        we keep it. ``grade`` turns that ``None`` into a single
        ``GradeStatus.ERROR`` for this one sample, so one bad judge call
        never takes down the whole run. We deliberately don't catch
        ``asyncio.CancelledError`` here (it isn't a regular ``Exception``),
        so cancelling a run still actually cancels it.
        """
        attempts = 0
        max_attempts = _MAXIMUM_PARSE_RETRIES + 1
        while attempts < max_attempts:
            attempts += 1
            try:
                response = await self._judge.judge(request)
            except Exception as error:
                redacted_message = self._truncate_candidate_output(
                    self._redact_candidate_output(str(error))
                )
                return None, {
                    "parse_attempts": attempts,
                    "judge_transport_error": type(error).__name__,
                    "judge_transport_error_message": redacted_message,
                }
            if response.parse_ok:
                return response, {"parse_attempts": attempts}
        return None, {"parse_attempts": attempts}

    def _result(
        self,
        sample: EvalSample,
        now: datetime,
        *,
        status: GradeStatus,
        score: float | None,
        hard_gate: bool,
        reason: str | None,
        calibration_ref: str | None = None,
        extra_evidence: dict[str, JsonValue] | None = None,
    ) -> GradeResult:
        evidence: dict[str, JsonValue] = dict(extra_evidence or {})
        if reason is not None:
            evidence["reason"] = reason
        return GradeResult(
            sample_id=sample.sample_id,
            grader=self._name,
            grader_type="judge",
            status=status,
            score=score,
            hard_gate=hard_gate,
            evidence=evidence,
            judge_calibration_ref=calibration_ref,
            created_at=now,
        )


def _stringify_input(payload: dict[str, JsonValue]) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(payload.items()))


def _stringify_output(output: dict[str, JsonValue] | None) -> str:
    if output is None:
        return ""
    return " ".join(f"{key}={value}" for key, value in sorted(output.items()))
