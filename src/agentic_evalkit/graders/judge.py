"""Calibrated model judge grading (design §9, plan Task 10 Steps 6-9).

"Model judges require a versioned calibration artifact containing model and
prompt fingerprints, held-out human labels, confusion matrix, TPR, TNR,
sample counts, thresholds, subgroup results when available, and expiry
policy. An uncalibrated or expired judge cannot gate a release" (design
§9). "Judge execution supports structured output, bounded retries,
parse-failure reporting, position/order checks, and abstention. The
framework never silently converts judge errors into task failures" (design
§9).

``JudgeGrader`` enforces every one of those conditions before it will ever
set ``hard_gate=True`` on a returned :class:`GradeResult`:

- the judge's transport call must not raise (a raised transport error is an
  operational failure, mapped to ``GradeStatus.ERROR`` for that one sample,
  never an aborted run -- ADR-0008 / ADR-0020);
- the judge's structured response must carry an ``OK``
  :class:`JudgeResponseStatus` -- a ``REFUSED`` envelope is a non-verdict and
  maps to ``GradeStatus.ABSTAIN``; a ``TIMEOUT``/``RATE_LIMITED``/``ERROR``
  envelope is operational and maps to ``GradeStatus.ERROR`` (ADR-0020);
- the judge's live ``fingerprint`` must equal ``CalibrationArtifact.judge_fingerprint``;
- the calibration must not be expired (an expired artifact is unusable
  outright: the result is ``GradeStatus.UNAVAILABLE``);
- the ratified project floor must hold on the point estimates: TNR >= 0.95 and
  TPR >= 0.85 (decision D-1) -- a sub-floor calibration is affirmatively bad
  evidence and likewise demotes the result to ``GradeStatus.UNAVAILABLE``, so a
  lax caller-supplied ``threshold`` can never gate below the floor;
- the 95% Wilson *lower bound* of TNR and TPR must each clear the same project
  floor: a point estimate can sit above the floor while its lower bound does
  not, meaning the held-out sample is too small to *prove* the rate clears the
  floor. Such a calibration is not affirmatively bad (that is the UNAVAILABLE
  demotion above) but its evidence is insufficient to gate, so -- like an
  absent/stale age -- it blocks gating while advisory grading continues
  (ADR-0020, superseding ADR-0007's point-estimate-only floor);
- the calibration must be dated (timezone-aware ``calibrated_at``) and within
  the ratified maximum age of 90 days -- an undated or stale artifact cannot
  prove its age, so it can never gate, though it may still grade advisorily
  (D-1 as amended 2026-07-04: absent evidence blocks gating; only
  affirmatively bad evidence is UNAVAILABLE);
- held-out positives (TP+FN) and negatives (TN+FP) must each be >= 30;
- TPR and TNR must each be >= ``CalibrationArtifact.threshold``;
- a reversed-order ("position-bias") probe must agree with the primary verdict.
  The probe is issued *only* on the gating path -- when ``gate=True`` and the
  calibration is usable -- so the uncalibrated/advisory path costs exactly one
  judge call per sample (ADR-0020); the probe is kept for calibrated ``FAIL``
  samples too, so a position-bias ``reason`` survives even when the primary
  verdict is not ``PASS``. A probe transport raise is itself a gate-blocking
  reason, never a propagated exception;
- the judge must return a parseable, non-abstained structured response
  (parse failures retry at most twice, i.e. three attempts total; a transport
  raise terminates immediately and does *not* consume the parse-retry budget).

Any single failed condition demotes the result to an advisory (non-gating)
outcome; several are also distinct non-PASS statuses so the *reason* survives
into ``evidence["reason"]`` rather than being collapsed into a bare FAIL. When
the judge supplies a ``rationale`` it is persisted -- redacted and truncated
exactly as ``candidate_output`` is (ADR-0018) -- to ``evidence["judge_rationale"]``
purely as recorded evidence; it is never read by any gating decision.
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

# Parse failures retry at most this many times (plan Task 10 KEY
# REQUIREMENTS: "parse retries <= 2"), i.e. up to 3 total judge calls.
_MAXIMUM_PARSE_RETRIES = 2

# Approximate cap, in characters (not bytes -- an approximation, not a
# byte-precise UTF-8 bound; that level of precision isn't needed for a
# truncation heuristic), on the stringified `candidate_output` sent to a
# caller-supplied `JudgeClient` (ADR-0018). Mirrors `runner.py`'s own
# `_LARGE_OUTPUT_THRESHOLD_BYTES` (8192) as an already-reasoned, familiar
# bound; defined independently here rather than imported, since that
# constant is private to `runner.py`.
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
    """Transport/response envelope for one :class:`JudgeResponse` (ADR-0020).

    A fixed vocabulary -- a ``StrEnum`` rather than a boolean or free string,
    per the wire-status rule ADR-0002 applies everywhere -- so a caller-supplied
    ``JudgeClient`` can signal *why* a verdict is absent without collapsing the
    distinction into a task pass/fail. ``OK`` is the only status that reaches
    verdict grading; every other status short-circuits to a non-gating outcome
    in :meth:`JudgeGrader.grade` (``REFUSED`` -> ``GradeStatus.ABSTAIN`` because
    a refusal is a non-verdict, never a task failure; ``RATE_LIMITED``/
    ``TIMEOUT``/``ERROR`` -> ``GradeStatus.ERROR`` because they are operational
    failures kept separate from task failures, ADR-0008).
    """

    OK = "ok"
    REFUSED = "refused"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    ERROR = "error"


class JudgeResponse(FrozenModel):
    """A judge's structured verdict for one :class:`JudgeRequest`.

    ``parse_ok=False`` means the judge's raw output could not be parsed
    into a structured verdict at all (never treated as a task failure --
    see module docstring). ``abstained=True`` means the judge parsed fine
    but explicitly declined to render a verdict.

    ``status`` is the response envelope (ADR-0020): additive, optional, and
    defaulting to :attr:`JudgeResponseStatus.OK` so every existing
    ``JudgeClient`` that never sets it keeps its exact prior meaning
    (``schema_version`` stays ``"1"``). ``rationale`` is optional free-text the
    judge may attach explaining its verdict; it is recorded as evidence only
    (redacted/truncated first) and is never read by any gating decision.
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
    """Grades an execution using a calibrated (or advisory-only) model judge.

    Args:
        judge: The provider-neutral :class:`JudgeClient` to query.
        calibration: The calibration artifact backing this judge
            configuration, or ``None`` if uncalibrated. An uncalibrated
            judge always grades in advisory mode (``hard_gate`` is always
            ``False``, and the ``GradeResult`` carries no calibration
            reference).
        gate: Caller's intent to allow this judge to gate a release *if*
            every calibration/consistency condition is met. Even when
            ``True``, any single failed condition demotes the result to
            advisory-only.
        pass_score_threshold: Minimum judge ``score`` counted as a pass,
            once a verdict is otherwise trustworthy.
        name: Stable grader identifier reported on the ``GradeResult``.
        redaction_policy: Policy applied to the stringified
            ``candidate_output`` before it is sent to the judge (never to
            ``prompt`` or ``reference`` -- see the note below). Defaults to
            :data:`~agentic_evalkit.reporters.base.DEFAULT_REDACTION_POLICY`,
            so secret-shaped substrings never reach a caller-supplied
            ``JudgeClient`` by default. Pass ``None`` (or a
            ``RedactionPolicy()`` with empty ``secret_patterns``) to opt out.
        max_candidate_output_chars: Maximum length, in characters, of the
            stringified ``candidate_output`` sent to the judge; longer
            values are truncated with a trailing marker. Defaults to
            ``_DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS`` (8192). Pass ``None`` to
            disable truncation.

    Since ADR-0018, the ``candidate_output`` a ``JudgeClient`` actually
    receives may differ from ``execution.output``'s literal content:
    secret-shaped substrings are redacted and an overlong string is
    truncated before it crosses the process boundary into ``judge.judge()``.
    A ``JudgeClient`` implementation must not assume byte-for-byte fidelity
    between the two. ``prompt`` and ``reference`` are never altered -- both
    are dataset/task-authored content this framework itself controls, not
    target-controlled output.

    Judge-call cost (ADR-0020): the reversed-order position-bias probe -- a
    second judge call -- is issued *only* on the gating path, i.e. when
    ``gate=True`` **and** the calibration is usable (no calibration failure).
    The uncalibrated/advisory path (``gate=False``, or any calibration failure)
    therefore makes exactly one judge call per graded sample, not two. A judge
    transport call that raises is mapped to a single ``GradeStatus.ERROR``
    sample -- one raising ``JudgeClient`` never aborts the whole run (ADR-0008).
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
            # `prompt`/`reference` are dataset/task-authored content this
            # framework itself controls, never the system-under-test's own
            # output -- only `candidate_output` is redacted/truncated before
            # crossing the process boundary (see `_prepare_candidate_output`).
            prompt=_stringify_input(sample.input),
            candidate_output=candidate_output,
            reference=sample.reference,
        )

        response, retry_evidence = await self._judge_with_bounded_retries(request)
        if response is None:
            return self._missing_response_result(
                sample, now, retry_evidence, candidate_output_evidence
            )

        # A parsed response may still carry a rationale (recorded evidence
        # only, redacted/truncated like candidate_output -- ADR-0018 -- and
        # never read by gating).
        response_evidence = self._response_evidence(response)

        # Each check below returns a non-gating `GradeResult` the instant one
        # applies, or `None` to fall through to the next -- checked via `is
        # not None`, deliberately not truthiness, so this sequence can never
        # be fooled by a future `GradeResult.__bool__`/`__len__` override.
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
        # Position-bias probe on the gating path only (ADR-0020): issued when
        # ``gate=True`` AND the calibration is usable (no calibration failure),
        # regardless of whether the primary verdict is PASS -- so a calibrated
        # FAIL sample is still probed and a position-bias ``reason`` survives.
        # The advisory path (``gate=False``, or any calibration failure) makes
        # exactly one judge call per sample, never a second probe call.
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
        """Build the ERROR result for a transport raise or exhausted parse retries.

        A transport raise and a parse-retry exhaustion both land here but are
        distinct operational failures (ADR-0020). The bounded-retries helper
        records ``judge_transport_error`` on the former and only
        ``parse_attempts`` on the latter, so one raising ``JudgeClient``
        yields one graded ERROR sample rather than aborting the run.
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
        """Return the non-gating result for a non-OK envelope, or ``None`` if OK.

        Short-circuits BEFORE fingerprint/abstention handling (ADR-0020): a
        refusal is a non-verdict (ABSTAIN, never a task FAIL); a
        rate-limit/timeout/error envelope is operational (ERROR, kept
        separate from task failure per ADR-0008). Never gating either way;
        the reason names the status.
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
        """Return an ERROR result if ``response``'s fingerprint doesn't match the live judge's."""
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
        """Return an ABSTAIN result if the judge explicitly abstained, else ``None``."""
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
        """Return an UNAVAILABLE result if the calibration is affirmatively bad, else ``None``.

        Affirmatively bad calibration evidence makes grading capability
        UNAVAILABLE outright (plan Task 10 Step 6 for expiry; ratified
        decision D-1 for the project floor): an expired artifact is
        unusable, and a sub-floor TNR/TNR overclaim is precisely the
        failure R-003 exists to prevent. This runs BEFORE the advisory path
        so bad evidence never yields an advisory PASS. Absent or
        insufficient evidence (undated/stale ``calibrated_at``, or a Wilson
        lower bound below the floor) is different: it only blocks gating,
        via ``usability_failure_reason`` elsewhere (D-1 as amended
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
        """Stringify, redact, then truncate ``execution.output`` for the judge.

        Applied only to ``candidate_output`` -- the system-under-test's own
        words. Deliberately NOT applied to ``prompt``
        (``_stringify_input(sample.input)``) or ``reference``
        (``sample.reference``): both are dataset/task-authored content this
        framework itself controls, not target-controlled output, mirroring
        the same target's-own-words-vs-framework-authored-content
        distinction ``reporters.base._redact_execution``'s docstring draws
        for precisely this reason.

        Redaction runs BEFORE truncation, never the reverse: truncating
        first risks cutting a secret-shaped pattern in half at the boundary
        and letting the un-redacted remainder through.

        Returns the final string plus an evidence dict that only carries
        ``candidate_output_redacted``/``candidate_output_truncated``/
        ``candidate_output_original_chars`` keys when that step actually
        fired -- mirroring ``HarnessGrader``'s "only add the key when
        applicable" convention for ``evidence["harness_error"]``.
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
        """Recorded (non-gating) evidence extracted from a parsed response.

        Currently just the optional ``rationale`` (ADR-0020): a judge's own
        free-text explanation of its verdict. Being judge output that can echo
        target-controlled content, it is redacted then truncated exactly as
        ``candidate_output`` is (ADR-0018) before it is persisted to
        ``evidence["judge_rationale"]``. Only added when a rationale is present,
        mirroring the "only add the key when applicable" convention the
        candidate-output evidence keys already follow. This is evidence only:
        no gating decision ever reads a rationale (or any confidence-like
        content) -- design §9's objective-first ordering forbids it.
        """
        if response.rationale is None:
            return {}
        redacted_rationale = self._truncate_candidate_output(
            self._redact_candidate_output(response.rationale)
        )
        return {"judge_rationale": redacted_rationale}

    def _redact_candidate_output(self, value: str) -> str:
        """Replace every secret-pattern match in ``value`` with ``"[REDACTED]"``.

        A pure string -> string function mirroring
        ``agentic_evalkit.reporters.base._redact_string`` and
        ``EvalRunner._redact`` (``runner.py``): this module cannot import
        either private helper (``runner.py``'s own docstring explains why it
        doesn't import ``reporters.base``'s -- the same reasoning applies
        here), so the same substitution behavior is reimplemented locally
        against the same public :class:`RedactionPolicy` contract.
        """
        redacted = value
        for pattern in self._compiled_secret_patterns():
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted

    def _truncate_candidate_output(self, value: str) -> str:
        """Cut ``value`` to ``self._max_candidate_output_chars`` plus a marker.

        ``None`` disables truncation entirely (the class default,
        ``_DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS``, does not). The appended
        marker makes it unambiguous to a human or the judge model that this
        is not the verbatim full output.
        """
        limit = self._max_candidate_output_chars
        if limit is None or len(value) <= limit:
            return value
        omitted = len(value) - limit
        return f"{value[:limit]}...[truncated, {omitted} chars omitted]"

    def _compiled_secret_patterns(self) -> tuple[re.Pattern[str], ...]:
        """Compile ``self._redaction_policy.secret_patterns``, or none at all.

        Mirrors ``EvalRunner._compiled_secret_patterns`` (``runner.py``):
        returns an empty tuple both when the policy was explicitly set to
        ``None`` (opting out of judge-bound redaction) and when a policy was
        supplied with no ``secret_patterns`` of its own. The constructor
        default is :data:`~agentic_evalkit.reporters.base.DEFAULT_REDACTION_POLICY`,
        which does carry patterns, so the ordinary path compiles those.
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
        """Issue a reversed-order probe and require verdict agreement.

        A judge whose verdict flips when option order is swapped exhibits
        position bias and can never gate, even with otherwise-valid
        calibration (plan Task 10 Step 8). A probe transport raise is itself a
        gate-blocking reason and never propagates out to abort the run
        (ADR-0020): the raised message is redacted then truncated (ADR-0018)
        before it is surfaced, since an exception message can echo target
        output.
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
        """Call the judge, retrying only *parse* failures (at most twice).

        A transport raise is not a parse failure: it terminates immediately
        (no retry storm) and does NOT consume the parse-retry budget (ADR-0020).
        On a raise this returns ``(None, evidence)`` where ``evidence`` carries
        ``judge_transport_error`` (the exception type name) and a bounded,
        redacted ``judge_transport_error_message`` -- an exception message can
        echo target-controlled output, so it is redacted then truncated exactly
        as ``candidate_output`` is (ADR-0018) before being persisted.
        ``grade`` maps that sentinel to a single ``GradeStatus.ERROR`` sample,
        so one raising ``JudgeClient`` never aborts the whole run.
        ``asyncio.CancelledError`` (a ``BaseException``, not an ``Exception``)
        is deliberately *not* caught, so run cancellation still propagates.
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
