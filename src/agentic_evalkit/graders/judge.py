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
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import Field, JsonValue, model_validator

from agentic_evalkit.models import EvalSample, GradeResult, GradeStatus, NormalizedExecutionResult
from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY, RedactionPolicy
from agentic_evalkit.stats import wilson_interval

# Minimum held-out positive/negative label counts a calibration must have
# before it is trusted to gate a release (plan Task 10 KEY REQUIREMENTS).
_MINIMUM_CLASS_SAMPLE_COUNT = 30

# Ratified project calibration floor (decision D-1, 2026-07-04). A caller-
# supplied ``CalibrationArtifact.threshold`` may be laxer than these, but it
# can never lower the bar below the project minimums: a calibration must clear
# ALL of these before it may hard-gate a release, independent of its own
# ``threshold``.
PROJECT_MIN_TNR = 0.95
PROJECT_MIN_TPR = 0.85
PROJECT_MAX_CALIBRATION_AGE_DAYS = 90

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


class CalibrationArtifact(FrozenModel):
    """Held-out human-labeled calibration evidence for one judge configuration.

    Attributes:
        calibration_id: Stable identifier for this calibration run.
        judge_fingerprint: Fingerprint of the exact model+prompt combination
            this calibration is valid for. A live judge with a different
            fingerprint can never use this artifact to gate.
        expires_at: Timestamp after which this calibration is stale and
            must not gate, regardless of how strong its historical TPR/TNR
            were.
        calibrated_at: Timestamp the held-out labels were collected. Optional
            and additive (schema_version stays "1"); when absent the artifact
            cannot prove it is within the ratified maximum age and is treated
            as unusable for gating (decision D-1).
        true_positive/true_negative/false_positive/false_negative: Confusion
            matrix counts from held-out human-labeled samples.
        threshold: Minimum TPR *and* TNR this calibration must clear.
        total_labeled/abstained_count/error_count: Additive, optional coverage
            evidence (ADR-0020) recording how many held-out samples were labeled
            in total and how many the judge abstained on or errored on during
            calibration. Recorded for auditability only -- no gate reads them
            yet; each defaults to ``None`` so ``schema_version`` stays ``"1"``,
            and non-negative when present.
    """

    calibration_id: str
    judge_fingerprint: str
    expires_at: datetime
    calibrated_at: datetime | None = None
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    threshold: float
    total_labeled: int | None = None
    abstained_count: int | None = None
    error_count: int | None = None

    @model_validator(mode="after")
    def _validate_counts(self) -> "CalibrationArtifact":
        for field_name in (
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        # Additive ADR-0020 coverage fields: non-negative when supplied, but
        # optional (``None`` means "not recorded"), so they are checked only
        # when present rather than folded into the required-count loop above.
        for optional_field_name in ("total_labeled", "abstained_count", "error_count"):
            optional_value = getattr(self, optional_field_name)
            if optional_value is not None and optional_value < 0:
                raise ValueError(f"{optional_field_name} must be non-negative")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {self.threshold}")
        return self

    @model_validator(mode="after")
    def _validate_calibrated_at(self) -> "CalibrationArtifact":
        """Reject timestamps that cannot support the age floor or expiry check.

        A naive ``expires_at`` or ``calibrated_at`` would make the age/expiry
        arithmetic against the UTC clock raise at grade time -- and a crash is
        not a demotion (D-1 is fail-closed, never fail-crashed). ``expires_at``
        is required on every artifact (unlike optional ``calibrated_at``), so
        it is validated unconditionally. A calibration taken after its own
        expiry is self-contradictory data and is rejected outright.
        """
        if self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        if self.calibrated_at is None:
            return self
        if self.calibrated_at.tzinfo is None:
            raise ValueError("calibrated_at must be timezone-aware")
        if self.calibrated_at > self.expires_at:
            raise ValueError("calibrated_at must not be after expires_at")
        return self

    @property
    def positive_count(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def negative_count(self) -> int:
        return self.true_negative + self.false_positive

    @property
    def true_positive_rate(self) -> float | None:
        if self.positive_count == 0:
            return None
        return self.true_positive / self.positive_count

    @property
    def true_negative_rate(self) -> float | None:
        if self.negative_count == 0:
            return None
        return self.true_negative / self.negative_count

    def is_expired(self, *, now: datetime | None = None) -> bool:
        return self.expires_at <= (now or datetime.now(UTC))

    def age_failure_reason(self, *, now: datetime | None = None) -> str | None:
        """Return why this calibration's age disqualifies it from gating, or
        ``None`` if it is dated and within the ratified maximum (decision D-1).

        An artifact with no ``calibrated_at`` cannot prove its age, so it can
        never gate -- a laxer caller cannot bypass the age floor by simply
        omitting the timestamp. Age failures block gating only; advisory
        grading continues (D-1 as amended 2026-07-04: absent evidence is not
        the same as affirmatively bad evidence).
        """
        if self.calibrated_at is None:
            return (
                f"calibration {self.calibration_id!r} has no calibrated_at; "
                f"cannot verify age within {PROJECT_MAX_CALIBRATION_AGE_DAYS} days"
            )
        effective_now = now or datetime.now(UTC)
        if self.calibrated_at > effective_now:
            # A future-dated calibration is self-contradictory evidence, not
            # merely "fresh": ``effective_now - calibrated_at`` would be
            # negative and never exceed the max-age bound below, silently
            # treating an impossible timestamp as trustworthy. Construction
            # only rejects calibrated_at *after* expires_at, which does not
            # catch a future calibrated_at still comfortably before expiry.
            return (
                f"calibration {self.calibration_id!r} calibrated_at "
                f"{self.calibrated_at.isoformat()} is in the future"
            )
        if effective_now - self.calibrated_at > timedelta(days=PROJECT_MAX_CALIBRATION_AGE_DAYS):
            return (
                f"calibration {self.calibration_id!r} age exceeds the maximum of "
                f"{PROJECT_MAX_CALIBRATION_AGE_DAYS} days"
            )
        return None

    def floor_failure_reason(self) -> str | None:
        """Return why this calibration sits below the ratified project floor
        (decision D-1), or ``None`` when it clears the floor or its rates are
        not yet statistically meaningful.

        A sub-floor calibration is affirmatively bad evidence: the judge's
        result demotes to ``GradeStatus.UNAVAILABLE`` outright, so a lax
        caller-supplied ``threshold`` can never gate below the project
        minimums. With fewer than the minimum held-out samples per class the
        rates are noise, not evidence, so the floor defers to
        ``usability_failure_reason``'s insufficient-sample report.
        """
        if (
            self.positive_count < _MINIMUM_CLASS_SAMPLE_COUNT
            or self.negative_count < _MINIMUM_CLASS_SAMPLE_COUNT
        ):
            return None
        tnr = self.true_negative_rate
        if tnr is not None and tnr < PROJECT_MIN_TNR:
            return f"calibration TNR={tnr} is below the project minimum {PROJECT_MIN_TNR}"
        tpr = self.true_positive_rate
        if tpr is not None and tpr < PROJECT_MIN_TPR:
            return f"calibration TPR={tpr} is below the project minimum {PROJECT_MIN_TPR}"
        return None

    def wilson_lower_bound_failure_reason(self) -> str | None:
        """Return why the 95% Wilson lower bound of TNR/TPR fails the floor, or
        ``None`` when both bounds clear it (ADR-0020, superseding ADR-0007).

        Distinct from :meth:`floor_failure_reason`: a *point* estimate below the
        floor is affirmatively bad evidence (UNAVAILABLE); a point estimate that
        clears the floor while its 95% Wilson *lower* bound does not is merely
        *insufficient* evidence -- the held-out sample is too small to prove the
        rate is above the floor. Insufficient evidence blocks gating only, so
        this reason is surfaced through :meth:`usability_failure_reason`
        alongside the age check while advisory grading continues. The
        :func:`~agentic_evalkit.stats.wilson_interval` helper is imported rather
        than reimplemented (unlike ``runner._redact``, which cannot import its
        sibling's private helper): ``wilson_interval`` is public
        (``agentic_evalkit.stats.__all__``) and ``stats`` imports nothing from
        ``graders``, so there is no import cycle.
        """
        tnr_lower, _ = wilson_interval(successes=self.true_negative, total=self.negative_count)
        if tnr_lower is not None and tnr_lower < PROJECT_MIN_TNR:
            return (
                f"calibration TNR 95% Wilson lower bound {tnr_lower:.4f} is below the project "
                f"minimum {PROJECT_MIN_TNR}: insufficient held-out evidence to gate"
            )
        tpr_lower, _ = wilson_interval(successes=self.true_positive, total=self.positive_count)
        if tpr_lower is not None and tpr_lower < PROJECT_MIN_TPR:
            return (
                f"calibration TPR 95% Wilson lower bound {tpr_lower:.4f} is below the project "
                f"minimum {PROJECT_MIN_TPR}: insufficient held-out evidence to gate"
            )
        return None

    def usability_failure_reason(self, *, now: datetime | None = None) -> str | None:
        """Return a human-readable reason this calibration cannot gate, or
        ``None`` if it clears every usability bar (design §9 / plan Task 10).
        """
        if self.is_expired(now=now):
            return f"calibration {self.calibration_id!r} expired at {self.expires_at.isoformat()}"
        if self.positive_count < _MINIMUM_CLASS_SAMPLE_COUNT:
            return (
                f"calibration has {self.positive_count} held-out positive samples, "
                f"below the required minimum of {_MINIMUM_CLASS_SAMPLE_COUNT}"
            )
        if self.negative_count < _MINIMUM_CLASS_SAMPLE_COUNT:
            return (
                f"calibration has {self.negative_count} held-out negative samples, "
                f"below the required minimum of {_MINIMUM_CLASS_SAMPLE_COUNT}"
            )
        tpr = self.true_positive_rate
        if tpr is None or tpr < self.threshold:
            return f"calibration TPR={tpr} is below threshold={self.threshold}"
        tnr = self.true_negative_rate
        if tnr is None or tnr < self.threshold:
            return f"calibration TNR={tnr} is below threshold={self.threshold}"
        # The age floor lives here, in the documented "can this calibration
        # gate" seam, so no caller can bypass it (D-1 as amended: undated or
        # stale artifacts never gate but may still grade advisorily). The
        # PROJECT_MIN_TNR/TPR *point* floor is deliberately NOT here -- a
        # sub-floor point estimate is unusable outright and is enforced as
        # UNAVAILABLE via ``floor_failure_reason`` in ``JudgeGrader.grade``.
        # The Wilson *lower-bound* floor, by contrast, is insufficient-evidence
        # (not affirmatively-bad) and blocks gating only, exactly like the age
        # check, so it belongs here alongside it (ADR-0020). Age is reported
        # first: a stale or undated artifact fails for a reason independent of
        # the confusion-matrix counts.
        age_reason = self.age_failure_reason(now=now)
        if age_reason is not None:
            return age_reason
        return self.wilson_lower_bound_failure_reason()


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
            # No usable response: a transport raise and a parse-retry
            # exhaustion both land here but are distinct operational failures
            # (ADR-0020). The bounded-retries helper records
            # ``judge_transport_error`` on the former and only
            # ``parse_attempts`` on the latter, so one raising ``JudgeClient``
            # yields one graded ERROR sample rather than aborting the run.
            if "judge_transport_error" in retry_evidence:
                none_reason = "judge transport call raised before a verdict was returned"
            else:
                none_reason = "judge response could not be parsed after bounded retries"
            return self._result(
                sample,
                now,
                status=GradeStatus.ERROR,
                score=None,
                hard_gate=False,
                reason=none_reason,
                extra_evidence={**candidate_output_evidence, **retry_evidence},
            )

        # A parsed response may still carry a rationale (recorded evidence
        # only, redacted/truncated like candidate_output -- ADR-0018 -- and
        # never read by gating).
        response_evidence = self._response_evidence(response)

        if response.status is not JudgeResponseStatus.OK:
            # Non-OK envelope short-circuits BEFORE fingerprint/abstention
            # handling (ADR-0020): a refusal is a non-verdict (ABSTAIN, never a
            # task FAIL); a rate-limit/timeout/error envelope is operational
            # (ERROR, kept separate from task failure per ADR-0008). Never
            # gating either way; the reason names the status.
            if response.status is JudgeResponseStatus.REFUSED:
                envelope_status = GradeStatus.ABSTAIN
                envelope_reason = (
                    f"judge declined to render a verdict (status {response.status.value!r})"
                )
            else:
                envelope_status = GradeStatus.ERROR
                envelope_reason = (
                    f"judge reported an operational failure (status {response.status.value!r})"
                )
            return self._result(
                sample,
                now,
                status=envelope_status,
                score=None,
                hard_gate=False,
                reason=envelope_reason,
                extra_evidence={
                    **candidate_output_evidence,
                    **retry_evidence,
                    **response_evidence,
                },
            )

        if response.fingerprint != self._judge_fingerprint():
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

        if response.abstained:
            return self._result(
                sample,
                now,
                status=GradeStatus.ABSTAIN,
                score=None,
                hard_gate=False,
                reason="judge explicitly abstained from rendering a verdict",
                extra_evidence={**candidate_output_evidence, **response_evidence},
            )

        if self._calibration is not None:
            # Affirmatively bad calibration evidence makes grading capability
            # UNAVAILABLE outright (plan Task 10 Step 6 for expiry; ratified
            # decision D-1 for the project floor): an expired artifact is
            # unusable, and a sub-floor TNR/TNR overclaim is precisely the
            # failure R-003 exists to prevent. This runs BEFORE the advisory
            # path so bad evidence never yields an advisory PASS. Absent or
            # insufficient evidence (undated/stale ``calibrated_at``, or a
            # Wilson lower bound below the floor) is different: it only blocks
            # gating, via ``usability_failure_reason`` below (D-1 as amended
            # 2026-07-04; ADR-0020).
            if self._calibration.is_expired(now=now):
                unusable_reason: str | None = (
                    f"calibration {self._calibration.calibration_id!r} expired at "
                    f"{self._calibration.expires_at.isoformat()}"
                )
            else:
                unusable_reason = self._calibration.floor_failure_reason()
            if unusable_reason is not None:
                return self._result(
                    sample,
                    now,
                    status=GradeStatus.UNAVAILABLE,
                    score=None,
                    hard_gate=False,
                    reason=unusable_reason,
                    extra_evidence={**candidate_output_evidence, **response_evidence},
                )

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
        if reversed_response.parse_ok and not reversed_response.abstained:
            if reversed_response.verdict != primary.verdict:
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
