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

- the judge's live ``fingerprint`` must equal ``CalibrationArtifact.judge_fingerprint``;
- the calibration must not be expired;
- the calibration must be dated and within the ratified maximum age of 90 days;
- held-out positives (TP+FN) and negatives (TN+FP) must each be >= 30;
- TPR and TNR must each be >= ``CalibrationArtifact.threshold``;
- the ratified project floor must hold: TNR >= 0.95 and TPR >= 0.85 (decision
  D-1), so a lax caller-supplied ``threshold`` can never gate below it;
- a reversed-order ("position-bias") probe must agree with the primary verdict;
- the judge must return a parseable, non-abstained structured response
  (parse failures retry at most twice, i.e. three attempts total).

Any single failed condition demotes the result to an advisory (non-gating)
outcome; several are also distinct non-PASS statuses so the *reason* survives
into ``evidence["reason"]`` rather than being collapsed into a bare FAIL.
"""

from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from pydantic import Field, JsonValue, model_validator

from agentic_evalkit.models import EvalSample, GradeResult, GradeStatus, NormalizedExecutionResult
from agentic_evalkit.models.base import FrozenModel

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


class JudgeResponse(FrozenModel):
    """A judge's structured verdict for one :class:`JudgeRequest`.

    ``parse_ok=False`` means the judge's raw output could not be parsed
    into a structured verdict at all (never treated as a task failure --
    see module docstring). ``abstained=True`` means the judge parsed fine
    but explicitly declined to render a verdict.
    """

    fingerprint: str
    verdict: str
    score: float | None
    parse_ok: bool
    abstained: bool


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
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {self.threshold}")
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
        """Return why this calibration is too old (or undated) to gate, or
        ``None`` if its age is within the ratified maximum (decision D-1).

        An artifact with no ``calibrated_at`` cannot prove its age and is
        rejected outright, so a laxer caller cannot bypass the age floor by
        simply omitting the timestamp.
        """
        if self.calibrated_at is None:
            return (
                f"calibration {self.calibration_id!r} has no calibrated_at; "
                f"cannot verify age within {PROJECT_MAX_CALIBRATION_AGE_DAYS} days"
            )
        if (now or datetime.now(UTC)) - self.calibrated_at > timedelta(
            days=PROJECT_MAX_CALIBRATION_AGE_DAYS
        ):
            return (
                f"calibration {self.calibration_id!r} age exceeds the maximum of "
                f"{PROJECT_MAX_CALIBRATION_AGE_DAYS} days"
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
        # Enforce the ratified project floor AFTER the artifact's own
        # threshold: a lax caller ``threshold`` can never lower the bar below
        # these project minimums (decision D-1). ``tpr``/``tnr`` are non-None
        # here since both class counts are >= 30.
        if tnr < PROJECT_MIN_TNR:
            return f"calibration TNR={tnr} is below the project minimum {PROJECT_MIN_TNR}"
        if tpr < PROJECT_MIN_TPR:
            return f"calibration TPR={tpr} is below the project minimum {PROJECT_MIN_TPR}"
        return None


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
    """

    def __init__(
        self,
        judge: JudgeClient,
        *,
        calibration: CalibrationArtifact | None,
        gate: bool,
        pass_score_threshold: float = 0.5,
        name: str = "judge@1",
    ) -> None:
        self._judge = judge
        self._calibration = calibration
        self._gate = gate
        self._pass_score_threshold = pass_score_threshold
        self._name = name

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        now = datetime.now(UTC)
        candidate_output = _stringify_output(execution.output)
        request = JudgeRequest(
            sample_id=sample.sample_id,
            prompt=_stringify_input(sample.input),
            candidate_output=candidate_output,
            reference=sample.reference,
        )

        response, parse_evidence = await self._judge_with_bounded_retries(request)
        if response is None:
            return self._result(
                sample,
                now,
                status=GradeStatus.ERROR,
                score=None,
                hard_gate=False,
                reason="judge response could not be parsed after bounded retries",
                extra_evidence=parse_evidence,
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
            )

        if response.abstained:
            return self._result(
                sample,
                now,
                status=GradeStatus.ABSTAIN,
                score=None,
                hard_gate=False,
                reason="judge explicitly abstained from rendering a verdict",
            )

        if self._calibration is not None:
            # An expired, stale (>90d), or undated calibration is not merely
            # "untrustworthy for gating" (hard_gate=False) -- the calibration
            # artifact itself is unusable, so grading capability is UNAVAILABLE
            # outright (plan Task 10 Step 6 for expiry; decision D-1 for the
            # ratified age floor). This runs BEFORE the usability/floor path so
            # an unusable artifact never yields an advisory PASS.
            if self._calibration.is_expired(now=now):
                stale_reason: str | None = (
                    f"calibration {self._calibration.calibration_id!r} expired at "
                    f"{self._calibration.expires_at.isoformat()}"
                )
            else:
                stale_reason = self._calibration.age_failure_reason(now=now)
            if stale_reason is not None:
                return self._result(
                    sample,
                    now,
                    status=GradeStatus.UNAVAILABLE,
                    score=None,
                    hard_gate=False,
                    reason=stale_reason,
                )

        calibration_failure = self._calibration_failure_reason()
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
            extra_evidence=parse_evidence,
        )

    def _judge_fingerprint(self) -> str:
        return self._judge.fingerprint

    def _calibration_failure_reason(self) -> str | None:
        if self._calibration is None:
            return "no calibration artifact was supplied; judge is advisory-only"
        if self._calibration.judge_fingerprint != self._judge_fingerprint():
            return (
                f"calibration fingerprint {self._calibration.judge_fingerprint!r} does not "
                f"match live judge fingerprint {self._judge_fingerprint()!r}"
            )
        return self._calibration.usability_failure_reason()

    async def _position_bias_failure_reason(
        self, request: JudgeRequest, primary: JudgeResponse
    ) -> str | None:
        """Issue a reversed-order probe and require verdict agreement.

        A judge whose verdict flips when option order is swapped exhibits
        position bias and can never gate, even with otherwise-valid
        calibration (plan Task 10 Step 8).
        """
        reversed_request = request.model_copy(update={"metadata": {"reversed": True}})
        reversed_response = await self._judge.judge(reversed_request)
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
        attempts = 0
        max_attempts = _MAXIMUM_PARSE_RETRIES + 1
        while attempts < max_attempts:
            attempts += 1
            response = await self._judge.judge(request)
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
