"""Calibration evidence for a judge configuration (design §9, ADR-0007, ADR-0020).

Split out of ``graders.judge`` so that module stays under this project's
800-line file ceiling after ADR-0020 added the Wilson lower-bound floor and
response envelope; ``CalibrationArtifact`` is fully self-contained (a
``FrozenModel`` plus pure validators/properties/failure-reason methods) and
has no dependency on anything else in ``judge.py``. ``judge.py`` re-exports
every public name below, so ``from agentic_evalkit.graders.judge import
CalibrationArtifact`` (used throughout this package and by external callers)
and module-qualified access (``judge.PROJECT_MIN_TNR``, used in tests)
continue to resolve unchanged.
"""

from datetime import UTC, datetime, timedelta

from pydantic import model_validator

from agentic_evalkit.models.base import FrozenModel
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
