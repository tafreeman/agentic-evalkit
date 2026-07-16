"""The proof that an AI judge is trustworthy enough to be relied on (design §9, ADR-0007, ADR-0020).

This used to live inside ``graders.judge``, but that file grew past this
project's 800-line-per-file limit once ADR-0020 added more checks to it. So
this class was moved here instead -- it doesn't depend on anything else in
``judge.py``, it's just a data model (a ``FrozenModel``) plus some plain
methods that answer "is this calibration still good enough to trust?"
``judge.py`` re-exports everything below under its own name, so existing
code that writes ``from agentic_evalkit.graders.judge import
CalibrationArtifact``, or that reaches for ``judge.PROJECT_MIN_TNR``
directly, keeps working exactly as before -- nothing outside this package
needs to change because of the move.
"""

from datetime import UTC, datetime, timedelta

from pydantic import model_validator

from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.stats import wilson_interval

# A calibration needs at least this many real "right" answers and this many
# real "wrong" answers tested by hand before we trust it enough to gate a
# release on.
_MINIMUM_CLASS_SAMPLE_COUNT = 30

# The project's own minimum bar for judge accuracy (decision D-1, made
# 2026-07-04). A caller can configure `CalibrationArtifact.threshold` to be
# stricter, but never looser than this -- every calibration has to clear
# ALL of these numbers before it's allowed to gate a release, no matter what
# its own `threshold` says.
PROJECT_MIN_TNR = 0.95
PROJECT_MIN_TPR = 0.85
PROJECT_MAX_CALIBRATION_AGE_DAYS = 90


class CalibrationArtifact(FrozenModel):
    """The record of how well this judge did against real, human-checked answers.

    The idea: before a run, someone (or some process) fed the judge a batch
    of examples where we already know the right answer, and recorded how
    often the judge got it right. This class is that record.

    Attributes:
        calibration_id: A stable name for this particular calibration run,
            so you can point back to it later.
        judge_fingerprint: A hash identifying the exact model + prompt this
            calibration was measured against. If today's judge has a
            different fingerprint, this calibration doesn't apply to it and
            can't be used to gate anything.
        expires_at: Once we're past this timestamp, this calibration is too
            old to gate a release, no matter how good its numbers were.
        calibrated_at: When these numbers were actually measured. This is
            optional (leaving it out doesn't break older calibration
            records, since ``schema_version`` stays ``"1"``) -- but if it's
            missing, we can't prove the calibration isn't stale, so it's
            treated as unusable for gating (decision D-1).
        true_positive: How many times, out of the held-out test examples,
            the judge correctly said "this is a good answer" when it really
            was good.
        true_negative: How many times the judge correctly said "this is a
            bad answer" when it really was bad.
        false_positive: How many times the judge said "good" when the
            answer was actually bad.
        false_negative: How many times the judge said "bad" when the answer
            was actually good.
        threshold: The judge's own configured pass bar for these
            true-positive/true-negative rates -- see ``PROJECT_MIN_TNR``/
            ``PROJECT_MIN_TPR`` above for the project-wide minimum this
            can't go below.
        total_labeled: How many held-out examples were tested in total, for
            the record. Optional (added later by ADR-0020, so leaving it out
            doesn't break older calibration records); nothing currently
            checks it before gating.
        abstained_count: Of those examples, how many the judge declined to
            answer during calibration. Same optional/record-only status as
            ``total_labeled``.
        error_count: Of those examples, how many the judge errored out on
            during calibration. Same optional/record-only status as
            ``total_labeled``.
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
        # These three fields were added later (ADR-0020) and are optional --
        # `None` just means "nobody recorded this" -- so we only check them
        # for being non-negative when a value is actually present, instead
        # of lumping them in with the always-required counts above.
        for optional_field_name in ("total_labeled", "abstained_count", "error_count"):
            optional_value = getattr(self, optional_field_name)
            if optional_value is not None and optional_value < 0:
                raise ValueError(f"{optional_field_name} must be non-negative")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {self.threshold}")
        return self

    @model_validator(mode="after")
    def _validate_calibrated_at(self) -> "CalibrationArtifact":
        """Reject any timestamp we wouldn't be able to safely compare against the clock later.

        A timestamp with no timezone attached (a "naive" datetime) would
        crash when we later try to compare it against the current UTC time
        -- and crashing is not an acceptable way to say "this calibration
        isn't good enough" (D-1's whole point is to fail safely, not to
        fail with a stack trace). ``expires_at`` is required on every
        calibration, so we always check it; ``calibrated_at`` is optional,
        so we only check its timezone when it's actually present. We also
        reject a calibration that claims to have been measured *after* its
        own expiry date -- that's contradictory data, not a real
        calibration.
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
        """Return why this calibration is too old to gate on, or ``None`` if its age is fine.

        If we don't know when this calibration was measured
        (``calibrated_at`` is missing), we can't prove it's recent enough,
        so it can't gate -- someone can't dodge the age requirement just by
        leaving the timestamp out. Either way, an age problem only blocks
        gating; the judge can still give an advisory grade (D-1, as amended
        2026-07-04: not knowing is treated differently from knowing it's
        bad).
        """
        if self.calibrated_at is None:
            return (
                f"calibration {self.calibration_id!r} has no calibrated_at; "
                f"cannot verify age within {PROJECT_MAX_CALIBRATION_AGE_DAYS} days"
            )
        effective_now = now or datetime.now(UTC)
        if self.calibrated_at > effective_now:
            # A calibration dated in the future is bad data, not just "very
            # fresh" -- if we let this through, `effective_now -
            # calibrated_at` would come out negative, which would never
            # trip the "too old" check below, so an impossible timestamp
            # would silently look trustworthy. The earlier validator only
            # rejects a `calibrated_at` that's after `expires_at`; a future
            # date that's still comfortably before expiry slips past that
            # check, so we catch it here instead.
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
        """Return why this calibration's accuracy is below our project-wide minimum, or ``None``.

        ``None`` covers two different "it's fine" cases: the accuracy is
        actually at or above our minimum, or there simply aren't enough
        held-out samples yet to say either way (with too few samples, the
        number itself is just noise -- that gets reported separately by
        ``usability_failure_reason`` instead). But if there ARE enough
        samples and the accuracy still falls short, that's solid proof the
        judge isn't good enough, and the result gets marked
        ``GradeStatus.UNAVAILABLE`` outright -- nobody can configure a
        looser ``threshold`` to get around this project-wide floor.
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
        """Check whether the accuracy numbers hold up even in a conservative worst case.

        This is a different, stricter check than :meth:`floor_failure_reason`.
        That method asks "is the raw accuracy number itself below our
        minimum?" -- and if so, that's solid proof of a bad judge
        (``UNAVAILABLE``). This method instead asks: "even if the raw number
        looks fine, is it based on so few examples that we can't really
        trust it?" We answer that using a "Wilson lower bound" -- a
        standard statistics technique that, given a rate and a sample size,
        computes a conservative floor for what the true rate could plausibly
        be. If even that conservative floor clears our minimum, we're
        confident; if it doesn't, the judge might still be fine, we just
        don't have enough evidence yet. That "not enough evidence" case
        blocks gating, but -- unlike an outright-bad accuracy number -- it
        doesn't mark the result ``UNAVAILABLE`` (see
        :meth:`usability_failure_reason`, where this check runs alongside
        the age check, ADR-0020, updating ADR-0007's original
        raw-accuracy-only version of this check). We import the actual math
        for this (:func:`~agentic_evalkit.stats.wilson_interval`) instead of
        rewriting it here, since it's already public and importing it
        doesn't create a circular import.
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
        """Return why this calibration can't gate a release, or ``None`` if it can."""
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
        # The age check lives here, in this one place everyone has to go
        # through to gate a release, so nobody can accidentally skip it
        # (D-1, as amended: a missing or stale calibration date blocks
        # gating but the judge can still give an advisory grade). The raw
        # PROJECT_MIN_TNR/TPR accuracy check deliberately does NOT live here
        # -- a genuinely bad accuracy number is worse than "not enough
        # evidence," so it's handled separately, as an outright UNAVAILABLE,
        # over in `JudgeGrader.grade`. The Wilson-lower-bound check, on the
        # other hand, IS "not enough evidence" rather than "proof it's bad,"
        # exactly like the age check, so it belongs right here next to it
        # (ADR-0020). We check age first, since a stale or missing date is a
        # separate problem from anything about the actual accuracy numbers.
        age_reason = self.age_failure_reason(now=now)
        if age_reason is not None:
            return age_reason
        return self.wilson_lower_bound_failure_reason()
