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

:func:`judge_authority` and its two result types live here too, and were
moved down from ``integrations.base`` by ADR-0024. They started there
because a host-platform export was the first thing that needed to label a
judge's authority, but nothing about the decision is integration-specific:
it reads a :class:`CalibrationArtifact` and nothing else, and it encodes
ADR-0007's decision D-1, which is a grading rule. Leaving it there would
have meant the ``calibrate`` command importing from
``agentic_evalkit.integrations``, which ADR-0022 forbids -- its dependency
arrow points outward only. ``integrations.base`` re-exports all three names,
so every existing import path still resolves and the public surface is
unchanged.
"""

from datetime import UTC, datetime, timedelta
from enum import StrEnum

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


class AuthorityLevel(StrEnum):
    """How much a judge's verdict is allowed to decide, given its calibration evidence.

    This is the three-way outcome of ADR-0007's decision D-1 (as amended
    2026-07-04), named so it can travel into a host platform's metadata as
    a label rather than as a bare boolean. The whole reason it has three
    members instead of two is that "we proved this judge is bad" and "we
    have no proof either way" are different facts about the world, and
    collapsing them loses the distinction that makes the control honest.

    - ``GATING``: full calibration evidence clears every floor, so this
      judge's verdict may block a release.
    - ``ADVISORY``: evidence is *absent* or too thin to prove reliability
      (no artifact, no ``calibrated_at``, too few held-out samples, a
      Wilson lower bound that does not clear the floor, a fingerprint that
      does not match the live judge). The verdict is still reported --
      it may well be right -- but it can never gate.
    - ``UNAVAILABLE``: evidence is *present and bad* (expired, or a
      measured TNR/TPR genuinely below the project floor on a sufficient
      sample). There is proof this judge should not be trusted here, so
      no verdict is reported at all.
    """

    GATING = "gating"
    ADVISORY = "advisory"
    UNAVAILABLE = "unavailable"


class JudgeAuthority(FrozenModel):
    """The verdict on a judge's own evidence, and why it came out that way.

    Attributes:
        level: How much this judge's verdict may decide -- see
            :class:`AuthorityLevel`.
        reason: Why, in words a reviewer can act on. ``None`` only when
            ``level`` is :attr:`AuthorityLevel.GATING`, where there is no
            failure to explain.
        calibration_id: Which calibration record produced this verdict.
            ``None`` when no artifact was supplied at all.
    """

    level: AuthorityLevel
    reason: str | None = None
    calibration_id: str | None = None


def judge_authority(
    calibration: CalibrationArtifact | None,
    *,
    judge_fingerprint: str | None = None,
    now: datetime | None = None,
) -> JudgeAuthority:
    """Decide what a judge's calibration evidence entitles its verdict to do.

    This is the one place that decision is made, so it is worth being exact
    about what it is and is not. It does not score anything, does not call a
    model, and does not look at a verdict -- it looks only at the *evidence
    about the judge* and returns how far that evidence reaches. A platform
    that already has a judge it likes can wrap it with this and gain the one
    property the judge was missing: an inability to block a release it has
    not earned the right to block. The ``calibrate`` command calls it on the
    artifact it has just measured, for the same reason and by the same route
    (ADR-0024).

    The order of checks below is load-bearing and matches
    :class:`~agentic_evalkit.graders.judge.JudgeGrader` exactly. Proof of a
    *bad* judge is established first, before anything else, so a judge with
    expired or sub-floor calibration can never slip out as an advisory pass;
    only after that does the weaker "not enough proof" family get
    considered. Every threshold is read from :class:`CalibrationArtifact`
    itself rather than recomputed here, so the project floors (TNR >= 0.95,
    TPR >= 0.85, age <= 90 days, the 95% Wilson lower bound) have exactly
    one definition in this codebase and this function cannot drift from it.

    Args:
        calibration: The judge's calibration record, or ``None`` if it has
            none. ``None`` is not an error -- it is the ordinary case for a
            judge someone just wrote, and it yields
            :attr:`AuthorityLevel.ADVISORY`.
        judge_fingerprint: The live judge's fingerprint (model + prompt). If
            given and it does not match what the calibration was measured
            against, the calibration describes a different judge and cannot
            gate this one. Left ``None`` to skip that check, which is right
            when the host platform does not expose a stable fingerprint.
        now: The moment to evaluate expiry and age against. Defaults to the
            current UTC time; tests pass a fixed value.

    Returns:
        A :class:`JudgeAuthority` carrying the level and the reason.
    """
    effective_now = now or datetime.now(UTC)

    if calibration is None:
        return JudgeAuthority(
            level=AuthorityLevel.ADVISORY,
            reason="no calibration artifact was supplied; judge is advisory-only",
        )

    # Tier one: solid proof the judge is unreliable. Expiry first, then a
    # measured rate genuinely below the project floor -- the two cases where
    # the evidence exists and is bad, rather than merely being thin.
    if calibration.is_expired(now=effective_now):
        return JudgeAuthority(
            level=AuthorityLevel.UNAVAILABLE,
            reason=(
                f"calibration {calibration.calibration_id!r} expired at "
                f"{calibration.expires_at.isoformat()}"
            ),
            calibration_id=calibration.calibration_id,
        )
    floor_reason = calibration.floor_failure_reason()
    if floor_reason is not None:
        return JudgeAuthority(
            level=AuthorityLevel.UNAVAILABLE,
            reason=floor_reason,
            calibration_id=calibration.calibration_id,
        )

    # Tier two: the evidence is not bad, it is just not enough. A judge here
    # still reports its verdict; it simply may not gate on it.
    if judge_fingerprint is not None and calibration.judge_fingerprint != judge_fingerprint:
        return JudgeAuthority(
            level=AuthorityLevel.ADVISORY,
            reason=(
                f"calibration fingerprint {calibration.judge_fingerprint!r} does not "
                f"match live judge fingerprint {judge_fingerprint!r}"
            ),
            calibration_id=calibration.calibration_id,
        )
    usability_reason = calibration.usability_failure_reason(now=effective_now)
    if usability_reason is not None:
        return JudgeAuthority(
            level=AuthorityLevel.ADVISORY,
            reason=usability_reason,
            calibration_id=calibration.calibration_id,
        )

    return JudgeAuthority(
        level=AuthorityLevel.GATING,
        calibration_id=calibration.calibration_id,
    )
