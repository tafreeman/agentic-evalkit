"""Finite-sample score-threshold calibration with exact arithmetic (ADR-0028).

A judge (or any scorer) emits a score; somebody has to pick the cut-off that
turns that score into "accept this answer". ``JudgeGrader`` defaults that
cut-off to ``0.5``, which is a guess. This module replaces the guess with a
threshold *selected on held-out calibration data* together with a stated,
checkable finite-sample guarantee, and it refuses to certify anything the
data cannot support.

What a certificate does and does not say -- read this before relying on one:

- The loss being controlled is ``loss * 1[score is on the accepted side]``:
  the **unconditional** mass of "accepted and wrong". Budget ``alpha`` bounds
  that mass over *all* items, **not** the error rate among the accepted ones,
  so ``alpha = 0.05`` is not "95% precision".
- ``CRC_EXPECTATION`` (conformal risk control) bounds the *expected* loss of a
  fresh item, averaged over the calibration draw as well. It is a marginal
  statement: one particular certificate can be worse than ``alpha``.
- ``RCPS_HIGH_PROB`` bounds the *true* risk by ``alpha`` with probability at
  least ``1 - delta`` over the calibration draw, for binary losses. It is the
  one to use when a certificate is shown to a reader as evidence.
- The two modes assume different things about the data. ``CRC_EXPECTATION``
  needs calibration and deployment items to be *exchangeable* (drawn the same
  way). ``RCPS_HIGH_PROB`` needs more: the items must be *independent* draws
  from the deployment distribution, because its binomial tail assumes the
  number of calibration errors is Binomial. Exchangeable but correlated items
  break it. If every loss were one shared Bernoulli(0.1) coin flip, 59 clean
  calibration items would occur with probability 0.9 and certify a threshold
  whose true risk is 0.1 at ``alpha = delta = 0.05``. Calibration sets with
  repeated attempts, shared prompts or shared contexts are correlated in
  exactly this way, so use one independent unit per item (ADR-0016 explains
  why repeated attempts are not independent).
- Nothing here can check either assumption; a changed judge, model, dataset
  or prompt also voids a certificate. Callers must pair a certificate with the
  provenance that produced it (ADR-0008, ADR-0015).

Scores are compared as floats (floats are totally ordered, so ties are exact).
``alpha``, ``delta`` and losses are converted to exact rationals, and a float
is read as the *decimal the caller wrote* (``0.3`` means ``3/10``, not the
nearest binary double), so a budget written as a decimal literal never moves
across a boundary. A *computed* float (``0.1 + 0.2``) is read as the decimal it
actually prints as, and a fraction with no short decimal (a third) should be
passed as a ``Fraction``.

A certificate is evidence for *choosing* a threshold. It is deliberately not
an input to any grader's hard gate: ``tests/contract/test_conformal_boundary``
pins that, because ADR-0007's gate only ever opens under a ratified
calibration floor.
"""

import math
from collections.abc import Sequence
from enum import StrEnum
from fractions import Fraction

from pydantic import Field, field_serializer, field_validator, model_validator

from agentic_evalkit.graders.calibration import AuthorityLevel
from agentic_evalkit.models.base import FrozenModel

# The high-probability mode does exact big-integer arithmetic whose size grows
# with the calibration set and the budget's denominator. Past this many items a
# hang is likelier than a useful certificate, so refuse loudly instead (an
# absurdly small budget ``alpha`` is almost always a typo).
MAX_EXACT_SAMPLES = 200_000

Rational = int | float | Fraction


# JSON numbers beyond 64 bits do not survive a parse (and Python refuses to turn
# an int past ~4300 digits into decimal text), so exact values that large travel
# as hexadecimal strings. Hex is a power-of-two base, which Python converts
# without that digit limit. Small values stay plain JSON numbers.
_JSON_INT_LIMIT = 2**63


def _encode_big_int(value: int) -> int | str:
    if -_JSON_INT_LIMIT < value < _JSON_INT_LIMIT:
        return value
    return hex(value)


def _decode_big_int(value: object) -> object:
    if isinstance(value, str):
        negative = value.startswith("-")
        digits = value[1:] if negative else value
        if digits.lower().startswith("0x"):
            parsed = int(digits, 16)
            return -parsed if negative else parsed
    return value


class ConformalMode(StrEnum):
    """Which finite-sample guarantee a certificate is making."""

    CRC_EXPECTATION = "crc_expectation"
    RCPS_HIGH_PROB = "rcps_high_prob"


class ExactRational(FrozenModel):
    """A reduced fraction that survives JSON without rounding.

    Attributes:
        numerator: Signed numerator.
        denominator: Strictly positive denominator, in lowest terms.
    """

    numerator: int
    denominator: int = Field(ge=1)

    @field_validator("numerator", "denominator", mode="before")
    @classmethod
    def _read_big_int(cls, value: object) -> object:
        return _decode_big_int(value)

    @field_serializer("numerator", "denominator", when_used="json")
    def _write_big_int(self, value: int) -> int | str:
        return _encode_big_int(value)

    @model_validator(mode="after")
    def _is_reduced(self) -> "ExactRational":
        if math.gcd(self.numerator, self.denominator) != 1:
            raise ValueError("ExactRational must be in lowest terms")
        return self

    @classmethod
    def from_fraction(cls, value: Fraction) -> "ExactRational":
        """Build from a ``Fraction`` (always reduced)."""
        return cls(numerator=value.numerator, denominator=value.denominator)

    def to_fraction(self) -> Fraction:
        """Return the exact value as a ``Fraction``."""
        return Fraction(self.numerator, self.denominator)


class ScoreObservation(FrozenModel):
    """One held-out calibration item: the scorer's score and the realised loss.

    Attributes:
        score: The scorer's output. Finite.
        loss: What accepting this item would cost, in ``[0, 1]``; ``0`` when
            the item is fine, ``1`` when it is wrong. ``RCPS_HIGH_PROB``
            additionally requires exactly ``0`` or ``1``.
    """

    score: float = Field(allow_inf_nan=False)
    loss: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("score")
    @classmethod
    def _no_negative_zero(cls, value: float) -> float:
        # -0.0 == 0.0 for ordering and ties, but not byte-for-byte in JSON; one
        # spelling keeps equal calibrations serialising identically.
        return value + 0.0


class ThresholdCertificate(FrozenModel):
    """The outcome of calibrating a score threshold, and how far it can be trusted.

    ``authority`` follows ADR-0007's two-tier rule: ``GATING`` means a
    threshold was certified; ``ADVISORY`` means the calibration set is too
    small to certify *anything* at this budget (absent evidence); and
    ``UNAVAILABLE`` means enough data exists but even the strictest observed
    threshold breaches the budget, so no observed threshold can be certified
    (accepting nothing always meets a budget, but is not a threshold). Only a
    ``GATING`` certificate has a ``threshold``, and the validator rejects any
    combination of fields that a calibration could not have produced.

    Attributes:
        mode: Which guarantee ``reason`` and the evidence fields refer to.
        authority: How far this certificate may be trusted.
        reason: Why, in words a reviewer can act on.
        higher_is_riskier: ``True`` accepts ``score <= threshold``; ``False``
            accepts ``score >= threshold``.
        alpha: The risk budget.
        delta: The failure probability (``RCPS_HIGH_PROB`` only).
        n_samples: Calibration items used.
        min_required_samples: The fewest items for which *any* certificate is
            possible at this budget, even if every item had zero loss.
        accepted_count: Calibration items on the accepted side of ``threshold``.
        threshold: The certified cut-off, or ``None`` when not ``GATING``.
        empirical_risk: Mean calibration loss of the accepted items over all
            ``n_samples`` (``None`` when not ``GATING``).
        inflated_empirical_risk: ``(sum of accepted losses + 1) / (n + 1)``,
            the quantity conformal risk control keeps at or below ``alpha``
            (``CRC_EXPECTATION``, ``GATING`` only).
        binomial_tail: ``P(Binomial(n, alpha) <= observed errors)``, the
            quantity kept at or below ``delta`` (``RCPS_HIGH_PROB``,
            ``GATING`` only).
    """

    mode: ConformalMode
    authority: AuthorityLevel
    reason: str
    higher_is_riskier: bool
    alpha: ExactRational
    delta: ExactRational | None = None
    n_samples: int = Field(ge=0)
    min_required_samples: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    threshold: float | None = Field(default=None, allow_inf_nan=False)
    empirical_risk: ExactRational | None = None
    inflated_empirical_risk: ExactRational | None = None
    binomial_tail: ExactRational | None = None

    @model_validator(mode="after")
    def _evidence_matches_authority(self) -> "ThresholdCertificate":
        zero, one = Fraction(0), Fraction(1)
        if not zero < self.alpha.to_fraction() < one:
            raise ValueError("alpha must be strictly between 0 and 1")
        if self.delta is not None and not zero < self.delta.to_fraction() < one:
            raise ValueError("delta must be strictly between 0 and 1")
        if (self.mode is ConformalMode.RCPS_HIGH_PROB) != (self.delta is not None):
            raise ValueError("delta must be set exactly for the rcps_high_prob mode")
        # The floor is a function of the budget alone, so recompute it rather than
        # trust the stored value: a forged floor would let a tiny set claim it had
        # enough data (unavailable) or a large one claim it had too little (advisory).
        floor = min_calibration_samples(
            self.alpha.to_fraction(),
            mode=self.mode,
            delta=None if self.delta is None else self.delta.to_fraction(),
        )
        if self.min_required_samples != floor:
            raise ValueError(
                f"min_required_samples is {self.min_required_samples}, but this budget "
                f"needs {floor}"
            )
        if self.authority is AuthorityLevel.GATING:
            self._check_gating()
        else:
            self._check_not_gating()
        return self

    def _check_not_gating(self) -> None:
        evidence = (self.empirical_risk, self.inflated_empirical_risk, self.binomial_tail)
        if self.threshold is not None or self.accepted_count != 0 or any(evidence):
            raise ValueError(
                "a non-gating certificate carries no threshold, acceptance or evidence"
            )
        enough_items = self.n_samples >= self.min_required_samples
        if (self.authority is AuthorityLevel.UNAVAILABLE) != enough_items:
            raise ValueError(
                "advisory means fewer items than the floor; unavailable means at least the floor"
            )

    def _check_gating(self) -> None:
        if self.threshold is None:
            raise ValueError("a gating certificate must carry a threshold")
        if self.n_samples < self.min_required_samples:
            raise ValueError("a gating certificate needs at least min_required_samples items")
        if not 1 <= self.accepted_count <= self.n_samples:
            raise ValueError("accepted_count must be between 1 and n_samples")
        if self.empirical_risk is None or not 0 <= self.empirical_risk.to_fraction() <= 1:
            raise ValueError("a gating certificate must carry an empirical risk in [0, 1]")
        # Every item loses at most 1, so the accepted items cannot lose more than they number.
        loss_sum = self.empirical_risk.to_fraction() * self.n_samples
        if loss_sum > self.accepted_count:
            raise ValueError("empirical_risk is larger than the accepted items could lose")
        if self.mode is ConformalMode.CRC_EXPECTATION:
            self._check_crc_statistic(loss_sum)
        else:
            self._check_rcps_statistic(loss_sum)

    def _check_crc_statistic(self, loss_sum: Fraction) -> None:
        inflated = self.inflated_empirical_risk
        if inflated is None or self.binomial_tail is not None:
            raise ValueError("a crc_expectation certificate carries the inflated risk only")
        if inflated.to_fraction() != (loss_sum + 1) / (self.n_samples + 1):
            raise ValueError("inflated_empirical_risk does not follow from empirical_risk")
        if inflated.to_fraction() > self.alpha.to_fraction():
            raise ValueError("inflated_empirical_risk exceeds alpha")

    def _check_rcps_statistic(self, loss_sum: Fraction) -> None:
        tail = self.binomial_tail
        if tail is None or self.inflated_empirical_risk is not None or self.delta is None:
            raise ValueError("an rcps_high_prob certificate carries the binomial tail only")
        if loss_sum.denominator != 1:
            raise ValueError("an rcps_high_prob empirical risk must be a whole number of errors")
        if self.n_samples > MAX_EXACT_SAMPLES:
            raise ValueError(
                f"{self.n_samples} calibration items is above the exact-arithmetic limit "
                f"of {MAX_EXACT_SAMPLES} for rcps_high_prob"
            )
        expected = _BinomialTail(self.n_samples, self.alpha.to_fraction()).at_most(int(loss_sum))
        if tail.to_fraction() != expected:
            raise ValueError("binomial_tail does not follow from empirical_risk and alpha")
        if tail.to_fraction() > self.delta.to_fraction():
            raise ValueError("binomial_tail exceeds delta")

    def permits(self, score: float) -> bool:
        """Say whether ``score`` is on the accepted side of a certified threshold.

        Fails closed: anything other than a ``GATING`` certificate and a
        finite score returns ``False``. Callers that must tell "too little
        data" from "no certifiable threshold" read :attr:`authority`.
        """
        if self.authority is not AuthorityLevel.GATING or self.threshold is None:
            return False
        if not math.isfinite(score):
            return False
        if self.higher_is_riskier:
            return score <= self.threshold
        return score >= self.threshold


def _exact(value: Rational, *, name: str) -> Fraction:
    """Read a number as the exact rational its decimal spelling denotes."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        # float.__repr__ so float subclasses (numpy scalars) read as plain decimals.
        return Fraction(float.__repr__(value))
    raise TypeError(f"{name} must be int, float or Fraction, got {type(value).__name__}")


def _open_unit(value: Rational, *, name: str) -> Fraction:
    exact = _exact(value, name=name)
    if not Fraction(0) < exact < Fraction(1):
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value!r}")
    return exact


def min_calibration_samples(
    alpha: Rational,
    *,
    mode: ConformalMode = ConformalMode.CRC_EXPECTATION,
    delta: Rational | None = None,
) -> int:
    """Return the fewest calibration items for which *any* certificate exists.

    This is the floor: with fewer items than this, even a calibration set in
    which every item has zero loss cannot certify the budget.

    - ``CRC_EXPECTATION``: the smallest ``n`` with ``1 / (n + 1) <= alpha``.
    - ``RCPS_HIGH_PROB``: the smallest ``n`` with ``(1 - alpha) ** n <= delta``
      (zero observed errors must already be enough evidence).

    Raises:
        ValueError: ``alpha`` or ``delta`` is outside ``(0, 1)``, ``delta`` is
            missing for ``RCPS_HIGH_PROB``, or the floor exceeds
            :data:`MAX_EXACT_SAMPLES`.
    """
    alpha_q = _open_unit(alpha, name="alpha")
    if mode is ConformalMode.CRC_EXPECTATION:
        if delta is not None:
            raise ValueError("delta only applies to the rcps_high_prob mode")
        # Smallest n with 1/(n+1) <= p/q is ceil(q/p) - 1 == (q - 1) // p.
        return (alpha_q.denominator - 1) // alpha_q.numerator
    if delta is None:
        raise ValueError("delta is required for the rcps_high_prob mode")
    delta_q = _open_unit(delta, name="delta")
    keep = Fraction(1) - alpha_q
    if keep <= delta_q:
        return 1
    if _floor_certainly_exceeds_limit(alpha_q, delta_q):
        raise ValueError(_floor_limit_message(alpha, delta))
    # Exact search, no floating point: grow until (1 - alpha) ** n clears delta,
    # then bisect between the last failing and first passing n.
    failing, passing = 1, 2
    while keep**passing > delta_q:
        if passing >= MAX_EXACT_SAMPLES:
            raise ValueError(_floor_limit_message(alpha, delta))
        failing, passing = passing, min(passing * 2, MAX_EXACT_SAMPLES)
    while passing - failing > 1:
        middle = (failing + passing) // 2
        if keep**middle <= delta_q:
            passing = middle
        else:
            failing = middle
    return passing


def _floor_limit_message(alpha: Rational, delta: Rational) -> str:
    return (
        f"alpha={alpha!r} at delta={delta!r} needs more than {MAX_EXACT_SAMPLES} calibration "
        "items, above the exact-arithmetic limit"
    )


def _floor_certainly_exceeds_limit(alpha: Fraction, delta: Fraction) -> bool:
    """Cheap lower bound on the floor, so absurd budgets fail before any big pow.

    ``(1 - alpha) ** n <= delta`` needs ``n >= ln(1/delta) / -ln(1 - alpha)``, and
    ``-ln(1 - alpha) <= alpha / (1 - alpha)``, so ``n >= ln(1/delta) * (1 - alpha) / alpha``.
    """
    log_inverse_delta = math.log(delta.denominator) - math.log(delta.numerator)
    approx_alpha = float(alpha)
    if approx_alpha == 0.0:
        return True
    try:
        bound = log_inverse_delta * (1.0 - approx_alpha) / approx_alpha
    except OverflowError:
        return True
    return bound > MAX_EXACT_SAMPLES


class _BinomialTail:
    """Exact ``P(Binomial(n, alpha) <= k)``, evaluated at non-decreasing ``k``.

    Terms are carried as integers over the common denominator ``b ** n`` for
    ``alpha = a / b``, so there is no beta function, no bisection and no
    rounding: ``P(Bin(n, alpha) <= k) <= delta`` is *equivalent* to the
    Clopper-Pearson upper limit for ``k`` errors being at most ``alpha``.
    Comparisons stay in integers (no ``Fraction`` normalisation per query); a
    ``Fraction`` is built once, for the threshold that wins.

    Attributes:
        numerator: The cumulative probability's numerator at the current ``k``.
        denominator: ``b ** n``, the common denominator.
    """

    def __init__(self, n: int, alpha: Fraction) -> None:
        self._n = n
        self._a = alpha.numerator
        self._c = alpha.denominator - alpha.numerator
        self.denominator: int = alpha.denominator**n
        self._term: int = self._c**n  # numerator of P(X == 0)
        self.numerator: int = self._term
        self._k = 0

    def _advance(self, k: int) -> None:
        target = min(k, self._n)
        if target < self._k:
            raise ValueError("_BinomialTail must be queried with non-decreasing k")
        while self._k < target:
            self._term = self._term * (self._n - self._k) * self._a // ((self._k + 1) * self._c)
            self.numerator += self._term
            self._k += 1

    def clears(self, k: int, delta: Fraction) -> bool:
        """Say whether ``P(Bin(n, alpha) <= k) <= delta``; ``k`` must not decrease."""
        self._advance(k)
        return self.numerator * delta.denominator <= delta.numerator * self.denominator

    def at_most(self, k: int) -> Fraction:
        """Return ``P(Bin(n, alpha) <= k)`` exactly; ``k`` must not decrease."""
        self._advance(k)
        return Fraction(self.numerator, self.denominator)


def calibrate_score_threshold(
    observations: Sequence[ScoreObservation],
    *,
    alpha: Rational,
    mode: ConformalMode = ConformalMode.CRC_EXPECTATION,
    delta: Rational | None = None,
    higher_is_riskier: bool = True,
) -> ThresholdCertificate:
    """Pick the loosest score threshold the calibration data can certify.

    Items are swept from the strictest threshold outward. Items tied on score
    are accepted or rejected together, never split. The result is the loosest
    threshold whose statistic still clears the bar. Both the empirical loss and
    the true risk only grow as the threshold loosens, so certifying the loosest
    such threshold certifies every stricter one at once and no multiplicity
    correction is needed within one call. Calibrating several strata or
    several budgets at once needs the caller to split ``delta`` (Bonferroni)
    itself.

    Args:
        observations: Held-out calibration items.
        alpha: The risk budget, strictly between 0 and 1. Floats are read as
            the decimal spelled (``0.3`` is ``3/10``); pass a ``Fraction`` for
            budgets that have no short decimal.
        mode: Which guarantee to certify. See the module docstring.
        delta: Failure probability; required for ``RCPS_HIGH_PROB`` and
            rejected for ``CRC_EXPECTATION``.
        higher_is_riskier: ``True`` for a risk score (accept ``score <=
            threshold``); ``False`` for a confidence score (accept ``score >=
            threshold``).

    Returns:
        A :class:`ThresholdCertificate`; check its ``authority`` before use.

    Raises:
        ValueError: A budget is out of range, ``delta`` is missing or
            unexpected for ``mode``, an ``RCPS_HIGH_PROB`` loss is not exactly
            ``0`` or ``1``, or an ``RCPS_HIGH_PROB`` floor or calibration set
            exceeds :data:`MAX_EXACT_SAMPLES`.
    """
    alpha_q = _open_unit(alpha, name="alpha")
    rcps = mode is ConformalMode.RCPS_HIGH_PROB
    if rcps and delta is None:
        raise ValueError("delta is required for the rcps_high_prob mode")
    if not rcps and delta is not None:
        raise ValueError("delta only applies to the rcps_high_prob mode")
    delta_q = _open_unit(delta, name="delta") if delta is not None else None

    n = len(observations)
    required = min_calibration_samples(alpha_q, mode=mode, delta=delta_q)
    swept = sorted(
        (
            (
                obs.score if higher_is_riskier else -obs.score,
                obs.score,
                _exact(obs.loss, name="loss"),
            )
            for obs in observations
        ),
        key=lambda item: item[0],
    )
    if rcps and any(loss not in (0, 1) for _, _, loss in swept):
        raise ValueError("rcps_high_prob requires every loss to be exactly 0 or 1")

    def refuse(authority: AuthorityLevel, reason: str) -> ThresholdCertificate:
        return ThresholdCertificate(
            mode=mode,
            authority=authority,
            reason=reason,
            higher_is_riskier=higher_is_riskier,
            alpha=ExactRational.from_fraction(alpha_q),
            delta=None if delta_q is None else ExactRational.from_fraction(delta_q),
            n_samples=n,
            min_required_samples=required,
            accepted_count=0,
        )

    if n < required:
        return refuse(
            AuthorityLevel.ADVISORY,
            f"insufficient_evidence: {n} calibration items, at least {required} are "
            f"needed before budget alpha={alpha_q} can be certified",
        )

    if rcps and n > MAX_EXACT_SAMPLES:
        raise ValueError(
            f"{n} calibration items is above the exact-arithmetic limit of "
            f"{MAX_EXACT_SAMPLES} for rcps_high_prob"
        )
    tail = _BinomialTail(n, alpha_q) if rcps else None
    best: tuple[float, int, Fraction, int] | None = None
    cumulative = Fraction(0)
    last_errors = -1
    tail_numerator = 0
    index = 0
    while index < n:
        bucket_key = swept[index][0]
        bucket_score = swept[index][1]
        while index < n and swept[index][0] == bucket_key:
            cumulative += swept[index][2]
            index += 1
        if tail is not None and delta_q is not None:
            errors = int(cumulative)
            if errors != last_errors:  # the statistic only moves when errors do
                if not tail.clears(errors, delta_q):
                    break
                last_errors = errors
                tail_numerator = tail.numerator
        elif (cumulative + 1) / (n + 1) > alpha_q:
            break
        best = (bucket_score, index, cumulative, tail_numerator)

    if best is None:
        return refuse(
            AuthorityLevel.UNAVAILABLE,
            f"no_certifiable_threshold: even the strictest observed threshold breaches "
            f"alpha={alpha_q} on {n} calibration items",
        )

    threshold, accepted, loss_sum, winning_numerator = best
    return ThresholdCertificate(
        mode=mode,
        authority=AuthorityLevel.GATING,
        reason="certified",
        higher_is_riskier=higher_is_riskier,
        alpha=ExactRational.from_fraction(alpha_q),
        delta=None if delta_q is None else ExactRational.from_fraction(delta_q),
        n_samples=n,
        min_required_samples=required,
        accepted_count=accepted,
        threshold=threshold,
        empirical_risk=ExactRational.from_fraction(loss_sum / n),
        inflated_empirical_risk=(
            None if rcps else ExactRational.from_fraction((loss_sum + 1) / (n + 1))
        ),
        binomial_tail=(
            ExactRational.from_fraction(Fraction(winning_numerator, tail.denominator))
            if tail is not None
            else None
        ),
    )
