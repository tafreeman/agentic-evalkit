"""Tests for finite-sample score-threshold calibration (ADR-0028)."""

import itertools
import math
import random
from fractions import Fraction
from typing import ClassVar

import pytest
from pydantic import ValidationError

import agentic_evalkit.graders.conformal as conformal_module
from agentic_evalkit.graders.calibration import AuthorityLevel
from agentic_evalkit.graders.conformal import (
    ConformalMode,
    ExactRational,
    Rational,
    ScoreObservation,
    ThresholdCertificate,
    calibrate_score_threshold,
    min_calibration_samples,
)

CRC = ConformalMode.CRC_EXPECTATION
RCPS = ConformalMode.RCPS_HIGH_PROB


def _obs(*pairs: tuple[float, float]) -> list[ScoreObservation]:
    return [ScoreObservation(score=score, loss=loss) for score, loss in pairs]


def _clean(count: int) -> list[ScoreObservation]:
    return _obs(*((index / count, 0.0) for index in range(count)))


class TestExactRational:
    def test_round_trips_through_json_without_rounding(self) -> None:
        value = ExactRational.from_fraction(Fraction(3, 10))
        restored = ExactRational.model_validate_json(value.model_dump_json())
        assert restored.to_fraction() == Fraction(3, 10)

    def test_rejects_unreduced_and_non_positive_denominators(self) -> None:
        with pytest.raises(ValidationError):
            ExactRational(numerator=2, denominator=4)
        with pytest.raises(ValidationError):
            ExactRational(numerator=1, denominator=0)


class TestMinCalibrationSamples:
    def test_crc_floor_is_exactly_the_smallest_n_with_one_over_n_plus_one_within_budget(
        self,
    ) -> None:
        for numerator, denominator in itertools.product(range(1, 12), range(2, 40)):
            if numerator >= denominator:
                continue
            alpha = Fraction(numerator, denominator)
            floor = min_calibration_samples(alpha)
            assert Fraction(1, floor + 1) <= alpha
            assert floor == 0 or Fraction(1, floor) > alpha

    def test_floats_are_read_as_the_decimal_that_was_typed(self) -> None:
        assert min_calibration_samples(0.05) == min_calibration_samples(Fraction(1, 20)) == 19

    def test_a_float_is_its_decimal_so_exact_thirds_need_a_fraction(self) -> None:
        assert min_calibration_samples(Fraction(1, 3)) == 2
        # 0.3333333333333333 is a hair below one third, so it asks for one more item.
        assert min_calibration_samples(1 / 3) == 3

    def test_crc_rejects_a_delta_it_cannot_use(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            min_calibration_samples(0.05, delta=0.05)

    def test_rcps_floor_is_the_smallest_n_whose_zero_error_tail_clears_delta(self) -> None:
        floor = min_calibration_samples(0.05, mode=RCPS, delta=0.05)
        assert floor == 59
        assert Fraction(95, 100) ** floor <= Fraction(5, 100) < Fraction(95, 100) ** (floor - 1)

    @pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5, float("nan"), float("inf")])
    def test_rejects_budgets_outside_the_open_unit_interval(self, bad: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            min_calibration_samples(bad)

    def test_rejects_bool_budget(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            min_calibration_samples(True)

    def test_rcps_needs_delta_and_refuses_absurd_floors(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            min_calibration_samples(0.05, mode=RCPS)
        with pytest.raises(ValueError, match="exact-arithmetic limit"):
            min_calibration_samples(1e-9, mode=RCPS, delta=0.05)


class TestCrcCalibration:
    def test_too_few_items_is_advisory_never_gating(self) -> None:
        cert = calibrate_score_threshold(_clean(18), alpha=0.05)
        assert cert.authority is AuthorityLevel.ADVISORY
        assert cert.threshold is None
        assert cert.min_required_samples == 19
        assert not cert.permits(0.0)

    def test_empty_calibration_set_is_advisory(self) -> None:
        cert = calibrate_score_threshold([], alpha=0.5)
        assert cert.authority is AuthorityLevel.ADVISORY
        assert cert.n_samples == 0

    def test_budget_at_the_floor_certifies_a_clean_set(self) -> None:
        cert = calibrate_score_threshold(_clean(19), alpha=0.05)
        assert cert.authority is AuthorityLevel.GATING
        assert cert.accepted_count == 19
        assert cert.inflated_empirical_risk is not None
        assert cert.inflated_empirical_risk.to_fraction() == Fraction(1, 20)

    def test_the_decimal_budget_is_not_moved_by_binary_rounding(self) -> None:
        # Seven clean items, then two that fail: the inflated risk after all nine is
        # exactly 3/10. float(0.3) is below 3/10, so reading the double instead of the
        # decimal would refuse the last two items.
        data = _obs(*((i / 10, 0.0) for i in range(7)), (0.7, 1.0), (0.8, 1.0))
        by_float = calibrate_score_threshold(data, alpha=0.3)
        by_fraction = calibrate_score_threshold(data, alpha=Fraction(3, 10))
        assert by_float == by_fraction
        assert by_float.accepted_count == 9
        assert by_float.threshold == 0.8

    def test_sweep_stops_at_the_first_uncertifiable_threshold(self) -> None:
        # A bad item at 0.5 breaches the budget; a clean run above it must not rescue it.
        data = _obs(
            *((i / 100, 0.0) for i in range(40)),
            (0.5, 1.0),
            *((0.6 + i / 100, 0.0) for i in range(40)),
        )
        cert = calibrate_score_threshold(data, alpha=Fraction(1, 50))
        assert cert.authority is AuthorityLevel.GATING
        assert cert.threshold == pytest.approx(0.39)
        assert cert.accepted_count == 40

    def test_strictest_threshold_breaching_the_budget_is_unavailable(self) -> None:
        data = _obs((0.0, 1.0), *((i / 100, 0.0) for i in range(1, 30)))
        cert = calibrate_score_threshold(data, alpha=0.05)
        assert cert.n_samples >= cert.min_required_samples
        assert cert.authority is AuthorityLevel.UNAVAILABLE
        assert cert.threshold is None
        assert not cert.permits(0.0)

    def test_tied_scores_are_accepted_or_rejected_together(self) -> None:
        data = _obs(*((0.1, 0.0) for _ in range(20)), (0.2, 1.0), (0.2, 0.0))
        cert = calibrate_score_threshold(data, alpha=Fraction(1, 20))
        assert cert.threshold == 0.1
        assert cert.accepted_count == 20  # the 0.2 pair is never split

    def test_confidence_scores_accept_the_high_side(self) -> None:
        data = _obs(*((1 - i / 100, 0.0) for i in range(30)), (0.4, 1.0))
        cert = calibrate_score_threshold(data, alpha=Fraction(1, 10), higher_is_riskier=False)
        assert cert.authority is AuthorityLevel.GATING
        assert cert.permits(1.0)
        assert cert.permits(cert.threshold or 0.0)
        assert not cert.permits(0.0)

    def test_delta_is_rejected_where_it_means_nothing(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            calibrate_score_threshold(_clean(30), alpha=0.1, delta=0.05)

    def test_calibration_is_deterministic_and_order_independent(self) -> None:
        data = _obs(*((i % 7 / 7, float(i % 5 == 0)) for i in range(60)))
        shuffled = list(data)
        random.Random(11).shuffle(shuffled)  # noqa: S311 -- seeded test data, not security
        assert calibrate_score_threshold(data, alpha=0.4) == calibrate_score_threshold(
            shuffled, alpha=0.4
        )


def _held_out_average(
    points: list[tuple[float, Fraction]], alpha: Fraction, *, higher_is_riskier: bool
) -> Fraction:
    """Average held-out loss over every choice of which point plays the test item."""
    total = Fraction(0)
    for held in range(len(points)):
        rest = [p for index, p in enumerate(points) if index != held]
        cert = calibrate_score_threshold(
            [ScoreObservation(score=s, loss=float(loss)) for s, loss in rest],
            alpha=alpha,
            higher_is_riskier=higher_is_riskier,
        )
        score, loss = points[held]
        total += loss if cert.permits(score) else Fraction(0)
    return total / len(points)


class TestCrcExpectationGuarantee:
    """The conclusion of conformal risk control, checked exactly and with heavy ties.

    For any fixed multiset of ``n + 1`` items and any budget ``alpha >= 1 / (n + 1)``,
    the loss of the left-out item, averaged over which item is left out, is at most
    ``alpha``. This is the finite-sample statement the expectation bound rests on
    (exchangeability makes the leave-one-out average equal the expectation). It is
    checked here against the *shipped* solver -- inclusive acceptance, tied scores
    and all -- over a tie-heavy score grid.
    """

    @pytest.mark.parametrize("higher_is_riskier", [True, False])
    def test_leave_one_out_average_never_exceeds_the_budget(self, higher_is_riskier: bool) -> None:
        rng = random.Random(20260726)  # noqa: S311 -- seeded test data, not security
        losses = (Fraction(0), Fraction(1, 2), Fraction(1))
        checked = 0
        for _ in range(250):
            size = rng.randint(3, 9)
            points = [(float(rng.randint(0, 3)), rng.choice(losses)) for _ in range(size)]
            floor = Fraction(1, size)  # n + 1 == size
            for alpha in (floor, floor + Fraction(1, 10), Fraction(1, 2), Fraction(9, 10)):
                if not floor <= alpha < 1:
                    continue
                average = _held_out_average(points, alpha, higher_is_riskier=higher_is_riskier)
                assert average <= alpha, (points, alpha)
                checked += 1
        assert checked > 400


class TestRcpsCalibration:
    def test_floor_then_certify(self) -> None:
        assert (
            calibrate_score_threshold(_clean(58), alpha=0.05, mode=RCPS, delta=0.05).authority
            is AuthorityLevel.ADVISORY
        )
        cert = calibrate_score_threshold(_clean(59), alpha=0.05, mode=RCPS, delta=0.05)
        assert cert.authority is AuthorityLevel.GATING
        assert cert.binomial_tail is not None
        assert cert.binomial_tail.to_fraction() == Fraction(95, 100) ** 59

    def test_tail_matches_a_direct_binomial_sum(self) -> None:
        data = _obs(*((i / 100, float(i == 85)) for i in range(100)))
        cert = calibrate_score_threshold(data, alpha=0.05, mode=RCPS, delta=0.05)
        assert cert.authority is AuthorityLevel.GATING
        assert cert.threshold == 0.99
        direct = sum(math.comb(100, k) * 0.05**k * 0.95 ** (100 - k) for k in range(2))
        assert cert.binomial_tail is not None
        assert float(cert.binomial_tail.to_fraction()) == pytest.approx(direct, rel=1e-12)
        assert cert.empirical_risk is not None
        assert cert.empirical_risk.to_fraction() == Fraction(1, 100)

    def test_stops_where_the_tail_stops_clearing_delta(self) -> None:
        # Two errors among 100 items put the tail at ~0.118 > 0.05: the sweep must stop at
        # the first threshold that includes the second error, however clean the rest is.
        data = _obs(*((i / 100, float(i in (40, 60))) for i in range(100)))
        cert = calibrate_score_threshold(data, alpha=0.05, mode=RCPS, delta=0.05)
        assert cert.threshold == 0.59
        assert cert.accepted_count == 60

    def test_no_certifiable_threshold_is_unavailable(self) -> None:
        data = _obs((0.0, 1.0), *((i / 100, 0.0) for i in range(1, 80)))
        cert = calibrate_score_threshold(data, alpha=0.05, mode=RCPS, delta=0.05)
        assert cert.authority is AuthorityLevel.UNAVAILABLE

    def test_requires_binary_losses_even_when_data_is_thin(self) -> None:
        with pytest.raises(ValueError, match="exactly 0 or 1"):
            calibrate_score_threshold(
                _obs((0.1, 0.5), (0.2, 0.0)), alpha=0.05, mode=RCPS, delta=0.05
            )

    def test_requires_delta(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            calibrate_score_threshold(_clean(80), alpha=0.05, mode=RCPS)


class TestThresholdCertificateContract:
    def test_non_finite_scores_are_never_permitted(self) -> None:
        cert = calibrate_score_threshold(_clean(30), alpha=0.1)
        assert cert.authority is AuthorityLevel.GATING
        assert not cert.permits(float("nan"))
        assert not cert.permits(float("inf"))
        assert not cert.permits(float("-inf"))

    def test_certificate_is_immutable_and_rejects_unknown_fields(self) -> None:
        cert = calibrate_score_threshold(_clean(30), alpha=0.1)
        with pytest.raises(ValidationError):
            cert.threshold = 1.0  # type: ignore[misc]
        with pytest.raises(ValidationError):
            ThresholdCertificate.model_validate({**cert.model_dump(), "bonus": 1})

    def test_json_round_trip_preserves_the_certificate(self) -> None:
        cert = calibrate_score_threshold(_clean(30), alpha=Fraction(1, 7))
        assert ThresholdCertificate.model_validate_json(cert.model_dump_json()) == cert

    def test_a_gating_certificate_must_carry_a_threshold(self) -> None:
        advisory = calibrate_score_threshold(_clean(3), alpha=0.05)
        with pytest.raises(ValidationError, match="threshold"):
            ThresholdCertificate.model_validate(
                {**advisory.model_dump(), "authority": AuthorityLevel.GATING}
            )

    def test_observations_reject_non_finite_scores_and_out_of_range_losses(self) -> None:
        with pytest.raises(ValidationError):
            ScoreObservation(score=float("nan"), loss=0.0)
        with pytest.raises(ValidationError):
            ScoreObservation(score=0.5, loss=1.5)


class TestCertificateCannotBeForged:
    """The validator rejects field combinations no calibration could produce."""

    @staticmethod
    def _rejects(base: ThresholdCertificate, **changes: object) -> None:
        with pytest.raises(ValidationError):
            ThresholdCertificate.model_validate({**base.model_dump(), **changes})

    def test_gating_certificates_reject_inconsistent_evidence(self) -> None:
        crc = calibrate_score_threshold(_clean(30), alpha=0.1)
        one_half = {"numerator": 1, "denominator": 2}
        self._rejects(crc, n_samples=0)  # fewer items than the floor
        self._rejects(crc, accepted_count=99)
        self._rejects(crc, accepted_count=0)
        self._rejects(crc, alpha={"numerator": -5, "denominator": 1})
        self._rejects(crc, empirical_risk=None)
        self._rejects(crc, inflated_empirical_risk=None)
        self._rejects(crc, inflated_empirical_risk={"numerator": 2, "denominator": 1})
        self._rejects(crc, binomial_tail=one_half)

    def test_rcps_certificates_reject_inconsistent_evidence(self) -> None:
        rcps = calibrate_score_threshold(_clean(80), alpha=0.05, mode=RCPS, delta=0.05)
        assert rcps.authority is AuthorityLevel.GATING
        self._rejects(rcps, binomial_tail=None)
        self._rejects(rcps, binomial_tail={"numerator": 1, "denominator": 2})  # above delta
        self._rejects(rcps, inflated_empirical_risk={"numerator": 1, "denominator": 100})
        self._rejects(rcps, delta=None)

    def test_non_gating_certificates_carry_nothing_and_match_the_floor(self) -> None:
        advisory = calibrate_score_threshold(_clean(3), alpha=0.05)
        self._rejects(advisory, threshold=1.0)
        self._rejects(advisory, accepted_count=1)
        self._rejects(advisory, empirical_risk={"numerator": 0, "denominator": 1})
        self._rejects(advisory, n_samples=advisory.min_required_samples)  # would be unavailable
        unavailable = calibrate_score_threshold(
            _obs((0.0, 1.0), *((i / 100, 0.0) for i in range(1, 30))), alpha=0.05
        )
        assert unavailable.authority is AuthorityLevel.UNAVAILABLE
        self._rejects(unavailable, n_samples=0)  # would be advisory


class TestInputRobustness:
    @pytest.mark.parametrize("alpha", [5e-324, Fraction(1, 10**400)])
    def test_absurdly_small_rcps_budgets_are_a_clear_value_error(self, alpha: Rational) -> None:
        with pytest.raises(ValueError, match="exact-arithmetic limit"):
            min_calibration_samples(alpha, mode=RCPS, delta=0.05)

    def test_extreme_but_representable_rcps_budgets_are_exact(self) -> None:
        assert min_calibration_samples(1 - Fraction(1, 10**30), mode=RCPS, delta=0.05) == 1
        tiny_delta = Fraction(1, 10**400)
        assert min_calibration_samples(Fraction(1, 2), mode=RCPS, delta=tiny_delta) == 1329

    def test_float_subclasses_read_as_plain_decimals(self) -> None:
        class Tagged(float):
            def __repr__(self) -> str:
                return f"Tagged({float.__repr__(self)})"

        assert min_calibration_samples(Tagged(0.05)) == 19

    def test_negative_zero_is_normalised_so_equal_calibrations_serialise_equally(self) -> None:
        clean = [(0.0, 0.0)] * 19 + [(5.0, 1.0)]
        with_negative = [(-0.0, 0.0)] + [(0.0, 0.0)] * 18 + [(5.0, 1.0)]
        first = calibrate_score_threshold(_obs(*clean), alpha=0.05)
        second = calibrate_score_threshold(_obs(*with_negative), alpha=0.05)
        assert first.model_dump_json() == second.model_dump_json()

    def test_a_large_calibration_set_with_an_awkward_budget_finishes(self) -> None:
        cert = calibrate_score_threshold(
            _clean(5000), alpha=Fraction(123456789, 10**9), mode=RCPS, delta=0.05
        )
        assert cert.authority is AuthorityLevel.GATING
        assert cert.accepted_count == 5000

    def test_oversized_rcps_inputs_are_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(conformal_module, "MAX_EXACT_SAMPLES", 10)
        with pytest.raises(ValueError, match="exact-arithmetic limit"):
            calibrate_score_threshold(_clean(11), alpha=0.5, mode=RCPS, delta=0.05)
        with pytest.raises(ValueError, match="exact-arithmetic limit"):
            min_calibration_samples(0.01, mode=RCPS, delta=0.05)


class TestRcpsCoverage:
    """RCPS: the chance of certifying a threshold whose true risk exceeds alpha is <= delta.

    Exact over every possible calibration sample of a known discrete population, so
    there is no sampling noise to hide a violation behind. The population is built so
    the loosest threshold's true risk (0.11) sits just above the budget (0.1), making
    the violation probability land close to ``delta`` instead of being vacuously small.
    """

    # (score, loss) -> probability. True risk of accepting score <= 0 is 0.01; <= 1 is 0.11.
    CELLS: ClassVar[dict[tuple[float, float], float]] = {
        (0.0, 0.0): 0.49,
        (0.0, 1.0): 0.01,
        (1.0, 0.0): 0.40,
        (1.0, 1.0): 0.10,
    }
    TRUE_RISK: ClassVar[dict[float, float]] = {0.0: 0.01, 1.0: 0.11}

    @staticmethod
    def _compositions(total: int, parts: int) -> list[tuple[int, ...]]:
        if parts == 1:
            return [(total,)]
        return [
            (head, *rest)
            for head in range(total + 1)
            for rest in TestRcpsCoverage._compositions(total - head, parts - 1)
        ]

    @pytest.mark.parametrize("higher_is_riskier", [True, False])
    def test_violation_probability_is_within_delta(self, higher_is_riskier: bool) -> None:
        alpha, delta = Fraction(1, 10), Fraction(1, 5)
        n = min_calibration_samples(alpha, mode=RCPS, delta=delta)
        cells = list(self.CELLS)
        violation = 0.0
        for counts in self._compositions(n, len(cells)):
            probability = float(math.factorial(n))
            observations: list[ScoreObservation] = []
            for (score, loss), count in zip(cells, counts, strict=True):
                probability *= self.CELLS[(score, loss)] ** count / math.factorial(count)
                signed = score if higher_is_riskier else -score
                observations += [ScoreObservation(score=signed, loss=loss)] * count
            cert = calibrate_score_threshold(
                observations,
                alpha=alpha,
                mode=RCPS,
                delta=delta,
                higher_is_riskier=higher_is_riskier,
            )
            if cert.authority is not AuthorityLevel.GATING or cert.threshold is None:
                continue
            threshold = cert.threshold if higher_is_riskier else -cert.threshold
            if self.TRUE_RISK[threshold] > alpha:
                violation += probability
        assert 0.1 < violation <= float(delta)


class TestStatisticsMustFollowFromTheRecordedRisk:
    """A certificate's bound must be the one its own empirical risk implies."""

    @staticmethod
    def _rejects(base: ThresholdCertificate, **changes: object) -> None:
        with pytest.raises(ValidationError):
            ThresholdCertificate.model_validate({**base.model_dump(), **changes})

    def test_crc_inflated_risk_must_equal_the_recomputed_value(self) -> None:
        crc = calibrate_score_threshold(_clean(30), alpha=0.1)
        everything_lost = {"numerator": 1, "denominator": 1}
        # The reported example: all 30 items lost, yet a tiny inflated risk is claimed.
        self._rejects(
            crc,
            empirical_risk=everything_lost,
            inflated_empirical_risk={"numerator": 1, "denominator": 31},
        )
        # A value that is merely different, and still within the budget, is no better.
        self._rejects(crc, inflated_empirical_risk={"numerator": 1, "denominator": 20})

    def test_rcps_tail_must_equal_the_recomputed_binomial_tail(self) -> None:
        rcps = calibrate_score_threshold(_clean(80), alpha=0.05, mode=RCPS, delta=0.05)
        self._rejects(rcps, binomial_tail={"numerator": 1, "denominator": 10**9})
        self._rejects(rcps, binomial_tail={"numerator": -1, "denominator": 2})
        # Half an error over 80 items is not a count of errors.
        self._rejects(rcps, empirical_risk={"numerator": 1, "denominator": 160})

    def test_empirical_risk_cannot_exceed_what_the_accepted_items_could_lose(self) -> None:
        crc = calibrate_score_threshold(_clean(30), alpha=0.1)
        self._rejects(
            crc,
            accepted_count=1,
            empirical_risk={"numerator": 1, "denominator": 2},
            inflated_empirical_risk={"numerator": 16, "denominator": 31},
        )

    def test_rcps_validation_does_not_recompute_past_the_exact_arithmetic_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rcps = calibrate_score_threshold(_clean(80), alpha=0.05, mode=RCPS, delta=0.05)
        monkeypatch.setattr(conformal_module, "MAX_EXACT_SAMPLES", 10)
        self._rejects(rcps)

    def test_honest_certificates_survive_revalidation(self) -> None:
        for data, kwargs in (
            (_clean(30), {"alpha": 0.1}),
            (_obs(*((i / 100, float(i == 85)) for i in range(100))), {"alpha": 0.2}),
            (
                _obs(*((i / 100, float(i == 85)) for i in range(100))),
                {
                    "alpha": 0.05,
                    "mode": RCPS,
                    "delta": 0.05,
                },
            ),
        ):
            cert = calibrate_score_threshold(data, **kwargs)  # type: ignore[arg-type]
            assert cert.authority is AuthorityLevel.GATING
            assert ThresholdCertificate.model_validate(cert.model_dump()) == cert


class TestLargeExactValuesRoundTripThroughJson:
    def test_a_five_thousand_item_rcps_certificate_round_trips(self) -> None:
        cert = calibrate_score_threshold(_clean(5000), alpha=0.05, mode=RCPS, delta=0.05)
        assert cert.binomial_tail is not None
        assert cert.binomial_tail.denominator > 2**64  # beyond what a JSON number can hold
        restored = ThresholdCertificate.model_validate_json(cert.model_dump_json())
        assert restored == cert

    def test_big_and_negative_rationals_round_trip(self) -> None:
        for value in (Fraction(-(7**5000), 5**5000), Fraction(5**6000, 7**6000), Fraction(-3, 7)):
            wire = ExactRational.from_fraction(value).model_dump_json()
            assert ExactRational.model_validate_json(wire).to_fraction() == value

    def test_small_values_stay_plain_json_numbers(self) -> None:
        wire = ExactRational.from_fraction(Fraction(3, 10)).model_dump_json()
        assert wire == '{"schema_version":"1","numerator":3,"denominator":10}'
