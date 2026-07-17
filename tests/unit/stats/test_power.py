"""Tests for ``required_sample_size``: the "how many samples do I need"
helper for comparing two pass rates (ADR-0016).

``required_sample_size`` computes its answer using a standard, exact
formula (a "closed form" -- see its own docstring in power.py) built only
from Python's standard library, with no external statistics packages.
Every expected value in these tests is recomputed right here, inline, from
``statistics.NormalDist`` at the same points on the bell curve
("quantiles") that the real implementation uses -- never a plain decimal
number that was pre-computed once and pasted in, which could just as
easily be wrong. This follows the same "recompute it live, do not trust a
hardcoded number" approach already used for
``test_wilson_interval_uses_normaldist_975_quantile`` in
``test_aggregate.py``.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import pytest

from agentic_evalkit.stats.power import required_sample_size


def _expected_n(baseline: float, mde: float, *, alpha: float = 0.05, power: float = 0.8) -> int:
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(power)
    p1, p2 = baseline, baseline + mde
    variance_term = p1 * (1.0 - p1) + p2 * (1.0 - p2)
    return math.ceil((z_alpha + z_power) ** 2 * variance_term / mde**2)


def test_required_sample_size_matches_inline_closed_form() -> None:
    result = required_sample_size(baseline_rate=0.5, minimum_detectable_effect=0.1)
    assert result == _expected_n(0.5, 0.1)


def test_required_sample_size_honors_alpha_and_power_overrides() -> None:
    result = required_sample_size(
        baseline_rate=0.3, minimum_detectable_effect=0.08, alpha=0.01, power=0.9
    )
    assert result == _expected_n(0.3, 0.08, alpha=0.01, power=0.9)


def test_required_sample_size_returns_positive_int() -> None:
    result = required_sample_size(baseline_rate=0.2, minimum_detectable_effect=0.1)
    assert isinstance(result, int)
    assert result >= 1


def test_required_sample_size_grows_as_effect_shrinks() -> None:
    # Asking to reliably detect a smaller improvement (a smaller "minimum
    # detectable effect") requires more samples than asking to detect a
    # bigger, more obvious one.
    smaller_effect = required_sample_size(baseline_rate=0.5, minimum_detectable_effect=0.05)
    larger_effect = required_sample_size(baseline_rate=0.5, minimum_detectable_effect=0.1)
    assert smaller_effect > larger_effect


def test_required_sample_size_grows_with_stricter_alpha_and_higher_power() -> None:
    base = required_sample_size(baseline_rate=0.5, minimum_detectable_effect=0.1)
    stricter_alpha = required_sample_size(
        baseline_rate=0.5, minimum_detectable_effect=0.1, alpha=0.01
    )
    higher_power = required_sample_size(
        baseline_rate=0.5, minimum_detectable_effect=0.1, power=0.95
    )
    assert stricter_alpha > base
    assert higher_power > base


@pytest.mark.parametrize("baseline", [0.0, 1.0, -0.1, 1.5])
def test_required_sample_size_rejects_out_of_range_baseline(baseline: float) -> None:
    with pytest.raises(ValueError, match="baseline_rate"):
        required_sample_size(baseline_rate=baseline, minimum_detectable_effect=0.05)


@pytest.mark.parametrize("mde", [0.0, -0.1, 0.6])
def test_required_sample_size_rejects_invalid_minimum_detectable_effect(mde: float) -> None:
    # 0.0 and -0.1 are rejected for being zero or negative (you cannot
    # target an "improvement" of zero or less); 0.6 is rejected because
    # baseline_rate (0.5) + mde (0.6) = 1.1, which is above the maximum
    # possible rate of 1.0.
    with pytest.raises(ValueError, match="minimum_detectable_effect"):
        required_sample_size(baseline_rate=0.5, minimum_detectable_effect=mde)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.01, 2.0])
def test_required_sample_size_rejects_out_of_range_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        required_sample_size(baseline_rate=0.5, minimum_detectable_effect=0.1, alpha=alpha)


@pytest.mark.parametrize("power", [0.0, 1.0, -0.5, 1.2])
def test_required_sample_size_rejects_out_of_range_power(power: float) -> None:
    with pytest.raises(ValueError, match="power"):
        required_sample_size(baseline_rate=0.5, minimum_detectable_effect=0.1, power=power)
