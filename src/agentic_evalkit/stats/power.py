"""Closed-form sample-size planning for a two-proportion comparison (ADR-0016).

Design section 10 calls for "predeclared subgroup slices with adequate sample
sizes"; C8 (statistical rigor) names a power calculation as the check that must
precede trusting a rate delta. :func:`required_sample_size` answers "how many
samples per arm do I need to detect an effect of at least this size?" using only
the standard library (:class:`statistics.NormalDist`) -- the same
no-numpy/scipy discipline the rest of :mod:`agentic_evalkit.stats` keeps.
"""

from __future__ import annotations

import math
from statistics import NormalDist

__all__ = ["required_sample_size"]


def required_sample_size(
    *,
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Return the per-arm sample size for a two-proportion z-test.

    Implements the standard closed form

        ``n = (z_{1-alpha/2} + z_{power})^2 * (p1(1-p1) + p2(1-p2)) / (p2-p1)^2``

    where ``p1 = baseline_rate`` and ``p2 = baseline_rate +
    minimum_detectable_effect``, with the two critical values taken from
    :meth:`statistics.NormalDist.inv_cdf` at the ``1 - alpha/2`` and ``power``
    quantiles -- the same ``NormalDist`` primitive the Wilson interval uses, at
    planning-time quantiles. The result is rounded up, since a fractional sample
    cannot be run.

    Args:
        baseline_rate: The control-arm success probability ``p1``, strictly
            between 0 and 1.
        minimum_detectable_effect: The smallest absolute increase in rate worth
            detecting; must be positive and small enough that ``baseline_rate +
            minimum_detectable_effect`` stays below 1.
        alpha: Two-sided significance level, strictly between 0 and 1
            (default 0.05).
        power: Desired power ``1 - beta``, strictly between 0 and 1
            (default 0.8).

    Returns:
        The minimum number of samples per arm, an ``int >= 1``.

    Raises:
        ValueError: If any argument is outside its documented range, in the same
            fail-fast style as :func:`agentic_evalkit.stats.wilson_interval`.
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(f"baseline_rate must satisfy 0 < baseline_rate < 1 (got {baseline_rate})")
    if not 0.0 < minimum_detectable_effect < 1.0 - baseline_rate:
        raise ValueError(
            "minimum_detectable_effect must satisfy "
            "0 < minimum_detectable_effect < 1 - baseline_rate "
            f"(got {minimum_detectable_effect} with baseline_rate={baseline_rate})"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must satisfy 0 < alpha < 1 (got {alpha})")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must satisfy 0 < power < 1 (got {power})")

    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(power)
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    variance_term = p1 * (1.0 - p1) + p2 * (1.0 - p2)
    numerator = (z_alpha + z_power) ** 2 * variance_term
    return math.ceil(numerator / (minimum_detectable_effect**2))
