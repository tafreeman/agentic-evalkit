"""Figuring out, ahead of time, how many samples you'd need to reliably
detect a real difference between two pass rates (ADR-0016).

Design section 10 calls for "predeclared subgroup slices with adequate
sample sizes" -- meaning: before you go looking at subgroups of your data,
you should have already worked out whether you're even testing with
enough samples to trust what you find. And C8 (this codebase's rule for
statistical rigor) requires this kind of "power calculation" before anyone
trusts a reported difference ("delta") between two pass rates.

"Statistical power" is the probability that, if a real difference of a
given size genuinely exists, your test will actually detect it rather than
missing it by chance -- testing with too few samples means you might
completely miss a real effect just because your sample was small and
noisy, not because the effect isn't there. :func:`required_sample_size`
answers the practical question this raises: "if I want to reliably detect
a difference of at least this size between two pass rates (imagine
running two variants -- 'arms' -- of an experiment side by side, each
with its own group of samples), how many samples do I need in *each*
group?"

It computes this with a standard, direct formula (a "closed form," meaning
an exact formula you can compute in one pass, rather than something you
have to estimate through trial and error or simulation), using only
Python's standard library (:class:`statistics.NormalDist`) -- the same
"no numpy/scipy dependency" discipline the rest of
:mod:`agentic_evalkit.stats` follows.
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
    """Return how many samples you'd need, in each of two groups, to
    reliably detect a real difference between their two success rates.

    This implements the standard formula for planning a "two-proportion
    z-test" -- the standard statistical test for "is this rate
    meaningfully different from that rate":

        ``n = (z_{1-alpha/2} + z_{power})^2 * (p1(1-p1) + p2(1-p2)) / (p2-p1)^2``

    where ``p1 = baseline_rate`` is the success rate you already have
    (e.g. your current system), and ``p2 = baseline_rate +
    minimum_detectable_effect`` is the success rate you're hoping to be
    able to detect (e.g. a new system that's actually better by that
    amount). The two ``z`` values are critical values from the normal
    (bell-curve) distribution -- looked up with
    :meth:`statistics.NormalDist.inv_cdf`, which answers "at what point on
    the bell curve have we covered this fraction of the distribution?"
    (that point is called a "quantile"). One lookup uses the ``1 -
    alpha/2`` fraction, the other uses the ``power`` fraction -- the exact
    same ``NormalDist`` building block :func:`agentic_evalkit.stats.wilson_interval`
    uses, just applied here to plan an experiment in advance rather than
    to summarize one that already happened. The formula's result is
    rounded up to a whole number, since you can't actually run a
    fractional sample.

    Args:
        baseline_rate: The success probability of your existing/control
            group (``p1`` in the formula above), strictly between 0 and 1.
        minimum_detectable_effect: The smallest absolute improvement in
            rate that you actually care about being able to detect -- e.g.
            0.05 means "I want to be able to detect at least a
            5-percentage-point improvement." Must be positive, and small
            enough that ``baseline_rate + minimum_detectable_effect``
            stays below 1.
        alpha: The "significance level" -- the chance you're willing to
            accept of concluding there's a real difference when actually
            there isn't one (a "false positive"). Strictly between 0 and 1
            (default 0.05, i.e. a 5% false-positive tolerance).
        power: The desired "statistical power" -- the chance you actually
            want of detecting the effect, given that it truly exists (i.e.
            ``1 - beta``, where ``beta`` is the chance of missing a real
            effect, a "false negative"). Strictly between 0 and 1 (default
            0.8, i.e. an 80% chance of catching a real effect of this size
            if it's really there).

    Returns:
        The minimum number of samples needed in each group, an ``int >= 1``.

    Raises:
        ValueError: If any argument is outside its documented range, in the
            same fail-fast style as :func:`agentic_evalkit.stats.wilson_interval`.
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
