"""Metrics for repeated attempts at the same task: ``pass@k`` and ``consistency@k``.

Design §10 and ADR-0008 distinguish two different questions you might ask
about a system that gets multiple attempts at the same sample:

- ``pass_at_k``: suppose a system is *allowed* ``k`` attempts at a task,
  and it only needs *any one* of them to succeed to be useful (like a
  developer allowed several tries to get a coding task right). Out of the
  attempts we actually ran, what's the probability that at least one of
  ``k`` randomly-chosen attempts among them would have succeeded? This is
  the standard, statistically "unbiased" way of estimating that (the same
  approach used in the Codex/HumanEval coding benchmarks) -- and
  importantly, it's computed mathematically from how many attempts we ran
  in total (``n``) and how many succeeded (``c``), *not* by actually going
  back and re-running the system ``k`` more times.
- ``consistency_at_k``: suppose instead a system must succeed on *every
  single one* of ``k`` attempts to be useful (e.g. you need it to behave
  reliably/deterministically every time, not just once). What's the
  probability all ``k`` attempts succeed, given how often a single attempt
  succeeds? This is a genuinely different question from ``pass_at_k`` --
  "at least one out of k" versus "all k out of k" -- and must never be
  reported as if it were a second ``pass@k`` number (Task 12 Step 5).

Both functions use only Python's standard library, no external math
packages. ``pass_at_k``'s formula involves counting combinations (how many
ways you can choose things from a group), which for a large number of
attempts can involve enormous intermediate numbers if computed the naive
way. To avoid that, it does the computation in "log space" (working with
the logarithms of the numbers instead of the numbers themselves, via
:func:`math.lgamma`) so that even a very large ``n`` (many repeated
trials) never requires Python to build the huge intermediate integers that
the more obvious approach, :func:`math.comb`, would require.
"""

from __future__ import annotations

import math

__all__ = ["consistency_at_k", "pass_at_k"]


def _log_binomial_coefficient(n: int, k: int) -> float:
    """Return ``log(C(n, k))`` -- the logarithm of "n choose k" -- computed
    via ``math.lgamma``.

    ``C(n, k)`` (read "n choose k") is the number of ways to pick an
    unordered group of ``k`` items out of ``n`` total; it's defined as
    ``C(n, k) = n! / (k! * (n-k)!)``. Rather than computing
    potentially-huge factorials directly, this uses ``math.lgamma`` (the
    log of the "gamma function," which is the standard continuous
    generalization of the factorial function) to work in log space
    instead: ``log(C(n, k)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)``.
    Callers of this function always guarantee ``0 <= k <= n``, which means
    ``C(n, k) >= 1`` always holds, so this returned value is always finite
    and never negative.
    """
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def pass_at_k(*, total_attempts: int, successful_attempts: int, k: int) -> float:
    """Return the probability that at least one of ``k`` sampled attempts
    succeeds -- i.e. the "pass@k" score.

    Computes the standard, statistically unbiased estimator for this:
    ``pass@k = 1 - C(n-c, k) / C(n, k)``, where ``n = total_attempts`` and
    ``c = successful_attempts``. In plain terms: this is "1 minus the
    probability that a random group of ``k`` attempts, drawn from
    everything we actually ran, would contain *zero* successes" -- which
    is exactly the same thing as "the probability it contains at least one
    success." It's computed via :func:`math.lgamma` in log space (see
    :func:`_log_binomial_coefficient`) so that a large ``n`` never
    requires Python to build huge intermediate integers.

    Args:
        total_attempts: Total number of attempts actually run for the
            sample (``n`` in the formula above). Must be at least 1.
        successful_attempts: How many of those attempts succeeded (``c``
            in the formula above). Must satisfy ``0 <= c <= n``.
        k: Number of attempts to hypothetically sample, without putting
            them back, out of the ``n`` total attempts (i.e. imagine
            drawing ``k`` of the ``n`` results out of a hat, without
            replacement). Must satisfy ``1 <= k <= n``.

    Returns:
        A probability in ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``successful_attempts`` or ``k`` violate their
            documented bounds relative to ``total_attempts``.
    """
    n = total_attempts
    c = successful_attempts
    if not (0 <= c <= n):
        raise ValueError(
            f"successful_attempts must satisfy 0 <= successful_attempts <= total_attempts "
            f"(got successful_attempts={c}, total_attempts={n})"
        )
    if not (1 <= k <= n):
        raise ValueError(f"k must satisfy 1 <= k <= total_attempts (got k={k}, total_attempts={n})")

    # If there aren't even enough failed attempts (n - c of them) to fill a
    # sample of size k, then any group of k attempts we could possibly draw
    # is guaranteed to include at least one success -- there simply aren't
    # enough failures to fill the whole group with failures alone. By
    # convention, C(n-c, k) -- "ways to choose k items from only n-c
    # available" -- is 0 when k is bigger than what's available, so pass@k
    # works out to exactly 1 in this case. We check for this directly
    # instead of just letting the formula run through lgamma, because
    # lgamma isn't defined for a negative first argument (which would
    # happen here).
    failed_attempts = n - c
    if failed_attempts < k:
        return 1.0

    log_ratio = _log_binomial_coefficient(failed_attempts, k) - _log_binomial_coefficient(n, k)
    return 1.0 - math.exp(log_ratio)


def consistency_at_k(*, success_probability: float, k: int) -> float:
    """Return the probability that all ``k`` independent attempts succeed.

    Computed as ``success_probability ** k`` -- i.e. multiplying the
    single-attempt success probability by itself ``k`` times, which is the
    standard way to get the probability of several independent events all
    happening. This answers a "reliability" question -- every one of ``k``
    attempts must pass -- which is a completely different question from
    ``pass_at_k`` ("at least one of ``k`` attempts is enough") and must
    not be reported as if it were a second ``pass@k`` metric.

    Args:
        success_probability: The single-attempt success probability
            (``p``). Must satisfy ``0.0 <= p <= 1.0``.
        k: Number of independent attempts that must all succeed. Must be
            at least 1.

    Returns:
        A probability in ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``success_probability`` or ``k`` violate their
            documented bounds.
    """
    if not (0.0 <= success_probability <= 1.0):
        raise ValueError(
            "success_probability must satisfy 0.0 <= success_probability <= 1.0 "
            f"(got {success_probability})"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1 (got {k})")

    return success_probability**k
