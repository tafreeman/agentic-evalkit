"""Repeated-trial reliability metrics: ``pass@k`` and consistency@k.

Design §10 and ADR-0008 distinguish two questions about repeated attempts
at the same sample:

- ``pass_at_k``: if a system is *allowed* ``k`` attempts and *any one*
  succeeding is useful, what is the probability at least one of ``k``
  randomly selected attempts (from the total attempts actually run)
  succeeds? This is the standard unbiased Codex/HumanEval-style estimator,
  computed from ``total_attempts`` (``n``) and ``successful_attempts``
  (``c``) rather than by re-running the system ``k`` times.
- ``consistency_at_k``: if a system must succeed on *every one* of ``k``
  attempts to be useful (e.g. determinism/reliability requirements), what
  is the probability all ``k`` succeed, given a single-attempt success
  probability? This is a different question from ``pass_at_k`` and must
  never be reported as a second ``pass@k`` metric (Task 12 Step 5).

Both functions use only the standard library. ``pass_at_k`` computes
``1 - C(n-c, k) / C(n, k)`` in log space via :func:`math.lgamma` so that
large ``n`` (e.g. many repeated trials) never requires constructing huge
intermediate integers the way :func:`math.comb` would.
"""

from __future__ import annotations

import math

__all__ = ["consistency_at_k", "pass_at_k"]


def _log_binomial_coefficient(n: int, k: int) -> float:
    """Return ``log(C(n, k))`` computed via ``math.lgamma``.

    ``C(n, k) = n! / (k! * (n-k)!)``, so
    ``log(C(n, k)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)``.
    Callers guarantee ``0 <= k <= n``, so ``C(n, k) >= 1`` always and this
    value is always finite and non-negative.
    """
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def pass_at_k(*, total_attempts: int, successful_attempts: int, k: int) -> float:
    """Probability at least one of ``k`` sampled attempts succeeds.

    Computes the unbiased estimator
    ``pass@k = 1 - C(n-c, k) / C(n, k)`` where ``n = total_attempts`` and
    ``c = successful_attempts``, using :func:`math.lgamma` in log space so
    large ``n`` never requires huge intermediate integers.

    Args:
        total_attempts: Total number of attempts run for the sample
            (``n``). Must be at least 1.
        successful_attempts: Number of those attempts that succeeded
            (``c``). Must satisfy ``0 <= c <= n``.
        k: Number of attempts hypothetically sampled without replacement
            from the ``n`` total attempts. Must satisfy ``1 <= k <= n``.

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

    # If there are not enough failed attempts (n - c) to fill a sample of
    # size k, then every k-sized sample must contain at least one success:
    # C(n-c, k) is conventionally 0 in that case (choosing more items than
    # exist), so pass@k = 1 exactly. Guard explicitly rather than relying on
    # lgamma, which is undefined for a negative first argument.
    failed_attempts = n - c
    if failed_attempts < k:
        return 1.0

    log_ratio = _log_binomial_coefficient(failed_attempts, k) - _log_binomial_coefficient(n, k)
    return 1.0 - math.exp(log_ratio)


def consistency_at_k(*, success_probability: float, k: int) -> float:
    """Probability all ``k`` independent attempts succeed.

    Computed as ``success_probability ** k``. This answers "every one of
    k attempts must pass" reliability, which is a distinct question from
    ``pass_at_k`` ("any one of k attempts is useful") and must not be
    reported as a second ``pass@k`` metric.

    Args:
        success_probability: Single-attempt success probability ``p``.
            Must satisfy ``0.0 <= p <= 1.0``.
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
