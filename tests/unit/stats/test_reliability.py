"""Tests for ``pass_at_k`` and ``consistency_at_k``: two different ways of
scoring a system that gets more than one attempt at the same task (design
section 10, ADR-0008).

``pass_at_k`` answers "would at least one of k attempts have succeeded?"
(useful when the system only needs one attempt out of several to work),
while ``consistency_at_k`` answers a stricter, different question: "would
every single one of k attempts have succeeded?" (useful when the system
needs to behave reliably every time). See ``reliability.py``'s own module
docstring for the full explanation of both, including why they are
genuinely different questions that must never be confused with each other.

These tests cover the plan's original known-value example (docs/plans/
2026-07-02-agentic-evalkit-initial-release.md, Task 12 Step 2) unchanged,
plus checks on each function's input rules from Task 12 Step 5: for
``pass_at_k``, the number of successes ``c`` must satisfy ``0 <= c <= n``
and ``k`` must satisfy ``1 <= k <= n`` (``n`` being the total attempts
actually run).
"""

from __future__ import annotations

import math

import pytest

from agentic_evalkit.stats.reliability import consistency_at_k, pass_at_k

# --- Step 2 (plan verbatim): known-value tests ------------------------------


def test_pass_at_k_known_values() -> None:
    assert pass_at_k(total_attempts=4, successful_attempts=1, k=1) == pytest.approx(0.25)
    assert pass_at_k(total_attempts=4, successful_attempts=1, k=4) == pytest.approx(1.0)


def test_consistency_at_k_requires_every_attempt_to_pass() -> None:
    assert consistency_at_k(success_probability=0.8, k=3) == pytest.approx(0.512)


# --- pass_at_k: additional known values -------------------------------------


def test_pass_at_k_zero_successes_is_zero() -> None:
    assert pass_at_k(total_attempts=5, successful_attempts=0, k=3) == pytest.approx(0.0)


def test_pass_at_k_all_successes_is_one() -> None:
    assert pass_at_k(total_attempts=5, successful_attempts=5, k=2) == pytest.approx(1.0)


def test_pass_at_k_matches_direct_combinatorics_for_small_n() -> None:
    # With n=10 total attempts, c=3 successes, and k=5: this recomputes
    # "1 - C(7,5)/C(10,5)" directly using Python's math.comb (i.e. "n
    # choose k" -- see reliability.py's _log_binomial_coefficient docstring
    # for what that means), as an independent check against the
    # implementation's own log-space computation.
    n, c, k = 10, 3, 5
    expected = 1 - (math.comb(n - c, k) / math.comb(n, k))
    assert pass_at_k(total_attempts=n, successful_attempts=c, k=k) == pytest.approx(expected)


def test_pass_at_k_handles_large_n_without_overflow() -> None:
    # A large n forces the implementation to use its log-space code path
    # (built on math.lgamma) instead of math.comb, since math.comb would
    # have to build astronomically large intermediate numbers here (see
    # reliability.py's module docstring for why). With only 1 success among
    # all n attempts, pass@k has a simple, exact answer: it is just the
    # probability that a random draw of k out of n attempts happens to
    # include that one successful attempt, which is k / n.
    n, c, k = 100_000, 1, 50_000
    result = pass_at_k(total_attempts=n, successful_attempts=c, k=k)
    assert 0.0 <= result <= 1.0
    assert result == pytest.approx(k / n)


def test_pass_at_k_is_exact_when_failures_are_fewer_than_k() -> None:
    # When there are fewer failed attempts (n - c) than k, there literally
    # are not enough failures to fill up a group of k attempts with
    # failures alone -- so every possible group of k must include at least
    # one success, and pass@k must be exactly 1.0. (In the formula, this
    # shows up as C(n-c, k) -- "ways to choose k items from fewer than k
    # available" -- being 0, by the usual mathematical convention.)
    result = pass_at_k(total_attempts=100_000, successful_attempts=60_000, k=50_000)
    assert result == pytest.approx(1.0)


# --- pass_at_k: validation ---------------------------------------------------


def test_pass_at_k_rejects_negative_successful_attempts() -> None:
    with pytest.raises(ValueError, match="successful_attempts"):
        pass_at_k(total_attempts=4, successful_attempts=-1, k=1)


def test_pass_at_k_rejects_successful_attempts_greater_than_total() -> None:
    with pytest.raises(ValueError, match="successful_attempts"):
        pass_at_k(total_attempts=4, successful_attempts=5, k=1)


def test_pass_at_k_rejects_k_less_than_one() -> None:
    with pytest.raises(ValueError, match="k"):
        pass_at_k(total_attempts=4, successful_attempts=1, k=0)


def test_pass_at_k_rejects_k_greater_than_total_attempts() -> None:
    with pytest.raises(ValueError, match="k"):
        pass_at_k(total_attempts=4, successful_attempts=1, k=5)


# --- consistency_at_k: additional known values ------------------------------


def test_consistency_at_k_of_one_equals_success_probability() -> None:
    assert consistency_at_k(success_probability=0.8, k=1) == pytest.approx(0.8)


def test_consistency_at_k_of_zero_probability_is_zero_for_k_greater_than_zero() -> None:
    assert consistency_at_k(success_probability=0.0, k=3) == pytest.approx(0.0)


def test_consistency_at_k_of_one_probability_is_one() -> None:
    assert consistency_at_k(success_probability=1.0, k=10) == pytest.approx(1.0)


# --- consistency_at_k: validation --------------------------------------------


def test_consistency_at_k_rejects_probability_below_zero() -> None:
    with pytest.raises(ValueError, match="success_probability"):
        consistency_at_k(success_probability=-0.1, k=1)


def test_consistency_at_k_rejects_probability_above_one() -> None:
    with pytest.raises(ValueError, match="success_probability"):
        consistency_at_k(success_probability=1.1, k=1)


def test_consistency_at_k_rejects_k_less_than_one() -> None:
    with pytest.raises(ValueError, match="k"):
        consistency_at_k(success_probability=0.5, k=0)
