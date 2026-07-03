# ADR-0008: Sample-Level Retention, Honest Intervals, and Provenance-Gated Comparison

## Status

Accepted

## Context

Design §7 (`docs/specs/2026-07-02-agentic-evalkit-design.md`) requires
`agentic-evalkit`'s statistics to be reproducible and hard to misread. Three
concerns drive this ADR:

1. Aggregates computed once and stored can silently drift from the per-sample
   truth; the retained samples must remain the source of record.
2. A point estimate ("62% pass") invites over-reading; binary rates need an
   honest interval, and an empty denominator must not fabricate one.
3. Comparing two runs is only meaningful when they measured the same thing;
   comparing runs with different datasets, adapters, graders, target
   policies, or sampling silently produces a nonsense delta.

Operational failures (timeouts, infrastructure errors, unavailable harnesses)
must never be counted as task failures, or every rate is wrong.

## Decision

- **Sample-level retention; recount, never trust a summary.** `aggregate_run`
  recomputes every count — pass, fail, partial, error, timeout, cancelled,
  abstain, unavailable — directly from `run.samples`, using both execution
  status and grade status. It never trusts a precomputed `run.summary`.
- **Wilson 95% intervals for binary rates.** `wilson_interval` uses
  `statistics.NormalDist().inv_cdf(0.975)` for the score interval; an empty
  denominator returns `None` bounds rather than a fabricated `(0, 0)` or a
  divide-by-zero. The Wilson upper bound is exactly `1.0` at `p=1.0` for any
  finite `n`, and the interval is reported as such.
- **Reliability in log space.** `pass_at_k = 1 - C(n-c, k) / C(n, k)`,
  computed with `math.lgamma` in log space to stay stable for large `n`, with
  inputs validated `0 <= c <= n` and `1 <= k <= n`. `consistency_at_k = p**k`
  captures all-attempt agreement, validated `0 <= p <= 1`, `k >= 1`.
- **Operational outcomes are separated from task outcomes.** Errors,
  timeouts, cancellations, and unavailable results are counted in their own
  buckets and are never folded into `failed`, so infrastructure problems
  cannot masquerade as the system-under-test failing the task.
- **Deterministic, seeded, paired bootstrap.** `compare_runs(left, right, *,
  bootstrap_samples=1000, seed)` pairs observations by `(sample_id, attempt)`,
  bootstraps a paired delta with a local `random.Random(seed)` (no global RNG
  state), and returns the estimate, the 2.5 / 97.5 percentiles, the paired
  count, and the seed. `bootstrap_samples` is validated to `[100, 10000]`.
- **Provenance-gated comparison.** `compare_runs` first checks that the two
  runs share resolved dataset id / revision / config / split, adapter, grader,
  target name/fingerprint policy, sampling (temperature/seed), and attempt
  count. On any mismatch it raises `IncompatibleRuns` listing **all**
  mismatches, not just the first, so an incomparable pair fails loudly and
  completely.

## Alternatives

1. **Store aggregates alongside the run and read them back.** Rejected:
   stored aggregates drift from the samples under any later reprocessing;
   recomputing from retained samples is the only self-consistent source.
2. **Report bare point estimates.** Rejected: a rate with no interval is
   routinely over-read; Wilson intervals (with explicit `None` on empty
   denominators) keep the uncertainty visible and honest.
3. **Compare any two runs the caller hands over.** Rejected: a delta between
   runs that measured different datasets or used different graders is
   meaningless; provenance gating turns that into a loud, enumerated error.
4. **A global-seed or unseeded bootstrap.** Rejected: it is not reproducible
   and couples independent comparisons through shared RNG state; a per-call
   `random.Random(seed)` returned in the result makes every comparison
   replayable.

## Consequences

- Any run's statistics can be regenerated exactly from its retained samples.
- Rates always carry an honest interval, and an empty cohort is visibly
  `None` rather than a fake zero-width interval.
- A comparison either measured the same thing or fails with a complete list
  of why it could not; there is no silent nonsense delta.
- Operational failures never deflate pass rates.

## Validation

- `tests/unit/stats/test_reliability.py` covers `pass_at_k` (including
  large-`n` log-space stability) and `consistency_at_k` against independently
  hand-computed values, with input-range validation.
- `tests/unit/stats/test_aggregate.py` covers recount-from-samples,
  operational-vs-task separation, and Wilson bounds including the empty-cohort
  `None` case and the `p=1.0` upper bound of exactly `1.0`.
- `tests/unit/stats/test_compare.py` covers `(sample_id, attempt)` pairing,
  the seeded reproducible bootstrap, and `IncompatibleRuns` enumerating all
  provenance mismatches.

## Supersession

Changing the interval method (e.g. to bootstrap or Jeffreys intervals for
binary rates), the bootstrap resampling scheme, or the set of provenance
fields that gate comparability is a material change and must supersede this
ADR with new validation.
