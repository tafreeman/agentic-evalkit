# ADR-0016: Cluster-Robust Intervals for Repeated Attempts (Addendum to ADR-0008)

## Status

Accepted

## Context

ADR-0008 fixed `pass_rate`'s interval to a 95% Wilson score interval computed
over `total = len(run.samples)`. It did not account for
`run.manifest.attempts > 1`: repeated attempts at the same `sample_id` are
correlated observations (same prompt, same task difficulty, same grader
idiosyncrasies), not independent Bernoulli trials. Treating each attempt as an
independent trial in the Wilson denominator is textbook pseudo-replication --
it inflates the effective N and reports a narrower, more-certain interval than
the data supports. `pass_at_k_by_sample` already special-cases the
repeated-attempt shape by grouping on `sample_id`, but `aggregate_run`'s
`pass_rate` did not.

`AggregateStats.score_mean` additionally carried no interval at all, even though
design section 10 promises every report shows "mean or rate with an appropriate
95% confidence interval" -- that promise was met for the rate but broken for the
mean. C8 (statistical rigor) also names a power / sample-size calculation as a
predeclaration check, which the package did not expose.

ADR-0008's own Supersession clause requires that "changing the interval method
(e.g. to bootstrap or Jeffreys intervals for binary rates) ... is a material
change and must supersede this ADR with new validation." This ADR is that new,
validated record. It follows the precedent already set by ADR-0011 (Offline
Resolution Cache) extending ADR-0010 (Offline Dataset Contract) as a separate,
later-numbered file that references the earlier one rather than editing it:
`docs/adr/0008-statistical-comparability.md` is not modified.

## Decision

- **When `attempts == 1`, behavior is unchanged.** `pass_rate`'s bounds are the
  same `wilson_interval` over the flat per-observation count as before, now
  additionally stamped `interval_method = IntervalMethod.WILSON`. Every existing
  single-attempt test passes unmodified. `wilson_interval` itself is untouched.
- **When `attempts > 1`, `pass_rate`'s bounds become cluster-robust.** Samples
  are grouped by `sample_id` (reusing `pass_at_k_by_sample`'s grouping, now
  factored into a shared `_attempts_by_sample_id`), each cluster contributes its
  pass proportion, and `clustered_interval` returns
  `mean(cluster_means) +/- z * stdev(cluster_means) / sqrt(m)` over the `m`
  clusters, stamped `interval_method = IntervalMethod.CLUSTER_ROBUST`.
  `numerator` / `denominator` / `value` stay the **exact pooled** passed/total
  counts -- only the interval's derivation changes, never the exact counts
  design section 10 requires.
- **A single cluster (`m < 2`) returns `(None, None)`.** The between-cluster
  variance is undefined with one cluster, so no fabricated zero-width interval is
  reported -- the same discipline as `wilson_interval`'s empty-denominator case.
- **`AggregateStats` gains an additive, optional `score_estimate:
  ContinuousEstimate | None`** carrying SEM and a 95% CI for `score_mean`,
  computed flat at `attempts == 1` and over per-`sample_id` mean scores at
  `attempts > 1`, and `None` when fewer than two scores are defined. Its `mean`
  is the exact pooled `score_mean` (no drift); because a score mean is not a
  probability, its bounds are not clamped to `[0, 1]`. `RateEstimate` gains an
  additive, optional `interval_method`. Both additions stay within
  `schema_version = "1"` per ADR-0002's additive-evolution rule.
- **A new stdlib-only `stats/power.py::required_sample_size`** implements the
  closed-form two-proportion z-test sample size via `statistics.NormalDist`,
  exported from `stats/__init__.py`, so a caller can size a run before trusting a
  delta.
- **Both shipped human-readable reporters now render the aggregates.**
  `reporters/markdown.py` renders `aggregates` as a Markdown table (pass rate,
  score mean, and `pass@k` when present) instead of a raw `str(dict)` repr, and
  `reporters/html.py` plus `templates/report.html.j2` gain an
  `{% if aggregates %}`-gated, visible "Uncertainty" section labeled by
  `interval_method`.

## Alternatives

1. **Redefine `pass_rate`'s numerator as "any-pass-per-sample."** Rejected: this
   just re-derives `pass_at_k` at `k = attempts` under a different name and breaks
   design section 10's exact-numerator/denominator requirement. Keeping the exact
   pooled counts and changing only the interval derivation is the minimal,
   additive, backward-compatible change.
2. **Bootstrap the clustered interval instead of a closed-form Wald interval.**
   Deferred, not rejected: the closed form keeps this change dependency-free and
   consistent with `wilson_interval`'s existing closed-form style. A future ADR
   may supersede this one if a cluster bootstrap proves necessary.
3. **Leave `score_mean` interval-free.** Rejected: it directly contradicts design
   section 10's explicit "mean or rate with an appropriate 95% confidence
   interval" promise.
4. **Label the continuous flat SEM interval with a named method.** Rejected: the
   `IntervalMethod` vocabulary names binary-rate constructions; a flat
   normal-approximation interval for a continuous mean corresponds to neither, so
   `score_estimate.interval_method` is honestly `None` in that case rather than
   mislabeled.

## Consequences

- Single-attempt runs (the common case) are byte-identical to today; no
  regression risk for existing consumers.
- Multi-attempt runs report an honestly wider, non-pseudo-replicated interval,
  machine-readably labeled with which method produced it, and a matching
  `score_estimate` for the mean.
- Cluster-robust SE over few distinct `sample_id`s is still only approximate;
  `aggregate_run` and `clustered_interval` document that caveat so a consumer
  never over-trusts a small-cluster interval.
- `docs/adr/0008-statistical-comparability.md` is not edited; this ADR is the
  validated supersession its own text requires for an interval-method change,
  following the ADR-0011-over-ADR-0010 precedent.
- No cross-package surface changes: the `Reporter.write(aggregates=...)`
  boundary is unchanged, and no ARP/sibling import is added.

## Validation

- `tests/unit/stats/test_aggregate.py`: the `WILSON`-vs-`CLUSTER_ROBUST` branch
  selection, byte-identical single-attempt bounds, `clustered_interval`'s
  closed form (recomputed inline from `statistics.fmean`/`stdev`), its
  single-cluster `(None, None)` case and its `[0, 1]` clamp, and the
  `score_estimate` population / `None` / no-drift rules -- every expected value
  computed inline, never a hardcoded decimal.
- `tests/unit/stats/test_power.py`: `required_sample_size` validated against a
  `statistics.NormalDist` closed form recomputed inside the test, plus each of
  its four invalid-input `ValueError` cases and its monotonicity in effect size,
  alpha, and power.
- `tests/unit/reporters/test_markdown_reporter.py` and `test_html_reporter.py`:
  the previously-untested `aggregates` rendering path in both formats, including
  the HTML visible-body assertion distinct from the embedded JSON blob and the
  aggregates-absent path.
- `tests/contract/test_dependency_boundary.py` and
  `tests/contract/test_provenance_drift.py` remain green, confirming no forbidden
  cross-package import and no drift in ADR-0008's provenance-field enumeration.

## Supersession

Changing the clustering unit (e.g. from `sample_id` to a coarser task-family
grouping), the interval method (e.g. to a cluster bootstrap), or the
`attempts > 1` threshold that gates clustering is a material change and must
supersede this ADR with new validation, per the same rule ADR-0008 established
for itself.
