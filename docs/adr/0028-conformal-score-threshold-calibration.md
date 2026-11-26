# ADR-0028: Score thresholds are calibrated with exact finite-sample guarantees, as evidence only

## Status

Accepted

## Context

ADR-0007 and ADR-0020 decide *whether a judge may gate*: held-out true-negative
and true-positive rates, with a Wilson lower bound, must clear ratified floors.
ADR-0024 measures those rates. None of them decides *where the cut-off sits*.
`JudgeGrader` turns a score into pass or fail at `pass_score_threshold`, which
defaults to `0.5` and which calibration measurement deliberately reuses
(ADR-0024) so the artifact describes the decision production actually makes. A
user who wants a different cut-off picks it by hand, with no statement of what
the choice risks.

Conformal risk control (expected loss) and risk-controlling prediction sets
(high-probability loss) are the standard finite-sample answers to "choose the
loosest threshold whose risk the held-out data can support". They fit this
package's stance directly: they state the minimum evidence below which nothing
may be certified (the `1 / (n + 1)` floor for the expectation form, the
zero-error tail for the high-probability form), and they say *which* guarantee
is being made. They are also easy to over-read. Three misreadings matter here:

- The controlled loss is `loss * 1[accepted]`, the unconditional mass of
  "accepted and wrong". A budget of `0.05` is not 95% precision among the
  accepted items.
- The expectation form is marginal over the calibration draw. A single
  certificate can be worse than its budget; only the high-probability form says
  otherwise about one certificate.
- Both assume calibration and deployment items are exchangeable. A changed
  judge, model, dataset or prompt voids a certificate, and nothing in a
  certificate can detect that.

A worked proposal for this capability was evaluated (a standalone
`conformal-eval-bounds` package: exact-rational calibrator, Clopper-Pearson
high-probability mode, multi-stratum aggregation, Lean 4 proofs and a Lean
differential-replay oracle). Its calibrator is sound, but three properties made
it unsuitable to adopt wholesale. Its result types are plain dataclasses, not
ADR-0002 wire models. Its Lean headline theorem is stated for a supremum
selector over a strictly-accepting, left-continuous loss, whereas the shipped
selector is an inclusive, tie-bucketed finite-grid sweep; the proposal itself
lists the tie-safe finite-grid bound as outstanding, so "proven in Lean" does
not yet describe the code that runs. And it would add a Lean 4 plus Mathlib
toolchain to a package whose single merge-blocking workflow is Python-only.

## Decision

This package ships `agentic_evalkit.graders.conformal`, a pure calibrator that
returns a typed certificate, and it keeps that certificate out of every
grader's gate.

- **API.** `calibrate_score_threshold(observations, *, alpha, mode, delta,
  higher_is_riskier)` returns a frozen `ThresholdCertificate`.
  `min_calibration_samples` returns the fewest items for which any certificate
  can exist. `ScoreObservation`, `ExactRational` and `ThresholdCertificate` are
  `FrozenModel`s (`frozen=True`, `extra="forbid"`, `schema_version="1"`), with
  `StrEnum` modes. Nothing is re-exported from `agentic_evalkit.graders`; the
  module is imported by name.
- **Two modes, named for what they guarantee.** `CRC_EXPECTATION` keeps
  `(sum of accepted losses + 1) / (n + 1) <= alpha`. `RCPS_HIGH_PROB` requires
  binary losses and keeps `P(Binomial(n, alpha) <= observed errors) <= delta`
  at every accepted threshold, which is the Clopper-Pearson upper limit staying
  at or below `alpha`, tested exactly instead of through a beta function.
- **Exact arithmetic.** Budgets, `delta` and losses are `Fraction`s. A float is
  read as the decimal it prints as (`0.3` is `3/10`), so a budget written as a
  decimal literal cannot move across a boundary through binary rounding; a
  computed float is read as exactly what it prints as, and a third is passed as
  `Fraction(1, 3)`. Scores stay floats, which are totally ordered, so ties are
  exact, and `-0.0` is normalised to `0.0`. High-probability calibration that
  would need more than `MAX_EXACT_SAMPLES` items, or a floor above it, is
  refused rather than approximated.
- **Two-tier authority, as in ADR-0007.** Too few items to certify anything at
  the budget is `ADVISORY` (absent evidence). Enough items, but even the
  strictest observed threshold breaches the budget, is `UNAVAILABLE`: no
  observed threshold can be certified (accepting nothing always meets a budget
  but is not a threshold). Only `GATING` carries a threshold, `permits` fails
  closed for everything else and for non-finite scores, and the certificate's
  validator rejects field combinations no calibration could produce.
- **Sweep.** Items are swept from the strictest threshold outward and tied
  scores are accepted or rejected as one bucket. The result is the loosest
  threshold whose statistic still clears the bar. The empirical loss and the
  true risk both only grow as the threshold loosens, so certifying that one
  threshold certifies every stricter one at once and no multiplicity correction
  is needed within a call; callers certifying several strata or budgets split
  `delta` themselves.
- **Evidence only.** A certificate is advice about *choosing* a threshold. No
  module under `src/` imports the calibrator, and
  `tests/contract/test_conformal_boundary.py` pins that. A certificate carries
  no TNR/TPR floor and therefore cannot open a hard gate (ADR-0007).
- **Language.** Documentation describes the guarantees as the published
  conformal risk control and RCPS results, checked here by exact enumerations
  (below). It does not describe them as formally verified.

## Alternatives

- **Adopt the proposal as written, Lean included.** Rejected: dataclass results
  break ADR-0002; the Lean workspace adds a second toolchain and a
  multi-gigabyte dependency cache to a Python library; and the proofs do not
  cover the inclusive, tied finite-grid selector that ships, so the label
  "proven" would overstate what is established. The proofs were not built or
  reviewed as part of this decision. They can live in their own repository and
  be cited once the tie-safe finite-grid bound exists.
- **Float Clopper-Pearson via a continued-fraction beta and bisection.**
  Rejected: the equivalent binomial-tail test is exact in integer arithmetic,
  shorter, and has no convergence tolerance to argue about.
- **Read a float budget as its binary value.** Rejected: `0.3` would silently
  mean a number slightly below three tenths and refuse a calibration that
  exactly meets the budget a reader typed.
- **Wire the certificate into `JudgeGrader` or `judge_authority`.** Rejected
  for now: it would make an unratified statistic a second way to open a gate.
  It needs its own ADR with ratified floors.
- **Strata, Bonferroni splitting and an unbounded upper threshold in the first
  cut.** Deferred: each adds surface without a caller. A caller can split
  `delta` across strata and calibrate each. Without an unbounded upper
  threshold, an item scoring above every calibration score is refused, which
  is the conservative side.

## Consequences

- A user can replace the `0.5` guess with a threshold the held-out data
  supports, and can see the exact quantity that was bounded.
- Nothing changes for any existing run, report, grader or schema; the module is
  inert until called.
- The expectation form can be misquoted as a per-certificate guarantee and the
  budget as a precision. The module docstring and `reason` strings say what is
  and is not claimed; reports that quote a certificate must carry its mode.
- A certificate does not travel with the provenance that makes it valid. Tying
  a certificate to the judge fingerprint and dataset identity that produced it
  (ADR-0008, ADR-0015) is follow-on work, and until then comparing a
  certificate across changed inputs is the caller's responsibility.
- Exact arithmetic costs time on very large calibration sets, bounded by
  `MAX_EXACT_SAMPLES`.

## Validation

- `tests/unit/graders/test_conformal.py` covers the floors, the two-tier
  authority outcomes, tie atomicity, both score orientations, the
  decimal-budget boundary, the exact binomial tail against a direct sum,
  forged-certificate rejection, and exotic inputs (absurd budgets, float
  subclasses, signed zero, large calibration sets).
- `TestCrcExpectationGuarantee` takes seeded random, tie-heavy multisets in both
  orientations, enumerates every choice of held-out item, and asserts the
  averaged held-out loss never exceeds the budget. That is the finite-sample
  statement the expectation guarantee rests on, checked against the shipped
  inclusive solver with exact rationals. It samples; it is not exhaustive over
  all multisets.
- `TestRcpsCoverage` enumerates every possible calibration sample of a known
  discrete population (both orientations) and asserts the probability of
  certifying a threshold whose true risk exceeds the budget is at most `delta`,
  while also being large enough (above 0.1) that the check is not vacuous.
- `tests/contract/test_conformal_boundary.py` asserts no module under `src/`
  imports the calibrator, that the scan detects every import form, and that
  every certificate model is a frozen, extra-forbidding wire model.
- `uv run pytest -m "not live"`, `uv run mypy`, `uv run ruff check .` and
  `uv run mkdocs build --strict`.

## Supersession

A later ADR that names this one may wire certificates into a grader or report,
add strata or provenance binding, or adopt machine-checked proofs once they
cover the shipped selector. Until a ratified floor for threshold risk exists, a
certificate may inform a threshold choice and must not be what opens a hard gate.
