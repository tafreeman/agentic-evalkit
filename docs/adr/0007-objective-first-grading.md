# ADR-0007: Objective-First Grading; Judges Gate Only When Calibrated

## Status

Accepted

## Context

Grading is where an evaluation framework is most tempted to overclaim.
Design §7 (`docs/specs/2026-07-02-agentic-evalkit-design.md`) requires
`agentic-evalkit` to order its evidence — cheap, deterministic, objective
checks first, subjective judgement last — and to prevent three specific
distortions:

1. A strong average silently **compensating for a failed hard requirement**
   (a solution that is elegant but does not compile).
2. A **subjective LLM judge** being trusted to gate a release without evidence
   that it agrees with humans on held-out data.
3. A component that **errored** being scored as if it had legitimately earned
   a low objective score (a silent zero), corrupting statistics.

## Decision

- **Objective-first evidence order.** Deterministic, programmatic graders
  produce the primary signal. `ExactMatchGrader` takes an injected extractor
  (it never imports `benchmarks`), applies Unicode NFC normalization,
  whitespace collapsing, opt-in case folding, and numeric canonicalization.
- **Atomic rubric criteria.** `RubricCriterion` / `Rubric` reject negative
  weights, duplicate criterion IDs, and zero-sum weights at construction, and
  require any broad/holistic criterion to set `requires_evidence=True`, so a
  vague criterion cannot silently carry weight.
- **Noncompensable hard gates.** `CompositeGrader` computes a score as the
  weighted mean over the **available numeric** sub-scores only; a failed hard
  gate is noncompensable — it forces a non-gating `FAIL` regardless of how
  high the other sub-scores are. A component that raises surfaces as `ERROR`,
  never as a silent zero.
- **Provider-neutral, calibration-gated judges.** `JudgeClient` is a
  provider-neutral protocol. A `JudgeGrader` may **gate** only when backed by
  a `CalibrationArtifact` that simultaneously: matches the current judge
  configuration by fingerprint equality; is not expired; was measured on at
  least 30 held-out human-labeled positives (TP+FN) **and** at least 30
  negatives (TN+FP); meets the required TPR and TNR thresholds; and passes a
  position-bias probe (a reversed-order second call must agree). Any failure —
  expired, fingerprint mismatch, insufficient sample, threshold miss, or
  position-bias disagreement — yields `GradeStatus.UNAVAILABLE` (with a reason
  in `evidence`, containing `"expired"` for the expiry case), never a pass.
  Uncalibrated judges cannot gate releases. Judge parse failures are retried
  at most twice (three attempts total) before abstaining.
- **Abstention is first-class.** A grader that cannot responsibly score
  abstains as a distinct outcome, separate from pass and fail.

## Alternatives

1. **Weight hard requirements very highly instead of gating.** Rejected: any
   finite weight is still compensable by a high-enough mean elsewhere; "must
   compile" is categorical, not a large coefficient.
2. **Trust a judge once it looks good on a few spot checks.** Rejected:
   without a held-out, sized, threshold-checked, bias-probed calibration
   artifact, judge agreement with humans is unmeasured, and an unmeasured
   judge gating a release is exactly the overclaim this ADR forbids.
3. **Treat a grader exception as a zero score.** Rejected: a silent zero is
   indistinguishable from a legitimate low score and poisons aggregate rates;
   `ERROR` keeps operational failure separate from task failure (ADR-0008).

## Consequences

- A run cannot report success while a hard requirement failed.
- A judge contributes to a gating decision only with auditable evidence that
  it tracks human labels on held-out data; otherwise it is explicitly
  `UNAVAILABLE`.
- Grader errors remain visible as errors, so ADR-0008's statistics separate
  operational failures from task failures cleanly.
- The position-bias probe costs a second judge call per graded sample — an
  intentional, documented cost for anyone wiring a billed judge provider.

## Validation

- `tests/unit/graders/test_exact_match.py` covers normalization and numeric
  canonicalization with an injected extractor.
- `tests/unit/graders/test_composite.py` includes the plan's verbatim
  `test_failed_hard_gate_cannot_be_averaged_away`, plus error-not-silent-zero
  and available-only weighted-mean behavior.
- `tests/unit/graders/test_rubric.py` covers rejection of negative/zero-sum
  weights, duplicate IDs, and evidence-free holistic criteria.
- `tests/unit/graders/test_judge.py` includes the verbatim
  `test_expired_calibration_cannot_gate`, plus fingerprint-mismatch,
  insufficient-sample, position-bias, and abstention cases — each asserting
  no gating pass is produced.

## Supersession

Changing the calibration minimums (held-out counts, TPR/TNR thresholds,
expiry), the position-bias policy, or allowing an uncalibrated judge to gate
under any condition is a material change and must supersede this ADR with new
calibration evidence.
