# ADR-0012: Grounded-citation probe — deterministic-primary, rubric-bound-judge grading family

## Status

Accepted (2026-07-09)

## Context

Host repositories need a reusable way to evaluate grounded question
answering: does an answer cite real documents from a trusted corpus, quote
them verbatim, cover all required evidence, and avoid echoing planted
do-not-cite distractor tokens? This is the NIST AREP grounded-citation
probe — three orthogonal axes: faithfulness (anti-hallucination),
completeness (anti-cherry-pick), sufficiency (anti-overreach) — and the
eval-validity literature shows scoring defects (weak verifiers, empty
responses counted as success, free-form numeric LLM ratings) distort
results more than capability differences do. The package already had every
needed primitive — `Rubric`/`RubricCriterion` (never yet bound to a live
judge request), `JudgeGrader` with the ADR-0007 calibration floor,
`CompositeGrader` hard gates, `exact`'s canonicalization, byte-sha256
dataset pinning — but no grading family composing them, and no adapter for
grounded-QA records.

## Decision

Add one grading family and one adapter, composed from existing primitives:

1. `models/grounding.py` — frozen wire contracts per ADR-0002.
   `GroundedCitationTask` validates fail-closed at load time: gold spans
   must be verbatim substrings of their documents, required evidence must
   name known documents, every document embeds a unique canary token. Gold
   is non-numeric by construction (document IDs and verbatim spans only).
2. `graders/grounding.py` — `GroundedCitationGrader`, the deterministic
   LLM-free primary tier: structured-contract, answer-nonempty,
   citation-present, citation-resolution, verbatim quote-faithfulness
   (reusing `exact`'s NFC/whitespace/case-fold steps, minus its
   numeric-shape rewrite — an equality rule that would corrupt substring
   containment), required-evidence coverage, and canary-leak. Binary
   primary outcome; per-check breakdown kept as auxiliary evidence, and a
   citation counts toward presence/coverage only when its normalized quote
   carries at least `MIN_SUBSTANTIVE_QUOTE_TOKENS` tokens (closing the
   degenerate one-word-quote pass found in adversarial review). The tier is
   labeled in every grade's evidence as a grounding-hygiene floor, not an
   answer-correctness verdict.
   `build_grounding_rubric()` expresses the three axes as atomic, binary,
   `requires_evidence=True` criteria; `RubricBoundJudgeClient` renders them
   into every judge request — the first production binding of the rubric
   module to `JudgeGrader`. The composed judge fingerprint covers the
   rubric content, so a rubric edit invalidates calibration like a model
   or prompt change. The reversed-order position-bias probe is made
   concrete (criteria enumerated in reverse). Judges are instructed to
   return per-criterion PASS/FAIL with evidence-citing rationales — never
   a numeric rating.
3. `build_grounded_citation_grader()` — a `CompositeGrader` with the
   deterministic tier hard-gated and the judge tier advisory at weight 0.0
   by default (verdict recorded in child evidence, score-inert). Judge
   hard-gating without a `CalibrationArtifact` is structurally impossible
   at construction — the judge component is wired gate-capable only when
   calibration evidence is supplied — and `JudgeGrader` still demotes at
   grade time unless the ratified floor holds (ADR-0007).
4. `benchmarks/grounding.py` — `GroundedCitationAdapter` projects records
   into `EvalSample`s with a strict oracle/input split: the target sees
   only the question and canary-field-stripped documents; required
   evidence, the canary registry, and gold spans live in grading-only
   metadata.
5. CLI registration: adapter `grounded-citation-tasks@1` and grader
   `grounded-citation@1`, with the judge tier wired to the packaged
   reference judge client (uncalibrated, advisory, weight 0.0), following
   the `judge-reference@1` precedent.

## Alternatives

- Judge-primary scoring: rejected — an uncalibrated judge gating is what
  ADR-0007 forbids; a deterministic primary tier has zero judge attack
  surface on the required gate.
- Exact-match of answers against gold spans: rejected — rigid span
  matching is a validity anti-pattern; outcome-based checks over cited
  evidence grade what was produced.
- Host-repo-local grading: rejected — the primitives live here, and
  duplicating grading policy in a host repo would fork the invariants.
- Extending `JudgeResponse` with per-criterion fields: deferred until a
  calibrated judge is wired; the rendered prompt already demands
  per-criterion verdicts.

## Consequences

- A host expresses a grounded-citation eval as a manifest: `local` dataset
  + `grounded-citation-tasks@1` + `grounded-citation@1` + any
  `ExecutionTarget`, revision-pinned and provenance-gated as usual
  (ADR-0008).
- The deterministic tier re-runs offline against a stored report's
  outputs, so a hermetic CI gate can re-grade committed golden outputs
  with no keys, network, or model.
- A judge FAIL can never move a composite score at weight 0.0; raising the
  weight or gating requires calibration evidence — the advisory boundary
  is structural.
- Canary tokens double as contamination tripwires: any echo — including a
  case-mangled one, since leak detection matches after containment
  normalization — hard-fails.
- `CompositeGrader` child entries now carry each child's own `evidence`,
  so the per-check audit trail survives into composite reports instead of
  being flattened to status/score.
- One more entry in each CLI known-component table; the adapter table's
  "every name matches a preset" comment no longer held and was corrected.

## Validation

- `tests/unit/graders/test_grounding.py`: every check's pass and fail
  branch, normalization, UNAVAILABLE/ABSTAIN separation, the
  degenerate-quote substance floor, rubric policy, prompt rendering, the
  concrete reversed-order probe, fingerprint composition, weight-0.0 score
  inertness, hard-gate propagation, and the calibration-requires-client
  construction guard.
- `tests/unit/benchmarks/test_grounding_adapter.py`: oracle/input split,
  `DatasetSchemaMismatch` on malformed records, each fail-closed task
  validator.
- `tests/integration/test_grounded_citation_manifest.py`: manifest-driven
  CLI run resolving both registrations end to end, with the hard gate
  firing as a graded task failure and zero operational errors.
- All run in the default hermetic suite (`pytest -m "not live"`).

## Supersession

Supersedes nothing. Revisit when a calibrated sufficiency-judge client
exists (judge weight/gating defaults, per-criterion `JudgeResponse`
extension) or when a second grounded dataset family outgrows this task
contract.
