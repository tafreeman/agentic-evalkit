# ADR-0013: Dataset contamination metadata and canary-leak detection

## Status

Accepted (2026-07-10)

## Context

The eval-validity literature names training-data contamination and
memorization as a first-order threat to score validity (C9), and flags it as
one of the thinnest, highest-value areas: a contaminated public benchmark can
overstate capability by rewarding memorization instead of the tested skill.
Before this record, the framework carried zero machine-readable signal about
that risk: `ResolvedDataset` and `DatasetPreset` had no field distinguishing
a freshly authored, never-published evaluation set from a widely mirrored
public benchmark, and both built-in presets (`gsm8k`, `swe-bench-verified`)
are exactly the long-public shape at highest risk. Meanwhile
`LocalDatasetProvider` (ADR-0010) already gave callers the structural
mechanism for the clean alternative — author rows locally, never publish
them — without documenting it as such, and ADR-0012's grounded-citation
grader shipped canary-leak detection as a private function, unavailable to
any other grader and at risk of semantic drift from any future reusable
helper.

## Decision

Make contamination risk typed, honest, and reusable — informative, never
enforcing:

1. `models/datasets.py` gains `ContaminationStatus` (`StrEnum`: `unknown` /
   `suspect` / `verified_clean` / `confirmed_contaminated`) and
   `ContaminationMetadata` (`FrozenModel`: `status`, `authored_after`,
   `public_since`, `canary_ids`, `held_out`, with a construction-time
   invariant rejecting `held_out=True` alongside a non-null `public_since`).
   `status` is an enum, never a boolean, so "never checked" stays
   distinguishable from "checked and clean" (ADR-0002). `held_out` here
   means the eval dataset itself was never published — explicitly not the
   ADR-0007 judge-calibration held-out corpus.
2. `contamination: ContaminationMetadata | None = None` is added to both
   `ResolvedDataset` and `DatasetPreset` — additive optional fields with
   safe defaults, no `schema_version` bump (ADR-0002's additive-evolution
   clause).
3. Both built-in presets are annotated
   `ContaminationMetadata(status=SUSPECT)`. No `authored_after` /
   `public_since` dates are asserted — neither date was verified from a
   primary source, and fabricating one would violate the repo's own
   factual discipline. No `canary_ids` — canaries only make sense for a
   set this project or its caller authored.
4. `graders/contamination.py` ships three pure, stdlib-only, policy-free
   functions: `normalize_for_containment` (Unicode NFC, whitespace
   collapse, case fold — deliberately without `exact`'s numeric-shape
   rewrite), `find_canary_leaks` (normalization-insensitive containment;
   never raises; returns the caller's original spellings), and
   `canary_leak_evidence` (a fixed `GradeResult.evidence`-shaped payload).
   They report; they never decide a `GradeStatus` or `hard_gate`.
5. `graders/grounding.py` (ADR-0012) now delegates its canary check to
   `find_canary_leaks` and imports the shared normalizer, so the package
   carries exactly one tripwire semantics.

## Alternatives

- *Reuse `ResolvedDataset.card_metadata`.* Rejected: `card_metadata` is
  populated verbatim from the provider's own API response
  (provider-attested); writing a framework-asserted label into it would
  conflate two provenance sources the ADR-0002 discipline keeps separate.
- *A plain `contamination_suspect: bool`.* Rejected on the grounds ADR-0002
  already rejects boolean status fields: a boolean cannot distinguish
  "never checked" from "checked and clean".
- *Case-sensitive canary matching.* Rejected (adversarial review finding,
  2026-07-09): the grounded-citation grader already detects case-mangled
  echoes via normalized matching; a case-sensitive helper would put two
  different tripwire semantics in one package, and leak detection would
  silently weaken if the grader ever adopted the helper.
- *Auto-thread `canary_ids` into `EvalSample.metadata` at projection time.*
  Deferred: `BenchmarkAdapter.prepare` structurally receives only a
  `SourceRecord`, never a `ResolvedDataset`; wiring dataset-level metadata
  through projection is a breaking protocol change that deserves its own
  superseding record. Until then, graders receive canaries by constructor
  injection or `GraderSpec.parameters` — both existing extension points.

## Consequences

- Both built-in presets are honestly labeled `SUSPECT`; a score on either
  carries a typed prompt that it cannot back a capability claim without an
  overlap or decontamination check first. Nothing refuses to run — the
  label informs, exactly like `ResolvedDataset.gated`.
- A caller building a private eval set has a documented, typed way to
  declare `held_out=True` and register canaries, with the local provider
  (`docs/guides/providers.md`) as the supported mechanism.
- One tripwire semantics package-wide: the grounded-citation grader and the
  reusable helper share one implementation and cannot drift.
- Purely additive: no dependency added, no protocol signature changed, no
  existing call site behaves differently.

## Validation

- `tests/unit/graders/test_contamination.py`: empty-input behavior, partial
  leak subsets, case- and whitespace-mangled echoes detected, evidence
  payload shapes, JSON round-trip of every return value, agreement between
  the helper and the grounded-citation grader on a case-mangled leak, and
  an end-to-end fake grader merging `canary_leak_evidence` into a
  `GradeResult` that round-trips.
- `tests/contract/test_models.py`: full-field `ContaminationMetadata`
  round-trip through `ResolvedDataset`, enum preservation, and the
  `held_out`/`public_since` construction-time invariant.
- `tests/unit/datasets/test_catalog.py`: both built-in presets assert
  `SUSPECT`, plus a loop test failing on any future preset shipped without
  a contamination annotation.
- All run in the default hermetic suite (`pytest -m "not live"`).

## Supersession

A future change that threads contamination metadata automatically into
`EvalSample`/grading (the deferred `BenchmarkAdapter.prepare` signature
change), or that derives `ContaminationStatus` values rather than accepting
caller assertions, must supersede this ADR.
