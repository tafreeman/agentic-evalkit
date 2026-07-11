# ADR-0015: Environment and Code Fingerprints Gate Run Comparability

## Status

Accepted

## Context

ADR-0008 gates `compare_runs` on eight manifest provenance fields (adapter,
grader, target name/fingerprint policy/fingerprint, sampling temperature/
seed, attempts) and states in its own Supersession clause that "changing ...
the set of provenance fields that gate comparability is a material change
and must supersede this ADR with new validation." `EvalRunManifest` has
carried `environment_fingerprint` and `code_fingerprint` fields since
ADR-0008, and `agentic_evalkit.provenance` (added afterward) now computes
and persists both on every `run` invocation -- but `compare_runs` has never
checked either, and `tests/contract/test_provenance_drift.py` explicitly
listed both as deliberately excluded from comparability. A `compare_runs`
call could therefore bootstrap a confident-looking delta between two runs
pinned to different interpreters, platforms, or `agentic-evalkit` versions
with no disclosure -- exactly the uncontrolled confound provenance-gating
exists to catch.

## Decision

`environment_fingerprint` and `code_fingerprint` join the eight existing
rows in `stats/compare.py`'s `_PROVENANCE_CHECKS` and
`EvalRunManifest.provenance_field_names()`, checked with the same None-vs-
None-is-fine / None-vs-pinned-is-a-mismatch semantics already used for
`target_fingerprint`. `compare_runs` gains a keyword-only
`allow_cross_environment: bool = False` parameter: when `True`, a mismatch
on *only* `environment_fingerprint` and/or `code_fingerprint` is waived
rather than raised, and the waived field name(s) are recorded on a new
`ComparisonResult.waived_provenance_fields: tuple[str, ...] = ()` so the
waiver is visible in every consumer (CLI table/JSON, canonical reports)
instead of silently dropped. The other eight fields are never waivable
through this flag. The CLI's `agentic-evalkit compare` command exposes the
parameter as `--allow-cross-environment`.

## Alternatives

1. **A single blanket `force=True` bypass of all provenance checks.**
   Rejected: collapses "I intentionally compared across machines" and "I
   don't know why this run differs" into one escape hatch, defeating
   ADR-0008's "fails loudly and completely" design.
2. **Leave `environment_fingerprint`/`code_fingerprint` unchecked
   (status quo).** Rejected: the data is already computed and printed in
   every report; leaving it unchecked is an unforced validity gap, not a
   deliberate scope boundary.
3. **A separate `compare_runs_across_environments()` function.** Rejected:
   duplicates the pairing/bootstrap logic ADR-0008 already centralized in
   one function for a smaller diff than a keyword-only opt-in provides.
4. **Reuse the existing but inert `EvalRunManifest.baseline_compatibility_rules`
   field.** Rejected: it is a per-manifest, authored-ahead-of-time
   declaration, not a per-comparison decision -- a caller doesn't know what
   they'll eventually diff a run against when authoring its manifest.

## Consequences

- A `compare_runs` caller who never opts in gets strictly more protection
  than before with zero code changes required on their part.
- A caller intentionally comparing across machines/interpreters/harness
  versions has an explicit, narrow, auditable way to say so, and that
  waiver travels with the result instead of living only in the caller's
  memory.
- `ComparisonResult` gains one additive optional-default field under
  `schema_version = "1"` (ADR-0002); no wire version bump.

## Validation

- `tests/contract/test_provenance_drift.py` (updated `_EXISTING_PROVENANCE_FIELDS`
  and `comparability_excluded`) keeps the declared<->checked binding
  falsifiable for the two new fields.
- `tests/unit/stats/test_compare.py` covers per-field mismatch rejection,
  None/None backward compatibility, None-vs-pinned rejection, the
  `allow_cross_environment` waiver scoped to exactly these two fields (not
  the other eight), the default-`False` regression case, and
  `ComparisonResult` JSON round-tripping with `waived_provenance_fields`
  both populated and empty.
- `tests/integration/test_cli.py` covers `agentic-evalkit compare
  --allow-cross-environment` end to end, including JSON output carrying
  `waived_provenance_fields` and the scoping proof against a non-waivable
  mismatch.

## Supersession

Changing which fields `allow_cross_environment` may waive, or adding a
second, differently-scoped escape hatch, must supersede this ADR (per
ADR-0008's own Supersession clause, which this ADR discharges for exactly
these two fields) with new validation -- following this ADR's own precedent
of citing and narrowing a prior decision rather than editing an
already-accepted one.
