# ADR-0002: Immutable Pydantic Contracts and Schema-Versioning Policy

## Status

Accepted

## Context

`agentic-evalkit` connects independently evolving layers — dataset providers,
benchmark adapters, execution targets, graders, statistics, and reporters —
through public wire contracts (design §5, `docs/specs/2026-07-02-agentic-evalkit-design.md`).
Those contracts cross process boundaries (subprocess targets, HTTP targets,
cached JSON pages, portable reports) and must remain stable, inspectable, and
safe to persist as evaluation evidence.

If contracts were mutable, a shared `EvalSample` or `NormalizedExecutionResult`
instance could be silently modified by one pipeline stage before a later stage
observes it, corrupting provenance without a stack trace. If contracts
silently accepted unknown fields, typos and drifted producer/consumer
versions would pass validation and fail invisibly downstream, and cached or
archived run evidence would not reveal what shape it was written in.

## Decision

Every public contract in `agentic_evalkit.models`:

- inherits a shared `FrozenModel` base built on Pydantic v2 `BaseModel` with
  `model_config = ConfigDict(frozen=True, extra="forbid")`;
- carries an explicit `schema_version: Literal["1"] = "1"` field so archived
  JSON is self-describing and forward compatibility can be checked without
  guessing;
- restricts values to JSON-compatible types (`pydantic.JsonValue`,
  `datetime`, `str`, `int`, `float`, `bool`, `None`) and uses tuples rather
  than mutable lists so an instance cannot be mutated in place through a
  contained collection;
- represents fixed-vocabulary fields (execution status, grade status) as
  `StrEnum` subclasses rather than booleans or free strings, so status is
  serialized as a readable string, cannot silently collapse into a boolean,
  and is exhaustively matchable;
- evolves **additively** within schema version `1`: new optional fields with
  safe defaults may be added without a version bump, but removing a field,
  renaming a field, changing a field's meaning, or changing a required field
  to a different type requires a new `schema_version` value and, for
  breaking wire changes, a new model version;
- performs no I/O in any model — construction, validation, and serialization
  are pure and side-effect free.

## Alternatives

1. **Mutable dataclasses with manual copy discipline.** Rejected: relies on
   every call site remembering to copy before mutating; a single missed
   `deepcopy` reintroduces shared-mutable-state bugs that are hard to
   reproduce and harder to test for.
2. **`extra="allow"` for forward compatibility.** Rejected: silently accepts
   typos and drifted fields as valid data, which is worse than a version
   bump for evaluation evidence that must be trustworthy years later. A
   consumer that needs true forward compatibility can be added deliberately
   as a new schema version instead.
3. **Plain booleans for status fields (e.g. `passed: bool`).** Rejected:
   design §5.4-§5.5 requires distinguishing `completed`/`timeout`/`cancelled`/
   `error` and `pass`/`fail`/`partial`/`error`/`abstain`/`unavailable`; a
   boolean cannot represent those without lossy conflation, which is exactly
   the failure mode design §9-§10 (objective hard gates, unavailable
   capability reporting) depends on avoiding.
4. **No `schema_version` field, relying on package version alone.** Rejected:
   package version tracks code, not wire shape; a patch release that only
   fixes a bug should not force reconsideration of archived JSON, but a
   deliberate breaking wire change must be detectable independent of
   `__version__`.

## Consequences

- Any attempted mutation of a model instance (`instance.field = value`) or
  construction with an unknown keyword raises `pydantic.ValidationError`
  immediately, at the point of the mistake rather than downstream.
- Producers and consumers on different `agentic-evalkit` versions can add
  optional fields without breaking each other as long as `schema_version`
  stays `"1"`; a `schema_version` bump is the explicit signal that old
  readers must not assume they understand the new shape.
- All public models round-trip through `model_dump_json()` /
  `model_validate_json()` with full equality, which is required for cache
  entries (ADR-0004), subprocess/HTTP wire payloads (ADR-0006), and portable
  reports (design §11.3) to be trustworthy.
- Collections in public models must be constructed as tuples (or frozen
  mappings) by callers; list-typed local data must be converted before it
  crosses into a model field.
- Because models cannot perform I/O, any field requiring computation (digest
  hashing, timestamp generation) must be computed by the caller before
  constructing the model, keeping the contract layer trivially testable and
  free of hidden side effects.

## Validation

- `tests/contract/test_models.py` asserts `DatasetRef` rejects both mutation
  and unknown fields (`test_models_are_frozen_and_forbid_unknown_fields`).
- `tests/contract/test_models.py` round-trips every public model through
  `model_dump_json()` / `model_validate_json()` and asserts equality,
  including `EvalSample` (`test_sample_round_trips_through_versioned_json`)
  and every model added in Task 2.
- `tests/contract/test_models.py` asserts `GradeStatus.ABSTAIN` is preserved
  as a distinct enum member rather than collapsing to a boolean
  (`test_grade_status_is_not_collapsed_to_boolean`).
- `uv run mypy` in strict mode confirms every model field is fully typed
  with no `Any` leakage.

## Supersession

A future breaking change to any schema-version-`1` contract must introduce a
new `schema_version` literal value (e.g. `"2"`) on the affected model(s) and
a migration note in `CHANGELOG.md`, rather than editing the meaning of
existing schema-version-`1` fields in place. This ADR governs the contracts
module until a schema-version-2 ADR supersedes the affected models.
