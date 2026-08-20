# Agentic EvalKit Low-Code Integration Layer Delivery Plan

**Status:** Proposed

**Created:** 2026-08-16

**Progress tracker:** `docs/plans/2026-08-16-arp-evalkit-evaluation-calibration-tracker.md`

## Decision

Deliver a configuration-first integration layer on top of the existing public
protocols. YAML and JSON are the primary user interfaces. A custom Python
component bundle remains available only for domain logic that cannot be
represented safely by the typed configuration model.

The architecture is **curated core plus explicit extension**:

- built-in target, mapping, and grading configurations cover common cases;
- the normal `agentic-evalkit run` command constructs the full runner;
- a manifest may explicitly select one trusted, versioned component factory;
- EvalKit never scans ambient installed plugins or imports an evaluated product;
- existing manifests and Python `EvalRunner` integrations remain supported.

## Definition of Low-Code

The first release meets the low-code claim only when all of these are true:

1. Standard callable, HTTP, subprocess, and MCP fixtures run from YAML or JSON
   with zero user-authored Python and no custom runner.
2. Local JSON/JSONL and Hugging Face rows can be projected into `EvalSample`
   values through typed field mappings.
3. Common objective correctness rules can be composed declaratively with real
   non-compensable hard gates.
4. A domain-specific integration needs at most one explicitly named component
   module and still uses the packaged CLI.
5. The evaluated system does not need to depend on EvalKit or change production
   behavior solely to be evaluated.
6. Every configuration, component, target, and grader identity is recorded in
   canonical provenance.

“Low-code” does not mean that EvalKit invents an oracle, converts an arbitrary
API automatically, or removes the need for a specialized sandbox when a task
mutates repositories or external state.

## Current Gap

EvalKit already supplies `CallableTarget`, `SubprocessTarget`, `HttpTarget`, and
`McpTarget`, along with reusable dataset, grader, reporter, provenance, and run
contracts. The general CLI path is nevertheless limited because:

- manifest `adapter` and `grader` fields resolve through hardcoded tables;
- custom datasets and graders require direct `EvalRunner` assembly;
- MCP is library-only rather than manifest-selectable;
- HTTP and callable targets assume fixed input/output shapes;
- only curated benchmark adapters and graders can be named without Python;
- prior entry-point discovery was removed and must not be silently revived.

Judge calibration is orthogonal. It establishes whether a model judge may be
authoritative; it does not make a system easier to invoke or its oracle easier
to specify.

## Configuration Contract

### One model, two serializations

YAML and JSON must decode into the same versioned Pydantic document and produce
the same canonical configuration digest. Features may not exist in only one
serialization.

Existing string references remain valid:

```yaml
adapter: gsm8k@1
grader: normalized-exact@1
```

New declarative objects are additive discriminated unions:

```yaml
schema_version: "2"
run_name: example-low-code

dataset:
  provider: local
  dataset_id: ./cases.jsonl

adapter:
  kind: field-map
  name: cases@1
  sample_id:
    path: id
  input:
    question:
      path: prompt
  reference:
    path: expected

target:
  kind: callable
  import_string: my_system.api:run
  call_style: keyword_arguments

grader:
  kind: composite
  name: objective@1
  checks:
    - kind: required-fields
      paths: [answer]
      hard_gate: true
    - kind: exact
      actual:
        output_path: answer
      expected:
        source: reference
      normalization: trim-casefold
      hard_gate: true
```

The final schema is fixed by ADR and tests. This example establishes the
required shape and safety properties, not every final field name.

### Safe field paths

Field selection supports object keys and bounded list indexes only. It does not
support method calls, arbitrary predicates, dynamic imports, Python evaluation,
shell expansion, or templates. Missing paths produce typed validation or grade
failures according to the component contract; they never silently become empty
values.

### Declarative objective graders

The first release includes:

- normalized exact equality;
- typed structural validation defined in the manifest;
- required and forbidden fields;
- numeric equality and tolerance;
- string equality, prefix, suffix, and containment;
- collection membership, subset, and cardinality;
- composite weighted scoring with explicit hard-gate children.

Open-ended regular expressions and arbitrary expressions are excluded from the
first slice. They add denial-of-service and interpretation risks without being
needed for the initial product pilots.

### Target mapping

Target configuration remains explicit:

| Target | Configuration-first support |
|---|---|
| Callable | Existing single-mapping call plus keyword-argument projection and JSON-compatible result normalization |
| HTTP | Existing EvalKit envelope plus direct-input request mode and bounded response-field extraction |
| Subprocess | Existing versioned JSONL envelope; alternate arbitrary CLI parsing is deferred |
| MCP | Existing stdio tool contract added to the manifest and CLI |

Credentials remain indirect. Manifests may name an environment-backed hook but
never contain a literal secret.

### Explicit extension escape hatch

When configuration cannot express the oracle, a manifest may name one trusted
factory:

```yaml
extensions:
  factory: my_project.evals:components
  api_version: "1"
  allow:
    - grader:workflow-invariants@1
```

The loader imports only that selected factory, validates its API version,
rejects reserved-name collisions, exposes only allowlisted components, and
records its distribution/module/configuration identity. It does not call
`importlib.metadata.entry_points()` or inspect unrelated installed packages.

## Architecture Invariants

1. **One-way dependency:** consumers import EvalKit; EvalKit never imports a
   product by name.
2. **Configuration first:** a built-in declarative component is preferred when
   it can express the behavior without weakening the oracle.
3. **Explicit execution boundary:** configured imports and extensions are
   trusted code and are disclosed before execution.
4. **No hidden interpretation:** configuration is data, not executable code.
5. **Objective-first authority:** deterministic checks gate before advisory or
   calibrated judges.
6. **Fail-visible operations:** error, timeout, cancellation, unavailable, and
   skipped execution cannot become evaluation passes.
7. **Identity-bound evidence:** canonical digests cover the resolved
   configuration and selected component identities.
8. **Backward compatibility:** current manifests and constructor-injected
   integrations remain valid throughout the migration window.

## Delivery Work Breakdown

### Phase 0 — Contract, ADR, and Red Tests

| ID | Task | Status | Acceptance evidence |
|---|---|---|---|
| LCI-001 | Record the configuration-first architecture and explicitly supersede ADR-0019 only for selected factories, not ambient discovery. | `NOT_STARTED` | Accepted ADR names trust boundary, loading algorithm, compatibility, and rejected alternatives. |
| LCI-002 | Freeze the versioned YAML/JSON manifest capability contract. | `NOT_STARTED` | Schema examples and JSON equivalents validate against the same model. |
| LCI-003 | Threat-model configured imports, field paths, credentials, target mapping, component collisions, and denial-of-service bounds. | `NOT_STARTED` | Security tests and mitigations are mapped to each boundary. |
| LCI-004 | Add red compatibility and end-to-end tests before implementation. | `NOT_STARTED` | Tests demonstrate current custom-component CLI failure and preserve all version-1 manifests. |

### Phase 1 — Unified Configuration and Component Resolution

| ID | Task | Status | Acceptance evidence |
|---|---|---|---|
| LCI-005 | Add additive string-or-object adapter and grader manifest models. | `NOT_STARTED` | Old strings and new typed objects round-trip without ambiguity. |
| LCI-006 | Add JSON manifest loading and canonical YAML/JSON equivalence. | `NOT_STARTED` | Equivalent documents produce identical resolved models and digests. |
| LCI-007 | Build a curated component registry with pure constructors. | `NOT_STARTED` | CLI and Python helpers resolve built-ins from one tested registry rather than duplicate tables. |
| LCI-008 | Implement selected-only component factory loading. | `NOT_STARTED` | Only the named factory imports; API mismatch, missing name, duplicate, and reserved collision fail typed. |
| LCI-009 | Add component allowlisting and resolved identity capture. | `NOT_STARTED` | Unlisted factory outputs are unreachable and identity appears in provenance. |
| LCI-010 | Improve validation errors and preflight disclosure. | `NOT_STARTED` | Errors identify exact configuration paths; preflight lists every executable import before run confirmation. |

### Phase 2 — Declarative Dataset Mapping and Objective Grading

| ID | Task | Status | Acceptance evidence |
|---|---|---|---|
| LCI-011 | Implement bounded field-path parsing and selection. | `NOT_STARTED` | Keys/indexes work; calls, predicates, traversal tricks, excessive depth, and malformed paths fail closed. |
| LCI-012 | Implement the field-map benchmark adapter. | `NOT_STARTED` | Local and Hugging Face rows map sample ID, input, reference, metadata, and source identity without Python. |
| LCI-013 | Implement declarative exact, string, field, and collection graders. | `NOT_STARTED` | Each returns typed details and distinguishes missing data from inequality. |
| LCI-014 | Implement declarative schema and numeric-tolerance graders. | `NOT_STARTED` | Boundary, NaN/infinity, type-confusion, and tolerance tests pass. |
| LCI-015 | Implement declarative composite and non-compensable hard gates. | `NOT_STARTED` | A weighted score cannot compensate for a failed required child. |
| LCI-016 | Bind resolved mapping and grader configuration to provenance. | `NOT_STARTED` | Any behavior-changing configuration edit changes the canonical digest and comparison identity. |

### Phase 3 — Target Completeness

| ID | Task | Status | Acceptance evidence |
|---|---|---|---|
| LCI-017 | Add MCP target configuration and CLI construction. | `NOT_STARTED` | YAML and JSON manifests run a pinned stdio MCP fixture without Python assembly. |
| LCI-018 | Add callable keyword-argument projection and bounded result normalization. | `NOT_STARTED` | Existing mapping mode remains unchanged; configured kwargs handle ordinary imported functions. |
| LCI-019 | Add HTTP direct-input and bounded response-extraction modes. | `NOT_STARTED` | Existing envelope remains default; both modes preserve retries, redaction, timeout, and status taxonomy. |
| LCI-020 | Validate subprocess configuration against the existing JSONL protocol. | `NOT_STARTED` | Conformance checker explains the required request/response envelope before a full run. |
| LCI-021 | Add target configuration identity and secret-safety regression tests. | `NOT_STARTED` | Credentials never enter manifests, digests, logs, or reports; behavior-changing target config changes identity. |

### Phase 4 — CLI Experience and Documentation

| ID | Task | Status | Acceptance evidence |
|---|---|---|---|
| LCI-022 | Extend `init` to scaffold callable, HTTP, subprocess, and MCP manifests in YAML or JSON. | `NOT_STARTED` | Generated files validate and identify which fields users must replace. |
| LCI-023 | Extend `validate`/`doctor` with component and target conformance checks. | `NOT_STARTED` | Default checks are non-executing; an explicit probe mode is clearly marked and bounded. |
| LCI-024 | Publish bring-your-own-system, configuration reference, trust model, extension, migration, and troubleshooting guides. | `NOT_STARTED` | Every configuration feature has one verified example; limitations sit beside capability claims. |
| LCI-025 | Add config-only runnable examples and remove custom-runner guidance from the primary path. | `NOT_STARTED` | Quickstarts use `agentic-evalkit run`; Python assembly remains documented as advanced API usage. |

### Phase 5 — Cross-System Conformance

| ID | Task | Status | Acceptance evidence |
|---|---|---|---|
| LCI-026 | Build neutral callable, HTTP, subprocess, and MCP conformance fixtures. | `NOT_STARTED` | All four run from YAML and equivalent JSON in clean-wheel Linux and Windows jobs. |
| LCI-027 | Run clean pinned pilots for ExecutionKit, Financial Scenario Engine, ARP, and the neutral service fixture. | `NOT_STARTED` | Three pilots are configuration-only; any fourth uses one explicit bundle and no custom runner. |
| LCI-028 | Add invalid/no-op/config-drift controls to every pilot. | `NOT_STARTED` | Valid controls pass, invalid controls fail, and behavior-changing config drift breaks comparability. |

### Phase 6 — Release and Adoption

| ID | Task | Status | Acceptance evidence |
|---|---|---|---|
| LCI-029 | Complete compatibility, security, type, lint, coverage, clean-wheel, and documentation gates. | `NOT_STARTED` | Required matrix passes at an exact release-candidate SHA with no unresolved high-risk finding. |
| LCI-030 | Publish the low-code-capable release and migrate the ARP objective gate onto it. | `NOT_STARTED` | Published wheel passes conformance smoke tests; ARP uses config-first integration or records why one explicit extension is unavoidable. |

## Documentation Deliverables

| Document | Audience | Required proof |
|---|---|---|
| Bring Your Own System quickstart | First-time evaluator | Clean install to canonical report using YAML, then JSON equivalent |
| Configuration reference | Integrators | Every field generated from or checked against the typed models |
| Target recipes | Python, HTTP, subprocess, MCP owners | Runnable fixture per target and explicit protocol limitations |
| Declarative mapping guide | Dataset owners | Local JSONL and Hugging Face examples with missing-field failures |
| Objective grading cookbook | Evaluation authors | Valid, invalid, hard-gate, and tolerance examples |
| Extension bundle contract | Advanced integrators | Selected-only factory, allowlist, collision, version, and provenance examples |
| Trust and security model | Security/release reviewers | Executable import boundary, credential handling, no ambient discovery, denial-of-service bounds |
| Migration guide | Existing EvalKit consumers | Current manifest and Python-driver paths mapped to additive low-code equivalents |
| Troubleshooting guide | Operators | Typed resolution, target, dataset, grader, and operational-status failures |

## Validation Matrix

| Layer | Required checks |
|---|---|
| Models | YAML/JSON parity, versioning, round-trip, unknown fields, legacy fixtures |
| Resolver | selected-only import, allowlist, API version, collisions, deterministic ordering |
| Mapping | missing paths, type confusion, depth/index bounds, source provenance |
| Grading | positive/negative controls, numeric boundaries, schema failures, hard-gate non-compensation |
| Targets | callable kwargs, HTTP modes/retries, subprocess protocol, MCP lifecycle, timeout/cancellation |
| Security | no ambient scanning, no expression execution, secret redaction, malicious config, output bounds |
| Evidence | config/component digests, comparison incompatibility, artifact references, operational statuses |
| Packaging | clean wheel, base dependency footprint, Linux/Windows, Python 3.11-3.14 |
| Documentation | commands executed from clean environments; outputs compared with documented expectations |

## Product Pilot Expectations

| Pilot | Preferred path | Domain-specific allowance |
|---|---|---|
| ExecutionKit | YAML/JSON callable target plus declarative cases and execution assertions | One explicit grader bundle only if execution trace invariants cannot be expressed safely |
| Financial Scenario Engine | YAML/JSON callable or subprocess target plus schema, numeric tolerance, and invariant checks | Deterministic financial oracle may be an explicit bundle if cross-field formulas exceed the first DSL |
| ARP | YAML/JSON callable target plus field mapping and objective workflow assertions | One ARP-owned component bundle for irreducible workflow semantics; no custom runner |
| Neutral HTTP/MCP fixture | Configuration only | No extension allowed; serves as the zero-code conformance proof |

These pilots validate portability. Their product repositories remain read-only
until their owners authorize consumer changes, and release evidence must use
clean worktrees at pinned SHAs.

## Release Gates

### Gate L0 — Contract

- [ ] ADR and versioned manifest contract accepted.
- [ ] YAML/JSON equivalence and legacy compatibility fixed by red tests.
- [ ] Trust boundaries and non-goals documented.

### Gate L1 — Configuration Runtime

- [ ] Declarative adapter and objective graders pass valid and invalid controls.
- [ ] All four target kinds run without a custom runner.
- [ ] No configuration path executes expressions, templates, or shell
      interpolation.

### Gate L2 — Extension Safety

- [ ] Only an explicitly named factory loads.
- [ ] Allowlist, API version, collision, and provenance rules pass adversarial tests.
- [ ] No ambient entry-point scan occurs.

### Gate L3 — Portability

- [ ] Neutral fixtures pass from equivalent YAML and JSON.
- [ ] ExecutionKit, Financial Scenario Engine, and ARP pilots produce canonical evidence.
- [ ] At least three of four product pilots are configuration-only.
- [ ] Every pilot includes a negative control.

### Gate L4 — Release

- [ ] Full QA, clean-wheel, platform, security, and documentation gates pass.
- [ ] Published artifact reproduces conformance evidence.
- [ ] ARP objective-gate integration uses the released surface.

## Stop Criteria

Do not release or advertise the layer as low-code if:

- a standard conformance fixture requires a custom runner or adapter class;
- YAML and JSON resolve differently;
- a component can load without explicit selection;
- configuration supports arbitrary code/expression evaluation or shell
  interpolation;
- an invalid/no-op control passes;
- a configuration change fails to change provenance identity;
- a timeout, error, unavailable dependency, cancellation, or skip becomes a
  passing grade;
- compatibility requires breaking current manifests without a versioned
  migration; or
- fewer than three product pilots satisfy the configuration-only target.

## Dependencies and Sequencing

- `docs/plans/2026-08-16-agentic-evalkit-master-delivery-plan.md` is the sole
  execution-order authority.
- LCI-001 through LCI-030 execute in numeric order with only one task active at
  a time.
- A phase starts only after every task and release gate in the preceding phase
  closes.
- Cross-system pilots begin only after the clean-wheel conformance step passes.
- ARP objective integration consumes the completed configuration layer rather
  than establishing a second permanent custom-runner pattern.
- Judge-backed configuration is not authoritative until the later calibration
  and human-evidence phases in the master plan close.

## Open Follow-On

Evaluate a language-neutral subprocess protocol for custom graders only after
real non-Python consumers demonstrate that the explicit Python factory and
existing harness boundaries are insufficient. It is not part of this release.
