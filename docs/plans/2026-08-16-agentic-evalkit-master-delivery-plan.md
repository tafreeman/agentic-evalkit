# Agentic EvalKit Master Delivery Plan

**Status:** Active master plan

**Created:** 2026-08-16

**Last updated:** 2026-08-16

**Execution model:** Strictly serial; one active task at a time

**Progress record:** `docs/plans/2026-08-16-arp-evalkit-evaluation-calibration-tracker.md`

## Authority

This is the sole execution-order authority for delivering the EvalKit low-code
configuration layer, validating and releasing calibrated judges, replacing
ARP's legacy evaluation package, testing the affected packages, and publishing
the final evidence.

The following documents supply implementation detail but do not define an
independent execution order:

- `docs/plans/2026-08-16-agentic-evalkit-low-code-integration-layer.md`
- `docs/plans/2026-07-04-arp-integration-analysis.md`
- `docs/plans/2026-08-16-arp-evalkit-evaluation-calibration-tracker.md`

If a supporting document conflicts with this plan's order or gates, this plan
wins. Point-in-time claims in the July integration analysis must be reverified
before implementation.

## Serial Execution Rule

1. Phases execute in the order M0 through M11.
2. A phase starts only after the preceding phase's gate is recorded `CLOSED`.
3. Within a phase, steps execute in numeric order.
4. Only one master-plan step may be `IN_PROGRESS` at any time.
5. A blocked step blocks every later step. Work does not jump to another phase.
6. Review, remediation, rerun, and evidence capture are part of the active step;
   they complete before the next step starts.
7. No legacy ARP evaluation code is deleted until M10.
8. No judge result is authoritative until M8 closes its authority gate.
9. Every package release or consumer cutover uses clean worktrees at exact SHAs.

## Final Outcome

Completion means the following conditions have been met in order and remain
true:

- EvalKit is a separately released package with YAML and JSON as its primary
  low-code integration interface.
- Standard callable, HTTP, subprocess, and MCP evaluations use the packaged CLI
  without a custom runner.
- Domain-specific logic uses declarative objective graders when possible and at
  most one explicitly selected extension bundle when necessary.
- Calibration artifacts are measured, statistically validated, configuration
  bound, and unable to acquire authority through missing or inconsistent data.
- ARP's deterministic and live evaluation paths use released EvalKit contracts.
- ARP's legacy `agentic-v2-eval` package, imports, build jobs, and documentation
  are removed only after parity, harvest, CI observation, and rollback gates.
- EvalKit, ARP, ExecutionKit, and Financial Scenario Engine demonstrations pass
  their required package and integration tests at pinned SHAs.
- Canonical evidence clearly separates execution status, objective grading,
  advisory judgment, and authoritative judgment.

## Scope Boundaries

In scope:

- configuration-first YAML/JSON manifests;
- declarative dataset mapping and objective grading;
- callable, HTTP, subprocess, and MCP target configuration;
- explicit selected-only extension factories;
- package tests, security tests, clean-wheel tests, docs, and releases;
- ARP objective and judge-backed evaluation migration;
- legacy-evaluation harvest, parity, CI cutover, and removal;
- read-only conformance demonstrations against ExecutionKit and Financial
  Scenario Engine before any separately authorized consumer changes.

Out of scope:

- automatically inferring correctness;
- ambient plugin discovery;
- weakening the one-way dependency boundary;
- provider-specific judge clients inside EvalKit core;
- replacing repository sandboxes or specialized benchmark harnesses;
- treating the unvalidated software-engineering fixtures as release evidence;
- deleting ARP's independent server evaluation pipeline or benchmark-data
  directories without a separate usage and ownership decision.

## Master Status

| Phase | Outcome | Status | Gate |
|---|---|---|---|
| M0 | Freeze and verify the starting state | `DONE` | G0 `CLOSED` |
| M1 | Accept the low-code and safety contract | `NOT_STARTED` | G1 `OPEN` |
| M2 | Implement the configuration-first runtime | `NOT_STARTED` | G2 `OPEN` |
| M3 | Validate the low-code release candidate | `NOT_STARTED` | G3 `OPEN` |
| M4 | Correct and validate calibration authority | `NOT_STARTED` | G4 `OPEN` |
| M5 | Test and publish the combined EvalKit release | `NOT_STARTED` | G5 `OPEN` |
| M6 | Integrate ARP's objective gate through released EvalKit | `NOT_STARTED` | G6 `OPEN` |
| M7 | Harvest and disposition legacy evaluation capabilities | `NOT_STARTED` | G7 `OPEN` |
| M8 | Calibrate and validate ARP judge-backed grading | `NOT_STARTED` | G8 `OPEN` |
| M9 | Cut over CI and complete the observation cycle | `NOT_STARTED` | G9 `OPEN` |
| M10 | Remove ARP's legacy evaluation package | `NOT_STARTED` | G10 `OPEN` |
| M11 | Run final cross-package validation and publish evidence | `NOT_STARTED` | G11 `OPEN` |

## M0 — Freeze and Verify the Starting State

**Purpose:** Replace stale narrative with reproducible baselines before changing
contracts or consumers.

1. **MDP-001 — Establish repository ownership and clean worktrees.** Record
   EvalKit, ARP, ExecutionKit, and Financial Scenario Engine repository paths,
   branches, exact SHAs, remotes, tags, dependency versions, and `git status`.
   Preserve all unrelated modified and untracked work.
2. **MDP-002 — Reconcile the EvalKit candidate branch.** Resolve the local
   branch being behind `origin/main`, identify the exact calibration commits,
   and create a clean candidate worktree without copying the unvalidated
   software-engineering fixtures.
3. **MDP-003 — Refresh the ARP evaluation usage map.** Re-run the import, CI,
   dependency, data, script, documentation, package, and workspace inventory
   from the July analysis against current ARP.
4. **MDP-004 — Capture EvalKit's package baseline.** Run the current required
   hermetic tests, lint, strict types, build, clean-wheel import/CLI smoke, and
   documentation checks; record commands and artifacts.
5. **MDP-005 — Capture ARP's legacy baseline.** Run the four deterministic
   golden workflows, bridge tests, legacy package tests, and current required
   ARP matrix; archive exact results and operational statuses.
6. **MDP-006 — Classify existing evidence.** Preserve the July 48-case reviewer
   run as `ADVISORY_ONLY`, record why it lacks calibration authority, and
   separate it from the unvalidated software-engineering fixture track.

### G0 — Baseline Gate

- [x] All four repositories have pinned starting SHAs and preserved worktree
      inventories.
- [x] EvalKit and ARP baseline commands and results are archived.
- [x] Current ARP legacy consumers and deletion targets are enumerated.
- [x] Advisory and unvalidated evidence is labeled accurately.

**Decision:** `CLOSED` — accepted 2026-08-16. Evidence:
`docs/release/2026-08-16-m0-baseline-acceptance.md`.

## M1 — Accept the Low-Code and Safety Contract

**Purpose:** Fix the configuration and trust boundaries before implementation.

1. **MDP-007 — Accept the configuration-first ADR.** Establish YAML/JSON as the
   primary interface, current public protocols as compatibility boundaries, and
   Python as an explicit escape hatch only.
2. **MDP-008 — Fix one versioned document model.** Define equivalent YAML and
   JSON serialization, additive string-or-object component fields, schema
   versioning, canonicalization, and validation-error paths.
3. **MDP-009 — Fix declarative mapping and grader scope.** Specify bounded field
   paths and exact, schema, field, numeric, string, collection, and composite
   hard-gate graders without expressions, templates, or shell interpolation.
4. **MDP-010 — Fix target configuration scope.** Specify callable mapping and
   keyword calls, HTTP envelope/direct-input modes, the existing subprocess
   JSONL protocol, and MCP manifest construction.
5. **MDP-011 — Fix the selected-only extension contract.** Specify explicit
   factory import, API version, allowlist, reserved-name collision behavior,
   trust disclosure, and prohibition of ambient entry-point scanning.
6. **MDP-012 — Add red contract and threat tests.** Demonstrate the current
   failures and lock compatibility, configuration safety, secret handling,
   denial-of-service bounds, and provenance identity before implementation.

### G1 — Contract Gate

- [ ] ADR and typed schema are accepted.
- [ ] Existing manifest and Python API compatibility is explicit.
- [ ] YAML/JSON equivalence is fixed by failing tests.
- [ ] Selected-only loading and non-executable configuration are fixed by
      failing security tests.

**Decision:** `OPEN`

## M2 — Implement the Configuration-First Runtime

**Purpose:** Make the packaged CLI the normal integration surface.

1. **MDP-013 — Implement the unified manifest loader.** Add YAML/JSON decoding,
   schema dispatch, legacy strings, declarative objects, and canonical digests.
2. **MDP-014 — Implement the curated component registry.** Replace duplicated
   CLI tables with deterministic constructors while preserving built-in names.
3. **MDP-015 — Implement bounded field selection and the field-map adapter.**
   Map sample ID, input, reference, metadata, and source identity from local and
   Hugging Face records with typed missing-path behavior.
4. **MDP-016 — Implement declarative primitive graders.** Deliver exact,
   string, field, collection, schema, and numeric-tolerance checks with typed
   details and negative controls.
5. **MDP-017 — Implement declarative composition.** Support weights and genuine
   non-compensable hard-gate children without changing existing grader behavior.
6. **MDP-018 — Implement complete target configuration.** Add callable
   projection, HTTP mapping, subprocess conformance checks, and MCP manifest/CLI
   construction while retaining timeout, retry, cancellation, and redaction
   semantics.
7. **MDP-019 — Implement selected-only extensions.** Load only the named
   factory, validate API version and allowlist, reject collisions, and never
   scan installed distributions.
8. **MDP-020 — Bind resolved behavior to evidence.** Include configuration,
   mapping, grader, target, extension, environment, and code identities in
   provenance and comparison compatibility.
9. **MDP-021 — Implement the CLI experience.** Extend `init`, `validate`, and
   `doctor` for YAML/JSON scaffolding, non-executing checks, optional bounded
   probes, exact error paths, and executable-import disclosure.
10. **MDP-022 — Complete implementation tests.** Make every M1 red test pass,
    run component unit/integration tests, and confirm legacy fixtures remain
    green.

### G2 — Runtime Gate

- [ ] Standard components resolve from YAML and equivalent JSON.
- [ ] Configuration does not evaluate arbitrary code, expressions, templates,
      or shell interpolation.
- [ ] Only an explicitly selected extension can import.
- [ ] Invalid/no-op controls fail and hard gates cannot be compensated.
- [ ] Resolved behavior changes alter provenance and comparison identity.

**Decision:** `OPEN`

## M3 — Validate the Low-Code Release Candidate

**Purpose:** Prove portability before adding calibration changes or modifying
ARP.

1. **MDP-023 — Build neutral conformance fixtures.** Create callable, HTTP,
   subprocess, and MCP systems evaluated from YAML and equivalent JSON without
   custom runners or adapter classes.
2. **MDP-024 — Validate package platforms.** Run the full EvalKit matrix from a
   clean candidate worktree across supported Python versions and required Linux
   and Windows jobs.
3. **MDP-025 — Validate the built wheel.** Install the wheel into clean
   environments and run manifest validation, all four target fixtures,
   canonical reporting, comparison, and error-taxonomy smoke tests.
4. **MDP-026 — Run the ExecutionKit conformance pilot.** Prefer existing
   importable callables plus configuration; permit no consumer change without
   separate authorization. Record any irreducible extension need.
5. **MDP-027 — Run the Financial Scenario Engine pilot.** Exercise deterministic
   schema, numeric tolerance, and invariant grading from configuration; permit
   no consumer change without separate authorization.
6. **MDP-028 — Execute the documentation.** Reproduce bring-your-own-system,
   manifest reference, mapping, objective-grading, target, extension, trust,
   migration, and troubleshooting instructions from clean environments.
7. **MDP-029 — Resolve every candidate finding.** Rerun the entire M3 sequence
   after remediation; do not carry known release defects into calibration work.

### G3 — Low-Code Candidate Gate

- [ ] Four neutral target fixtures pass from YAML and JSON.
- [ ] ExecutionKit and Financial Scenario Engine pilots produce canonical
      evidence without modifying their production code.
- [ ] Full source and clean-wheel package gates pass.
- [ ] Documentation commands reproduce exactly.
- [ ] No unresolved release-blocking defect remains.

**Decision:** `OPEN`

## M4 — Correct and Validate Calibration Authority

**Purpose:** Make calibration artifacts safe before packaging the combined
release.

1. **MDP-030 — Reconcile the calibration implementation.** Apply the reviewed
   calibration measurement commits to the clean candidate and record their
   exact identity.
2. **MDP-031 — Enforce count invariants.** Require `total_labeled` to equal all
   verdict, abstention, and error counts; reject impossible and inconsistent
   confusion matrices during construction and deserialization.
3. **MDP-032 — Bind exact judge behavior.** Include model/fingerprint, prompt
   and rubric digest, parser/schema version, pass threshold, redaction policy,
   and truncation/bounding configuration in calibration identity.
4. **MDP-033 — Fail incomplete artifacts advisory.** Ensure legacy or missing
   coverage/configuration fields cannot silently obtain hard-gate authority.
5. **MDP-034 — Validate statistics independently.** Check class floors, TPR/TNR,
   Wilson lower bounds, non-verdict rate, age, and authority outcomes against
   hand-computed fixtures.
6. **MDP-035 — Add adversarial authority tests.** Cover forged totals, stale
   artifacts, thin classes, threshold drift, preprocessing drift, fingerprint
   mismatch, missing fields, malformed judge envelopes, and order-bias failure.
7. **MDP-036 — Run the complete calibration matrix.** Execute models, CLI,
   serialization, grader consumption, compatibility, type, lint, and package
   tests after all fixes.

### G4 — Calibration Authority Gate

- [ ] Counts reconcile and statistical results match independent fixtures.
- [ ] Every behavior-changing judge setting is identity-bound.
- [ ] Missing, stale, thin, drifted, or forged artifacts remain advisory.
- [ ] Calibration and judge-grader test matrices pass.

**Decision:** `OPEN`

## M5 — Test and Publish the Combined EvalKit Release

**Purpose:** Release one package containing both the low-code surface and the
validated calibration feature before permanent ARP adoption.

1. **MDP-037 — Complete release documentation.** Align ADRs, manifest and CLI
   reference, migration guide, calibration guide, changelog, version, package
   metadata, and limitations.
2. **MDP-038 — Run source quality gates.** Execute the full non-live tests,
   coverage requirement, lint, strict type checks, contract tests, security
   tests, and strict documentation build from the release-candidate SHA.
3. **MDP-039 — Run clean-package gates.** Build sdist/wheel, inspect contents,
   install without the repository checkout, run CLI/help/doctor/validate/run/
   calibrate/report smoke tests, and execute YAML/JSON conformance.
4. **MDP-040 — Verify optional integrations.** Exercise each declared extra and
   confirm missing optional infrastructure reports typed `UNAVAILABLE` rather
   than a pass or crash.
5. **MDP-041 — Publish the versioned artifact.** Use the approved release
   channel, record tag and immutable artifact digests, and verify installation
   from that channel in a fresh environment.
6. **MDP-042 — Close the release acceptance record.** Link exact commands,
   reports, exceptions, versions, and reviewer approval.

### G5 — EvalKit Release Gate

- [ ] G1 through G4 are closed.
- [ ] Source and clean-package matrices pass at one exact SHA.
- [ ] Published YAML/JSON and calibration smoke tests pass.
- [ ] Package version, tag, wheel digest, and release evidence agree.

**Decision:** `OPEN`

## M6 — Integrate ARP's Objective Gate Through Released EvalKit

**Purpose:** Establish deterministic EvalKit evidence alongside the legacy gate
without deleting or disabling the rollback path.

1. **MDP-043 — Add the released dependency.** Update ARP's dependency range,
   lock, exact CI constraints, install paths, import smoke, and two-way boundary
   tests without a co-located EvalKit checkout.
2. **MDP-044 — Decouple unconditional runtime imports.** Move the small LLM
   protocol into ARP ownership and retain behavior while removing the legacy
   package's import-time control over the runtime.
3. **MDP-045 — Define the four-workflow objective contract.** Map code review,
   bug resolution, consensus review, and full-stack generation inputs,
   expected outputs, and non-compensable assertions.
4. **MDP-046 — Configure the objective evaluation.** Use YAML/JSON field mapping,
   target configuration, and declarative graders; allow one explicit ARP bundle
   only for documented irreducible semantics and never a custom runner.
5. **MDP-047 — Add valid, no-op, and invalid controls.** Prove discrimination
   and operational status handling for every workflow family.
6. **MDP-048 — Run side-by-side parity.** Execute the legacy and EvalKit gates
   against identical pinned inputs and workflow code; explain every difference.
7. **MDP-049 — Add a non-blocking EvalKit CI job.** Publish canonical reports
   and retain the legacy required job unchanged.
8. **MDP-050 — Complete the objective observation sample.** Record the required
   consecutive clean runs defined at M0 and resolve all flakes or drift before
   moving forward.

### G6 — ARP Objective Gate

- [ ] ARP installs released EvalKit from its release channel.
- [ ] Four workflow families and their negative controls run through the
      packaged configuration surface.
- [ ] No custom runner exists in the permanent integration.
- [ ] Side-by-side outcomes are equivalent or stronger and fully explained.
- [ ] Canonical provenance resolves to exact ARP and EvalKit identities.

**Decision:** `OPEN`

## M7 — Harvest and Disposition Legacy Evaluation Capabilities

**Purpose:** Preserve valuable behavior before deleting its only implementation.

1. **MDP-051 — Refresh the harvest inventory.** Recheck every prior H1-H12 item,
   caller, test, data dependency, and documentation claim against current ARP.
2. **MDP-052 — Port objective metrics.** Rewrite retained accuracy/F1,
   code-quality, performance, and non-compensatory aggregation behavior against
   EvalKit contracts with behavioral parity tests.
3. **MDP-053 — Port judge prompt packs and bias controls.** Preserve selected
   pattern, quality, standard, order, schema, and drift mechanisms as ARP-owned
   configuration or reviewed EvalKit contributions without copying ARP imports
   upstream.
4. **MDP-054 — Port benchmark definitions that retain value.** Move definitions
   and adapters onto EvalKit's provider contracts; do not port defective legacy
   transport/cache code.
5. **MDP-055 — Preserve the dual-gate operating pattern.** Document the
   deterministic required floor, explicit live execution, credential behavior,
   sampling, status, and report retention using implementation-backed wording.
6. **MDP-056 — Record written dispositions.** Mark every legacy capability
   ported, retained elsewhere, consciously dropped, or separately deferred with
   an owner and evidence.
7. **MDP-057 — Run harvest parity tests.** Demonstrate that retained behavior is
   covered before any legacy file is eligible for deletion.

### G7 — Harvest Gate

- [ ] Every legacy capability has a current written disposition.
- [ ] Retained metrics, prompts, controls, and benchmark definitions have tests.
- [ ] The deliberately dropped sandbox is not advertised as preserved.
- [ ] No unresolved production caller depends solely on code scheduled for M10.

**Decision:** `OPEN`

## M8 — Calibrate and Validate ARP Judge-Backed Grading

**Purpose:** Promote judge results only through independent human evidence.

1. **MDP-058 — Freeze the decision and labeling protocol.** Define the review
   quality decision, rubric, label schema, exclusions, blind labeling,
   disagreement handling, adjudication, and leakage controls.
2. **MDP-059 — Build the held-out corpus.** Target at least 100 positive and 100
   negative independently labeled examples; document source, license,
   deduplication, access, and split provenance.
3. **MDP-060 — Complete labeling and adjudication.** Measure agreement, resolve
   disagreements without judge-output exposure, and reconcile all counts.
4. **MDP-061 — Implement the ARP judge client and identity.** Bind the exact
   released model, prompt, rubric, parser, threshold, redaction, and bounding
   behavior required by G4.
5. **MDP-062 — Measure calibration.** Produce the artifact through the released
   EvalKit CLI, verify class/statistical coverage, and retain thin or failed
   artifacts as advisory evidence rather than discarding them.
6. **MDP-063 — Run discrimination and drift sentinels.** Cover low-quality,
   invalid, prompt-injection, order, fingerprint, threshold, preprocessing, and
   missing-field cases.
7. **MDP-064 — Make the authority decision.** Require all configured TPR/TNR
   Wilson floors, class coverage, non-verdict, age, identity, and bias checks;
   otherwise judges remain advisory and M8 stays open.
8. **MDP-065 — Rerun the 48-case reviewer evaluation.** Use clean pinned SHAs,
   require no errors/timeouts/skips, link every judge grade to the calibration
   artifact, and separate execution success from grading authority.
9. **MDP-066 — Publish the authority-aware evidence bundle.** Report objective,
   advisory, and authoritative outcomes distinctly.

### G8 — Judge Authority Gate

- [ ] Human corpus, labels, agreement, and adjudication pass independent review.
- [ ] Calibration authority clears every configured statistical and identity
      requirement.
- [ ] Sentinels fail or demote exactly as designed.
- [ ] The 48-case rerun contains no operational failure or hidden skip.
- [ ] Advisory output is never described as an authoritative pass.

**Decision:** `OPEN — judges remain advisory`

## M9 — Cut Over CI and Complete the Observation Cycle

**Purpose:** Make EvalKit the governed ARP evaluation path while rollback remains
available.

1. **MDP-067 — Add explicit required-live semantics.** Missing credentials or
   provider execution becomes visible `NOT_RUN` and nonzero when live execution
   is required.
2. **MDP-068 — Install final released versions.** Update ARP's dependency lock
   and constraints to the calibration-capable EvalKit release used in M8.
3. **MDP-069 — Promote the EvalKit objective gate to required.** Keep the legacy
   gate available only as the documented rollback comparator.
4. **MDP-070 — Configure live execution.** Run scheduled/manual provider-backed
   evaluation with secret safety, exact provider identity, and distinct
   provider-versus-grade failure semantics.
5. **MDP-071 — Repoint package and repository tooling.** Update CI installs,
   SBOM, dependency audit, deploy, pre-commit, development commands, ownership,
   and docs without deleting legacy source yet.
6. **MDP-072 — Complete one full observation cycle.** Exercise deterministic,
   live, failure-triage, artifact-retention, and rollback procedures for the
   duration fixed at M0.
7. **MDP-073 — Close all observation findings.** Rerun the full cycle after any
   gate-affecting fix.

### G9 — Cutover Gate

- [ ] EvalKit objective CI is required and stable.
- [ ] Required live execution cannot silently skip.
- [ ] One full observation cycle is complete, including a qualifying live run.
- [ ] Rollback has been exercised and documented.
- [ ] G6 through G8 remain closed on the final pinned versions.

**Decision:** `OPEN`

## M10 — Remove ARP's Legacy Evaluation Package

**Purpose:** Delete the replaced implementation only after its behavior,
evidence, and rollback conditions are satisfied.

1. **MDP-074 — Create the pre-removal recovery point.** Record the final legacy
   and EvalKit comparison bundles and create the approved recoverable tag or
   equivalent reference.
2. **MDP-075 — Rewrite remaining consumers.** Update evaluation examples,
   score/report scripts, cross-package tests, guarded scoring paths, data docs,
   and development tooling to use released EvalKit or their written disposition.
3. **MDP-076 — Delete `agentic-v2-eval`.** Remove its source package only after
   confirming no unresolved caller and no undispositioned harvest item remains.
4. **MDP-077 — Remove repository wiring.** Drop the workspace member, package
   build/deploy jobs, dependency watches, ownership rules, legacy CI jobs, and
   obsolete scoped hooks; relock and regenerate constraints.
5. **MDP-078 — Update architecture and user documentation.** Record EvalKit as
   the external evaluation/evidence plane, mark historical material accurately,
   and remove stale legacy-package instructions.
6. **MDP-079 — Run the ARP package matrix.** Execute tests, docs, pre-commit,
   no-LLM, dependency/lock, SBOM/audit, objective gate, judge gate, examples, and
   install/import tests with no legacy checkout present.
7. **MDP-080 — Run reference and boundary sweeps.** Confirm no production, CI,
   packaging, docs, or test dependency on `agentic-v2-eval` remains and the
   one-way EvalKit boundary still passes.

### G10 — Legacy Removal Gate

- [ ] G7 through G9 are closed.
- [ ] Recovery reference and comparison evidence are resolvable.
- [ ] Legacy source and repository wiring are absent.
- [ ] ARP installs, tests, documents, and evaluates successfully without the
      legacy package.
- [ ] No EvalKit module imports or vendors ARP, ExecutionKit, or product code.

**Decision:** `OPEN`

## M11 — Final Cross-Package Validation and Evidence Release

**Purpose:** Prove the completed system from published artifacts rather than
mutable worktrees or narrative claims.

1. **MDP-081 — Reinstall every published artifact cleanly.** Use exact package
   versions and empty environments; record artifact hashes and dependency
   resolutions.
2. **MDP-082 — Run EvalKit's final package matrix.** Execute source, wheel,
   platform, CLI, YAML/JSON, calibration, reporter, integration-extra, security,
   and documentation gates.
3. **MDP-083 — Run ARP's final package and evaluation matrix.** Execute all
   required repository gates, four objective workflows, controls, live-required
   behavior, judge authority checks, and legacy-absence sweeps.
4. **MDP-084 — Rerun ExecutionKit and Financial Scenario Engine pilots.** Use the
   published EvalKit package and configuration-only paths where demonstrated in
   M3; do not broaden consumer scope without authorization.
5. **MDP-085 — Verify evidence integrity.** Resolve every dataset, configuration,
   component, target, grader, calibration, environment, code, and artifact
   identity; reject broken references and incompatible comparisons.
6. **MDP-086 — Review capability claims.** Describe only runtime-verified and
   release-verified behavior; label advisory, unavailable, unvalidated, and
   separately planned software-engineering fixtures beside the relevant claim.
7. **MDP-087 — Publish the final acceptance record.** Include all gate decisions,
   exact commands, versions, SHAs, reports, exceptions, limitations, recovery
   reference, and independent approval.

### G11 — Final Acceptance Gate

- [ ] G0 through G10 remain closed at final published versions.
- [ ] EvalKit and ARP complete their final package matrices.
- [ ] ExecutionKit and Financial Scenario Engine conformance evidence resolves.
- [ ] All objective, advisory, authoritative, and operational outcomes are
      labeled correctly.
- [ ] Final claims are supported by immutable evidence.

**Decision:** `OPEN`

## Required Package Test Matrix

The active phase records the exact command versions before execution. A green
subset never substitutes for a required full gate.

| Package or boundary | Required validation |
|---|---|
| EvalKit source | Non-live pytest and coverage, contract tests, calibration tests, low-code unit/integration tests, lint, strict types, security/adversarial tests, strict docs |
| EvalKit distribution | sdist/wheel inspection, clean install, import/version, CLI commands, YAML/JSON parity, four target fixtures, reports, calibration artifact production and consumption |
| EvalKit optional integrations | Declared extras installed at compatible versions; real integration surface exercised; absent infrastructure returns typed unavailable status |
| ARP repository | Current required tests, docs, pre-commit, no-LLM, lock/constraint parity, dependency audit, SBOM, imports, examples, objective and live/judge gates |
| ARP migration boundary | Installed EvalKit contains no product imports; ARP contains no vendored/shadow EvalKit; legacy package references reach zero at M10 |
| ExecutionKit and Financial Scenario Engine | Their current required package tests plus published-EvalKit conformance pilot; consumer changes require separate authorization |
| Evidence | Canonical reports, redaction, artifact resolution, provenance/config/calibration identity, comparison compatibility, and operational status taxonomy |

## Stop Conditions

The active phase stays open when any of these occurs:

- a later phase starts before the active gate closes;
- more than one master-plan task is marked `IN_PROGRESS`;
- YAML and JSON resolve differently;
- a standard target needs a custom runner or hidden adapter class;
- configuration executes arbitrary expressions, templates, or shell
  interpolation;
- an extension loads without explicit selection and allowlisting;
- an invalid/no-op/known-bad control passes;
- a timeout, error, cancellation, unavailable dependency, or skipped live run
  becomes a pass;
- calibration counts do not reconcile or live judge behavior differs from the
  artifact identity;
- an advisory judge result is promoted or described as authoritative;
- a legacy capability lacks a written disposition;
- package, clean-wheel, docs, security, boundary, or consumer gates fail;
- provenance or artifact references cannot be resolved; or
- unrelated user work would be overwritten or deleted.

## Tracker Update Protocol

After each master step:

1. record the master task ID, exact SHA, command, result, artifact path, and date;
2. update the corresponding E0-E4 tracker row only when its acceptance criteria
   are satisfied;
3. keep the current phase `IN_PROGRESS` until its gate closes;
4. close the gate in this plan and update the Master Status table;
5. only then mark the next phase `IN_PROGRESS`.

## Change Log

| Date | Change | Decision |
|---|---|---|
| 2026-08-16 | Completed MDP-001 through MDP-006 and closed G0. | Four repository identities and worktree inventories are pinned; EvalKit and ARP baselines are archived; ARP consumers/deletion targets are refreshed; advisory and unvalidated evidence are separated. M1 remains `NOT_STARTED`. |
| 2026-08-16 | Created the consolidated master delivery plan. | One strict M0-M11 sequence governs low-code configuration, EvalKit package testing/release, ARP objective migration, calibration and judge validation, legacy harvest/removal, CI observation, and final evidence. |
