# ARP EvalKit Evaluation, Calibration, and Release Tracker

**Status:** Active planning tracker

**Created:** 2026-08-16

**Last updated:** 2026-08-16

**Scope owner:** EvalKit maintainer and ARP maintainer

**Source plans:**

- `docs/plans/2026-08-16-agentic-evalkit-master-delivery-plan.md`
- `docs/plans/2026-07-04-arp-integration-analysis.md`
- `docs/plans/2026-08-16-agentic-evalkit-low-code-integration-layer.md`

**Historical sprint status:** `_bmad-output/implementation-artifacts/sprint-status.yaml` (preserved; not updated by this tracker)

## Purpose

This file tracks the work required to deliver the configuration-first low-code
integration layer, demonstrate that Agentic EvalKit can run and correctly
evaluate ARP workflows, validate and release the judge-calibration feature,
calibrate any ARP judge-backed graders, and retire the legacy ARP gate only
after equivalent or stronger evidence exists.

The master delivery plan is the sole execution-order authority. This file is a
task and evidence catalog: only the row corresponding to the active master-plan
step may be `IN_PROGRESS`, and no second workstream starts before the active
master gate closes.

The objective workflow gate and the calibrated-judge gate are separate proof
tracks:

- Objective grading is the first release-blocking proof. It must run the four
  existing ARP golden workflow families through EvalKit with deterministic,
  authoritative hard gates.
- Judge grading is a later proof. It remains advisory until the calibration
  implementation is released and an independent, held-out, human-labeled ARP
  corpus produces an authoritative artifact bound to the live judge config.
- The unvalidated software-engineering fixtures under
  `examples/software_engineering_baseline/` are a separate future evaluation
  track. They are not evidence for the ARP integration gates in this tracker.

## How to Use This Tracker

1. Change only the status for work supported by linked evidence.
2. Add the exact commit SHA, command, run ID, report path, and date to the
   Evidence column before marking an item `DONE`.
3. Update the progress summary and change log in the same change.
4. Do not interpret `DONE` as proof beyond the individual row's acceptance
   criteria.
5. Do not mark a live or judge-backed task complete when it was skipped because
   credentials, labels, or calibration authority were unavailable.
6. Run publication and cutover gates from clean worktrees at pinned SHAs. Local
   mutable worktrees are useful for development but are not release evidence.

### Status Vocabulary

| Status | Meaning |
|---|---|
| `NOT_STARTED` | No accepted implementation or evidence exists. |
| `IN_PROGRESS` | Implementation or evidence collection has begun. |
| `BLOCKED` | Progress requires a named external decision or dependency. |
| `REVIEW` | Implementation exists and awaits independent review or gate execution. |
| `DONE` | Row-specific acceptance criteria are met and evidence is linked. |
| `NOT_APPLICABLE` | The row was intentionally removed from scope with a recorded decision. |

### Evidence Vocabulary

| Label | Meaning |
|---|---|
| `STRUCTURAL_VERIFIED` | Code or artifacts were inspected, but the behavior was not executed. |
| `RUNTIME_VERIFIED` | The named behavior was executed in a controlled local or CI environment. |
| `LIVE_VERIFIED` | The provider-backed path ran with required credentials and did not skip. |
| `RELEASE_VERIFIED` | The published package and consumer pin were installed and exercised. |
| `ADVISORY_ONLY` | Evidence demonstrates plumbing or measurement but has no gating authority. |
| `NOT_RUN` | No qualifying execution evidence exists. |

## Progress Summary

Update these counts whenever task statuses change.

| Workstream | Done | Total | Current gate status |
|---|---:|---:|---|
| E0 — Low-code integration layer | 0 | 12 | `NOT_STARTED` |
| E1 — Objective EvalKit ARP gate | 0 | 8 | `NOT_STARTED` |
| E2 — Calibration validation and release | 0 | 9 | `NOT_STARTED` |
| E3 — ARP judge calibration and evidence | 0 | 9 | `NOT_STARTED` |
| E4 — CI cutover and legacy retirement | 0 | 7 | `NOT_STARTED` |
| **Total planned implementation** | **0** | **45** | **Open** |

Historical and prerequisite evidence is tracked separately below and does not
increase these implementation totals.

## Current Baseline Snapshot

| Baseline ID | Status | Evidence level | Current evidence | Interpretation |
|---|---|---|---|---|
| BASE-01 | `DONE` | `RUNTIME_VERIFIED`, `ADVISORY_ONLY` | `reports/2026-07-26-agent-workflow-eval/`: 48/48 executions completed and passed; all 48 grades have `hard_gate=false`; no calibration references | Proves ARP reviewer-to-EvalKit plumbing for one `review_code` step. It does not prove judge authority or full workflow coverage. |
| BASE-02 | `DONE` | `RUNTIME_VERIFIED` | ARP `27bcdbb63d558a6b32fe90b7459f1eaef6db5ac1`; `scripts/eval_gate.py` targeted run: four golden workflows, scores `1.0`, `0.8`, `0.8`, `0.8`; aggregate `0.85`; `docs/release/2026-08-16-m0-baseline-acceptance.md` | Proves the legacy objective baseline still runs. It does not prove EvalKit integration. |
| BASE-03 | `DONE` | `RUNTIME_VERIFIED` | ARP `27bcdbb63d558a6b32fe90b7459f1eaef6db5ac1`; EvalKit bridge targeted and boundary tests: 28 passed; `docs/release/2026-08-16-m0-baseline-acceptance.md` | Proves the bridge's tested scoring and callable-target boundaries. It is not wired into the golden or live gate. |
| BASE-04 | `DONE` | `STRUCTURAL_VERIFIED` | ARP declares `agentic-evalkit>=0.3.0,<0.4.0`; CI constraints pin `0.3.0`; installed CLI has no `calibrate` command | Confirms the ARP consumer is on a release that predates the calibration measurement feature. |
| BASE-05 | `DONE` | `RUNTIME_VERIFIED` | EvalKit `main` and clean candidate are reconciled at `fcbcd365e1f58eb4ee7f6392fd89d4d208a0b28d`; 1,147 non-live tests pass at 94.16% coverage; lint, format, strict types, build, strict docs, and isolated-wheel smoke pass; `docs/release/2026-08-16-m0-baseline-acceptance.md` | Calibration measurement code is reconciled and has a package baseline. It is still versioned `0.3.0`, unpublished as a calibration-capable release, and not yet safe for authority until M4. |
| BASE-06 | `DONE` | `STRUCTURAL_VERIFIED` | `examples/software_engineering_baseline/` is untracked and explicitly unvalidated | Keep separate from ARP proof and published capability claims. |

## Dependency and Critical Path

```text
M0 baseline
  -> M1-M3 low-code contract, runtime, and candidate
  -> M4 calibration authority
  -> M5 combined EvalKit package release
  -> M6 ARP objective integration
  -> M7 legacy capability harvest
  -> M8 ARP judge calibration
  -> M9 CI cutover and observation
  -> M10 legacy package removal
  -> M11 final cross-package evidence
```

- The sequence above is mandatory and strictly ordered.
- Only one tracker row may be `IN_PROGRESS`.
- The permanent ARP driver uses the released E0 configuration surface rather
  than establishing another custom runner pattern.
- Judge calibration uses the combined released package and completes before CI
  cutover or legacy removal.
- Legacy retirement requires harvest closure, objective parity, judge
  validation, required CI, and the full observation cycle.

## E0 — Configuration-First Low-Code Integration Layer

**Detailed plan:**
`docs/plans/2026-08-16-agentic-evalkit-low-code-integration-layer.md`

**Outcome:** Standard Python, HTTP, subprocess, and MCP systems can be
evaluated from equivalent YAML or JSON through the packaged CLI. Domain-specific
logic uses at most one explicitly selected extension bundle and never requires
a custom runner.

| ID | Task | Status | Owner | Acceptance criteria | Evidence |
|---|---|---|---|---|---|
| EVK-LCI-001 | Accept the configuration-first ADR, versioned contract, threat model, and red compatibility tests. | `NOT_STARTED` | EvalKit architecture + QA | ADR preserves one-way dependencies and supersedes prior plugin discovery only for explicitly selected factories; red tests fix YAML/JSON parity and legacy behavior. | `NOT_RUN` |
| EVK-LCI-002 | Implement one typed manifest model with equivalent YAML and JSON loading. | `NOT_STARTED` | EvalKit | Existing string component names and new declarative objects validate and round-trip; equivalent serializations have the same canonical digest. | `NOT_RUN` |
| EVK-LCI-003 | Replace CLI-only hardcoded tables with a curated, constructor-based component registry. | `NOT_STARTED` | EvalKit | Built-ins resolve through one tested registry without ambient environment scanning or public-protocol breakage. | `NOT_RUN` |
| EVK-LCI-004 | Implement safe declarative field mapping for local and Hugging Face records. | `NOT_STARTED` | EvalKit | Sample identity, input, reference, metadata, and source identity map without Python; malformed, missing, or unsafe paths fail typed. | `NOT_RUN` |
| EVK-LCI-005 | Implement declarative objective graders and composite hard gates. | `NOT_STARTED` | EvalKit + QA | Exact, schema, field, numeric, string, collection, and composite checks pass adversarial controls; failed hard gates cannot be compensated. | `NOT_RUN` |
| EVK-LCI-006 | Implement the explicit selected-only extension factory escape hatch. | `NOT_STARTED` | EvalKit | Only the configured factory imports; API version, allowlist, collision, and trust-boundary checks pass; no entry-point scan occurs. | `NOT_RUN` |
| EVK-LCI-007 | Complete manifest/CLI target support and safe input/output projection. | `NOT_STARTED` | EvalKit | Callable, HTTP, subprocess, and MCP conformance fixtures run through configuration; existing protocols, retries, redaction, timeouts, and cancellation semantics remain intact. | `NOT_RUN` |
| EVK-LCI-008 | Add configuration scaffolding, validation, doctor, and non-executing conformance checks. | `NOT_STARTED` | EvalKit | CLI generates YAML or JSON for all targets, reports exact configuration errors, and discloses executable imports before run confirmation. | `NOT_RUN` |
| EVK-LCI-009 | Bind resolved configuration and component identity into evidence and comparability. | `NOT_STARTED` | EvalKit | Behavior-changing mapping, grader, target, or extension changes alter provenance and block invalid comparisons. | `NOT_RUN` |
| EVK-LCI-010 | Publish and execute the full low-code documentation set. | `NOT_STARTED` | Docs + QA | Bring-your-own-system, config reference, target recipes, mapping, grading, extension, trust, migration, and troubleshooting guides reproduce from clean installs. | `NOT_RUN` |
| EVK-LCI-011 | Run cross-platform clean-wheel conformance and product pilots. | `NOT_STARTED` | QA + product owners | YAML and JSON fixtures pass on Linux/Windows; ExecutionKit, Financial Scenario Engine, ARP, and a neutral service produce canonical evidence; at least three pilots are configuration-only. | `NOT_RUN` |
| EVK-LCI-012 | Release the low-code-capable EvalKit version and make ARP consume it. | `NOT_STARTED` | EvalKit + ARP | Published wheel passes conformance; ARP objective integration uses configuration-first components or records why one explicit bundle is necessary; no custom runner remains. | `NOT_RUN` |

## E1 — Objective EvalKit ARP Workflow Gate

**Outcome:** The four existing ARP golden workflow families run through
EvalKit with objective, non-compensable hard gates and canonical provenance.

| ID | Task | Status | Owner | Acceptance criteria | Evidence |
|---|---|---|---|---|---|
| EVK-ARP-001 | Freeze the objective-gate contract and map each existing golden case to its workflow input, expected output, and authoritative assertions. | `NOT_STARTED` | ARP + EvalKit | Mapping covers code review, bug resolution, consensus review, and full-stack generation; subjective judge dimensions are excluded from the required gate. | `NOT_RUN` |
| EVK-ARP-002 | Implement an ARP-owned EvalKit driver using public `ExecutionTarget`, `DatasetProvider`, and `Grader` contracts. | `NOT_STARTED` | ARP | Driver loads the existing four golden cases, invokes the real workflow entry points, and does not require EvalKit to import ARP internals. | `NOT_RUN` |
| EVK-ARP-003 | Implement workflow-specific objective graders and genuine hard gates. | `NOT_STARTED` | ARP | Each case asserts required output values/schema/behavior; a high weighted score cannot compensate for a failed required assertion. | `NOT_RUN` |
| EVK-ARP-004 | Add valid, no-op, and deliberately invalid controls. | `NOT_STARTED` | QA | Valid controls pass; no-op and invalid controls fail for the intended reason; any invalid control passing blocks the gate. | `NOT_RUN` |
| EVK-ARP-005 | Bind canonical provenance and artifacts. | `NOT_STARTED` | EvalKit + ARP | Report identifies dataset, case, ARP SHA, EvalKit version/SHA, target, grader, dependencies, config, seed, timestamps, and artifact digests. | `NOT_RUN` |
| EVK-ARP-006 | Add bounded execution and operational-failure semantics. | `NOT_STARTED` | ARP | Timeout, exception, unavailable dependency, and cancellation are distinct from evaluation failure; none can be reported as a pass. | `NOT_RUN` |
| EVK-ARP-007 | Add required deterministic CI at clean pinned SHAs. | `NOT_STARTED` | ARP | Network-free job runs all four cases plus controls, publishes the canonical bundle, and fails closed on any hard-gate or operational failure. | `NOT_RUN` |
| EVK-ARP-008 | Produce and independently review the first qualifying objective proof bundle. | `NOT_STARTED` | QA / reviewer | Clean run passes all valid cases, fails all negative controls, contains resolvable provenance, and satisfies Gate A below. | `NOT_RUN` |

## E2 — Calibration Feature Validation and Release

**Outcome:** EvalKit ships a versioned calibration measurement and authority
contract that is safe for downstream judge gating.

The current remote-tracking implementation is an input to this workstream, not
release evidence. Before release, resolve the remaining authority questions:

- enforce `total_labeled == TP + TN + FP + FN + abstained + errors`;
- bind the calibration artifact to every behavior-changing judge setting,
  including pass threshold, prompt/rubric identity, parser/schema version,
  redaction policy, and truncation/bounding configuration;
- decide and encode whether artifacts missing the new coverage/config fields
  are always advisory rather than inheriting legacy gating behavior.

| ID | Task | Status | Owner | Acceptance criteria | Evidence |
|---|---|---|---|---|---|
| EVK-CAL-001 | Reconcile local `main` with the reviewed remote calibration commits in a clean worktree and record the exact candidate SHA. | `NOT_STARTED` | EvalKit | Candidate is based on an explicit SHA; the user's untracked fixtures and unrelated changes are not included. | `STRUCTURAL_VERIFIED`: local branch is three commits behind `origin/main`; candidate not yet executed. |
| EVK-CAL-002 | Run the complete hermetic calibration, grader, CLI, model, serialization, and compatibility test matrix. | `NOT_STARTED` | EvalKit | All required tests, lint, type checks, clean-wheel install, and CLI smoke checks pass from the candidate artifact. | `NOT_RUN` |
| EVK-CAL-003 | Enforce labeled-count invariants. | `NOT_STARTED` | EvalKit | Construction and deserialization reject inconsistent totals and impossible confusion-matrix/count combinations; adversarial tests cover each path. | `NOT_RUN` |
| EVK-CAL-004 | Bind exact judge and preprocessing configuration identity. | `NOT_STARTED` | EvalKit | Authority check compares model/fingerprint, prompt and rubric digest, parser/schema version, pass threshold, redaction policy, and truncation/bounding settings. Any drift demotes to advisory. | `NOT_RUN` |
| EVK-CAL-005 | Define safe compatibility semantics for older/incomplete calibration artifacts. | `NOT_STARTED` | EvalKit | Missing coverage or configuration identity cannot silently obtain hard-gate authority; decision is documented and tested. | `NOT_RUN` |
| EVK-CAL-006 | Add adversarial authority tests. | `NOT_STARTED` | QA | Tests cover forged totals, stale artifact, class-floor failure, Wilson-bound failure, excessive non-verdicts, config drift, reversed label/order behavior, missing fields, and fingerprint mismatch. | `NOT_RUN` |
| EVK-CAL-007 | Validate measurement statistics and thresholds against hand-computed fixtures. | `NOT_STARTED` | QA / statistical reviewer | Confusion counts, rates, Wilson lower bounds, non-verdict rate, class floors, and authority outcome match independent expected values. | `NOT_RUN` |
| EVK-CAL-008 | Complete release documentation and publish the calibration-capable EvalKit version. | `NOT_STARTED` | EvalKit | ADR/API docs/changelog/version are aligned; published wheel installs cleanly; `calibrate` smoke test and grader consumption pass against the published artifact. | `NOT_RUN` |
| EVK-CAL-009 | Update ARP's dependency range and exact CI constraint to the released version. | `NOT_STARTED` | ARP | Lock/constraints are consistent; ARP installs from the release channel, not a co-located checkout; bridge and objective gates pass. | `NOT_RUN` |

## E3 — ARP Judge Calibration and Evidence

**Outcome:** Any judge-backed ARP grader has measured, configuration-bound
authority over a held-out corpus; otherwise it is reported as advisory.

Target corpus size is at least 100 independently labeled positive examples and
100 negative examples unless a statistical review approves a different design.
The implementation minimum remains 30 examples per class, but this is only an
eligibility floor. With zero false positives, 73 negative examples are needed
for the 95% Wilson TNR lower bound to clear a `0.95` floor.

| ID | Task | Status | Owner | Acceptance criteria | Evidence |
|---|---|---|---|---|---|
| EVK-JDG-001 | Define the ARP review-quality decision, rubric, label schema, exclusion rules, and adjudication protocol. | `NOT_STARTED` | ARP + QA | Human labelers can reach a binary reference decision without seeing judge outputs; ambiguous/excluded examples are explicitly handled. | `NOT_RUN` |
| EVK-JDG-002 | Build an independent held-out corpus with provenance and leakage controls. | `NOT_STARTED` | ARP | Target is at least 100 positive and 100 negative examples; source, license, split, deduplication, and access controls are recorded. | `NOT_RUN` |
| EVK-JDG-003 | Complete blind labeling, agreement analysis, and adjudication. | `NOT_STARTED` | Human reviewers | Labels are independent of judge results; disagreements are adjudicated; class counts and exclusions are auditable. | `NOT_RUN` |
| EVK-JDG-004 | Implement the ARP `JudgeClient` and stable behavior fingerprint. | `NOT_STARTED` | ARP | Fingerprint and artifact identity capture every setting named in EVK-CAL-004; credentials and raw sensitive content are not persisted. | `NOT_RUN` |
| EVK-JDG-005 | Measure calibration on held-out labels and generate the signed/digested artifact. | `NOT_STARTED` | ARP + EvalKit | Artifact is produced by the released calibration feature; counts reconcile; age is at most 90 days; non-verdict rate is at most 5%. | `NOT_RUN` |
| EVK-JDG-006 | Make an explicit authority decision. | `NOT_STARTED` | QA / statistical reviewer | TPR/TNR Wilson lower bounds and all identity/coverage requirements clear the configured floors; otherwise status remains advisory and blockers are recorded. | `NOT_RUN` |
| EVK-JDG-007 | Rerun the 48-case reviewer evaluation at clean pinned SHAs. | `NOT_STARTED` | ARP | Full run has no errors/timeouts/skips; each case links its calibration artifact and reports whether the judge grade was gating or advisory. | `NOT_RUN` |
| EVK-JDG-008 | Add invalid, low-quality, prompt-injection, and config-drift sentinels. | `NOT_STARTED` | QA | Sentinels demonstrate discrimination and fail-safe demotion; a known-bad sentinel passing as authoritative blocks publication. | `NOT_RUN` |
| EVK-JDG-009 | Publish an authority-aware reviewer evidence bundle. | `NOT_STARTED` | ARP + EvalKit | Report separates execution success, objective grades, advisory judge grades, and authoritative judge grades; no advisory result is described as a pass gate. | `NOT_RUN` |

## E4 — CI Cutover and Legacy Retirement

**Outcome:** ARP uses EvalKit as its governed evaluation path, live skips are
visible, and the legacy gate is removed only after parity and rollback evidence.

| ID | Task | Status | Owner | Acceptance criteria | Evidence |
|---|---|---|---|---|---|
| EVK-CUT-001 | Add explicit required-live semantics. | `NOT_STARTED` | ARP | A `--require-live` or equivalent mode reports `NOT_RUN` and exits nonzero when credentials/provider execution are absent; optional local runs may still skip visibly. | `NOT_RUN` |
| EVK-CUT-002 | Add scheduled/manual provider-backed execution. | `NOT_STARTED` | ARP | Live job records provider/model identity, does not expose secrets, distinguishes provider failure from grade failure, and proves it did not skip. | `NOT_RUN` |
| EVK-CUT-003 | Run side-by-side legacy/EvalKit parity. | `NOT_STARTED` | ARP + QA | Same pinned inputs and workflow code run through both paths; expected differences are explained; EvalKit preserves or strengthens required outcomes. | `NOT_RUN` |
| EVK-CUT-004 | Make the EvalKit objective job required and preserve rollback. | `NOT_STARTED` | ARP | Required branch/CI gate uses EvalKit; legacy path remains available for one observation cycle with documented rollback instructions. | `NOT_RUN` |
| EVK-CUT-005 | Complete one full successful CI observation cycle. | `NOT_STARTED` | ARP | Required objective gate is stable across the agreed cycle; artifact retention and failure triage have been exercised. | `NOT_RUN` |
| EVK-CUT-006 | Retire the legacy gate and stale documentation. | `NOT_STARTED` | ARP | Harvest inventory is dispositioned; dependencies/scripts/workflows/docs are removed or redirected; no production or CI import remains. | `NOT_RUN` |
| EVK-CUT-007 | Archive the release evidence and final decision. | `NOT_STARTED` | EvalKit + ARP | Gate L and Gate A-D decisions, exact SHAs, package versions, commands, reports, exceptions, and rollback outcome are stored together and independently reviewable. | `NOT_RUN` |

## Release and Publication Gates

### Gate L — Low-Code Integration Release

- [ ] Equivalent YAML and JSON configurations resolve identically.
- [ ] Callable, HTTP, subprocess, and MCP fixtures run through the packaged CLI
      without custom runners or adapter classes.
- [ ] Declarative mappings and objective graders pass valid, invalid, and
      non-compensable hard-gate controls.
- [ ] Explicit extensions are selected-only, allowlisted, identity-bound, and
      never discovered by scanning the environment.
- [ ] Existing manifests and public Python integrations remain compatible.
- [ ] Clean-wheel pilots cover ExecutionKit, Financial Scenario Engine, ARP,
      and a neutral service; at least three are configuration-only.
- [ ] Documentation commands reproduce from a clean install.

**Decision:** `OPEN`

**Release candidate:** Not yet selected.

### Gate A — Objective ARP Proof

- [ ] All four golden workflow families execute through EvalKit.
- [ ] Required behavior is enforced by objective hard gates.
- [ ] Valid controls pass and all no-op/invalid controls fail.
- [ ] No timeout, error, unavailable dependency, cancellation, or skipped live
      path is counted as a pass.
- [ ] Canonical bundle contains resolvable, pinned provenance.
- [ ] Required deterministic CI passes from a clean worktree.

**Decision:** `OPEN`

**Evidence bundle:** Not yet produced.

### Gate B — Calibration Feature Release

- [ ] Count invariants and exact configuration identity are enforced.
- [ ] Incomplete/legacy artifacts cannot silently become authoritative.
- [ ] Adversarial, statistical, compatibility, and clean-package tests pass.
- [ ] Documentation and version metadata describe actual behavior.
- [ ] Published artifact installs and runs the calibration smoke test.
- [ ] ARP consumes the published artifact at an exact CI-resolved version.

**Decision:** `OPEN`

**Release candidate:** Not yet selected.

### Gate C — ARP Judge Authority

- [ ] Human-labeled corpus and protocol pass independent review.
- [ ] Class floors, Wilson lower bounds, maximum non-verdict rate, and maximum
      age all pass.
- [ ] Live configuration exactly matches the calibration artifact.
- [ ] Negative and config-drift sentinels fail or demote as designed.
- [ ] The rerun separates execution status from grading authority.

**Decision:** `OPEN — judge results remain advisory`

**Calibration artifact:** Not yet produced.

### Gate D — Legacy Retirement

- [ ] Gate A is closed.
- [ ] Gate L is closed.
- [ ] Side-by-side parity is accepted.
- [ ] EvalKit required CI completes one successful observation cycle.
- [ ] Harvest inventory is fully dispositioned.
- [ ] Rollback and evidence archive are complete.

**Decision:** `OPEN`

**Earliest retirement:** After the observation cycle; no date committed.

## Stop and Rollback Criteria

Stop publication, authority promotion, or cutover when any of the following is
true:

- an invalid/no-op/known-bad control passes;
- YAML and JSON configurations resolve differently;
- a standard target requires a custom runner, or a component loads without
  explicit selection;
- declarative configuration executes arbitrary expressions, templates, or
  shell interpolation;
- any required case has an execution error, timeout, unavailable dependency,
  cancellation, or unacknowledged skip;
- report artifacts or provenance references cannot be resolved;
- the calibration corpus is unavailable, leaked into tuning, inadequately
  labeled, or below required class/statistical coverage;
- confusion counts do not reconcile with `total_labeled`;
- the live judge config differs from the calibration artifact;
- TPR/TNR Wilson lower bounds, non-verdict rate, age, position-bias, or
  fingerprint requirements fail;
- a calibration or judge result obtains hard-gate authority through missing or
  legacy fields;
- the new objective gate produces unexplained behavior weaker than the legacy
  gate; or
- the required CI gate cannot complete the agreed observation cycle reliably.

Rollback means restoring the last known required gate while preserving failed
run artifacts for diagnosis. It does not mean deleting or rewriting evidence.

## Immediate Next Action

G0 is closed. The next authorized master-plan action is **MDP-007** in M1.
Do not start EVK-LCI-001, EVK-CAL-001, or EVK-ARP-001 out of master-plan order.

## Decisions Required

| Decision ID | Decision | Needed by | Owner | Status |
|---|---|---|---|---|
| DEC-01 | Exact configuration fields that form calibration identity, including threshold and preprocessing settings | EVK-CAL-004 | EvalKit architecture | `OPEN` |
| DEC-02 | Compatibility behavior for calibration artifacts missing new coverage/config fields | EVK-CAL-005 | EvalKit architecture + QA | `OPEN` |
| DEC-03 | ARP objective assertions for each of the four golden workflows | EVK-ARP-003 | ARP maintainer | `OPEN` |
| DEC-04 | Human labeling protocol, corpus sources, and reviewer independence | EVK-JDG-002 | ARP + QA | `OPEN` |
| DEC-05 | Required CI observation-cycle duration before legacy retirement | EVK-CUT-005 | ARP release owner | `OPEN` |
| DEC-06 | Whether a later release needs a language-neutral custom-grader protocol beyond the existing subprocess and harness boundaries | After EVK-LCI-012 | EvalKit architecture | `DEFERRED` |

## Change Log

| Date | Change | Evidence / decision |
|---|---|---|
| 2026-08-16 | Completed MDP-001 through MDP-006 and closed G0. | `docs/release/2026-08-16-m0-baseline-acceptance.md`; M1 remains `NOT_STARTED`. |
| 2026-08-16 | Subordinated tracker execution order to the master delivery plan and removed multi-workstream activation. | Only one M0-M11 task may be active; all rows reset to `NOT_STARTED`. |
| 2026-08-16 | Added E0 configuration-first low-code integration workstream and Gate L; made the released E0 surface the permanent path for ARP objective integration. | `docs/plans/2026-08-16-agentic-evalkit-low-code-integration-layer.md`; user preference establishes YAML/JSON over custom code. |
| 2026-08-16 | Created separate tracker; preserved historical BMAD sprint status and separated objective, calibration-release, judge-authority, and cutover tracks. | Current repository and runtime baseline summarized in BASE-01 through BASE-06. |
