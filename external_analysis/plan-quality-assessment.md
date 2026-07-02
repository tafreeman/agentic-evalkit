# agentic-evalkit Initial Release — Plan Quality Assessment

**Assessor:** Claude Code (independent analysis)
**Date:** 2026-07-02
**Documents reviewed:**
- `README.md`
- `docs/specs/2026-07-02-agentic-evalkit-design.md` (design specification, 20 sections)
- `docs/plans/2026-07-02-agentic-evalkit-initial-release.md` (16-task implementation plan)
- `external_analysis/agentic-evalkit-plan-review.md` (prior external review)

---

## Executive Summary

The plan is **well-constructed at its core** — the architectural decisions are sound, the methodological discipline is high, and the grading/statistical rigor exceeds what most evaluation tooling delivers. However, it carries **scope risk** (16 tasks under the label "initial release"), **missing public-package artifacts**, and an **absent integration demonstration** for its motivating use case (evaluating ARP). These are addressable without architectural changes.

**Overall verdict: GO — with conditions.**

---

## 1. What Is Good

### 1.1 The Central Architectural Decision Is Correct

The plan inverts the eval↔runtime dependency: instead of evaluation logic living inside ARP or ExecutionKit, `agentic-evalkit` becomes the *consumer* of those systems through a neutral `ExecutionTarget` protocol. The invariant `agentic-evalkit -X-> ARP/EK` is enforced by AST-based dependency-boundary tests and clean-wheel tests run outside all source trees.

`★ Insight ─────────────────────────────────────`
This is an application of the Dependency Inversion Principle at the system level. The framework depends on abstractions (ExecutionTarget protocol), not concretions (ARP, EK). This means ARP and EK can evolve independently, and any system with a callable, subprocess, or HTTP surface becomes evaluable — not just ARP. The same pattern appears in pytest's plugin architecture and OpenTelemetry's provider model.
`─────────────────────────────────────────────────`

### 1.2 ADR-Before-Code Discipline

Nine Architecture Decision Records are mapped to governing tasks, each committed *before* the production code it governs. This is unusually rigorous and ensures:

- Decisions are documented and reviewable independently of implementation
- The rationale for each architectural choice survives the original authors
- Future contributors can understand *why* the system looks the way it does

### 1.3 Immutable Contract Design

Pydantic v2 frozen models with `schema_version`, `extra="forbid"`, and explicit versioning policy. Key design choices:

- `FrozenModel` base class enforces immutability at the framework level — no accidental mutation of evaluation data
- `schema_version: Literal["1"]` bakes versioning into every wire contract
- Tuples instead of mutable lists in public models
- JSON-compatible value types throughout

`★ Insight ─────────────────────────────────────`
The `extra="forbid"` setting on Pydantic models is load-bearing for an evaluation framework. It means a typo in a manifest field (`concurreny: 4` instead of `concurrency: 4`) fails at parse time with a clear error, rather than silently accepting the field and using a default. In evaluation contexts where reproducibility matters, silent configuration errors are worse than loud failures.
`─────────────────────────────────────────────────`

### 1.4 Grading Integrity Is Genuinely Well-Designed

The grading system has several non-obvious correctness properties:

| Property | How it's enforced |
|---|---|
| Hard gates cannot be averaged away | `CompositeGrader` test explicitly asserts FAIL when any hard gate fails, regardless of weighted score |
| Uncalibrated judges cannot gate releases | `JudgeGrader` requires non-expired calibration with minimum 30+30 held-out labels and TPR/TNR above threshold |
| Missing harness ≠ passing grade | `UnavailableHarnessExecutor` returns typed `unavailable`, never a substitute score |
| Objective evidence beats model judgment | Explicit evidence ordering in design §9: authoritative verifier → executable tests → schema validation → exact match → domain metric → calibrated judge → human review |

### 1.5 Statistical Rigor

Many evaluation tools report naive accuracy with no uncertainty. This plan includes:

- **Wilson 95% confidence intervals** for binary rates (not normal approximation — Wilson is correct for small samples and proportions near 0 or 1)
- **Deterministic seeded bootstrap** for paired deltas (reproducible, no hidden randomness)
- **Explicit attempt accounting** so missing attempts cannot inflate `pass@k`
- **Rejection of incompatible comparisons** with enumerated reasons rather than misleading deltas
- **Separate counting** of pass/fail/partial/error/timeout/cancelled/abstain/unavailable — no collapsing into a single "score"

### 1.6 Test-First TDD Throughout

Every task follows red-green-refactor with explicit failing test code provided in the plan. This is unusually thorough for a plan document and demonstrates that the contracts are testable before implementation begins.

### 1.7 Honest Scope Control

The plan explicitly defers the Docker SWE-bench verifier (Slice 5) to a follow-on plan while still shipping the *contracts* (Tasks 8-9) that the verifier will implement. This avoids the highest-risk, highest-dependency work until the framework is proven.

### 1.8 Clean Architecture Boundaries

The file map enforces separation of concerns:

> "models do not perform I/O, providers do not grade, targets do not know benchmark types, graders do not execute targets, and reporters consume completed run models only."

This is a well-articulated set of architectural rules that prevents the common eval-framework anti-pattern of everything depending on everything.

### 1.9 Security Consciousness

- Remote dataset code disabled by default
- Credential redaction from logs and artifacts
- Bounded subprocess output (size limits enforced before parsing)
- No host execution of untrusted benchmark code
- Field-level redaction in reports
- AST-based forbidden-import enforcement

---

## 2. What Is Lacking

### 2.1 No ARP Integration Demonstration (HIGH)

The plan proves the framework with GSM8K + an echo target. There is no end-to-end demonstration of evaluating an actual ARP workflow. The motivating use case — "evaluate agentic systems" where ARP is the primary system — is absent from the initial release evidence.

**Risk:** The framework may be technically correct but practically irrelevant if the first real integration reveals gaps in the target protocol or result normalization.

### 2.2 Missing Public-Package Artifacts (HIGH)

If the decision is to publish this as a public package, the following are absent from the file map:

- `LICENSE` — legal requirement for distribution
- `CHANGELOG.md` — user-facing change tracking
- `CONTRIBUTING.md` — contributor onboarding
- `SECURITY.md` — vulnerability reporting process
- PyPI trusted-publishing workflow — the plan builds wheels (`uv build`) but has no publish step

### 2.3 No Positioning or Differentiation (MEDIUM)

The README is one sentence. The evaluation-tooling space includes Inspect (UK AISI), Harbor, LightEval (Hugging Face), OpenAI Evals, and Langfuse. There is no statement of why `agentic-evalkit` exists alongside these tools.

The real differentiator — **ARP/EK-neutral boundary + objective-first grading with authoritative-harness separation** — is implicit in the architecture but never stated as a value proposition.

### 2.4 No Migration Path (MEDIUM)

Existing ARP evaluation code exists. The plan states "do not modify ARP or EK" and "migrating ARP's existing evaluation code is a non-goal for the initial release." This is fine as a scope boundary, but there is no discussion of:

- How existing ARP eval code *relates* to the new framework
- Whether the two systems coexist, or one replaces the other
- What the eventual migration strategy looks like

### 2.5 Missing Error Recovery Patterns (MEDIUM)

The plan classifies errors well (typed provider errors, typed grader errors, typed target errors) but does not address:

- **Partial run resumption:** If a 1000-sample run fails at sample 847, can the user resume from 847?
- **Checkpointing:** Are intermediate results persisted so a crash doesn't lose all progress?
- **Retry policy for transient failures:** The HTTP target has retry logic, but the runner pipeline itself has no retry strategy for provider flakes or target timeouts.

### 2.6 No Performance or Scalability Targets (LOW)

No discussion of:

- Expected throughput (samples/second) for each target type
- Memory behavior with datasets of 10K, 100K, or 1M rows
- Streaming vs. in-memory semantics for `iter_records`
- Cache size management or eviction policy

### 2.7 Limited Subgroup Analysis Design (LOW)

Subgroup support is mentioned ("predeclared sample tags") but not designed in detail:

- How are subgroups declared in the manifest?
- What minimum sample size triggers a warning?
- Are intersectional subgroups supported?
- How do subgroups interact with paired comparison?

### 2.8 No Observability for the Framework Itself (LOW)

The events system (Task 11) emits progress events, but there is no discussion of:

- Structured logging for the framework when used as a library
- Metrics export (Prometheus/OpenTelemetry)
- Tracing across provider → adapter → target → grader boundaries

### 2.9 Coverage Gate Is Aggressive (LOW)

The plan sets `fail_under = 90` (branch-aware). For an initial release including a Jinja2 HTML reporter, Typer CLI surface, and async I/O paths, 90% branch coverage is aggressive. The user's own global standard targets 80%.

---

## 3. What Should Be Changed

### 3.1 Split Slice 4 into 4a and 4b (CRITICAL)

The prior review identified this correctly. Slice 4 cannot be deferred wholesale because the CLI `run` command (Task 14) and the runner (Task 11) consume statistics and reporters.

**Proposed split:**

| Slice 4a (keep with Slice 3 — required for working `run`) | Slice 4b (defer, reevaluate at checkpoint) |
|---|---|
| Objective graders only (exact, schema, composite, rubric) | Calibrated judges (`JudgeGrader`, `CalibrationArtifact`) |
| Runner with minimal counts summary | Full statistics (Wilson CIs, bootstrap, `pass@k`, paired compare) |
| Minimal `JsonReporter` (canonical JSON output) | Markdown + HTML reporters |
| | `compare` and rich `report` CLI commands |

**Checkpoint:** After `agentic-evalkit run` works end-to-end on GSM8K and emits canonical JSON with an objective grade, pause and decide: continue to 4b for full v1, or cut v0.1 and roll 4b into v0.2.

### 3.2 Add Public-Package Hygiene (HIGH)

Add to Task 1 and Task 15 file maps:

- `LICENSE` (MIT or Apache 2.0)
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `.github/workflows/publish.yml` — PyPI trusted publishing via OIDC, no long-lived tokens

### 3.3 Add Positioning Statement (HIGH)

Add to README and design §1 (Objective) a clear statement of why `agentic-evalkit` exists:

> "Existing evaluation frameworks couple dataset access, grading, and reporting to specific agent platforms or model-provider SDKs. `agentic-evalkit` separates evaluation from the system under test through a neutral `ExecutionTarget` protocol, making any callable, subprocess, or HTTP system evaluable without framework lock-in. Its objective-first grading policy ensures deterministic checks gate releases before model judges are consulted."

### 3.4 Separate Development Gate from Release Gate for Hugging Face (HIGH)

- **Development gate (strict):** Live Hugging Face integration must pass at least once before claiming Task 6 complete. The test in Step 6 remains as written.
- **Release gate (resilient):** Acceptance criteria #4 and #6 are reworded so a transient HF outage at release time is a recorded known-issue, not a release block. The weekly `live-provider.yml` workflow provides ongoing evidence.
- **Add frictionless-access criterion:** `pip install agentic-evalkit` → `agentic-evalkit init --preset gsm8k` → `agentic-evalkit run` must work with zero manual file-hunting, zero `datasets`/`pyarrow`/Docker installation.

### 3.5 Add ARP Integration Example (MEDIUM)

At minimum, add a documented example (in `docs/guides/`) showing an ARP workflow evaluated through the subprocess or HTTP target. This does not need to be an automated test in v1, but it proves the motivating use case is achievable.

### 3.6 Fix Naming Issues (MEDIUM)

- **Citation:** `arXiv:2506.06287` is the FutureSearch Deep Research Bench paper (RetroSearch is the frozen-web environment within it, not the paper title). Relabel accordingly.
- **Notation:** `pass^k` for "consistent success" is non-standard. The literature uses `pass@k` universally and describes consistency in prose. Adopt standard naming.
- **Internal codenames:** The dependency test forbids `agentic_v2`, `tools`, `executionkit` — these appear to be internal package names. Ensure they do not leak into public documentation, examples, or error messages.

### 3.7 Reduce Coverage Gate (LOW)

Change `fail_under = 90` to `fail_under = 80` for the initial release, matching the user's global standard. 90% can be a v1.1 target after the HTML reporter and CLI surface stabilize.

### 3.8 Add Performance Boundaries (LOW)

Add a section to the design or a `docs/guides/performance.md` documenting:

- Expected memory behavior with large datasets
- Streaming semantics of `iter_records`
- Cache size guidance and eviction policy
- Concurrency limits and their rationale

### 3.9 Add Migration Considerations (LOW)

Add a brief section to the design acknowledging existing ARP evaluation code and stating the coexistence strategy for the initial release period.

---

## 4. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Scope creep — 16 tasks is too large for "initial release" | Medium | High | 4a/4b split with reevaluation checkpoint |
| Live HF dependency blocks development | Medium | Medium | Separate dev gate from release gate |
| First real ARP integration reveals target protocol gaps | Medium | High | Add documented ARP integration example in v1 |
| 90% branch coverage gate causes rushed, low-quality tests | Medium | Low | Reduce to 80% for v1 |
| Published without positioning → low adoption | High | Medium | Add differentiation statement before release |
| Missing LICENSE blocks distribution | High | High | Add before any public commit |

---

## 5. Go / No-Go Decision

### Verdict: **GO — with conditions**

The plan is architecturally sound, methodologically disciplined, and the central decision (invert the eval↔runtime dependency) is the correct one. The grading and statistical rigor exceed what most evaluation tooling delivers. The concerns identified are about scope management, public-package hygiene, and missing integration demonstrations — not about architectural validity.

### Conditions for Go

1. **Apply the 4a/4b split** with a formal reevaluation checkpoint after GSM8K runs end-to-end with JSON output.
2. **Add public-package artifacts:** LICENSE, CHANGELOG.md, CONTRIBUTING.md, SECURITY.md, and PyPI publish workflow.
3. **Add positioning/differentiation statement** to README and design §1.
4. **Separate HF development gate from release gate** and add frictionless-access acceptance criterion.
5. **Fix naming issues:** citation label, `pass^k` → standard notation, internal codename hygiene.
6. **Reduce coverage gate** to 80% branch for v1 (or keep 90% but explicitly acknowledge the risk).
7. **Add at minimum a documented ARP integration example** to prove the motivating use case.

### What Must NOT Change

These elements are load-bearing and should be defended:

- The `agentic-evalkit -X-> ARP/EK` dependency invariant and its AST-based enforcement
- Immutable frozen Pydantic contracts with `schema_version`
- Objective-first grading evidence order and non-compensable hard gates
- The rule that uncalibrated/expired judges cannot gate releases
- The rule that missing authoritative harness returns `unavailable`, never a substitute score
- Wilson confidence intervals and deterministic seeded bootstrap
- Rejection of incompatible run comparisons with enumerated reasons
- ADR-before-code sequencing

---

## 6. Comparison with Prior Review

The [prior external review](agentic-evalkit-plan-review.md) reached the same GO verdict and identified several of the same issues (4a/4b split, public-package artifacts, HF gate separation, naming fixes). This independent assessment confirms those findings and adds:

- **Missing ARP integration demonstration** as a distinct concern (the prior review noted the motivating use case was absent but didn't flag it as a gap)
- **Missing migration path** discussion
- **Error recovery patterns** (checkpointing, resumption) as a gap
- **Performance/scalability targets** as a gap
- **Observability** for the framework itself as a gap
- **Coverage gate** recommendation to reduce to 80%

The prior review's seven recommended plan adjustments are all endorsed by this assessment.

---

## Appendix: Design Specification Quality

The design specification (`docs/specs/2026-07-02-agentic-evalkit-design.md`) is independently strong:

- **Section 2 (System Boundary):** Clearly delineates what the framework owns vs. does not own. The dependency invariant is stated as a formal invariant with a diagram.
- **Section 3 (Approaches Considered):** Documents rejected alternatives with rationale — essential for future contributors who will ask "why not just make it an ARP plugin?"
- **Section 9 (Grading and Rubrics):** The evidence ordering is explicit and testable. The hard-gate semantics are unambiguous.
- **Section 10 (Aggregation and Statistical Validity):** Specifies *which* intervals, *which* bootstrap method, and *which* compatibility checks — not just "we'll do statistics."
- **Section 14 (Validation Strategy):** Maps evidence types to test categories — makes the acceptance criteria auditable.
- **Section 18 (Source-Derived Principles):** Cites primary sources with URLs — allows verification of design influences.

The one weakness in the spec is the absence of a "why this exists" narrative in §1 (Objective). The objective states *what* the framework does but not *why* a developer would choose it over existing alternatives.
