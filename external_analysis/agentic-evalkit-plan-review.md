# `agentic-evalkit` Initial Release — Plan Review

**Reviewer:** External analysis (Claude Code)
**Date:** 2026-07-02
**Inputs reviewed:**
- `README.md`
- `docs/superpowers/specs/2026-07-02-agentic-evalkit-design.md` (design spec, 20 sections)
- `docs/superpowers/plans/2026-07-02-agentic-evalkit-initial-release.md` (16-task implementation plan)
**External references verified:** DeepResearch Bench ([arXiv:2506.11763](https://arxiv.org/pdf/2506.11763)), Deep Research Bench / RetroSearch ([arXiv:2506.06287](https://arxiv.org/pdf/2506.06287))

---

## 1. What the plan proposes

Spin `agentic-evalkit` out as an **independent, host-neutral evaluation library + CLI** that depends on neither ARP nor ExecutionKit (EK). ARP/EK are evaluated only through stable callable/subprocess/HTTP target adapters. The initial release (design Slices 1–4) covers:

- Immutable Pydantic contracts (samples, execution, grades, runs, manifests)
- Local + Hugging Face dataset providers with a content-addressed cache
- GSM8K runnable quickstart (objective grading)
- SWE-bench Verified *projection + prediction-export* contracts (without the Docker verifier)
- Objective + calibrated-judge graders with composite hard gates
- Statistical aggregation: confidence intervals, `pass@k`, paired comparison
- Portable JSON / JSONL / Markdown / HTML reports
- Full Typer CLI

Slice 5 (the official containerized SWE-bench harness) is deliberately deferred to a follow-on plan.

The central architectural move is a **dependency inversion**: today evaluation logic lives inside ARP (the runtime) or EK (`executionkit.evals`); the design makes eval the *consumer* of those systems via a neutral `ExecutionTarget` protocol. The invariant `agentic-evalkit -X-> ARP/EK` is enforced by AST-based dependency-boundary tests and clean-wheel tests run outside all source trees.

---

## 2. Strengths

1. **The core decision is correct.** Decoupling evaluation from the runtime platform is a sound architectural pattern; coupling eval to the system-under-test is a recurring anti-pattern (you cannot fairly benchmark a platform using its own internals). The `ExecutionTarget` boundary and the dependency invariant are well-reasoned and concretely enforced.

2. **Methodological discipline is high.** ADR-before-code (9 ADRs, each mapped to a governing task), test-first red/green/refactor per task, immutable frozen Pydantic models with `schema_version`, and gates that explicitly must not be weakened to make CI green. Acceptance criteria (design Section 17) are concrete and testable.

3. **Grading integrity is genuinely well-designed.** Objective-first evidence ordering, non-compensable hard gates (a failed hard gate cannot be averaged away — Task 10 has an explicit test for this), calibrated judges that *cannot gate* when expired/uncalibrated/under-sample, and `authoritative_grader_unavailable` instead of a substitute score. The rule "generic rubric scoring must never be labeled `SWE-bench resolved`" shows the author understands benchmark validity.

4. **Statistical validity is non-trivial and correct.** Wilson intervals (not naive normal approximations), seeded deterministic bootstrap for paired deltas, explicit attempt accounting so missing attempts cannot inflate `pass@k`, and rejection of incompatible comparisons with reasons rather than misleading deltas. Many eval tools skip this.

5. **Honest scope control.** Deferring the Docker SWE-bench verifier to a follow-on plan (while still shipping the *contracts* it will use) avoids the highest-risk, highest-dependency work until the framework is proven.

6. **Reference provenance checks out.** The Section 18 citations were verified. One minor mislabel: `2506.06287` is listed as "RetroSearch" but is actually the FutureSearch *Deep Research Bench* paper, in which RetroSearch is the frozen-web environment — not the paper title. Not load-bearing, but worth correcting in a "principles" section.

---

## 3. Concerns and risks (pre-decision)

1. **"Initial release" is large.** 16 tasks, 9 ADRs, ~50+ source modules, 3 target adapters, a judge-calibration system, a statistical engine, a cache, 2 providers, and 4 report formats. The label "initial release" understates the scope.
2. **Intent ambiguous: internal tool vs. published public library.** Clean-wheel tests, an MkDocs Material docs site, curated guides, and a polished CLI are overhead only justified if the package is meant to be *published*. The differentiator vs. Inspect/Harbor/lighteval/OpenAI Evals needs to be explicit.
3. **The motivating use case is not demonstrated in the initial release.** The release proves the framework with GSM8K + an echo target; there is no end-to-end "evaluate ARP" demonstration.
4. **Live Hugging Face as a release gate couples release to an external service.** Mitigated by a separate weekly workflow that "does not retry indefinitely," but a release could still block on an HF outage.
5. **Coverage gate at 90% branch** is aggressive for an initial release including an HTML reporter (Jinja2) and broad CLI surface. The reviewer's own global standard targets 80%.
6. **Minor terminology nit.** `pass^k` for "consistent success" (= `p**k`) is non-standard; the literature reserves `pass@k` and uses "consistency" in prose. Math is correct; notation is cosmetic.
7. **`asyncio.create_subprocess_exec` on Windows** (in the CI matrix) works on 3.11+ with the default ProactorEventLoop; flagging as the one platform-specific surface to watch.

---

## 4. User decisions and their impact

Three decisions were made by the repository owner:

| # | Decision | Impact |
|---|---|---|
| 1 | **Published package** | Docs-site / clean-wheel / polish overhead justified. Adds public-package hygiene items the plan does not yet name. |
| 2 | **Full release, but hold Slice 4 last and reevaluate** | Requires a 4a/4b split so the runner/CLI dependency chain stays intact; introduces a formal reevaluation checkpoint. |
| 3 | **HF not a release blocker; frictionless dataset access mandatory in v1** | Separate the development-verification gate (strict, must pass live once) from the release gate (HF outage = known issue, not a block). Make zero-config dataset access an explicit acceptance criterion. |

### Decision 1 — Published package

The plan supports this, but to be release-ready as a *public* package, add to Task 1 / Task 15:

- **`LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`** — none are in the file map.
- **A PyPI release workflow** — the plan builds the wheel (`uv build`) but has no publish step. Add trusted-publishing (OIDC) to GitHub Actions with signed artifacts; no long-lived PyPI tokens.
- **A positioning statement.** The current README is one sentence. For a public package entering a space with Inspect, Harbor, lighteval, and OpenAI Evals, the README/design needs an explicit "why this, not those." The real differentiator is the **ARP/EK-neutral boundary + objective-first grading with authoritative-harness separation**.
- **Do not leak internal codenames.** Task 15's dependency test forbids importing `agentic_v2`, `tools`, `executionkit` — those look like internal package names. Extend "do not expose internal machine paths in published pages" to examples/docs.
- **Fix public-facing naming:** relabel the `2506.06287` citation; adopt standard `pass@k` / consistency naming instead of `pass^k`.

### Decision 2 — Full release, Slice 4 last, reevaluate at checkpoint

Slice 4 cannot be deferred wholesale without breaking the end-to-end path, because the CLI `run`/`compare`/`report` commands (Task 14) and the runner's output (Task 11) *consume* the statistics (Task 12) and reporters (Task 13). The resolution is a split:

**4a — keep with Slice 3 (required for a working `run`):**
- Objective graders only (exact, schema, composite, rubric). The calibrated-judge half of Task 10 is *not* needed for GSM8K.
- Runner (Task 11) with its **minimal counts summary** (`total/failed/errors` — already what its test asserts).
- A **minimal `JsonReporter`** so `run` emits canonical JSON.

**4b — hold last, then reevaluate (the deferrable rich stuff):**
- Calibrated judges (`JudgeGrader`, `CalibrationArtifact`) — the judge half of Task 10.
- Full statistics: Wilson CIs, seeded bootstrap, paired `compare`, `pass@k`/consistency — Task 12's richness beyond basic counts.
- Markdown + HTML reporters — the non-JSON half of Task 13.
- `compare` and rich `report` CLI commands — the Task 14 parts that depend on 4b.

This produces a clean **reevaluation checkpoint**: once `agentic-evalkit run` works end-to-end on GSM8K and emits canonical JSON with an objective grade, pause and decide — continue to 4b for the full v1, or cut v0.1 (objective-only, JSON-only) and roll 4b into v0.2. The decision is not pre-committed; the checkpoint is a gate, not a definite split.

The plan's Task 11 runner test already only requires the minimal summary, so 4a is consistent with the written tests — no test weakening needed.

### Decision 3 — HF not a release blocker; frictionless access mandatory v1

This sharpens, rather than relaxes, the dataset requirements:

- **Separate development-verification from release gate.** Task 6 Step 6 ("if HF unavailable, rerun before claiming task complete; do not replace the live gate with mocks") should remain a strict **development** gate — live integration must be proven at least once. But acceptance criteria #4 and #6 should be reworded so a *release* is not blocked by an HF outage at release time: live evidence is "verified by the weekly `live-provider.yml` workflow and on-demand," and a current HF outage is a recorded known-issue, not a release block.
- **Frictionless access becomes a hard v1 acceptance criterion.** `pip install agentic-evalkit` → `agentic-evalkit init --preset gsm8k` → `agentic-evalkit run` must work with **zero importer code, zero file-hunting, zero `datasets`/`pyarrow`/Docker**. The `doctor` HF healthcheck and classified provider errors become load-bearing UX, since a user on a flaky network must get a clear `dataset_rate_limited` / `dataset_provider_unavailable` message, not a stack trace.

---

## 5. Final verdict

**Proceed.**

The plan is architecturally sound, methodologically disciplined, and the central decision (invert the eval↔runtime dependency) is the right one. The grading and statistical rigor are above the bar for most eval tooling. The owner's three decisions remove the open framing questions:

- **Published** → overhead justified; add LICENSE / publish-workflow / positioning / naming fixes.
- **Slice 4 last + reevaluate** → adopt the 4a/4b split so the runner/CLI dependency chain stays intact; checkpoint after GSM8K-runs-and-emits-JSON.
- **HF not a blocker, frictionless access mandatory** → split the development-verification gate (strict, must pass live once) from the release gate (HF outage = known issue, not a block); make zero-config dataset access an explicit acceptance criterion.

None of this changes the architecture or the core contracts — only sequencing, gate semantics, and public-package hygiene. The design itself was already sound; the remaining work is plan-text adjustments.

### Recommended plan adjustments (not yet applied)

1. Split Slice 4 into 4a (minimal summary + JSON report, stays with Slice 3) and 4b (judges, full stats, Markdown/HTML, `compare`); insert a reevaluation checkpoint after 4a.
2. Reword acceptance criteria #4 and #6 to separate development-verification (strict) from release gate (HF outage = known issue).
3. Add an explicit frictionless-dataset-access acceptance criterion.
4. Add `LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md` to the Task 1/15 file maps.
5. Add a PyPI trusted-publishing release workflow.
6. Add a positioning/"why this not Inspect/Harbor/lighteval" section to the README and design.
7. Relabel the `2506.06287` citation; adopt standard `pass@k` / consistency naming.

These adjustments are pending owner approval; no plan or spec files were modified during this review.