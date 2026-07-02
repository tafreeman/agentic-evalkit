# agentic-evalkit Initial Release — Plan Modifications

**Purpose:** This document consolidates every analysis performed on the design spec and implementation plan — the internal review, and four independent external reviews (`agentic-evalkit-plan-review.md`, `plan-quality-assessment.md`, `gemini35flassh_evaluation_report.md`, `opus47_evaluation_report.md`) — into one explicit list of changes to apply before or during execution.

**Status:** Applied to the design spec and implementation plan on 2026-07-02.

**Source documents synthesized:**
- `docs/specs/2026-07-02-agentic-evalkit-design.md`
- `docs/plans/2026-07-02-agentic-evalkit-initial-release.md`
- `external_analysis/agentic-evalkit-plan-execution-report.md`
- `external_analysis/agentic-evalkit-plan-review.md`
- `external_analysis/plan-quality-assessment.md`
- `external_analysis/gemini35flassh_evaluation_report.md`
- `external_analysis/opus47_evaluation_report.md`

All five analyses independently reached **GO / Execute**. None found an architectural flaw. The modifications below are sequencing, hygiene, gate-semantics, and risk-mitigation changes — not redesigns.

## Resolution record

- **Canonical name:** `agentic-evalkit` / `agentic_evalkit` / `agentic-evalkit`. The ARP-local `agentic-v2-eval` draft was verified, but it is not authoritative for this standalone repository and was not modified, per the explicit no-ARP/no-EK-change boundary.
- **HTTPX:** The claim that HTTPX 1.0 is stable was rejected after checking PyPI. Stable is 0.28.1; 1.0 is a development prerelease. The plan now pins `httpx>=0.28.1,<1`.
- **License:** MIT was selected because both companion repositories use MIT.
- **Sequencing:** Slice 4 is split into required 4a and deferred 4b, with a recorded `SHIP_V0_1`/`CONTINUE_FULL_V1` checkpoint after a clean-wheel GSM8K JSON run.
- **Delivery:** The implementation is grouped into three independently mergeable milestones rather than one long-lived branch.
- **Release evidence:** Live Hugging Face evidence remains mandatory during provider development and scheduled monitoring. A classified transient outage at final release time becomes a known issue when offline contracts, clean-wheel checks, and prior live evidence are healthy.
- **Public hygiene:** License, changelog, contribution/security policies, OIDC trusted publishing, positioning, and a real HTTP-agent example are included.
- **Coverage:** The initial branch-aware floor is 80%; achieved coverage is reported, and any later increase is evidence-based.
- **Risk mitigations:** Windows JSONL framing, Dataset Viewer retries, Windows cache caveat, 1,000-sample bootstrap default with a 10,000 cap, precise Typer pin, and log-space `pass@k` are incorporated.
- **Terminology:** `pass^k` was replaced by an explicitly named all-attempt consistency metric, and the FutureSearch/RetroSearch citation label was corrected.
- **Deferred items:** ADR-0010, run resumption, performance/eviction targets, detailed subgroup syntax, and framework observability remain post-v1 work and are listed in the plan index/acceptance task.

---

## 1. Blocking — resolve before Task 1 begins

### 1.1 Resolve the package-naming divergence
**Raised by:** `opus47_evaluation_report.md` (M-1)

The evalkit design/plan use `agentic-evalkit` / `agentic_evalkit`. A companion spec in the `agentic-runtime-platform` repository (`docs/superpowers/specs/2026-07-02-evaluation-framework-design.md`) reportedly describes the same package as `agentic-v2-eval` / `agentic_v2_eval` and expects ARP to import it under that name.

**Applied resolution:** The companion ARP spec was verified and uses the legacy name. `agentic-evalkit` is canonical for the new repository, distribution, import, and CLI. The ARP draft is not authoritative for this standalone architecture and was not modified because ARP/EK changes are explicitly out of scope. Integration is through target protocols, not an ARP import path.

**Note:** This claim could not be independently verified from this workspace, since only the `agentic-evalkit` repository is open here. Verify against the actual ARP repository before treating this as confirmed; if the companion spec does not exist or already agrees on `agentic-evalkit`, this item is void.

### 1.2 Verify the HTTPX version constraint
**Raised by:** `opus47_evaluation_report.md` (P-5)

Task 1's `pyproject.toml` step pins `httpx` below 1. The external report claimed HTTPX 1.0 was stable.

**Applied resolution:** PyPI was checked and lists 0.28.1 as stable while 1.0 is a development prerelease. The plan pins `httpx>=0.28.1,<1` and records the verification rather than adopting the incorrect proposed range.

---

## 2. Scope and sequencing

### 2.1 Split Slice 4 into 4a (required) and 4b (deferred, checkpointed)
**Raised by:** `agentic-evalkit-plan-review.md`, `plan-quality-assessment.md` (both call this **critical**)

Slice 4 cannot be deferred wholesale — the CLI `run`/`compare`/`report` commands (Task 14) and the runner (Task 11) consume the statistics (Task 12) and reporters (Task 13).

**Modification:** Split Task 10, 12, 13, 14 work as follows:

| Keep with Slice 3 (4a — required for a working `run`) | Defer, reevaluate at checkpoint (4b) |
|---|---|
| Objective graders only: exact, schema, composite, rubric (Task 10 minus judge) | Calibrated judges: `JudgeGrader`, `CalibrationArtifact` (rest of Task 10) |
| Runner (Task 11) with its already-minimal counts summary | Full statistics: Wilson CIs, seeded bootstrap, paired `compare`, `pass@k`/consistency (rest of Task 12) |
| Minimal `JsonReporter` only (part of Task 13) | Markdown + HTML reporters (rest of Task 13) |
| | `compare` and rich `report` CLI commands (part of Task 14) |

Insert a formal **reevaluation checkpoint** after `agentic-evalkit run` works end-to-end on GSM8K and emits canonical JSON with an objective grade. At that checkpoint, decide whether to continue into 4b for a full v1 or ship v0.1 (objective-only, JSON-only) and roll 4b into v0.2. Task 11's runner test already only asserts the minimal summary, so this split requires no test weakening.

### 2.2 Split the 16-task plan into milestone PRs
**Raised by:** `opus47_evaluation_report.md` (M-2)

A single 73KB/16-task/9-ADR plan is a large unit of delivery and review.

**Modification:** Execute and land the plan as three milestone groupings instead of one continuous branch:
- Milestone A: Tasks 1–7 (foundation, contracts, plugins, cache, providers, catalog/presets)
- Milestone B: Tasks 8–11 plus the JSON and runnable-CLI portions of Tasks 13–14; finish with the recorded v0.1 checkpoint
- Milestone C: deferred judge/statistics/rich-report/compare portions plus Tasks 15–16; move deferred work to v0.2 when the checkpoint selects `SHIP_V0_1`

This aligns naturally with the 4a/4b checkpoint in §2.1, since Milestone B is where that checkpoint falls.

---

## 3. Release-gate semantics

### 3.1 Separate the Hugging Face development gate from the release gate
**Raised by:** `agentic-evalkit-plan-review.md`, `plan-quality-assessment.md`

**Modification:**
- **Development gate (unchanged, strict):** Task 6 Step 6 still requires live Hugging Face evidence at least once before Task 6 is marked complete. Do not replace this with mocks.
- **Release gate (reworded):** Reword design §17 acceptance criteria #4 and #6 so a transient Hugging Face outage at release time is a recorded known issue, not a release blocker. Ongoing evidence comes from the weekly `live-provider.yml` workflow (Task 15) and on-demand runs, not from a hard gate at release.

### 3.2 Add an explicit frictionless-dataset-access acceptance criterion
**Raised by:** `agentic-evalkit-plan-review.md`, `plan-quality-assessment.md`

**Modification:** Add a new criterion to design §17: `pip install agentic-evalkit` → `agentic-evalkit init --preset gsm8k` → `agentic-evalkit run` must succeed with zero importer code, zero manual file-hunting, and zero `datasets`/`pyarrow`/Docker installation. Make the `doctor` Hugging Face healthcheck and classified provider errors (Task 3, Task 6) load-bearing for this criterion so network issues surface as a clear error code, not a stack trace.

---

## 4. Public-package hygiene

**Raised by:** `agentic-evalkit-plan-review.md`, `plan-quality-assessment.md` (both **high priority**, contingent on this being a published package)

**Modification:** Add to the Task 1 and Task 15 file maps:
- `LICENSE` (MIT or Apache 2.0)
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `.github/workflows/publish.yml` — PyPI trusted publishing via OIDC; no long-lived API tokens

---

## 5. Coverage gate

**Raised by:** `plan-quality-assessment.md`, `opus47_evaluation_report.md` (both flag `fail_under = 90` as aggressive for an initial release)

**Modification:** Change Task 1's `[tool.coverage.report]` from `fail_under = 90` to `fail_under = 80` for the initial release, matching the project's established 80% standard. Optionally raise back to 90% at Task 15 once the HTML reporter and CLI surface stabilize, per `opus47_evaluation_report.md`'s progressive-raise suggestion.

---

## 6. Engineering risk mitigations (execution-level, no plan-text change required)

**Raised by:** `gemini35flassh_evaluation_report.md`, `opus47_evaluation_report.md`

Apply these during implementation of the named tasks:

| # | Task | Risk | Mitigation |
|---|---|---|---|
| 6.1 | Task 9 (`SubprocessTarget`) | Windows line-ending/buffering differences (`\r\n`) can split or corrupt JSONL reads | Decode subprocess output as UTF-8 using an explicit line-based reader (e.g. `StreamReader.readline()`), and strip both `\r` and `\n` before parsing JSON. Do not do raw chunk-reads without line reassembly. |
| 6.2 | Task 6, Task 15 (`live-provider.yml`) | Hugging Face Dataset Viewer rate limits (429) or transient 5xx errors can fail CI | Add exponential backoff/retry at the HTTP client level for live tests and the weekly workflow. Keep local/dev tests offline-capable via captured fixtures; `@pytest.mark.live` remains the strict gate. |
| 6.3 | Task 12 (paired bootstrap) | Pure-Python bootstrap at 10,000 samples can take 5–10s per `compare` call | Default `bootstrap_samples` to 1,000, expose `--bootstrap-samples` (up to 10,000) as a CLI/API override. |
| 6.4 | Task 4 (`DatasetCache`) | `Path.replace()` is not guaranteed atomic on all Windows filesystems | Add a note to ADR-0004 documenting this platform caveat and confirm the cache test suite runs and passes on the Windows CI job. |
| 6.5 | Task 1 (`pyproject.toml`) | `typer<1` is a broad, under-specified range | Pin to `typer>=0.12,<1`. |
| 6.6 | Task 12 (`pass_at_k`) | Naive `comb(n-c, k) / comb(n, k)` can overflow/slow down for large `n` | Use log-space computation (e.g., `math.lgamma`-based) for large sample counts. |

---

## 7. Documentation and positioning

**Raised by:** `agentic-evalkit-plan-review.md`, `plan-quality-assessment.md`

**Modification:** Add a short "why this exists" statement to `README.md` and design §1 (Objective), naming the real differentiator versus Inspect, Harbor, LightEval, OpenAI Evals, and Langfuse:

> "Existing evaluation frameworks couple dataset access, grading, and reporting to specific agent platforms or model-provider SDKs. `agentic-evalkit` separates evaluation from the system under test through a neutral `ExecutionTarget` protocol, making any callable, subprocess, or HTTP system evaluable without framework lock-in. Its objective-first grading policy ensures deterministic checks gate releases before model judges are consulted."

**Modification:** Add one documented example (in `docs/guides/`) showing a real agentic system evaluated through the subprocess or HTTP target, proving the motivating use case beyond GSM8K + an echo target. This does not need to be an automated test in v1.

---

## 8. Naming and terminology fixes

**Raised by:** `agentic-evalkit-plan-review.md`, `plan-quality-assessment.md`

**Modification:**
- Relabel the `arXiv:2506.06287` citation in design §18 — it is the FutureSearch *Deep Research Bench* paper; RetroSearch is the frozen-web environment described within it, not the paper's title.
- Replace the non-standard `pass^k` notation with standard `pass@k` plus a "consistency" description in prose, per common literature usage.
- Confirm internal codenames (`agentic_v2`, `tools`, `executionkit` in Task 15's dependency-boundary test) never leak into public docs, examples, or CLI error messages.

---

## 9. Deferred / optional (do not block initial release)

These were raised as gaps but are explicitly **not** required before execution:

- **Async-first ADR (`opus47_evaluation_report.md` M-8):** Add `ADR-0010` documenting why the provider/target protocol is async even for local, synchronous providers.
- **Run resumption/checkpointing (`plan-quality-assessment.md`, `opus47_evaluation_report.md` M-7):** No partial-run resume in v1. Design `EvalRunResult` so it does not preclude appending results later.
- **Performance/scalability targets (`plan-quality-assessment.md`):** Document expected throughput, memory behavior at 10K–1M rows, and cache eviction policy in a follow-up `docs/guides/performance.md`.
- **Subgroup analysis detail (`plan-quality-assessment.md`):** Defer detailed manifest syntax for predeclared subgroup tags and minimum-sample-size warnings past v1.
- **Framework observability (`plan-quality-assessment.md`):** Defer structured logging/metrics/tracing across provider → adapter → target → grader boundaries past v1.
- **Migration-path narrative (`plan-quality-assessment.md`):** Add a brief note acknowledging any existing evaluation code and the coexistence strategy during the initial release period, once/if such code is confirmed to exist.
- **Dataset Viewer fallback documentation (`opus47_evaluation_report.md` M-4):** Document the `parquet` extra as the escape hatch for datasets the Dataset Viewer cannot serve.

---

## 10. What must NOT change

All five analyses agree these are load-bearing and should be defended against any future simplification pressure:

- The `agentic-evalkit -X-> ARP/EK` dependency invariant and its AST-based enforcement (Task 15).
- Immutable, frozen Pydantic contracts with `schema_version` and `extra="forbid"` (Task 2).
- The objective-first grading evidence order and non-compensable hard gates (Task 10, design §9).
- The rule that uncalibrated or expired judges cannot gate releases (Task 10).
- The rule that a missing authoritative harness returns `unavailable`, never a substitute score (Task 8).
- Wilson confidence intervals and deterministic seeded bootstrap (Task 12).
- Rejection of incompatible run comparisons with enumerated reasons rather than misleading deltas (Task 12).
- ADR-before-code sequencing (all tasks).

---

## 11. Summary of changes by affected file

| File | Change |
|---|---|
| `docs/specs/2026-07-02-agentic-evalkit-design.md` §1 | Add positioning/"why this exists" statement (§7) |
| `docs/specs/2026-07-02-agentic-evalkit-design.md` §17 | Reword criteria #4, #6; add frictionless-access criterion (§3) |
| `docs/specs/2026-07-02-agentic-evalkit-design.md` §18 | Fix citation label, `pass^k` → `pass@k` (§8) |
| `docs/plans/2026-07-02-agentic-evalkit-initial-release.md` Task 1 | LICENSE/CHANGELOG/CONTRIBUTING/SECURITY, `fail_under=80`, `typer>=0.12,<1`, verify httpx pin (§1.2, §4, §5, §6.5) |
| `docs/plans/2026-07-02-agentic-evalkit-initial-release.md` Tasks 10, 12, 13, 14 | Apply 4a/4b split and checkpoint (§2.1) |
| `docs/plans/2026-07-02-agentic-evalkit-initial-release.md` Task 4 | ADR-0004 Windows atomicity note (§6.4) |
| `docs/plans/2026-07-02-agentic-evalkit-initial-release.md` Task 6, 15 | Live-test retry/backoff, dev-vs-release gate separation (§3.1, §6.2) |
| `docs/plans/2026-07-02-agentic-evalkit-initial-release.md` Task 9 | Windows-safe subprocess line reading (§6.1) |
| `docs/plans/2026-07-02-agentic-evalkit-initial-release.md` Task 12 | Bootstrap default 1,000/configurable, log-space `pass_at_k` (§6.3, §6.6) |
| `docs/plans/2026-07-02-agentic-evalkit-initial-release.md` Task 15 | Add publish workflow, positioning in docs, host-neutral HTTP-agent example, milestone split (§2.2, §4, §7) |
| `README.md` | Add positioning statement (§7) |
