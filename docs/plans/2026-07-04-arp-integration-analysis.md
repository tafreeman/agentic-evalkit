# agentic-evalkit ARP Integration Analysis and Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> This is an **analysis and plan**, produced 2026-07-04 without modifying ARP. Every task executor must re-verify the "Verified current state" section against the live repositories before starting a task — every fact below is a point-in-time snapshot and file:line references drift. Task text such as "the legacy gate" refers to the artifacts named in that section; do not infer paths from memory.
>
> Naming note: this document deliberately names `agentic-v2-eval`, `agentic_v2`, `tools.agents`, and `executionkit`. That is safe **only** under `docs/plans/` — `tests/contract/test_public_docs.py` exempts `docs/plans`, `docs/specs`, and `docs/adr` from the codename scan, and user-facing docs must never carry these names. Use backtick paths, not Markdown links, so `mkdocs build --strict` stays clean (this page is built but not in `nav`).

**Goal:** Make `agentic-runtime-platform` (ARP) a consumer of `agentic-evalkit`, re-platform ARP's CI evaluation gates onto it, harvest every piece of ARP eval tooling with standalone value (into evalkit or into ARP-side adapters), and only then deprecate and remove the legacy `agentic-v2-eval` package — with a rollback boundary after every phase.

**Architecture:** The dependency is strictly one-way: ARP imports `agentic_evalkit`; `agentic_evalkit` never imports `agentic_v2`, `tools`, or `executionkit` (enforced upstream by `tests/contract/test_dependency_boundary.py`, and belt-and-braces by a new ARP-side mirror test in Task 3). ARP integrates through evalkit's public structural protocols — `ExecutionTarget`, `Grader`, `JudgeClient`, `DatasetProvider` — driven from small ARP-owned driver modules, because the shipped evalkit CLI hardcodes its adapter/grader/provider tables and has no plugin discovery wired yet. Distribution follows ARP's own ADR-023 Option A′ precedent for ExecutionKit: consume a published release, not a co-located checkout.

**Tech stack:** Python 3.11+, uv workspace (ARP) + `uv export` → `ci-constraints.txt` → pip installs in CI, Hatchling, Pydantic v2, Typer, GitHub Actions on both repos, PyPI trusted publishing (evalkit `publish.yml`, currently inert).

---

## Scope and non-goals

**In scope:** adding the dependency; decoupling the two unconditional ARP production imports; re-platforming the golden and live CI eval gates; translating the live rubrics; the harvest inventory and its dispositions; CI/tooling/docs cutover; retiring `agentic-v2-eval`.

**Out of scope (recorded as follow-on gates, not silently dropped):**

- `tools/agents/benchmarks/` retirement. ARP's server dataset routes and docs reference its directories (e.g. `tools/agents/benchmarks/gold_standards/`), and it has consumers beyond eval. This plan *harvests* its benchmark registry into evalkit presets (Task 8) but does not delete the package.
- The ARP server evaluation pipeline (`agentic_v2/server/evaluation_scoring.py`, `judge.py`) and the React evaluation/dataset UI. They are architecturally independent of `agentic-v2-eval` today (verified — the dashboard consumes the server pipeline, not the package) and keep working untouched. Re-pointing them at evalkit run JSON is a future plan.
- The ADR-010/011/012 commit-driven A/B eval harness (`tools/commit_eval/`). Verified: **no code exists** — all three ADRs are `Proposed` design intent. Their value enters this plan only as design reference (Task 8, item H8).
- The official SWE-bench Docker executor (already gated separately in `docs/plans/README.md`).

**Guardrails observed while producing this plan:** ARP and ExecutionKit were treated read-only; nothing was added to ARP's `pyproject.toml`/`uv.lock`; nothing under `agentic-v2-eval/` was touched; no commits were made.

---

## Verified current state (2026-07-04)

### agentic-evalkit (this repo)

- Package `agentic-evalkit` 0.1.0, import `agentic_evalkit`, CLI `agentic-evalkit`, Python `>=3.11`, Hatchling, MIT. Runtime deps: pydantic 2, typer, rich, pyyaml, huggingface-hub, httpx, jinja2. Extras `parquet`/`judges`/`swebench` are **empty placeholders** (ADR-0009).
- Remote `https://github.com/tafreeman/agentic-evalkit.git` (**private**, GitHub Free — no branch protection). Tag `v0.1.0` → commit `3a8fef3`, which **is** on `origin/main`. Local `main` is **one unpushed, test-only commit ahead** (`2c5591d`). The handoff claim "tagged at current HEAD" has drifted; the tag itself is safe to pin today.
- Release evidence: `docs/release/initial-release-acceptance.md` — all 17 criteria PASS under `CONTINUE_FULL_V1`; 416 tests, 88.91% branch coverage, mypy strict clean, clean-wheel and live-HF gates passed.
- **Offline contract gap (confirmed in source, post-dates the acceptance audit):** `_bmad-output/implementation-artifacts/deferred-work.md` is accurate. `run --offline` and `datasets preview --offline` silently drop the flag (`del offline` in `src/agentic_evalkit/cli/datasets.py` ~line 103; `cli/runs.py` and `runner.py` never forward it); `DatasetCatalog` honors `offline=True` only in `preview`, while `search`/`resolve`/`iter_records` raise `OfflineCacheMiss` unconditionally — which also blanket-rejects the network-free `local` provider; `OfflineCacheMiss` cannot distinguish "warm the cache and retry" from "categorically uncacheable".
- **No LLM provider client ships.** `JudgeGrader` takes a caller-implemented `JudgeClient` protocol; objective graders (`ExactMatchGrader`, `SchemaGrader`, `CompositeGrader`/`WeightedGrader` with genuine non-compensable hard gates) are deterministic. The default test suite is hermetic (`-m "not live"` baked into addopts); no API key is ever required.
- **Judges cannot gate without calibration:** live fingerprint must equal the calibration's `judge_fingerprint`, calibration unexpired, ≥30 held-out positive and ≥30 negative labels, TPR/TNR floors, and a reversed-order position-bias probe agreeing with the primary verdict — otherwise the grade is demoted to advisory. ARP has no calibration datasets today.
- Runner: manifest-driven (`manifest.py`), `asyncio.TaskGroup` bounded concurrency, typed run events, canonical JSON to `<output-dir>/<run_id>.json` with artifact spill; reporters JSON/JSONL/Markdown/HTML; `compare` = paired bootstrap with a strict compatibility gate; CLI exit codes 0/2/3/4/5/130.
- **CLI extension gap:** `cli/runs.py` hardcodes `_KNOWN_ADAPTERS`/`_KNOWN_GRADERS`; `cli/datasets.py` hardcodes the two providers; `load_plugins()` exists but is wired to nothing. Downstream custom graders/targets work **only via the library API** (construct `EvalRunner` yourself). This shapes the whole integration: ARP drives evalkit from Python, not from the packaged CLI.
- `publish.yml` (PyPI trusted publishing via OIDC, fires on GitHub release) exists and is inert until the PyPI project + environment are configured.

### ARP touchpoints of `agentic-v2-eval` (the cutover blast radius)

Only **two unconditional production imports** exist outside the package:

1. `agentic-workflows-v2/agentic_v2/models/llm.py:5` — `from agentic_v2_eval.interfaces import LLMClientProtocol` (the runtime's single hard import; breaks at import time if the package vanishes).
2. `scripts/eval_gate.py:66-67` — `load_rubric` + `Scorer`; powers both CI eval gates.

Guarded/degradable touchpoints: `agentic_v2/scoring/step_scoring.py:39-40` (try/except → `_EVAL_AVAILABLE`, per-step rubric scoring silently disables), `scripts/score-trace.py` (guarded dev CLI), `examples/05_evaluation.py` (hard import, example only), `tests/e2e/test_cross_package.py` + two guarded tests in `agentic-workflows-v2/tests/`, `agentic_v2/devex/workspace_test_runner.py` (shells out to the package's pytest).

CI/tooling surface: `.github/workflows/eval-package-ci.yml` (test/lint/type-check/build jobs on ubuntu+windows × 3.11/3.12, plus `eval-golden-gate` — every push/PR, `AGENTIC_NO_LLM=1`, `python scripts/eval_gate.py --cases datasets/default/golden_cases.json --threshold 0.80` — and `eval-live-gate` — `run-live-eval` label or nightly cron, `--live`, median-of-3 through `agentic_v2.workflows.run_workflow`, exits 0 cleanly when no provider key resolves); install lines in `ci.yml`, `sbom.yml`, `dependency-audit.yml`; `deploy.yml` builds and stages the legacy wheel in release assets; `dependabot.yml` directory watch; `CODEOWNERS` rule; root `pyproject.toml` `[tool.uv.workspace] members`; `justfile` `setup`/`test`; the pre-commit mypy hook scoped `^agentic-v2-eval/` (the repo's only mypy hook).

Data: `datasets/default/golden_cases.json` + four golden `*_output.json` snapshots are **inputs** to the gate (they stay regardless); repo-root `runs/` is written by the runtime and read by the server — the eval package never writes it.

Docs: ~40 markdown files reference the package, including `CLAUDE.md`, `CONTRIBUTING.md`, the five files under `docs/evaluation/`, `docs/architecture-eval.md`, `docs/deep-dive-agentic-v2-eval.md`. Known stale claims to fix in passing: `docs/integration-architecture.md` asserts the runtime does not import the eval package (contradicted by `llm.py:5` and `step_scoring.py`); `CONTRIBUTING.md:96` still says "35 known findings" for mypy (it is 0); ADR-017 ratifies query-param dataset routes while shipped code uses `{dataset_id:path}`.

### The legacy package as it actually is (not as documented)

`agentic-v2-eval` v0.3.0, 4,407 source LOC, 273 tests passing, 86.27% coverage, mypy strict clean. Corrections to the handoff brief, verified in source:

- **The sandbox has zero production callers repo-wide** — exercised only by its own 25 tests. Its README oversells it: there is **no import blocklist** (only an OS command-*name* blocklist + word-boundary arg scan) and **no resource limits** (only a wall-clock `subprocess.run(timeout=...)`; no rlimit/cgroup/network isolation). There is no caller behavior to preserve.
- **`datasets.py` is a lazy bridge, not a loader** — the real HF/GitHub/cache logic lives in `tools/agents/benchmarks/` (outside the package). The package's declared `agentic-tools` dependency (`{ workspace = true }`) is the reverse coupling ARP's own design doc targets for removal.
- Rubric YAML keys `thresholds`/`hard_gates`/`levels` are **decorative** — `Scorer` reads only `name`/`weight`/`description`/`min_value`/`max_value`. Real thresholding lives in `--fail-under` and `scripts/eval_gate.py`. Evalkit's `CompositeGrader` hard gates are therefore a strict upgrade, not a port.
- The four evaluators (`LLMEvaluator`, `PatternEvaluator`, `QualityEvaluator`, `StandardEvaluator`) are all LLM judges, are not wired into any runner/CLI path, and do not share one `Evaluator` interface. Deterministic scoring lives in `metrics/` (accuracy/F1, code-quality heuristics, performance percentiles) — pure functions with no evalkit equivalent yet.
- No resumability, run manifests, provenance, statistics, or comparison exist anywhere in the package — evalkit is a strict superset on the runner/reporting/stats axis.

### Cross-repo facts

- `executionkit` ↔ eval coupling: **zero in both directions** (fresh grep both repos). EK stays out of scope.
- `evalkit` references anywhere in ARP: **zero** (tracked + untracked re-verified). This is a from-scratch adoption.
- ARP's `docs/superpowers/specs/2026-07-02-evaluation-framework-design.md` (branch `docs/eval-framework-design`, unmerged) designed this exact adoption but prescribed evolving `agentic-v2-eval` **in place**. The standalone package was instead built as this repository — the design doc's §4 invariant, provider/grader/statistics architecture, and acceptance criteria map almost 1:1 onto what evalkit shipped. Task 10 marks that doc superseded-by-evalkit rather than leaving two competing plans.
- ARP CI dependency pipeline: `uv.lock` → `just update-constraints` (`uv export --locked --no-hashes --no-emit-workspace --all-packages --all-extras -o ci-constraints.txt`) → every CI job installs with `pip install -e ... -c ci-constraints.txt`, and a `lockfile-constraints` job fails on drift. **pip does not accept direct-URL requirements in constraints files**, so a git-URL dependency would be exported into `ci-constraints.txt` and break every install step — see Distribution mechanics.
- Precedent: `agentic-workflows-v2/pyproject.toml` ADR-023 Option A′ — ExecutionKit is consumed as the **published PyPI release** through an optional extra (`ek = ["executionkit>=0.1.0,<0.3.0"]`) with guarded imports and coverage-omitted bridge modules, "so the workspace resolves cleanly on any checkout and in CI without a co-located EK clone."

---

## Capability diff — every ARP usage, dispositioned

| # | ARP usage (evidence) | evalkit coverage | Disposition |
|---|---|---|---|
| 1 | `LLMClientProtocol` import in `agentic_v2/models/llm.py:5` | Not needed — 10-line structural protocol | **Decouple:** inline the protocol into `agentic_v2` (Task 4). Removes the only hard runtime import. |
| 2 | `Scorer` + rubric YAML weighted scoring (`eval_gate.py`, `step_scoring.py`, `score-trace.py`) | `CompositeGrader`/`WeightedGrader` — weighted mean over definitive children plus **real** non-compensable hard gates | **Covered, with translation** (Tasks 5, 7). Weights carry over; legacy "thresholds/hard_gates" YAML keys were never enforced, so parity is against observed behavior, not the YAML. |
| 3 | 8 packaged rubrics (`agentic_v2_eval/rubrics/*.yaml`) | Manifest + grader construction (library API; CLI grader table is hardcoded) | **Translate the live ones** (`code` for the gate; whatever `step_scoring` resolves). Judge-prompt packs (`quality`, `prompt_pattern`, `prompt_standard`) move with the judges (Task 7). Archive the rest. |
| 4 | CI golden gate (`eval-golden-gate` job → `eval_gate.py` `derive_criteria()` over `datasets/default/`) | Local provider + `CallableTarget` + composite objective graders + canonical JSON + typed exit codes | **Covered — re-platform with a side-by-side parity window** (Task 5). Deterministic and network-free by construction. |
| 5 | CI live gate (median-of-3 real-model runs via `run_workflow`, clean skip without keys) | Attempts/sampling policy, per-attempt sample results, pass@k/pass^k | **Mostly covered; ARP-side glue** computes the median-of-3 weighted score from evalkit sample results and preserves the credential-degradation contract (Task 6). |
| 6 | Batch/streaming runners | `EvalRunner` (bounded async concurrency, events, manifests, provenance, artifact spill) | **Covered — superset.** Neither side has resumability (evalkit defers it to v0.2). |
| 7 | JSON/Markdown/HTML reporters | JSON (canonical) + JSONL + Markdown + HTML, shared redaction | **Covered.** |
| 8 | LLM judge evaluators (Pattern/Quality/Standard/LLM) | `JudgeGrader` (calibration-gated, provider-neutral, **no client shipped**) | **Gap + harvest:** ARP implements `JudgeClient` over its model client; prompt packs become ARP-side judge configurations; judges stay **advisory** until calibration datasets exist (Task 7; Risk R5). |
| 9 | Deterministic metrics (`metrics/accuracy.py`, `quality.py`, `performance.py`) | No equivalent | **Harvest:** port as objective graders — ARP-side first, upstream candidates (Tasks 7-8). |
| 10 | Sandboxed subprocess executor (`sandbox/`) | No equivalent (evalkit `SubprocessTarget` solves a different problem: target execution) | **Accepted gap — drop.** Zero callers, no import blocklist, no resource limits. Recorded as prior art for the SWE-bench Docker executor plan; do not port as-is. |
| 11 | Dataset loading via `tools.agents.benchmarks` bridge (9-benchmark registry; HF loader with verified defects) | Catalog with `local` + `huggingface` providers (fixes the exact defect class: explicit config/split, typed errors never collapsing to `[]`, content-addressed revision-aware cache) — but only 2 presets (`gsm8k`, `swe-bench-verified`) | **Partial:** transport/caching is covered and better; benchmark **breadth** (HumanEval, MBPP, CodeClash, …) is a gap → harvest the registry as evalkit presets/adapters (Task 8, H4). |
| 12 | Legacy CLI (`evaluate`/`report`, exit 0/1/2) | `validate`/`run`/`report`/`compare`, exit 0/2/3/4/5/130 | **Covered** — richer, typed. Gate scripts map thresholds to exit semantics explicitly. |
| 13 | `examples/05_evaluation.py`, e2e cross-package test, guarded unit tests | n/a | **Rewrite** against evalkit during cutover (Tasks 9, 11). |
| 14 | `eval-package-ci.yml` package jobs; `deploy.yml` legacy wheel; sbom/audit/dependabot/CODEOWNERS/pre-commit/justfile/devex entries | evalkit's own repo CI already covers package QA | **Retire/re-point mechanically** (Tasks 9, 12). |
| 15 | Server eval pipeline + React UI | Independent of the package today | **Out of scope** — unaffected by this migration; future follow-on may consume evalkit run JSON. |

---

## Harvest inventory — ARP eval tooling of value

Nothing below may enter evalkit by copy-paste import: every upstream port is a rewrite against evalkit contracts, lands via evalkit's own plan/ADR/TDD process, and must keep `tests/contract/test_dependency_boundary.py` green. "ARP-side" means it lives in ARP and *uses* evalkit — the boundary only forbids the other direction. **Task 12 (deleting the legacy package) is gated on every row below reaching its disposition or being explicitly re-dispositioned in writing.**

| ID | Asset | Location | Why it's valuable | Disposition |
|---|---|---|---|---|
| H1 | Self-consistency ensembling + majority vote (deterministic `canonical_key` bucketing, "below-threshold is data, not an error") | `agentic_v2/engine/consensus.py` (255 LOC, **zero non-stdlib deps**) | Textbook-clean Wang-et-al. self-consistency; most toolkits lack it | **Upstream to evalkit** (stats/aggregation utility or attempt-policy helper). Easiest win. |
| H2 | Bias-aware LLM judge mechanisms: seeded criteria-order shuffling, swapped-order consistency probe, calibration-drift MAE vs human fixtures, strict judge-output schema validation | `agentic_v2/scoring/judge.py` (608 LOC; light coupling to ARP model client + `normalize_score`) | The most distinctive asset — complements evalkit's calibration gate (which *checks* calibration) with *drift detection* and richer bias probes | **Upstream the algorithms** into evalkit's judge module as a rewrite; ARP keeps its server judge until then. |
| H3 | Non-compensatory tiered gate: `gate_passed` as hard conjunction, CI-score-as-tiebreaker-only, lexicographic round selection, exponential recency decay (ADR-009) | `agentic_v2/scoring/multidimensional_scoring.py` (561 LOC) + `workflows/lib/ci_calculator` | Real, tested "all dimensions must pass" discipline beyond evalkit's per-grader hard gates | **Upstream the math** (tier/tiebreaker/decay) as evalkit aggregation policies; keep ARP's research-tier semantics ARP-side. |
| H4 | Benchmark registry: 9 definitions (SWE-bench ×3, HumanEval ×2, MBPP ×2, CodeClash, custom-local) with papers/leaderboards/citations/metrics + presets | `tools/agents/benchmarks/datasets.py`, `registry.py` (~2,700 LOC package, no `agentic_v2` imports) | Curated breadth evalkit lacks (2 presets today) | **Upstream as evalkit presets + adapters**, fixing the verified loader defects in the port (missing HF config param, 16-hex cache keys, exceptions collapsed to `[]`, `datasets/` shadowing footgun). Evalkit's provider stack **replaces** the legacy loaders — port definitions, not transport. |
| H5 | Deterministic metrics: accuracy/precision/recall/F1/confusion, code-quality/lint/complexity heuristics, latency-percentile scoring | `agentic-v2-eval/src/agentic_v2_eval/metrics/` (620 LOC, pure) | Objective grader material evalkit doesn't have | **Port to ARP-side objective graders now** (Task 7); **upstream candidates** after API review (Task 8). |
| H6 | Judge prompt packs + pattern conformance: `PatternEvaluator` (per-pattern phase/state-machine data, median-of-N, 256KB ReDoS-guard parsing), `StandardEvaluator`, `QualityEvaluator`, and `quality.yaml`/`prompt_pattern.yaml`/`prompt_standard.yaml` | `agentic-v2-eval/src/agentic_v2_eval/evaluators/` + `rubrics/` | Agentic-pattern structural judging is rare; prompt packs are reusable IP | **Port to ARP-side `JudgeClient`-backed graders** (Task 7); pattern-conformance judging is an evalkit upstream candidate once calibrated. |
| H7 | Dual eval-gate CI pattern: deterministic key-free floor on every commit + opt-in median-of-N live gate (label/nightly) + clean exit-0 degradation without credentials; plus `derive_criteria()`'s signal extraction and missing-criteria hard-fail | `scripts/eval_gate.py` (569 LOC) + `eval-package-ci.yml` jobs + `datasets/default/README.md` | Excellent CI design; the glue is ARP-specific | **Replicate the pattern** in the re-platformed gates (Tasks 5-6) and **document it** in an evalkit CI guide (Task 8). Code itself retires with the gate. |
| H8 | ADR-010/011/012 commit-driven A/B harness design (git-worktree isolation, sequential A/B, hexagonal core, two-layer scoring) — **design only, never built** | `docs/adr/ADR-010/011/012` | Well-reasoned blueprint; evalkit `compare` already covers the statistical core | **Design reference only.** Mark the ADRs superseded-or-deferred in Task 10; no port. |
| H9 | Structured human-escalation (`HandoffSummary`, sink protocol, escalation-never-masks-failure) | `agentic_v2/governance/escalation.py` (168 LOC, zero deps) | Under-served pattern; eval-adjacent | **Optional upstream** — low priority; stays ARP-side if evalkit scope says no. |
| H10 | Golden-workflow engine-regression harness (volatile-key stripping, deterministic mock diffing) and its documented division of labor vs score gating | `agentic-workflows-v2/tests/test_golden_workflow.py`, `datasets/default/README.md` | Keeps "did the DAG change" separate from "did quality regress" | **Stays in ARP** untouched; the division-of-labor doc note survives the gate re-platform. |
| H11 | Semantic dataset→workflow field mapping | `agentic_v2/scoring/dataset_matching.py` (602 LOC, ARP-coupled) | Good "bring your own dataset" UX idea | **Reference only** for evalkit `field_mapping` UX; no port. |
| H12 | Sandbox subprocess executor | `agentic-v2-eval/src/agentic_v2_eval/sandbox/` | See diff row 10 — weaker than documented, zero callers | **Consciously dropped.** Its 25-test contract is archived as prior art for the SWE-bench Docker executor plan. |

---

## Distribution mechanics decision

> **Decision (2026-07-04, provisional): hold on PyPI; proceed with Option B (private git dependency).** Option A remains the recommended end state — adopt it later by publishing and swapping the one dependency line (the `eval` extra), once the API is proven and public release is acceptable. The concrete Option B setup checklist is below; the A/B analysis is retained for that future switch.

**Recommendation — Option A: publish `agentic-evalkit` to PyPI and consume it as a normal pinned dependency**, mirroring ARP's ADR-023 Option A′ ExecutionKit precedent (`ek = ["executionkit>=0.1.0,<0.3.0"]`).

Why A wins:

1. **It dissolves the CI-auth blocker entirely.** Public PyPI needs no credentials anywhere.
2. **It is the only option that survives ARP's constraints pipeline unmodified.** `uv export` writes non-workspace dependencies into `ci-constraints.txt`; pip rejects direct-URL requirements in constraints files, so a git-sourced dependency breaks every `pip install -c ci-constraints.txt` step unless the export is post-processed. A PyPI release exports as an ordinary `agentic-evalkit==0.1.x` pin.
3. Dependabot can watch it; the `lockfile-constraints` job keeps working; SBOM/audit jobs resolve it like any dependency.
4. The package was **built for this**: MIT chosen explicitly for public release, `SECURITY.md`, clean-wheel gate, mkdocs site, and an inert `publish.yml` already configured for PyPI trusted publishing.

**Cost of A (user decision required):** publishing makes the source public and is effectively irreversible (PyPI does not truly unpublish). The GitHub repo may stay private, but the code is out. This is recorded as Open Question Q1 — **do not execute Task 1's publish steps without the owner's explicit go.**

**Fallback — Option B: private git dependency** (if the code must stay private):

- Declare the git URL as a **PEP 508 direct reference in the `eval` extra itself** — `eval = ["agentic-evalkit @ git+https://github.com/tafreeman/agentic-evalkit@v0.1.1"]`. This is load-bearing: `[tool.uv.sources]` is uv-only and pip ignores it, so a bare name there would send pip's CI install to PyPI (where the package does not exist). uv honors the direct reference and locks the exact commit SHA; a `[tool.uv.sources]` entry is redundant and may be omitted.
- **Named CI-auth blocker:** ARP's Actions `GITHUB_TOKEN` is repo-scoped and **cannot read** `tafreeman/agentic-evalkit`. Provision a fine-grained PAT (Contents: Read, that single repo), store it as an ARP Actions secret (e.g. `EVALKIT_READ_TOKEN`), and add `git config --global url."https://x-access-token:${EVALKIT_READ_TOKEN}@github.com/".insteadOf "https://github.com/"` **before every install step in every workflow that installs Python deps** (`ci.yml`, `eval-package-ci.yml` successor, `sbom.yml`, `dependency-audit.yml`, `deploy.yml`). PAT expiry becomes a recurring CI-outage risk owned by the user. A per-repo deploy key (SSH) is the non-expiring alternative at the cost of SSH plumbing in CI.
- **Constraints landmine mitigation:** add `--no-emit-package agentic-evalkit` to the `uv export` invocations in `justfile` `update-constraints`/`relock` *and* in the `lockfile-constraints` CI job, so the URL line never enters `ci-constraints.txt`; pip then resolves evalkit from the extra's pinned URL directly.
- Dependabot cannot update it; local dev works unchanged (the machine already has push credentials).

**Rejected:** git submodule (same auth problem, plus worktree/untracked-file friction and workspace noise); vendoring (defeats the boundary, guarantees drift, double maintenance); private package index (infrastructure overkill for a solo project).

### Option B setup checklist (the chosen path)

Owner actions (GitHub console — cannot be scripted from this repo):

- [x] **Create a fine-grained PAT** — provisioned 2026-07-04 with the access below: resource owner `tafreeman`; repository access → Only select repositories → `tafreeman/agentic-evalkit`; permissions → Repository permissions → **Contents: Read-only** (Metadata: Read-only auto-added; nothing else); expiration per policy (≤366 days — this is the rotation cost).
- [x] **Store it as an ARP Actions secret** — set 2026-07-04. The plan's CI snippet assumes the secret name `EVALKIT_READ_TOKEN`; if a different name was used, the workflow step must be aligned to it. The token value never enters source, the plan, or chat.
- [ ] *(Not taken — no-expiry alternative)* A **read-only deploy key** (SSH) on the evalkit repo, private key as `EVALKIT_DEPLOY_KEY`, `ssh://git@github.com/...` dependency URL + SSH-agent step in CI, would eliminate PAT rotation at the cost of SSH plumbing. Q1a resolved in favor of the PAT; revisit only if annual rotation becomes a burden.

Code changes (Task 3 wires these):

- [ ] **Dependency** — the direct-URL `eval` extra above in `agentic-workflows-v2/pyproject.toml`.
- [ ] **Constraints export** — add `--no-emit-package agentic-evalkit` to `uv export` in `justfile` targets `update-constraints` and `relock`, and to the `lockfile-constraints` CI job, so the URL line never reaches `ci-constraints.txt`. After relock, verify evalkit's transitive deps (huggingface-hub, jinja2, …) *do* appear pinned in the constraints (they are emitted normally; only evalkit itself is stripped).
- [ ] **CI git auth** — before the first dep-install step in every workflow that installs Python deps (`ci.yml`, the `eval-package-ci.yml` successor, `sbom.yml`, `dependency-audit.yml`, `deploy.yml`), add a narrow-scoped rewrite so the token only ever reaches the evalkit URL:

  ```yaml
  - name: Authenticate to private agentic-evalkit
    env:
      EVALKIT_READ_TOKEN: ${{ secrets.EVALKIT_READ_TOKEN }}
    run: >
      git config --global
      url."https://x-access-token:${EVALKIT_READ_TOKEN}@github.com/tafreeman/agentic-evalkit".insteadOf
      "https://github.com/tafreeman/agentic-evalkit"
  ```

  `git config --global` persists for the whole job, so one step per job covers all its install steps; it also covers uv (which shells out to git). Local dev needs no rewrite — the developer machine already holds push credentials.

Verification (Task 3 Step 4): a CI dry-run (scratch PR or `gh workflow run`) installs `agentic-workflows-v2[eval]` on a runner with no ambient evalkit checkout and imports `agentic_evalkit`; `ci-constraints.txt` contains no `git+` line; `lockfile-constraints` stays green.

Recurring risk owned by the user: PAT expiry silently fails all five workflows at once (Risk R2). Put a rotation reminder on the calendar, or take the deploy-key path.

**Version policy (either option):** push `2c5591d`, cut **v0.1.1** including it (and Task 2's offline fixes if ready), and have ARP pin `agentic-evalkit>=0.1.1,<0.2` with the lock holding the exact version. Pinning `v0.1.0` today is acceptable if v0.1.1 waits.

---

## Milestones and rollback boundaries

| Milestone | Tasks | Merge boundary / rollback point |
|---|---|---|
| A — evalkit preflight | 1, 2 | evalkit repo only; ARP untouched. Rollback: don't consume the new release. |
| B — dependency + decoupling | 3, 4 | One ARP PR. Rollback: revert the PR; legacy package still fully authoritative. |
| C — parity gates (non-blocking) | 5, 6 | ARP PR; new gates run alongside, legacy gates still gate merges. Rollback: delete the new jobs. |
| D — harvest | 7, 8 | Independent ARP-side and evalkit-side PRs. Rollback: per-PR revert. |
| E — cutover | 9, 10 | ARP PR + ADR; legacy gates demoted to `workflow_dispatch` for ≥1 cycle. Rollback: re-promote legacy jobs (one-line `if:` change). |
| F — retirement | 11, 12 | Tag `pre-evalkit-removal` immediately before Task 12's PR. Rollback: revert or restore from tag. |

Each milestone starts from an updated `main`, lands as its own conventional-commit PR (`feat(eval)`, `ci(eval)`, `docs(adr)`, …), and runs ARP's full local matrix (`just test && just docs && pre-commit run --all-files`) before merge.

---

## Phase A — evalkit-side preflight

### Task 1: Release hygiene and distribution setup

**Files (evalkit repo):**
- Modify: `CHANGELOG.md`
- Create: GitHub release `v0.1.1` (tag)
- Configure: PyPI project + `pypi` environment (Option A only; repository settings, no code)

- [ ] **Step 1: Push the stranded commit** — `git push origin main` (currently `2c5591d` test-only, local-only). Expected: `origin/main == main`.
- [ ] **Step 2: Distribution — Option B (decided, Q1).** Tag `v0.1.1` from `main`; provision the fine-grained PAT (Contents: Read on `tafreeman/agentic-evalkit`) and add it as ARP Actions secret `EVALKIT_READ_TOKEN` — per the Option B setup checklist. No PyPI publish and no `publish.yml` run for now (deferred with Option A). *If Option A is later adopted:* verify the name on PyPI, configure trusted publishing (PyPI project ↔ repo ↔ `pypi` environment), and cut a GitHub release to trigger `publish.yml`.
- [ ] **Step 3: Prove installability from a clean environment** — Option A: `pip install agentic-evalkit==0.1.1 && agentic-evalkit --help && agentic-evalkit datasets curated`; Option B: same via the authenticated git URL. Expected: exit 0, curated list shows `gsm8k` and `swe-bench-verified`.
- [ ] **Step 4: Record the decision** in `CHANGELOG.md` and, if Option A, note the public-release fact in `README.md`.

**Risks:** publishing is irreversible (A); PAT lifecycle (B). **Testing:** the clean-env install in Step 3 *is* the gate. **Rollback:** none needed — ARP unaffected.

### Task 2: Reconcile the offline contract (upstream fix)

Not a blocker for adoption (the ARP golden gate is local-file + deterministic and never touches the network even without the flag), but **required before any ARP CI job claims enforced hermeticity or uses `--offline`**, because today the CLI accepts the flag and silently ignores it — worse than absent.

**Files (evalkit repo):** `src/agentic_evalkit/datasets/catalog.py`, `cli/datasets.py`, `cli/runs.py`, `runner.py`, `errors.py`, matching tests; new ADR if the provider contract changes.

- [ ] **Step 1: Design decision (small ADR):** providers declare network independence (the `local` provider is exempt from offline rejection); `OfflineCacheMiss` gains a discriminator (e.g. `retryable: bool`) separating "warm the cache" from "categorically uncacheable".
- [ ] **Step 2: Thread `offline` end-to-end** — CLI `run --offline`/`datasets preview --offline` must reach `DatasetCatalog` per-call (delete the `del offline`); runner's catalog adapter forwards it through `resolve`/`iter_records`.
- [ ] **Step 3: Tests:** `run --offline` over a local dataset succeeds with zero network; `run --offline` over an uncached HF dataset fails with the typed error, not silence; truncate unbounded query text in `OfflineCacheMiss` context while there.
- [ ] **Step 4: Ship in v0.1.1 or v0.1.2** and bump ARP's floor accordingly.

**Testing:** evalkit's own suite + a socket-guard test (no network syscalls) for the local path. **Rollback:** evalkit release revert; ARP pins the prior version.

---

## Phase B — ARP dependency and decoupling

### Task 3: Add the dependency behind an `eval` extra, with an ARP-side boundary test

**Files (ARP):**
- Modify: `agentic-workflows-v2/pyproject.toml` (add `eval = ["agentic-evalkit>=0.1.1,<0.2"]` to `[project.optional-dependencies]`; Option B additionally `[tool.uv.sources]`)
- Modify: `uv.lock`, `ci-constraints.txt` (via `just update-constraints`; Option B: add `--no-emit-package agentic-evalkit` to `justfile` and the `lockfile-constraints` job first)
- Modify: `justfile` `setup` (install `[dev,server,langchain,tracing,eval]`), CI install lines that need the extra
- Create: `agentic-workflows-v2/tests/contract/test_evalkit_boundary.py`
- Create: `agentic-workflows-v2/tests/contract/test_evalkit_import.py`

- [ ] **Step 1: Write the failing boundary test first.** Locate the installed `agentic_evalkit` via `importlib.util.find_spec`, AST-scan every module file for imports rooted in `{"agentic_v2", "tools", "executionkit"}` (mirror of evalkit's own `test_dependency_boundary.py`), and additionally assert no `agentic_evalkit` module exists inside the ARP source tree (anti-vendoring guard). Expected: fails with "module not installed".
- [ ] **Step 2: Add the extra + relock.** `uv lock`, `just update-constraints`; verify `ci-constraints.txt` gained an ordinary pin (A) or no evalkit line at all (B). Run the `lockfile-constraints` check locally.
- [ ] **Step 3: Import smoke test** — `import agentic_evalkit; agentic_evalkit.__version__ == "0.1.1"`, plus constructing an `EvalRunner` with a trivial callable target against a 2-row local dataset, asserting a canonical JSON report is produced. Keep it key-free and offline (local provider).
- [ ] **Step 4: Full local matrix** — `just test && just docs && pre-commit run --all-files`; both new tests green; `AGENTIC_NO_LLM=1` unit suite unaffected.

**Dependencies:** Task 1. **Risks:** R2/R6 (Option B plumbing). **Rollback:** revert the PR — no production code references evalkit yet.

### Task 4: Remove the unconditional runtime import

**Files (ARP):**
- Modify: `agentic-workflows-v2/agentic_v2/models/llm.py` (define `LLMClientProtocol` locally or in `agentic_v2/models/interfaces.py`; stop importing from `agentic_v2_eval`)
- Modify: `agentic-workflows-v2/agentic_v2/scoring/step_scoring.py` (repoint the guarded import at the ARP-internal protocol/scorer seam; behavior unchanged this task)
- Tests: adjust the two guarded test modules if their skip conditions referenced the old import

- [ ] **Step 1:** Inline the 10-line structural protocol; keep the name and signature identical so implementers don't change.
- [ ] **Step 2:** Verify with the legacy package **uninstalled** in a scratch venv: `import agentic_v2` succeeds; `step_scoring` degrades exactly as before (`_EVAL_AVAILABLE = False`).
- [ ] **Step 3:** Full suite green with the package still installed (no behavior change either way).

**Why now:** after this task, deleting `agentic-v2-eval` can no longer break the runtime at import time — the blast radius shrinks to scripts/CI/docs. **Rollback:** revert; import restored.

---

## Phase C — gate parity

### Task 5: Re-platform the golden gate on evalkit (side-by-side)

**Files (ARP):**
- Create: `evals/golden/manifest.yaml` (or keep configuration in the driver — executor's choice, document it)
- Create: `scripts/eval_gate_evalkit.py` — an ARP-owned driver using the **library API**: `local` provider over `datasets/default/golden_cases.json`, a `CallableTarget` that loads each case's referenced `*_output.json` (no live execution), and a `CompositeGrader` whose children reproduce `derive_criteria()` exactly (Correctness = success_rate/100; Completeness = expected-steps fraction; Code Quality = required `code_metrics` keys fraction; Efficiency = retry penalty; Security = 1.0 unless `error_type` leaked) with `code` rubric weights, `hard_gate=True` standing in for the legacy missing-criteria hard-fail
- Create: `scripts/compare_eval_gates.py` (parity harness: runs both gates, diffs per-case weighted scores, tolerance 1e-6)
- Modify: `.github/workflows/eval-package-ci.yml` — add job `eval-golden-gate-evalkit` (non-blocking: `continue-on-error: true` initially)
- Create: `agentic-workflows-v2/tests/test_eval_gate_evalkit.py` (unit tests for the criterion functions against the four committed goldens)

- [ ] **Step 1:** Write the criterion functions + unit tests first (fixtures are the committed goldens — deterministic, key-free).
- [ ] **Step 2:** Implement the driver; map gate semantics to exit codes: below threshold → exit 2 (invalid/regression), grader/target errors → exit 5, mirroring evalkit CLI conventions. Emit the canonical run JSON to a CI artifact for inspection.
- [ ] **Step 3:** Parity: `python scripts/compare_eval_gates.py` — Expected: identical pass/fail verdicts on all four cases and per-case score deltas within tolerance; **commit the parity output** as evidence in the PR description.
- [ ] **Step 4:** Land the CI job non-blocking; watch ≥3 consecutive green runs on `main` before Task 9 promotes it.

**Dependencies:** Tasks 3-4. **Risks:** R7 (score-semantics drift — the parity harness is the control). **Rollback:** remove the job + scripts; legacy gate untouched throughout.

### Task 6: Re-platform the live gate

**Files (ARP):** extend `scripts/eval_gate_evalkit.py` (`--live`); modify the `eval-live-gate` successor job.

- [ ] **Step 1:** `CallableTarget` wrapping `agentic_v2.workflows.run_workflow` (this import lives in the ARP driver — correct direction); manifest `attempts: 3`.
- [ ] **Step 2:** Median-of-3 per case computed by the driver from evalkit's per-attempt sample results (evalkit's pass@k/pass^k become additional reported signals, not the gate).
- [ ] **Step 3:** Preserve the degradation contract exactly: `AGENTIC_NO_LLM=1` or no resolvable provider key → log + exit 0. Test both branches with env manipulation via `monkeypatch`.
- [ ] **Step 4:** One manual `workflow_dispatch`/labeled run with real keys before cutover; attach the run JSON artifact as evidence.

**Rollback:** same as Task 5.

---

## Phase D — harvest execution

### Task 7: ARP-side ports (metrics + judges)

**Files (ARP):**
- Create: `agentic-workflows-v2/agentic_v2/evalkit_adapters/` (name at executor's discretion): `judge_client.py` (implements evalkit's `JudgeClient` over ARP's model client; fingerprint = model+prompt hash), `objective_graders.py` (accuracy/F1, code-quality, performance-percentile criteria as evalkit `Grader`s — ported from H5), `judge_packs/` (the H6 prompt YAMLs relocated and re-keyed)
- Tests mirroring each module; deterministic via injected fake judge clients

- [ ] **Step 1:** Port `metrics/` functions as graders with their existing unit-test expectations carried over (behavioral parity).
- [ ] **Step 2:** Implement `JudgeClient`; wire `PatternEvaluator`/`StandardEvaluator`/`QualityEvaluator` prompt packs as judge configurations. All judge-backed graders run **advisory-only** (never `hard_gate=True`) until Step 3's data exists.
- [ ] **Step 3:** Start the calibration corpus: ≥30 positive / ≥30 negative human-labeled held-out cases per judge intended to gate (evalkit enforces these floors). This is human work — track it as its own backlog item; do not fake labels.
- [ ] **Step 4:** Coverage: 80% on changed lines; `AGENTIC_NO_LLM=1` suite green.

**Dependencies:** Task 3. **Risks:** R5. **Rollback:** modules are additive; revert freely.

### Task 8: Upstream contributions to evalkit

Each item is its own evalkit-repo PR following evalkit's TDD/ADR conventions; the dependency-boundary and public-docs contract tests are the non-negotiable gates. Ordered by value/effort:

- [ ] **Step 1 (H1):** self-consistency/majority-vote aggregation utilities (pure rewrite; likely `stats/` or a new `ensembles` module + ADR if public contracts grow).
- [ ] **Step 2 (H4):** benchmark presets from the ARP registry (HumanEval/+, MBPP/-sanitized, CodeClash) — definitions + adapters onto evalkit's provider stack; **fix the verified defects in the port**; each preset needs an adapter + oracle validation per ADR-0005.
- [ ] **Step 3 (H2):** judge bias mechanisms — seeded criteria shuffling, swap-consistency probe (evalkit already probes order; extend to criteria-level), calibration-drift MAE reporting against the calibration artifact.
- [ ] **Step 4 (H3):** non-compensatory tier aggregation + recency-decay/lexicographic-selection math as aggregation policies.
- [ ] **Step 5 (H7-doc):** a CI-integration guide documenting the dual-gate pattern (deterministic floor + opt-in live median-of-N + credential degradation) with the ARP gates as the worked example — **written without ARP codenames** (this lands in `docs/guides/`, which the codename scan covers).
- [ ] **Step 6:** After Steps 1-4 land, evaluate replacing ARP-side ports from Task 7 with the upstream versions (ARP floor bump).

**Rollback:** per-PR; nothing in ARP depends on these landing.

---

## Phase E — cutover

### Task 9: CI and tooling cutover

**Files (ARP):** `.github/workflows/eval-package-ci.yml` (rename/replace: drop the package test/lint/type-check/build jobs — evalkit's own repo CI owns package QA; promote the evalkit gates to blocking; demote legacy gates to `workflow_dispatch`), `ci.yml`/`sbom.yml`/`dependency-audit.yml` (swap `pip install -e "agentic-v2-eval/[dev]"` for the `eval` extra), `deploy.yml` (stop building/staging the legacy wheel), `justfile` (`setup`/`test`), `.pre-commit-config.yaml` (retarget or extend the mypy hook — decide whether the new `evalkit_adapters` modules join the strict-mypy scope; do not silently lose the repo's only mypy hook), `agentic_v2/devex/workspace_test_runner.py` (drop the legacy PACKAGES entry), `.github/dependabot.yml` + `CODEOWNERS` (Task 12 removes; here just stop depending on them).

- [ ] **Step 1:** Promote `eval-golden-gate-evalkit` to required; demote the legacy golden job to `workflow_dispatch` (keep it runnable for one full cycle).
- [ ] **Step 2:** Sweep every install line; verify each touched workflow with a PR dry-run (`gh workflow run` where applicable or a scratch PR).
- [ ] **Step 3:** `just setup && just test && just docs` green from a fresh venv.

**Rollback point:** re-promoting the legacy jobs is a one-line `if:` change; keep it that way until Task 12.

### Task 10: Documentation and ADR

**Files (ARP):**
- Create: `docs/adr/ADR-0XX-adopt-agentic-evalkit.md` (allocate the next free number at execution time; 004-006 stay unused). Contents: the adoption decision, the one-way boundary invariant now enforced from **both** sides, the distribution decision (A or B with its auth story), the harvest dispositions table (H1-H12) by reference, and the legacy-package retirement criteria. This change swaps a subsystem and moves a dependency boundary — ARP's own rules require the ADR.
- Modify: `CLAUDE.md` (the "offline evaluation harness lives in `agentic-v2-eval/`" orientation sentence), `README.md`, `CONTRIBUTING.md` (also fix the stale "35 known findings"), `.claude/rules/ci.md` + `testing.md`, `docs/evaluation/*` (5 files), `docs/architecture-eval.md`, `docs/deep-dive-agentic-v2-eval.md` (mark superseded/historical), `docs/integration-architecture.md` (fix the already-false "runtime does not import the eval package" claim to describe the new reality), `docs/NO_LLM_MODE.md`, `docs/GLOSSARY.md`, onboarding/overview docs.
- Modify: `docs/superpowers/specs/2026-07-02-evaluation-framework-design.md` — status header → superseded: the standalone package shipped as `agentic-evalkit` (external repo) rather than by evolving `agentic-v2-eval` in place; §4's invariant and architecture are realized upstream.
- Modify: `docs/adr/ADR-010/011/012` — status → deferred/superseded per H8; add a note on ADR-017's path-vs-query discrepancy (either supersede it or fix the docs — executor surfaces it, owner decides).

- [ ] **Step 1:** ADR first, then the doc sweep. `just docs` (link/fence checker) green is the gate; grep the tree for `agentic-v2-eval` afterwards and triage every remaining hit (allowed: CHANGELOG history, the ADR itself, archived docs explicitly marked historical).

**Rollback:** docs-only; revert freely.

---

## Phase F — retirement

### Task 11: Migrate examples, tools, and baselines

**Files (ARP):** rewrite `examples/05_evaluation.py` against evalkit (keep the no-API-key guarantee); port or retire `scripts/score-trace.py` (recommend: port — it becomes a thin wrapper over evalkit `report`); regenerate `datasets/default/README.md` to describe the evalkit-based gate; **keep** `golden_cases.json` + the four `*_output.json` files (they are gate *inputs*, format unchanged); archive one final legacy-gate run output next to one evalkit-gate run output as the comparability record (R7).

- [ ] **Step 1:** Rewrites + tests; `python examples/05_evaluation.py` exits 0 key-free.
- [ ] **Step 2:** Confirm nothing outside the legacy package's own tree still imports `agentic_v2_eval` — `grep -r "agentic_v2_eval" --include="*.py"` returns hits only under `agentic-v2-eval/`.

### Task 12: Deprecate and remove `agentic-v2-eval`

**Gate (all must hold):** every harvest row H1-H12 at its disposition or re-dispositioned in writing; evalkit gates blocking and green ≥1 full cycle including one real live-gate run; Tasks 4, 9, 10, 11 merged; Task 3's boundary tests green.

**Files (ARP):** delete `agentic-v2-eval/`; root `pyproject.toml` (drop the workspace member); `uv.lock` + `ci-constraints.txt` (relock); `.github/dependabot.yml` (drop the directory watch); `.github/CODEOWNERS` (drop the rule); `.pre-commit-config.yaml` (drop or retarget the scoped hook); remove the legacy `workflow_dispatch` jobs; final doc-reference sweep.

- [ ] **Step 1:** Tag the pre-removal state: `git tag pre-evalkit-removal && git push --tags`.
- [ ] **Step 2:** Remove workspace member + directory; `uv lock && just update-constraints`; expected: workspace resolves, `lockfile-constraints` green.
- [ ] **Step 3:** Full matrix: `just test && just docs && pre-commit run --all-files`; CI fully green including the evalkit gates; `AGENTIC_NO_LLM=1` suite green.
- [ ] **Step 4:** PR titled `refactor(eval)!: remove agentic-v2-eval in favor of agentic-evalkit`, body linking the ADR and the parity evidence.

**Rollback:** revert the PR or restore from `pre-evalkit-removal`. After one further clean cycle, the tag may be dropped.

---

## Risk register

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | **One-way boundary invariant erodes** (someone vendors evalkit into ARP, patches it in place, or contributes ARP-importing code upstream) | The entire architectural premise fails silently | Upstream AST contract test already enforces the evalkit side. **Yes — ARP gets its own guard** (Task 3): an ARP-side mirror test that AST-scans the *installed* `agentic_evalkit` for forbidden roots and asserts no in-repo shadow module. Catches vendored forks and bad upgrades at ARP CI time. |
| R2 | **Private-repo CI auth (Option B chosen):** ARP's Actions token cannot read `tafreeman/agentic-evalkit` | Every CI job that installs deps fails | Human half DONE (2026-07-04): fine-grained PAT (Contents: Read) provisioned + stored as `EVALKIT_READ_TOKEN`. Remaining (Task 3): narrow-scoped insteadOf rewrite in **five** workflows; verify via CI dry-run. **Recurring:** PAT expiry re-triggers this — rotation reminder, or switch to a deploy key. |
| R3 | **Public exposure (Option A) — DORMANT:** PyPI publishing is effectively irreversible | Source becomes public | Not on the current path — Q1 decided to hold PyPI. Re-arm this risk only if Option A is later adopted; it requires an explicit publish decision at that point. |
| R4 | **Offline contract gap:** `--offline` is accepted and silently ignored; `local` provider blanket-rejected under offline | False confidence in hermeticity; blocks any HF-dataset CI eval | Scoped precisely: the golden gate is network-free *by construction* (local files, callable target, objective graders) — adoption is not blocked. Task 2 fixes enforcement upstream before any ARP job claims or uses `--offline`. |
| R5 | **Judge calibration debt:** evalkit hard-blocks judge gating without ≥30/≥30 held-out human labels, fingerprint match, unexpired calibration | ARP's judge-based evaluators (H6) cannot gate releases on day one | Correct behavior, not a bug — judges run advisory until the Task 7 Step 3 corpus exists. Never work around it by mislabeling judge output as objective. |
| R6 | **pip-constraints landmine (Option B):** `uv export` writes the git URL into `ci-constraints.txt`; pip rejects URL lines in `-c` files | Every `pip install -c` step in CI breaks on the first relock | `--no-emit-package agentic-evalkit` added to *all three* export sites (justfile ×2, `lockfile-constraints` job) in the same PR as the dependency (Task 3 Step 2 verifies). |
| R7 | **Comparability break:** legacy weighted scores and evalkit `GradeResult`s are different score systems; historical trend lines reset | Regression signal gap during transition | Side-by-side parity window with a tolerance-checked comparison harness (Task 5 Step 3); archive one paired run as the conversion record (Task 11); thresholds re-derived from parity data, not guessed. |
| R8 | **Version/tag drift:** `v0.1.0` excludes `2c5591d`; the offline fix will land after; floating pins drift | ARP builds against unexpected code | Task 1 pushes and cuts v0.1.1; ARP pins `>=0.1.1,<0.2` with uv.lock holding the exact version; upgrades are deliberate relocks. |
| R9 | **Capability gaps open at deletion time** | Value silently lost — the failure mode this plan exists to prevent | Task 12's gate is the harvest table itself: every H-row dispositioned or explicitly re-dispositioned in writing before deletion. Sandbox (H12) is the one *conscious* drop, with its evidence recorded. |
| R10 | **Docs blast radius (~40 files) + pre-existing stale claims** | Contributors and agents follow dead instructions (CLAUDE.md's orientation line is loaded into every session) | Task 10 sweeps with `just docs` as the gate and fixes the three already-stale claims (integration-architecture, CONTRIBUTING mypy count, ADR-017 mismatch) while in there. |
| R11 | **Solo-dev sequencing:** concurrent sessions share working trees; long-lived migration branches rot | Half-migrated states on `main` | Six small PR-sized milestones, each independently green and revertible; no mega-branch; re-check `git status`/`log` before each commit. |

---

## Open questions and recorded assumptions

- **Q1 — DECIDED (2026-07-04):** hold on PyPI; proceed with **Option B** (private git dependency, PAT + constraints workarounds — see the Option B setup checklist). Revisit Option A (PyPI) later once the API is proven and public release is acceptable; switching is a one-line dependency change. **Q1a — DECIDED (2026-07-04):** fine-grained PAT (Contents: Read), provisioned and stored as ARP Actions secret `EVALKIT_READ_TOKEN`. Standing cost: PAT rotation before expiry (Risk R2). Deploy-key alternative remains available if rotation becomes a burden.
- **Q2:** Cut v0.1.1 before ARP pins (recommended), or pin v0.1.0 now and upgrade later? *Assumed: v0.1.1 first.*
- **Q3:** `step_scoring.py`'s future — port per-step scoring onto evalkit `CompositeGrader` (async integration in the engine listener needs a spike) or keep a minimal ARP-internal weighted scorer? *Assumed: decouple in Task 4, decide the port during Task 7; either satisfies the migration.*
- **Q4:** Does ARP ever want HF-dataset evals in CI (network), or local-only? *Assumed: local-only until Task 2 lands; HF-in-CI would also need cache-warming strategy.*
- **Q5:** Should evalkit's CLI grow entry-point plugin discovery so ARP could use `agentic-evalkit run` directly instead of library drivers? *Assumed: not required — library drivers are the supported path (ADR-0009 marks CLI discovery as intended-future); a separate evalkit plan if wanted.*
- **Q6:** ADR-010/011/012 commit-eval harness — build on top of evalkit someday, or retire the ADRs? *Assumed: retire to design-reference status (Task 10); evalkit `compare` covers the statistical core.*
- **Q7:** Who produces the judge calibration labels (Task 7 Step 3), and on what timeline? Human labeling is the long pole for judge gating. *Assumed: owner; judges stay advisory indefinitely until done.*
- **Q8:** `tools/agents/benchmarks/` retirement — after H4 presets land upstream, does ARP delete it? *Out of this plan's scope; needs its own usage map first (the server references its directories).*
- **Q9:** ARP's `docs/eval-framework-design` branch is unmerged — merge it (with the Task 10 superseded-status edit) or fold its content into the new ADR? *Assumed: merge with the status edit, preserving the design history.*

## References (verify at execution time)

- Evidence for every claim above: `agentic-evalkit` @ `main` (`2c5591d`) — `docs/release/initial-release-acceptance.md`, `_bmad-output/implementation-artifacts/deferred-work.md`, `tests/contract/test_dependency_boundary.py`, `tests/contract/test_public_docs.py`; ARP @ branch `docs/eval-framework-design` — `docs/superpowers/specs/2026-07-02-evaluation-framework-design.md`, `.github/workflows/eval-package-ci.yml`, `scripts/eval_gate.py`, `datasets/default/README.md`, `agentic-workflows-v2/pyproject.toml` (ADR-023 Option A′ comment), `justfile`.
- Style/process template: `docs/plans/2026-07-02-agentic-evalkit-initial-release.md`; follow-on-gate convention: `docs/plans/README.md`.

## Execution notes

- Do not begin Task 3 before Task 1's distribution decision is made and installable.
- Treat every ARP change as one small conventional-commit PR per milestone; run `just test && just docs && pre-commit run --all-files` before each merge.
- Never weaken the boundary tests (either repo) to make a step green; a boundary-test failure is a design failure, not a test problem.
- Preserve sample-level evidence in the new gates — no aggregate may discard per-case statuses (evalkit's contracts enforce this; keep the ARP drivers honest too).
- Upstream contributions (Task 8) must read evalkit's design/ADRs first and land through its own TDD process; ARP conventions do not apply in the evalkit repo.
