# Spec: Container-backed HarnessExecutor so SWE-bench Verified runs end-to-end

**Slug:** `swebench-harness-executor`
**Status:** Proposed
**Repo:** `agentic-evalkit`
**Criteria:** C3 (outcome-based scoring), C10 (sandbox/tool-use fidelity); touches C7 (anti-gaming), C8 (statistical rigor)
**Grounded against:** `main` @ `0c967a1` (2026-07-09)

> **Refresh, 2026-07-10.** This spec's ADR number is renumbered **0012 → 0014**: ADR-0012 was
> taken by the grounded-citation probe (merged 2026-07-09) and ADR-0013 by contamination
> metadata + canaries. Every `0012` reference below (acceptance criterion 9, the ADR stub) should
> be read as `0014`; re-verify the next free number against
> `tests/contract/test_adrs.py::REQUIRED_ADR_PREFIXES` at implementation time, per this spec's own
> renumbering discipline. Its technical claims were re-verified against post-merge main: still
> accurate (no concrete `HarnessExecutor`, empty `swebench` extra, neither `swebench-verified@1`
> nor `swebench-harness@1` registered in `cli/runs.py` — implementing this also corrects the
> adapter-table comment's "every preset-referenced name resolves here" claim, which the
> swe-bench-verified preset currently falsifies). Upstream package versions (swebench 4.1.0,
> docker 7.1.0) were live-checked 2026-07-09; re-verify at implementation time.

## Problem

`agentic-evalkit` ships `SweBenchVerifiedAdapter` (`src/agentic_evalkit/benchmarks/swebench.py`)
and the versioned `HarnessRequest`/`HarnessResult`/`HarnessExecutor` contracts
(`src/agentic_evalkit/benchmarks/harness.py`, ADR-0005), but **no concrete
`HarnessExecutor` implementation exists in the repository** — a full-text
search of `src/` for `HarnessExecutor` turns up only the protocol itself,
`UnavailableHarnessExecutor`, and the test-only `FakeHarnessExecutor`. The
`swebench` optional-dependency group in `pyproject.toml` is a declared-empty
placeholder (`swebench = []`, ADR-0009). Consequently:

- **C3 (outcome-based scoring) is not just weak, it is absent.** Every run of
  the `swe-bench-verified` preset can only ever produce
  `HarnessStatus.UNAVAILABLE` / `GradeStatus.UNAVAILABLE` — there is no code
  path in this framework, today, that can apply a candidate patch and run the
  real `FAIL_TO_PASS`/`PASS_TO_PASS` suite to produce an authoritative
  `resolved` verdict. `docs/guides/swebench.md` ("Why authoritative grading
  returns `unavailable`, not a substitute score") documents this as a
  deliberate, honest interim state, not a hidden gap.
- **C8 (statistical rigor) has no substrate for this benchmark.**
  `stats/aggregate.py::pass_at_k_by_sample` already implements the
  never-fabricate-0.0-or-1.0 discipline C8 calls for (a sample "never
  successfully attempted `k` times has no defined `pass@k` estimate"), but
  because every SWE-bench attempt today grades `UNAVAILABLE`, this
  already-correct statistics code has zero real attempts to compute over for
  this benchmark. The gap is entirely upstream, in grading, not in the stats
  layer.
- **C10 (sandbox fidelity) is the literal, named blocker.** Design §7.1
  (`docs/specs/2026-07-02-agentic-evalkit-design.md`) requires the follow-on
  executor to "record harness version, image digests, patch application, test
  logs, resolution status, resources, and typed infrastructure failures" and
  to "pass one gold-patch and one intentionally invalid-patch smoke test
  through the same production path" before authoritative grading can be
  trusted — i.e. sandbox fidelity must itself be validated, not assumed
  (exactly C10's requirement). None of that exists yet.
- The repository's **own roadmap already scopes this exact change** and has
  explicitly unblocked it: `docs/plans/README.md` ("Follow-on gate: official
  SWE-bench Docker executor") states the Docker executor "may begin only now
  that the initial acceptance audit has passed," and
  `docs/release/initial-release-acceptance.md` (2026-07-03,
  `CONTINUE_FULL_V1`) confirms that audit passed with all 13 design-§17
  criteria marked `PASS`. `docs/guides/swebench.md`'s "The follow-on Docker
  executor" section already publicly commits to the same four requirements
  this spec formalizes.
- ADR-0005's own **Supersession clause** pre-declares the governance path:
  "Adding an in-repository authoritative execution harness (a real SWE-bench
  sandbox)... is a material change and must supersede this ADR with its own
  isolation and validation evidence." This spec's ADR stub honors that.

This is squarely a validity gap, not a capability gap: the framework already
has the correct typed contracts and the correct honest fallback
(`UnavailableHarnessExecutor`); what is missing is the one component that
turns "we refuse to fabricate a score" into "we can actually earn one."

## Research basis

- **C3 / outcome-based scoring, Anthropic + AISI + tau-bench pattern.** The
  executor grades what the official harness actually observed (patch
  applied, `FAIL_TO_PASS`/`PASS_TO_PASS` test results), not a trajectory or
  tool-call trace; trajectory data (build logs, patch diff) stays auxiliary
  evidence attached to the verdict, never the verdict itself — matching this
  repo's own `HarnessResult.logs`/`evidence` vs. `resolved` split already
  encoded in the contract.
- **C10 / AISI Inspect sandboxing pattern — fidelity must be validated.** The
  concrete adoptable instance is design §7.1's own requirement: one
  known-resolved (gold) patch and one intentionally-invalid patch must both
  pass through the *identical* production `execute()` code path and produce
  `resolved=True` / `resolved=False` respectively. This is this repo's
  local instantiation of "validate the sandbox, don't just build it."
- **C7 / anti-gaming, ADR-0008 pattern already proven in this repo.**
  `tests/unit/benchmarks/test_harness.py::test_generic_grade_cannot_claim_resolved_without_a_harness_result`
  already proves, at the type level, that a `GradeResult` alone can never
  smuggle out a `resolved` claim. The new grader-side mapping (below) is the
  runtime enforcement of that same discipline: `HarnessStatus.ERROR`/
  `UNAVAILABLE` must never fold into `GradeStatus.FAIL` (ADR-0008:
  operational failures are never task failures).
- **Anti-pattern this spec deliberately avoids repeating:** OpenAI retired
  SWE-bench Verified as a reported benchmark over validity concerns, and the
  original SWE-bench→Verified regrading moved GPT-4o's *measured* score from
  16% to 33.2% on the *same* model and scaffold purely by fixing task/grading
  defects. A harness that silently degrades (fabricated pass, or a
  false-negative from a flaky sandbox) would reproduce exactly that failure
  mode inside this framework. The gold/invalid-patch validation gate above is
  the direct countermeasure.
- **Upstream facts verified live (not asserted from memory), 2026-07-09:**
  the official evaluation package is `swebench` on PyPI, latest **4.1.0**
  (`requires-python >=3.10`), invoked as
  `python -m swebench.harness.run_evaluation --predictions_path ... --dataset_name ... --run_id ...`;
  its own `requires` list includes `docker`, `datasets`, `GitPython`,
  `ghapi`, `modal`, `tenacity`, among others (source:
  `SWE-bench/SWE-bench` `pyproject.toml`, GitHub `main`). Its per-instance
  report (`swebench/harness/grading.py::get_eval_report`) returns
  `{"patch_is_None", "patch_exists", "patch_successfully_applied", "resolved", "tests_status": {"FAIL_TO_PASS": {"success": [...], "failure": [...]}, "PASS_TO_PASS": {...}}}`
  — this maps directly onto `HarnessResult.resolved` (from `"resolved"`) and
  `HarnessResult.evidence` (the rest). The Docker Python SDK (`docker` on
  PyPI) is currently **7.1.0** (`requires-python >=3.8`). Both are
  compatible with this repo's `requires-python = ">=3.11"` floor.
- **Cross-repo motivation, correctly scoped.** `agentic-workflows-v2/agentic_v2/scoring/evalkit_bridge.py`
  in `agentic-runtime-platform` exists but, as verified against its current
  `main`, wires only `Rubric`/`RubricCriterion`/`CallableTarget` — a grep for
  `HarnessExecutor`, `HarnessResult`, or `pass@k` in that module returns zero
  matches, and its own docstring says it "does not wire into any ARP call
  site yet (that is Slice C)." The sibling `arp-benchmark-grading-integrity`
  spec's "prerequisite" claim is therefore about a **not-yet-built** ARP
  integration, not an existing one: this evalkit-side executor is a
  necessary but not yet sufficient upstream condition for authoritative
  SWE-bench `pass@k` inside ARP.

## Proposed change

Extend, do not replace, the existing boundary (ADR-0005: adapters project,
harness executors verify). Two new modules close the gap; everything else is
registration and packaging.

1. **`src/agentic_evalkit/benchmarks/swebench_docker.py` (new)** —
   `SweBenchDockerHarnessExecutor`, a structural `HarnessExecutor`
   implementation (satisfies the existing `runtime_checkable` protocol, no
   base class, no changes to `HarnessRequest`/`HarnessResult`/
   `HarnessStatus`). Responsibilities:
   - **Capability preflight.** Probe for an importable `swebench`/`docker`
     and a reachable Docker daemon *inside* `execute()`, not at module import
     time (the module itself must stay importable with zero extras installed
     so `cli/runs.py` and `benchmarks/__init__.py` never gain an unconditional
     `import docker`). On failure, return `HarnessStatus.UNAVAILABLE`,
     `resolved=None`, with an install/start-Docker hint — reusing the exact
     message shape `UnavailableHarnessExecutor` already establishes.
   - **Delegate to the official harness**, not a reimplementation: shell out
     to (or drive programmatically) `swebench.harness.run_evaluation` against
     a one-row predictions file built from `HarnessRequest.prediction` (the
     exact three-key export `SweBenchVerifiedAdapter.export_prediction`
     already produces), scoped to `HarnessRequest.timeout_seconds` /
     `resource_limits`.
   - **Map the upstream report** (`get_eval_report` shape, verified above)
     onto `HarnessResult`: `resolved` -> `resolved`; `patch_exists` /
     `patch_successfully_applied` / `tests_status` -> `evidence`; build-time
     image digests -> `image_digests`; anything that raises before a verdict
     exists (image pull failure, timeout, OOM) -> `HarnessStatus.ERROR` with
     `error` populated, never a guessed `resolved`.
   - **Reuse `ArtifactStore`** (`src/agentic_evalkit/artifacts.py`, already
     built for "large execution outputs, target logs, harness evidence") for
     build/test logs and the applied patch diff instead of inlining raw text
     into `HarnessResult.evidence`/`logs`.
   - Inject the Docker client/transport (mirroring how `_load_http_target` in
     `cli/runs.py` injects an `httpx.AsyncClient`) so unit tests can
     substitute a fake client and drive every status/error branch without a
     real daemon.
2. **`src/agentic_evalkit/graders/harness.py` (new)** — `HarnessGrader`, the
   currently-missing link between `runner.py`'s `Grader.grade(sample,
   execution) -> GradeResult` call and a `HarnessExecutor`. No such bridge
   exists today (`graders/` has `exact.py`, `judge.py`, `composite.py`,
   `rubric.py` — none reference `harness`). Follows `ExactMatchGrader`'s
   already-established injected-callable pattern exactly: constructed with a
   `HarnessExecutor` plus a benchmark-neutral `predictor: Callable[[EvalSample,
   NormalizedExecutionResult], dict[str, JsonValue]]` (for SWE-bench, a thin
   closure over `SweBenchVerifiedAdapter().export_prediction`), so this module
   stays grading-policy-only, matching `exact.py`'s stated design principle.
   Maps `HarnessResult` -> `GradeResult`:
   - `UNAVAILABLE` -> `GradeStatus.UNAVAILABLE`, `hard_gate=False`.
   - `ERROR` -> `GradeStatus.ERROR`, `hard_gate=False` (ADR-0008: never a
     task failure).
   - `COMPLETED` + `resolved=True/False` -> `GradeStatus.PASS`/`FAIL`,
     `hard_gate=True` (this is the one branch where a real, earned verdict
     exists).
3. **`pyproject.toml`** — populate the empty extra:
   `swebench = ["swebench>=4.1,<5", "docker>=7.1,<8"]` (versions verified
   live against PyPI above, not asserted). Regenerate `uv.lock`.
4. **`src/agentic_evalkit/cli/runs.py`** — add `SweBenchVerifiedAdapter()` to
   `_KNOWN_ADAPTERS["swebench-verified@1"]`; add a `"swebench-harness@1"`
   entry to `_build_known_graders()` constructed via a guarded import
   (`try`/`except ImportError`) that falls back to
   `UnavailableHarnessExecutor("install agentic-evalkit[swebench]")` — the
   same graceful-degradation idiom the module's docstring already documents
   for `judge-reference@1`. `agentic-evalkit --help` and every existing CLI
   test must keep passing unmodified against the base install.
5. **`docs/guides/swebench.md`** — update "The follow-on Docker executor"
   (currently future-tense) to document the shipped capability with a real
   usage example. **`docs/plans/README.md`** — mark the "Follow-on gate:
   official SWE-bench Docker executor" entry resolved, linked to the new ADR.
6. **New opt-in CI workflow** (`.github/workflows/live-swebench.yml`,
   `workflow_dispatch` + optional weekly `schedule`, mirroring
   `live-provider.yml`'s shape) installs `--extra swebench`, requires Docker
   (present on `ubuntu-latest` GitHub-hosted runners), and runs
   `tests/live -m live -k swebench`. Left **out** of the always-on
   `ci.yml`/`packaging` jobs — those must stay Docker-free and fast, and
   `live-provider.yml`'s own docstring already scopes it to the Hugging Face
   path specifically, so this is a new, parallel workflow, not an addition to
   that one.

**Explicit non-goals (to prevent scope drift):**

- No change to `HarnessRequest`/`HarnessResult`/`HarnessStatus` — the whole
  point of the existing contract (design §7.1, ADR-0005 Consequences) is that
  the executor is a pure additive implementation.
- No change to `DatasetPreset.readiness` / `required_capabilities` on the
  `swe-bench-verified` preset (`datasets/presets.py`) — `readiness` describes
  what the *base* install can do, and `required_capabilities=("swebench",)`
  already gates authoritative grading correctly; readiness is not meant to
  flip at runtime based on what happens to be installed.
- No remote/Modal-backed execution (`swebench`'s `modal` dependency notwithstanding)
  — local Docker only, matching the sketch and `docs/plans/README.md`'s
  "Docker / image resource preflight" framing.
- No wiring into ARP — `evalkit_bridge.py` integration is out of scope here
  (see Cross-repo note above); this spec only unblocks it.

## Acceptance criteria

1. `uv pip install agentic-evalkit[swebench]` installs `swebench>=4.1,<5` and
   `docker>=7.1,<8`; the base (no-extras) install and
   `tests/integration/test_clean_wheel.py` remain unchanged — neither package
   importable in that bare-wheel venv.
2. `SweBenchDockerHarnessExecutor` satisfies `isinstance(x, HarnessExecutor)`
   against the existing `runtime_checkable` protocol with **zero** diffs to
   `harness.py`'s public models.
3. Hermetic (`-m "not live"`) unit tests, using an injected fake Docker
   client (no real daemon), prove: (a) missing extra or unreachable daemon ->
   `HarnessStatus.UNAVAILABLE`, `resolved=None`, actionable message; (b)
   image-pull/build/timeout failures -> `HarnessStatus.ERROR`,
   `resolved=None`, `error` populated; (c) a successful run maps the upstream
   report's `resolved`/`patch_exists`/`patch_successfully_applied`/
   `tests_status` fields onto `HarnessResult` exactly as specified above.
4. Hermetic unit tests for `HarnessGrader`, using the **existing**
   `FakeHarnessExecutor` (no new test double needed), prove all four
   `HarnessResult` outcomes map to the correct `GradeResult.status`/
   `hard_gate`, and that `UNAVAILABLE`/`ERROR` never produce
   `GradeStatus.FAIL` (extends
   `test_generic_grade_cannot_claim_resolved_without_a_harness_result`'s
   discipline to the grader layer).
5. A manifest naming adapter `swebench-verified@1` and grader
   `swebench-harness@1` validates and resolves via `cli/runs.py`'s known
   tables (today it does not: neither name is registered).
6. Live, Docker-backed evidence (`tests/live/test_swebench_harness_live.py`,
   `@pytest.mark.live`, excluded from default CI): one gold-patch fixture
   instance resolves `True`; one intentionally-invalid-patch fixture instance
   resolves `False`; both through the identical `execute()` path (design
   §7.1's explicit smoke-test requirement). Runs only via the new opt-in
   workflow, never in `ci.yml`.
7. `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy`
   (strict) pass on all new/changed files; the repository-wide 80%
   branch-coverage floor (`tool.coverage.report.fail_under = 80`) is met
   without disabling it, with `# pragma: no cover` limited to the literal
   blocking Docker Engine API call site only if the injected-client seam
   cannot cover it any other way.
8. `tests/contract/test_dependency_boundary.py` and
   `tests/contract/test_public_docs.py` pass unmodified (no ARP /
   `agentic-tools` / ExecutionKit import introduced anywhere in the new
   modules or docs).
9. `docs/adr/0012-swebench-docker-harness-executor.md` is accepted and
   explicitly supersedes ADR-0005 per ADR-0005's own Supersession clause;
   `docs/guides/swebench.md` and `docs/plans/README.md` are updated as
   described above; `uv run mkdocs build --strict` still passes.

## ADR stub — ADR-0012: Container-Backed SWE-bench Harness Executor

**Status:** Proposed (supersedes ADR-0005's Supersession clause: "Adding an
in-repository authoritative execution harness... must supersede this ADR
with its own isolation and validation evidence.")

- **Context:** ADR-0005 established the adapter/harness split and shipped
  `UnavailableHarnessExecutor` as the only executor, deliberately deferring
  the real Docker executor to "a new, separate plan" gated on the initial
  release's acceptance audit (`docs/plans/README.md`). That audit passed
  2026-07-03 (`docs/release/initial-release-acceptance.md`,
  `CONTINUE_FULL_V1`), unblocking this work.
- **Decision (carried forward from ADR-0005, unchanged):** adapters still
  never verify; a missing/unavailable harness still returns typed
  `unavailable`, never a substitute score; `HarnessStatus` stays a
  three-member closed enum (`completed` / `unavailable` / `error`) —
  `SweBenchDockerHarnessExecutor` does not introduce a fourth status.
- **Decision (new):** `SweBenchDockerHarnessExecutor` drives the official
  `swebench` package (pinned `>=4.1,<5`) in Docker, behind the `swebench`
  extra; a `HarnessGrader` (new, `graders/harness.py`) is the sanctioned way
  a `HarnessResult` becomes a `GradeResult`, and it is the only grader
  allowed to set `hard_gate=True` from a harness verdict. Capability absence
  (extra not installed, daemon unreachable) and infrastructure failure
  (image pull, timeout, OOM) are handled at runtime inside `execute()`, never
  at import time.
- **Isolation evidence (this ADR's obligation per its own Supersession
  clause):** [STUB — fill in at implementation time] container
  resource limits applied per `HarnessRequest.resource_limits`; no
  host-filesystem writes outside the artifact store; network egress scoped
  to what the official harness itself requires.
- **Validation evidence:** [STUB — fill in at implementation time, citing
  real test paths/line numbers once merged, matching ADR-0005's own
  Validation section style] hermetic mapping tests
  (`tests/unit/benchmarks/test_swebench_docker_executor.py`,
  `tests/unit/graders/test_harness_grader.py`) plus the live gold/invalid-patch
  smoke test (`tests/live/test_swebench_harness_live.py`).
- **Consequences:** SWE-bench Verified becomes a fully `runnable`-grade
  benchmark for anyone who installs the `swebench` extra and has Docker;
  `pass_at_k_by_sample` (`stats/aggregate.py`) gains real substrate to
  compute over for this benchmark for the first time; no public contract
  changed.
- **Supersession:** introducing a fourth `HarnessStatus`, or changing how
  `resolved` is derived from the upstream report, would itself require
  superseding this ADR.

## Effort estimate: L

Two new modules across two packages (a container-orchestrating executor and
the previously-nonexistent harness-to-grader bridge), a populated
optional-dependency group with lockfile regeneration, CLI registration with a
graceful-degradation import guard, a new opt-in live CI workflow requiring
Docker, hermetic tests for every status/error branch plus a live gold/invalid
-patch smoke test, two docs updates, and a new ADR that must carry its own
isolation and validation evidence per ADR-0005's explicit Supersession
clause. This is exactly the scope the repository's own roadmap already
describes as needing "a new, separate plan" (`docs/plans/README.md`), not an
incremental patch.
