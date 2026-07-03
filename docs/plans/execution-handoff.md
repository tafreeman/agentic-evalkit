# Execution handoff — agentic-evalkit initial release

**Updated:** 2026-07-02, after a session interruption (spend limit) cut three in-flight agents.
**Audience:** the orchestrating agent (any session) resuming plan execution. The plan itself is
`docs/plans/2026-07-02-agentic-evalkit-initial-release.md`; the design is
`docs/specs/2026-07-02-agentic-evalkit-design.md`.

## Decisions in force

- Checkpoint decision is pre-made: **CONTINUE_FULL_V1** (user said "complete all the work"). Still record `docs/release/v0.1-checkpoint.md` at Task 14 Step 9.
- Single branch `feature/milestone-a`; merge to `main` at milestone boundaries (after Task 7, after Task 14 Step 9, after Task 16).
- Subagents: sonnet, general-purpose, prompts per `docs/plans/agent-prompts/`. Agents NEVER run git and NEVER touch files outside their ownership list; every agent first reads `agent-prompts/COMMON.md`.
- Orchestrator runs authoritative gates before each commit: `uv run pytest -m "not live" --cov` / `ruff check .` / `ruff format --check .` / `mypy`, then commits with the plan's conventional message (no attribution trailers).

## State: done and committed (all gates green, 34 tests, coverage 94%)

| Work | Commit |
|---|---|
| Plan hardenings + review follow-ups (docs) | 83910c6, 03c0fd2, 09c96f7 |
| Task 1 foundation (+ `--version` test, CI 3.11-3.14, `invoke_without_command=True` fix) | d3c5b68 |
| Task 2 contracts (19 models) | dceb303 |
| RunSummary/samples reconciliation + `datasets/base.py` provider protocol + package `__init__` stubs | e2d9b25 |
| Task 3 errors/plugins (17 error classes; note: `tests/__init__.py` was required for entry-point loading) | HEAD |

## What is left (dispatch order)

1. **Now, in parallel** (all dependencies satisfied; disjoint files): Task 4 (cache), Task 5 (local provider), Task 6 (HF provider) — prompts exist in `agent-prompts/`; Task 8 (benchmarks, REDO — died mid-run), Task 13 (reporters, REDO — died before writing files), Task 9 (targets), Task 10 (graders, full incl. judges), Task 12 (stats) — condensed notes below.
2. After 4+5+6: Task 7 (catalog/presets). After 9+10: Task 11 (runner) — can run parallel with 7.
3. After 7, 11, 12, 13: Task 14 (CLI, full Steps 1-11). Orchestrator itself executes Step 9: clean-wheel install, run doctor/init/run on GSM8K, record CONTINUE_FULL_V1, commit checkpoint doc. Merge milestone B.
4. Task 15 (docs, workflows, dependency-boundary/clean-wheel/live gates — orchestrator runs the full verification matrix incl. `-m live`). Then Task 16 (acceptance audit). Merge to main.

## Condensed orchestration notes (tasks without prompt files)

Every agent also gets: repo path, branch, "read COMMON.md conventions" (copy its rules inline if regenerating prompts), plan section name, and the mandatory report-back callback.

- **Task 8**: use ValueError for adapter row-validation (do NOT import errors.py per original design of this task — it is now committed, so importing DatasetSchemaMismatch is also acceptable). HarnessRequest/HarnessResult subclass `models.FrozenModel`; statuses "completed"/"unavailable"/"error" + `resolved: bool | None`; `export_prediction(sample, patch, model_name_or_path="agentic-evalkit-target")` → exactly the 3 official keys; FakeHarnessExecutor lives in harness.py (documented test-only); Gsm8kAdapter emits grader=GraderSpec(name="normalized-exact@1"), adapter="gsm8k@1"; extract_final_answer: text after final "####", strip commas, "5.0"→"5".
- **Task 13**: owns `reporters/**` + `tests/unit/reporters/**`. `Reporter.write(run, destination, *, aggregates=None, generated_at=None) -> Path`; do NOT import stats. JSON envelope top-level keys: schema_version, run_id, provenance {dataset_id, dataset_revision (=resolved_dataset.revision), config, split, adapter, grader, target_name, environment_fingerprint, code_fingerprint}, manifest, resolved_dataset, summary, samples, started_at, finished_at, generated_at; sorted keys, indent 2, atomic replace; serialize via model_dump(mode="json"). RedactionPolicy + apply_redaction in base.py (model_copy-based). HTML: Jinja2 autoescape, package template report.html.j2, one self-contained file, no external URLs; byte-identical output with fixed generated_at. Shared fixture `_run_with_pass_error_timeout_and_provenance()` in tests/unit/reporters/conftest.py (revision "abc"; three samples: completed+pass grade, error+no grade, timeout+no grade).
- **Task 9**: verbatim plan snippets. CallableTarget fingerprint starts "callable:{name}:"; sync via asyncio.to_thread + asyncio.timeout. SubprocessTarget: create_subprocess_exec, one JSON line, readline-based CRLF-safe reads, byte bounds, concurrent stderr drain, kill+await on timeout, CRLF split-write fixture. HttpTarget: injected AsyncClient, retry only connection/429/502-504 (Retry-After honored), redact authorization headers, deadline → TIMEOUT.
- **Task 10 (full)**: ExactMatchGrader with injected extractor (do NOT import benchmarks; GSM8K wiring happens in Task 14 via adapter's GraderSpec). Composite semantics per verbatim test: score = weighted mean over available numeric scores (0.8), status FAIL + hard_gate True when any hard gate fails. Rubric validation rules per plan Step 4. Judge (Steps 6-9): CalibrationArtifact needs positives (TP+FN) ≥ 30 AND negatives (TN+FP) ≥ 30, TPR/TNR ≥ threshold, fingerprint equality, non-expired (expired → GradeStatus.UNAVAILABLE with evidence["reason"] containing "expired"); parse retries ≤ 2; position-bias/fingerprint/abstention tests must show none produce a gating pass.
- **Task 12**: Wilson via statistics.NormalDist().inv_cdf(0.975); pass_at_k = 1 − C(n−c,k)/C(n,k) via math.lgamma in log space, validated 0≤c≤n, 1≤k≤n; consistency_at_k = p**k; aggregate_run recounts from run.samples (grade/execution statuses), exact numerator/denominator, None bounds on empty; compare_runs(left, right, *, bootstrap_samples=1000 [100-10000], seed) — check dataset id/revision/config/split, adapter, grader, target policy, sampling, attempts; raise IncompatibleRuns listing ALL mismatches; pair by sample+attempt; random.Random(seed); return estimate, 2.5/97.5 percentiles, paired count, seed.
- **Task 7**: owns datasets/catalog.py, presets.py, appends datasets/__init__.py exports. Presets exactly per plan (gsm8k runnable / swe-bench-verified prediction_export + required capability "swebench"). Route by ref.provider (KeyError "provider 'missing'"); cache decoration on preview via exact CacheKey (hit → decode, miss → provider+write); offline=True never calls a provider; registering a plugin named like a built-in → PluginCompatibilityError.
- **Task 11**: owns artifacts.py, events.py, runner.py. Type the catalog parameter as a LOCAL Protocol (resolve + iter_records) — don't import datasets.catalog. ArtifactStore content-addressed sha256 (test pins digest of b"same": 0967115f2813a3541eaef77de9d9d5773f1c0c04314b0bbfe4ff3b3b1c55b5d5), sidecar metadata, atomic bounded writes. Runner: the 12 numbered requirements in plan Step 5, TaskGroup + semaphore, injected clock/ID factories, result.summary is RunSummary, grade only COMPLETED executions.
- **Task 14 (full)**: exit codes 0/2/3/4/5/130 (dataset not found → 4, per hardened taxonomy + verbatim test). manifest.py: yaml.safe_load only, ManifestValidationError (already in errors.py), no env interpolation. Commands: doctor, datasets curated/search/inspect/preview/pull, init, validate, run, compare, report; --format table|json, --offline. Packaged examples/zero_target.py returns "0". Orchestrator does Step 9 checkpoint itself.
- **Task 15/16**: per plan; orchestrator runs live-provider and clean-wheel gates; public-docs hygiene test scans for internal codenames; acceptance audit maps all 17 criteria to evidence.

## Contracts cheat sheet (committed, do not re-derive)

- `agentic_evalkit.models`: DatasetRef, ResolvedDataset, SourceRecord, SearchHit, SearchPage, SamplePage, GraderSpec, EvalSample, ExecutionStatus, ExecutionRequest, NormalizedExecutionResult, GradeStatus, GradeResult, DatasetSelection, SamplingPolicy, EvalRunManifest, RunSummary, SampleResult, EvalRunResult, FrozenModel. EvalRunResult: run_id, manifest, resolved_dataset, samples tuple, summary RunSummary, started_at, finished_at.
- `agentic_evalkit.errors`: AgenticEvalkitError (stable snake_case .code, secret-redacting str/repr) + DatasetNotFound, DatasetConfigRequired, DatasetSplitNotFound, DatasetAccessDenied, DatasetLicenseRejected, DatasetIntegrityError, DatasetSchemaMismatch, DatasetProviderUnavailable, UnsafeCodeRequired, DatasetRateLimited, OfflineCacheMiss, PluginCompatibilityError, TargetFailure, TargetTimeout, GraderError, IncompatibleRuns, ManifestValidationError.
- `agentic_evalkit.datasets.base`: DatasetProvider protocol (async search/resolve/preview, iter_records async-iterator, healthcheck; api_version) + ProviderHealth.
- `agentic_evalkit.plugins.load_plugins(group, expected_api_version)`.

Remove this file (and agent-prompts/) from the repo before the final release audit, or move under a non-published location — it is orchestration state, not product documentation.
