# Common rules for all agentic-evalkit task agents

Repository: C:\Users\tandf\source\agentic-evalkit (branch: feature/milestone-a).
You implement ONE task from docs/plans/2026-07-02-agentic-evalkit-initial-release.md.

REQUIRED READING (before writing any file):
1. Your task's ENTIRE section in the plan. Test/code snippets there are verbatim requirements and must pass unmodified.
2. docs/specs/2026-07-02-agentic-evalkit-design.md — at minimum §5-§10.

AVAILABLE, COMMITTED INFRASTRUCTURE:
- agentic_evalkit.models exports: DatasetRef, ResolvedDataset, SourceRecord, SearchHit, SearchPage, SamplePage, GraderSpec, EvalSample, ExecutionStatus, ExecutionRequest, NormalizedExecutionResult, GradeStatus, GradeResult, DatasetSelection, SamplingPolicy, EvalRunManifest, RunSummary, SampleResult, EvalRunResult, FrozenModel.
  EvalRunResult shape: run_id, manifest, resolved_dataset, samples: tuple[SampleResult, ...], summary: RunSummary(total/passed/failed/partial/errors/timeouts/cancelled/abstained/unavailable), started_at, finished_at.
- agentic_evalkit.errors: AgenticEvalkitError base (stable snake_case .code) with subclasses DatasetNotFound, DatasetConfigRequired, DatasetSplitNotFound, DatasetAccessDenied, DatasetLicenseRejected, DatasetIntegrityError, DatasetSchemaMismatch, DatasetProviderUnavailable, UnsafeCodeRequired, DatasetRateLimited, OfflineCacheMiss, PluginCompatibilityError, TargetFailure, TargetTimeout, GraderError, IncompatibleRuns, ManifestValidationError. Import; NEVER edit errors.py.
- agentic_evalkit.datasets.base: DatasetProvider protocol (async search/resolve/preview + iter_records async-iterator + healthcheck, api_version) and ProviderHealth. Import; do not recreate.
- Package __init__.py stubs exist for datasets/benchmarks/targets/graders/stats/reporters — append exports, keep the docstring, only for packages your task file says you own.

EXECUTION RULES:
- TDD: write the failing test first, run it, observe the expected failure, implement minimally, observe pass, refactor.
- Environment: uv 0.11.23, Python 3.13. Commands: `uv run pytest <your tests> -v`, `uv run ruff check <your paths>`, `uv run ruff format <your paths>` then `--check`, `uv run mypy <your src paths>` (strict: annotate everything).
- Other agents work concurrently in other directories. Scope lint/type/test commands to YOUR paths; if repo-wide runs fail in files you don't own, ignore those and note it.
- NEVER run git commands. NEVER touch files outside your task file's ownership list. NEVER edit pyproject.toml unless your task file explicitly allows it.
- NEVER import agentic_v2, tools, executionkit, datasets (the HF library), or pyarrow. Never set trust_remote_code.
- Do not weaken tests, skip assertions, or replace live gates with mocks to get green.

CALLBACK (mandatory): When your task is complete or if you hit a blocker, report back to the orchestrator: what you changed, whether tests passed (paste final pytest/ruff/mypy summary lines), the exports you added, and any issues that need attention before this work can be merged.
