# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Repeated-attempt runs (`manifest.attempts > 1`) now report an honest,
  non-pseudo-replicated `pass_rate` interval: bounds are computed
  cluster-robustly over per-`sample_id` clusters
  (`agentic_evalkit.stats.clustered_interval`) instead of treating every
  correlated attempt as an independent Wilson trial, and a new
  `IntervalMethod` enum stamps which construction (`wilson` vs
  `cluster_robust`) produced them. `score_mean` gains a matching
  `score_estimate` (SEM and 95% CI), a stdlib-only
  `agentic_evalkit.stats.required_sample_size` power helper is added, and both
  the Markdown and HTML reporters now render the `aggregates` block visibly (a
  real table and an "Uncertainty" section) rather than a raw dict repr or
  hidden JSON. The exact pooled numerator/denominator/value are unchanged and
  single-attempt runs are byte-identical to before. See
  [ADR-0016](docs/adr/0016-cluster-robust-intervals-for-repeated-attempts.md).
- `datasets pull` now persists the resolved dataset identity (`name@revision`)
  to a new offline resolution cache, so `run --offline`/`datasets
  inspect|preview --offline` can resolve a Hugging Face-backed dataset
  without contacting the provider again, as long as it was resolved online
  at least once first. `iter_records` under `--offline` also now serves an
  already-cached page (from `pull` or `preview`) at the exact `(offset,
  limit)` the runner requests, instead of unconditionally rejecting.
  Previously `run --offline` was categorically unusable for any
  Hugging-Face-backed dataset, even immediately after `pull`. See
  [ADR-0011](docs/adr/0011-offline-resolution-cache.md).
- Canonical run reports (`run` and `report`) now carry a real `aggregates`
  block: the Wilson-bounded pass rate, resource distributions, and (for
  manifests with repeated attempts) per-sample `pass@k`, computed via
  `agentic_evalkit.stats.build_report_aggregates`. Previously every
  reporter accepted an `aggregates` parameter but nothing in the CLI ever
  populated it.
- `run` now computes and persists `environment_fingerprint`,
  `code_fingerprint`, and `target_fingerprint` on every manifest via
  `agentic_evalkit.provenance`. Previously these were declared, versioned
  wire fields that no production code path ever populated, so every real
  run's report carried `null` in all three.
- Two opt-in graders, `judge-reference@1` and `composite-reference@1`, are
  now selectable from a manifest's `grader` field, backed by a new packaged
  `agentic_evalkit.examples.ReferenceJudgeClient` (a deterministic,
  network-free `JudgeClient`). Makes the calibrated-judge and composite
  grading pipelines runnable end to end from the quickstart without an LLM
  provider configured. Both are wired in permanently uncalibrated, so
  neither can ever hard-gate a release.
- `compare_runs` now gates comparability on `environment_fingerprint` and
  `code_fingerprint` alongside the existing eight provenance fields, so two
  runs pinned to different interpreters, platforms, or `agentic-evalkit`
  builds can no longer be silently diffed into a confident-looking delta.
  A caller who intentionally compares across environments can opt in with
  the new keyword-only `allow_cross_environment` parameter (`compare
  --allow-cross-environment` on the CLI); the waived field(s) are recorded
  on the new `ComparisonResult.waived_provenance_fields` rather than
  dropped. See
  [ADR-0015](docs/adr/0015-environment-and-code-fingerprints-gate-comparability.md).

## [0.1.1] - 2026-07-06

### Fixed

- Made the default `not live` CLI test suite fully hermetic by replacing its
  remaining Dataset Viewer-backed run path with a canned provider, while
  retaining provider-error and exit-code coverage. Real provider CLI checks
  now live under `tests/live/` and run only in the opt-in live workflow.
- The datasets CLI and runner now honor `--offline` end-to-end; previously
  the flag was accepted but silently ignored. The network-free `local`
  provider is exempt from offline rejection, and `OfflineCacheMiss` gains a
  retryable discriminator distinguishing a warm-the-cache miss from a
  categorically uncacheable one.

### Changed

- Package identity for public release: project URLs now point at
  `github.com/tafreeman/agentic-evalkit`, author metadata set. README
  documents the reserved optional extras.

## [0.1.0] - 2026-07-03

### Added

- Repository foundation: packaging metadata, CI matrix, MIT license,
  contributor and security policies, and the initial `agentic-evalkit` CLI
  entry point.
- Immutable, versioned Pydantic contracts for datasets, samples, execution
  results, grades, and run manifests/results (`agentic_evalkit.models`).
- Typed error hierarchy (`agentic_evalkit.errors`) and versioned
  Python entry-point plugin discovery (`agentic_evalkit.plugins`).
- Content-addressed, checksum-verified dataset cache with offline support.
- Built-in `local` dataset provider (JSON/JSONL/CSV/YAML) and
  `huggingface` provider (Hub search plus Dataset Viewer integration, no
  `datasets`/`pyarrow`/`trust_remote_code` dependency).
- Curated dataset catalog with verified `gsm8k` and `swe-bench-verified`
  presets.
- Benchmark adapter and harness contracts, including the GSM8K adapter,
  the SWE-bench Verified adapter and official prediction export, and a
  typed `unavailable` harness result for the deferred Docker executor.
- Host-neutral execution targets: `CallableTarget`, `SubprocessTarget`
  (structured JSONL), and `HttpTarget`, all normalizing to one
  `NormalizedExecutionResult` contract.
- Objective graders (`ExactMatchGrader`, `SchemaGrader`), noncompensable
  hard-gated `CompositeGrader`, atomic rubrics, and calibrated
  `JudgeGrader` with expiry/fingerprint/held-out-sample enforcement.
- Reproducible `EvalRunner` pipeline with content-addressed run artifacts
  and ordered progress events.
- Statistical aggregation: Wilson confidence intervals, `pass@k`,
  all-attempt consistency at `k`, and paired-bootstrap run comparison with
  explicit incompatible-run rejection.
- Canonical JSON, JSONL, self-contained HTML, and Markdown reporters with
  a shared redaction policy.
- A runnable CLI: `doctor`, `datasets curated/search/inspect/preview/pull`,
  `init`, `validate`, `run`, `compare`, and `report`, with a stable
  exit-code policy and no traceback on typed provider/evaluation errors.
- Documentation: architecture-decision records 0001-0009, the quickstart,
  providers, graders, targets, SWE-bench, and HTTP agent example guides,
  and a strict MkDocs Material site.
- Release gates: a dependency-boundary AST scan, an ADR shape/consistency
  check, a public-documentation codename hygiene scan, and a clean-wheel
  integration test that installs the built wheel into an isolated
  environment outside the repository.
- Scheduled live Hugging Face provider evidence
  (`.github/workflows/live-provider.yml`) and a PyPI trusted-publishing
  release workflow (`.github/workflows/publish.yml`), inert until
  configured.
