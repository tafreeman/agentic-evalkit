# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Made the default `not live` CLI test suite fully hermetic by replacing its
  remaining Dataset Viewer-backed run path with a canned provider, while
  retaining provider-error and exit-code coverage. Real provider CLI checks
  now live under `tests/live/` and run only in the opt-in live workflow.

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
