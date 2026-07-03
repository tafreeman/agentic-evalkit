# agentic-evalkit

`agentic-evalkit` is a standalone evaluation toolkit for agentic systems. It
combines dynamic dataset discovery, typed evaluation contracts,
benchmark-valid grading, calibrated judges, statistical reporting, and a
developer-friendly Python API and CLI.

`agentic-evalkit` separates datasets, grading, and reporting from the system
under test through callable/subprocess/HTTP targets, and objective checks
gate before judges. Existing evaluation frameworks couple dataset access,
grading, and reporting to specific agent platforms or model-provider SDKs;
this package's neutral `ExecutionTarget` protocol makes any callable,
subprocess, or HTTP system evaluable without framework lock-in.

**Coexistence note:** legacy evaluation code may remain in host
repositories. This package neither imports nor migrates it — integration
happens only through the public `ExecutionTarget` protocol described above.

## Identity

- Distribution and repository: `agentic-evalkit`
- Python package: `agentic_evalkit`
- CLI: `agentic-evalkit`

## Get started

```bash
pip install agentic-evalkit
agentic-evalkit doctor
agentic-evalkit init --preset gsm8k --output eval.yaml
agentic-evalkit run eval.yaml --limit 5 --yes
```

This resolves the curated GSM8K preset from Hugging Face, runs five samples
through the packaged smoke target, grades them with a normalized exact-match
grader, and writes a canonical JSON report — with no importer code, manual
dataset download, `datasets`, `pyarrow`, or Docker required. Continue with
the [quickstart guide](guides/quickstart.md) for the full walkthrough.

## Guides

- [Quickstart](guides/quickstart.md) — install to first report, including the
  standalone `report` command for self-contained HTML.
- [Providers](guides/providers.md) — local JSON/JSONL/CSV/YAML formats,
  Hugging Face authentication, the content-addressed cache and `--offline`
  mode, the `parquet` extra fallback, and the provider plugin entry point.
- [Graders](guides/graders.md) — the objective-first evidence order, hard
  gates that cannot be averaged away, calibrated-judge requirements, and
  abstention/error semantics.
- [Targets](guides/targets.md) — the callable, subprocess-JSONL, and HTTP
  execution targets, plus credential hooks and timeout/retry policy.
- [SWE-bench](guides/swebench.md) — the preview/prediction-export workflow
  available today, the typed `unavailable` harness result, and the
  follow-on Docker executor boundary.
- [HTTP agent example](guides/http-agent-example.md) — evaluating a real
  tool-using agent over HTTP: request/response mapping, an authentication
  hook, timeouts, an objective schema grader, and a canonical report.

## Repository boundary

This project does not modify or import Agentic Runtime Platform or
ExecutionKit internals. Those systems may be evaluated through stable
callable, subprocess, or HTTP target adapters — see
[ADR-0001](adr/0001-standalone-boundary.md) and
[ADR-0006](adr/0006-execution-target-boundary.md).

## Reference

- [Architecture specification](specs/2026-07-02-agentic-evalkit-design.md)
- [Implementation plan](plans/2026-07-02-agentic-evalkit-initial-release.md)
- Architecture Decision Records: [0001](adr/0001-standalone-boundary.md) ·
  [0002](adr/0002-immutable-versioned-contracts.md) ·
  [0003](adr/0003-provider-plugins-and-hugging-face-baseline.md) ·
  [0004](adr/0004-content-addressed-dataset-cache.md) ·
  [0005](adr/0005-benchmark-adapters-and-harnesses.md) ·
  [0006](adr/0006-execution-target-boundary.md) ·
  [0007](adr/0007-objective-first-grading.md) ·
  [0008](adr/0008-statistical-comparability.md) ·
  [0009](adr/0009-optional-dependencies-and-plugins.md)
