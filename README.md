# agentic-evalkit

`agentic-evalkit` is a standalone evaluation toolkit for agentic systems. It combines dynamic dataset discovery, typed evaluation contracts, benchmark-valid grading, calibrated judges, statistical reporting, and a developer-friendly Python API and CLI.

`agentic-evalkit` separates datasets, grading, and reporting from the system under test through callable/subprocess/HTTP targets, and objective checks gate before judges. Existing evaluation frameworks couple dataset access, grading, and reporting to specific agent platforms or model-provider SDKs; this package's neutral `ExecutionTarget` protocol makes any callable, subprocess, or HTTP system evaluable without framework lock-in.

**Coexistence note:** legacy evaluation code may remain in host repositories. This package neither imports nor migrates it — integration happens only through the public `ExecutionTarget` protocol described above.

See [the architecture specification](docs/specs/2026-07-02-agentic-evalkit-design.md) for the full design, or jump straight to the [quickstart guide](docs/guides/quickstart.md).

## Identity

- Distribution and repository: `agentic-evalkit`
- Python package: `agentic_evalkit`
- CLI: `agentic-evalkit`

## Quickstart

```bash
pip install agentic-evalkit
agentic-evalkit doctor
agentic-evalkit init --preset gsm8k --output eval.yaml
agentic-evalkit run eval.yaml --limit 5 --yes
```

This resolves the curated GSM8K preset from Hugging Face, runs five samples
through the packaged smoke target, grades them with a normalized exact-match
grader, and writes a canonical JSON report. No importer code, manual dataset
download, `datasets`, `pyarrow`, or Docker is required. See
[docs/guides/quickstart.md](docs/guides/quickstart.md) for the full walkthrough,
including the standalone `report` command that regenerates a self-contained
HTML report from that JSON.

## Optional extras

The `parquet`, `judges`, and `swebench` extras (e.g.
`pip install agentic-evalkit[swebench]`) are currently reserved placeholders
per [ADR-0009](docs/adr/0009-optional-dependencies-and-plugins.md): installing
them adds nothing today. They reserve the capability names that
`agentic-evalkit doctor` reports on, ahead of their own future
implementation.

## Documentation

- [Quickstart](docs/guides/quickstart.md) — install to first report
- [Providers](docs/guides/providers.md) — local formats, Hugging Face auth, cache/offline, plugins
- [Graders](docs/guides/graders.md) — objective-first order, hard gates, calibrated judges
- [Targets](docs/guides/targets.md) — callable, subprocess, and HTTP execution targets
- [SWE-bench](docs/guides/swebench.md) — preview/prediction workflow and the harness boundary
- [HTTP agent example](docs/guides/http-agent-example.md) — evaluating a real HTTP agent endpoint

## Repository boundary

This project does not modify or import Agentic Runtime Platform or ExecutionKit internals. Those systems may be evaluated through stable callable, subprocess, or HTTP target adapters — see [ADR-0001](docs/adr/0001-standalone-boundary.md) and [ADR-0006](docs/adr/0006-execution-target-boundary.md).
