# agentic-evalkit

`agentic-evalkit` is a standalone evaluation toolkit for agentic systems. It combines dynamic dataset discovery, typed evaluation contracts, benchmark-valid grading, calibration-gated judge evidence, statistical reporting, and a developer-friendly Python API and CLI. The judge-evidence piece is the full calibration-gating machinery — `CalibrationArtifact`, TNR/TPR floors, expiry, a position-bias probe — real and tested, but it ships no LLM provider client: callers supply their own `JudgeClient`, and the packaged reference judge is deterministic and permanently advisory, structurally unable to gate a release.

`agentic-evalkit` separates datasets, grading, and reporting from the system under test through callable/subprocess/HTTP targets, and objective checks gate before judges. Existing evaluation frameworks couple dataset access, grading, and reporting to specific agent platforms or model-provider SDKs; this package's neutral `ExecutionTarget` protocol makes any callable, subprocess, or HTTP system evaluable without framework lock-in.

**Why not promptfoo, Inspect, DeepEval, Braintrust, or LangSmith?** Those tools solve the eval *workflow* problem well, and for prompt-level CI assertions or red-teaming you should prefer them. This package solves the eval *validity* problem — calibration-gated judges, provenance-gated comparison, typed operational-vs-task failure separation, authoritative-verifier boundaries, and contamination tripwires — none of which they document as first-class concepts. The verified comparison and recorded build-vs-buy decision live in [docs/prior-art.md](docs/prior-art.md).

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

## Python API

The CLI is built on a small, curated Python API for integrations that need
more than the built-in presets: wrap your own system as a target, then
describe a run with a typed manifest.

```python
from agentic_evalkit import CallableTarget, DatasetRef, EvalRunManifest, EvalRunner

def my_system(sample_input: dict) -> dict:
    return {"answer": solve(sample_input["question"])}

target = CallableTarget(my_system, name="my-system")
manifest = EvalRunManifest(
    run_name="quickstart", adapter="gsm8k@1", grader="normalized-exact@1",
    target_name="my-system", dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
)
```

`EvalRunner(...).run(manifest)` then drives dataset resolution, execution,
and grading end to end. `CallableTarget` satisfies `ExecutionTarget` —
agentic-evalkit's only system-under-test boundary — which is also exported
at the top level for anyone implementing a custom target. Everything
else — additional targets, graders, reporters, dataset providers,
benchmark adapters, and statistics — is one import away under its own
subpackage (`agentic_evalkit.graders`, `agentic_evalkit.reporters`, and so
on); see [the HTTP agent example](docs/guides/http-agent-example.md) for a
complete, runnable Python-API script with the catalog/adapter/grader/
artifact-store wiring this snippet omits for brevity.

## Optional extras

The `swebench` extra (`pip install agentic-evalkit[swebench]`, pulling in
`swebench>=4.1,<5` and `docker>=7.1,<8`) is the only extra `agentic-evalkit`
declares. It backs `SweBenchDockerHarnessExecutor`, the container-based
SWE-bench Verified harness executor landed in
[ADR-0014](docs/adr/0014-swebench-docker-harness-executor.md): with the
extra installed and a reachable Docker daemon, `swebench-harness@1` grades a
real resolved/unresolved verdict instead of reporting `unavailable`. The
base install still ships without Docker or any model-provider SDK — see
[ADR-0009](docs/adr/0009-optional-dependencies-and-plugins.md) for the
extras policy.

## Documentation

- [Quickstart](docs/guides/quickstart.md) — install to first report
- [CLI reference](docs/guides/cli-reference.md) — commands, options, offline behavior, and exit codes
- [Providers](docs/guides/providers.md) — local formats, Hugging Face auth, cache/offline
- [Graders](docs/guides/graders.md) — objective-first order, hard gates, calibrated judges
- [Targets](docs/guides/targets.md) — callable, subprocess, and HTTP execution targets
- [SWE-bench](docs/guides/swebench.md) — preview/prediction workflow and the harness boundary
- [HTTP agent example](docs/guides/http-agent-example.md) — evaluating a real HTTP agent endpoint

## Repository boundary

This project does not modify or import Agentic Runtime Platform or ExecutionKit internals. Those systems may be evaluated through stable callable, subprocess, or HTTP target adapters — see [ADR-0001](docs/adr/0001-standalone-boundary.md) and [ADR-0006](docs/adr/0006-execution-target-boundary.md).
