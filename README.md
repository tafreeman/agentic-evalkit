# agentic-evalkit

`agentic-evalkit` helps teams answer a practical question: did an AI agent
perform well enough, and what evidence supports that result?

Use it from the command line or Python to run repeatable evaluations, apply
objective checks, compare results, and generate reports for review. It can
evaluate Python code, command-line programs, and HTTP services without tying
them to a specific AI framework.

Objective checks run before optional model-based judging. The package includes
tested support for calibrated judge evidence while leaving model-provider
selection to the caller. The built-in reference judge is deterministic and
advisory only; it cannot approve a release.

**Why not promptfoo, Inspect, DeepEval, Braintrust, or LangSmith?** Those
tools are strong at the eval *workflow* — prompt-level CI checks,
red-teaming, experiment tracking — and are usually the better choice for
that job. This package solves a narrower problem instead: eval *validity*,
making a result structurally hard to overstate through calibration-gated
judges, provenance-gated comparisons, a strict split between operational
and task failures, authoritative-verifier boundaries, and
dataset-contamination tripwires — none of which those tools document as
first-class concepts. See the [prior-art review](docs/prior-art.md) for the
verified comparison and the build-vs-buy decision behind it.

Start with the [quickstart guide](docs/guides/quickstart.md). For design
boundaries and comparisons with other tools, see the
[architecture specification](docs/specs/2026-07-02-agentic-evalkit-design.md)
and [prior-art review](docs/prior-art.md).

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

This project imports no host-repo internals — systems are reached only through the public `ExecutionTarget` protocol (callable, subprocess, or HTTP adapters); see [ADR-0001](docs/adr/0001-standalone-boundary.md) and [ADR-0006](docs/adr/0006-execution-target-boundary.md).
