# Saved experiments

Saved experiments make a multi-arm evaluation replayable.  An experiment YAML
records a name, purpose, current benchmark assessment, the run arms, and files
that define provenance.  Each arm is either an EvalKit manifest or a pinned
HELM preparation/evaluation request.  A run directory records immutable input
identity, per-arm status, checksums, canonical native reports, and checkpoints.

Use the dedicated commands rather than the one-off `run` command:

```powershell
uv run --no-sync agentic-evalkit experiments validate <experiment.yaml>
uv run --no-sync agentic-evalkit experiments run <experiment.yaml> --run-dir <new-directory>
uv run --no-sync agentic-evalkit experiments resume <run-directory>
uv run --no-sync agentic-evalkit experiments status <run-directory> --json
uv run --no-sync agentic-evalkit experiments compare <run-directory>
```

`resume` verifies the input identity and completed-report checksums before it
does any work.  If the config, declared evidence, dataset, target source, or
EvalKit source changes, start a new run directory.  `compare` accepts exactly
two completed native arms.  It rejects HELM arms because HELM evidence has its
own native schema and is not a task-grade result from EvalKit.

## Native manifest arms

A manifest arm can use `field-map@1` to project nested JSON into samples.
Paths are literal map keys and zero-based list indexes.  The tiny tracked
fixtures in the repository's `examples/experiments` directory
show two single-subprocess arms and a hand-off-shaped subprocess request.
They are regression-only smoke fixtures, not capability evidence.

For a real target, use a JSONL subprocess that reads one request line and
writes one response line.  The response must repeat `sample_id` and place its
structured answer under `output`.  A local `components` factory is available
when field mapping is insufficient; it must be an explicitly named trusted
`.py` file and return `ComponentBundle(adapter, grader)`.  Validation never
loads that factory, but execution does, so record its file as provenance.

## Benchmark eligibility

`regression_only` permits operational regression checks only.  A capability
experiment requires a `conditional` assessment, source URLs, task-validation
evidence, and a benchmark review no older than 90 days.  `blocked` definitions
cannot run.  These labels constrain the claim, not the harness: a clean run
means the configured process completed; it does not repair contamination,
task defects, or a changed grader.

Keep downloaded datasets, provider caches, reports, checkpoints, and model
outputs outside the repository.  Record revision, source URL, license, digest,
and limitations beside those local artifacts.  Do not promote a deterministic
smoke pass into a model or workflow claim.

## HELM preparation

Use a separate, pinned HELM Python environment.  A HELM arm specifies the
Python executable, exact `crfm-helm` version, native run entry, output/cache
root, and phase.  `phase: prepare` invokes HELM with both `--cache-instances`
and `--cache-instances-only`, so it creates native instance evidence without
model inference.  It still may download the scenario's public source data.

HELM 0.5.16 uses colons in generated run directories for entries with model
arguments. These entries require a Linux/POSIX environment; native Windows
rejects those directory names. The adapter records the failed process without
claiming successful preparation. Python runs in UTF-8 mode for portable config
decoding.

Keep the `helm-result.json`, HELM log, and native instance evidence together
in the external run directory. Installing HELM alone does not populate a
dataset or verify that a scenario can complete preparation.

## ARP SWE A/B boundary

Connect ARP through its existing `bridge.py` subprocess and an explicit local
component factory for its case adapter and pytest grader. Keep grading in
unique, revision-pinned worktrees and remove them after each attempt. Record
the factory, worker, workflow, runtime and oracle files in `provenance_files`.
Use positive and negative controls to check grading before running a model.
Preserve ARP's cost ceiling and substitution checks. A rejected model or an
interrupted workflow is an operational error, not a failed repair.

## Study measures

`agentic-evalkit experiments report <run-directory> --output-dir <new-directory>`
exports verified native-arm summaries as JSON, CSV, and Markdown. It does not
run models. Existing export directories are refused.

A domain grader supplies `evidence.study` with `expected_action` (`accept` or
`reject`), `actual_action` (`accepted` or `rejected`), and `answer_correct`
(boolean, required for accepted answers). Optional nonnegative measures:
`numeric_absolute_error`, `retry_count`, `fallback_count`. Missing labels are
reported as missing; PASS does not imply acceptance. Repeated attempts are
reported descriptively, with task-level intervals only for complete tasks.

Subprocess responses may include `telemetry` with nonnegative `input_tokens`,
`output_tokens`, `cost_usd`, `latency_ms`, plus `model_name`, `trace_refs`,
`tool_calls`, and `environment_metadata`. Unknown or malformed telemetry fails
validation. Missing telemetry stays unavailable, not zero. These fields do
not automatically instrument the target; its adapter must supply them.
