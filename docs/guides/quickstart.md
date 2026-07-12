# Quickstart

This walkthrough takes you from an empty virtual environment to a canonical
JSON evaluation report, then to a self-contained HTML report. Every command
below is copy-pasteable; run them from any empty directory outside this
repository.

## 1. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate — Linux/macOS: source .venv/bin/activate
pip install agentic-evalkit
```

The base install includes Hugging Face dataset discovery — you do not need
`datasets`, `pyarrow`, or Docker for this walkthrough.

## 2. Check your environment

```bash
agentic-evalkit doctor
```

`doctor` checks the Python version, cache read/write permissions, Hugging
Face Dataset Viewer reachability, and optional-capability availability
(`swebench`). Each check reports `ok`, `warning`, or `error` with a
remediation string. In a fresh environment you should see:

- `python_version`: ok
- `cache_read_write`: ok
- `huggingface_health`: ok (Dataset Viewer reachable)
- `capability_swebench`: warning (not installed — expected; install
  `agentic-evalkit[swebench]` plus a running Docker daemon to enable
  authoritative SWE-bench grading, per
  [ADR-0014](../adr/0014-swebench-docker-harness-executor.md))

`doctor` exits nonzero only if a check reports `error`.

## 3. See what's available without touching the network

```bash
agentic-evalkit datasets curated --format json
```

This lists the built-in, verified presets — `gsm8k` and
`swe-bench-verified` — entirely offline; it only reads the package's
built-in preset table. See [the providers guide](providers.md) for
searching and inspecting arbitrary Hugging Face datasets.

## 4. Generate a manifest from the GSM8K preset

```bash
agentic-evalkit init --preset gsm8k --output eval.yaml
```

This writes a manifest pinning:

- dataset `huggingface:openai/gsm8k` (config `main`, split `test`);
- adapter `gsm8k@1` (projects `question`/`answer` rows and normalizes the
  reference answer);
- grader `normalized-exact@1`;
- the packaged `agentic_evalkit.examples.zero_target` demo target, wired in
  automatically because no callable/subprocess/HTTP target was supplied.

`zero_target` is a transport/pipeline smoke target — it always answers
`"0"`, so it always fails GSM8K grading. It exists to prove the pipeline
runs end to end before you wire in a real system under test (see
[the targets guide](targets.md) for how to point `run` at your own
callable, subprocess, or HTTP target).

Validate the manifest before running it:

```bash
agentic-evalkit validate eval.yaml
```

## 5. Run the evaluation

```bash
agentic-evalkit run eval.yaml --limit 5 --yes
```

`--limit 5` overrides the manifest's sample count for a fast smoke run;
`--yes` skips the interactive confirmation prompt (required automatically
in noninteractive contexts such as CI). `run` prints a one-line preflight
summary, streams live progress, then prints separated outcome counts and
the canonical JSON report path:

```text
preflight: dataset=huggingface:openai/gsm8k config=main split=test adapter=gsm8k@1 grader=normalized-exact@1 target=CallableTargetConfig limit=5 attempts=1 concurrency=1
outcomes: total=5 passed=0 failed=5 partial=0 errors=0 timeouts=0 cancelled=0 abstained=0 unavailable=0
report: agentic-evalkit-runs/<run_id>.json
```

`passed=0 failed=5` is expected with the demo target — `zero_target` never
produces the right GSM8K answer. What this run verifies is the *pipeline*:
live dataset resolution, target execution, objective grading, and report
generation, all through one manifest and no importer code.

The canonical JSON report's top-level keys are `schema_version`, `run_id`,
`provenance`, `manifest`, `resolved_dataset`, `summary`, `samples`,
`started_at`, `finished_at`, and `generated_at`. `provenance` pins
`dataset_id`, `dataset_revision`, `config`, `split`, `adapter`, `grader`,
`target_name`, `environment_fingerprint`, and `code_fingerprint` — every
field needed to reproduce or compare the run later.

## 6. Regenerate a self-contained HTML report

```bash
agentic-evalkit report agentic-evalkit-runs/<run_id>.json --format html
```

`report` reads the canonical JSON — the single source of truth — and
regenerates a JSONL, Markdown, or self-contained HTML report from it. The
HTML report embeds its CSS and JSON data in one file, loads no remote
scripts or fonts, and provides filter buttons for outcome categories, so it
can be opened directly in a browser or attached to a pull request with no
server required.

## 7. Compare two runs

Once you have two canonical run files (for example, before/after a change
to your target), compare their paired success rates with a seeded
bootstrap interval:

```bash
agentic-evalkit compare run-a.json run-b.json --bootstrap-samples 1000 --seed 0
```

`compare` rejects incompatible runs (different dataset revision, adapter,
grader, target, or sampling policy) with every mismatch listed, rather than
producing a misleading delta.

## Provider failures are typed, not tracebacks

```bash
agentic-evalkit datasets inspect hf:missing/not-found
```

exits with a stable error code and no traceback:

```text
error [dataset_access_denied] access denied calling is-valid ... (HTTP 401)
```

Every CLI command that talks to a provider follows this policy: a typed
`AgenticEvalkitError` prints its stable `code` and message and maps to one
of a small set of exit codes (0 success, 2 invalid input, 3 missing
capability, 4 provider/dataset error, 5 evaluation completed with
infrastructure errors, 130 cancelled) — never a raw Python traceback unless
you pass `--debug`.

## Next steps

- [Providers](providers.md) — search and inspect any Hugging Face dataset,
  use local files, and work offline.
- [Graders](graders.md) — how objective checks gate before judges.
- [Targets](targets.md) — wire `run` to your own system under test.
- [SWE-bench](swebench.md) — the coding-agent preset and its harness
  boundary.
- [HTTP agent example](http-agent-example.md) — a complete worked example
  evaluating a real HTTP-based agent.
