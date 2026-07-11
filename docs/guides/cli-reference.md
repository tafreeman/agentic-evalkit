# CLI reference

The `agentic-evalkit` command-line interface discovers and caches datasets,
creates and validates evaluation manifests, runs evaluations, and turns the
resulting canonical JSON into reports or paired comparisons.

Install the package and confirm the executable is available:

```bash
pip install agentic-evalkit
agentic-evalkit --version
agentic-evalkit --help
```

The CLI requires Python 3.11 or newer. Run
`agentic-evalkit <command> --help` for the options supported by the installed
version.

## Command map

| Command | Purpose |
| --- | --- |
| `doctor` | Check Python, cache, provider, and optional-capability readiness. |
| `datasets curated` | List the built-in dataset presets without network access. |
| `datasets search` | Search a provider's dataset catalog. |
| `datasets inspect` | Resolve a dataset locator and display its metadata. |
| `datasets preview` | Preview one page of raw dataset records. |
| `datasets pull` | Resolve and cache one immutable dataset page. |
| `init` | Create a manifest from a curated preset. |
| `validate` | Validate a manifest without executing it. |
| `run` | Execute a manifest and write canonical JSON. |
| `report` | Regenerate JSONL, Markdown, or HTML from canonical JSON. |
| `compare` | Compare compatible runs with a paired bootstrap interval. |

## Global conventions

Commands that display structured data use a Rich table by default. Pass
`--format json` to `doctor`, `compare`, or a `datasets` command when its help
lists the option. JSON is written to stdout, which makes these commands useful
in scripts:

```bash
agentic-evalkit doctor --format json
agentic-evalkit datasets curated --format json
agentic-evalkit compare run-a.json run-b.json --format json
```

Commands that accept `--debug` normally convert expected failures into a
stable error code and a concise message. Use `--debug` when diagnosing an
unexpected failure and a full traceback is useful.

Dataset locators use a provider prefix. The built-in forms are:

- `hf:<owner>/<dataset>` for Hugging Face, for example `hf:openai/gsm8k`
- `local:<path>` for a local JSONL, JSON, YAML, or CSV file

Use `--config` and `--split` on `datasets preview` or `datasets pull` when the
provider cannot select them automatically.

## Check the environment

```bash
agentic-evalkit doctor
agentic-evalkit doctor --offline
```

`doctor` checks the Python version, cache read/write access, Hugging Face
health, optional capabilities, and judge calibration. `--offline` skips
network checks. The command exits `3` if any check has error status, so it can
serve as a setup gate in automation.

## Work with datasets

Start with the verified presets:

```bash
agentic-evalkit datasets curated
```

Search the default Hugging Face provider, resolve a result, and preview three
records:

```bash
agentic-evalkit datasets search "grade school math" --limit 5
agentic-evalkit datasets inspect hf:openai/gsm8k
agentic-evalkit datasets preview hf:openai/gsm8k --config main --split test --limit 3
```

`search` defaults to the `huggingface` provider and 20 results. `preview`
defaults to offset `0` and limit `3`.

Cache an exact page before an offline run:

```bash
agentic-evalkit datasets pull hf:openai/gsm8k \
  --config main --split test --offset 0 --limit 100
agentic-evalkit datasets preview hf:openai/gsm8k \
  --config main --split test --offset 0 --limit 100 --offline
```

`pull` records the provider-resolved revision plus the exact config, split,
offset, and limit. It is a snapshot, not a subscription to later dataset
updates. An offline lookup succeeds only when the requested resolution and
page are already cached; otherwise it exits with a classified cache-miss
error. Set `AGENTIC_EVALKIT_CACHE_DIR` to place the cache in a specific
directory. See [Providers](providers.md) for cache layout, authentication,
and concurrent-writer guidance.

## Create and validate a manifest

```bash
agentic-evalkit init --preset gsm8k --output eval.yaml
agentic-evalkit validate eval.yaml
```

`init` requires `--preset`; discover valid names with `datasets curated`. It
refuses to overwrite an existing file unless `--force` is present. The
generated manifest uses the package's smoke target so the full pipeline can
be exercised before replacing it with a callable, subprocess, or HTTP target.

`validate` checks manifest parsing and typed constraints only. It does not
resolve a dataset, contact a target, or start an evaluation.

## Run an evaluation

```bash
agentic-evalkit run eval.yaml --limit 5 --yes
```

Before execution, `run` prints the resolved dataset selection, adapter,
grader, target type, sample limit, attempts, and concurrency. In an
interactive terminal it asks for confirmation; pass `--yes` to skip the
prompt. Non-interactive runs must pass `--yes`.

Useful options are:

- `--limit <n>` overrides the manifest's sample limit for this run.
- `--output-dir <path>` selects the report and artifact root. The default is
  `agentic-evalkit-runs/`.
- `--offline` requires the exact dataset resolution and page to be cached.
- `--debug` exposes a traceback for diagnosis.

A successful run writes a redacted canonical JSON report named from its run
ID and prints the path. Environment, code, and target fingerprints are
computed at execution time. A run that completes with sample errors or
timeouts still writes its report, then exits `5`.

## Generate another report format

Given canonical JSON from `run`, create a Markdown, JSONL, or self-contained
HTML report:

```bash
agentic-evalkit report agentic-evalkit-runs/<run-id>.json --format markdown
agentic-evalkit report agentic-evalkit-runs/<run-id>.json --format jsonl
agentic-evalkit report agentic-evalkit-runs/<run-id>.json --format html
```

Without `--output`, the source suffix is replaced with `.md`, `.jsonl`, or
`.html`. `report` reapplies the default redaction policy and recomputes report
aggregates from the run data. Use `--output <path>` to choose another
destination.

## Compare compatible runs

```bash
agentic-evalkit compare run-a.json run-b.json \
  --bootstrap-samples 1000 --seed 0
```

`compare` reports the paired success-rate estimate and its 2.5th and 97.5th
bootstrap percentiles. The bootstrap is deterministic for the same inputs and
seed. `--bootstrap-samples` accepts values from 100 through 10,000 and defaults
to 1,000.

The runs must agree on dataset identity and revision, adapter, grader, target,
and sampling policy. An incompatibility lists every mismatch and exits `2`
instead of producing a misleading comparison.

## Exit codes

| Code | Meaning |
| ---: | --- |
| `0` | Success. |
| `2` | Invalid input, manifest, schema, or run comparison. |
| `3` | Missing capability or a failing `doctor` check. |
| `4` | Provider, dataset, integrity, rate-limit, or offline-cache error. |
| `5` | Evaluation or other infrastructure error. |
| `130` | The user cancelled an interactive run. |

Scripts should check both the exit code and the generated report. In
particular, exit `5` from `run` can accompany a valid report that preserves
the evidence for failed or timed-out samples.
