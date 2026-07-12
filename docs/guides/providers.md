# Providers

`agentic-evalkit` ships two dataset providers in the base install: `local`
(files already on disk) and `huggingface` (Hub search plus Dataset Viewer
integration). Both implement the same async `DatasetProvider` protocol
(`search`, `resolve`, `preview`, `iter_records`, `healthcheck`), so a
Python caller and the CLI use identical code paths.

## Local files

The `local` provider reads four formats from an allow-listed set of root
directories:

- **JSON** — a top-level list of objects, or an object containing a
  `records` list;
- **JSONL** — one JSON object per nonblank line;
- **CSV** — parsed with `csv.DictReader` (header row required);
- **YAML** — a list of objects, parsed with `yaml.safe_load`.

Every row is validated as a JSON object and assigned a zero-based string
row ID plus a canonical-JSON SHA-256 digest, so the same logical row
produces the same digest regardless of source format. The dataset
`revision` is the SHA-256 of the raw file bytes — any byte-level change to
the source file is a different, immutable revision.

Inspect a local file:

```bash
agentic-evalkit datasets inspect local:./my-dataset.jsonl
agentic-evalkit datasets preview local:./my-dataset.jsonl --limit 3
```

Local roots are not recursively indexed — `search` against the `local`
provider always returns an empty page. Point commands at the exact file
path you want to use.

## Contamination and held-out datasets

Both built-in presets (`gsm8k`, `swe-bench-verified`) are long-public,
widely mirrored benchmarks and carry
`contamination=ContaminationMetadata(status=ContaminationStatus.SUSPECT)`
(ADR-0013): their scores must not back a capability claim without an
overlap or decontamination check first. `SUSPECT` is informative, not
enforcing — the framework never refuses to run a suspect preset; it refuses
to let the risk stay unlabeled.

The supported pattern for a defensible held-out set is the local provider:
author your own rows, keep them unpublished, and declare that provenance:

```python
from agentic_evalkit.models import ContaminationMetadata

metadata = ContaminationMetadata(
    held_out=True,
    canary_ids=("TRIPWIRE-ALPHA-001",),
)
```

- `held_out=True` records that the dataset itself was never published, so
  it cannot appear in any model's pretraining corpus by construction. (This
  is not the judge-calibration held-out corpus from the calibration floor —
  see the field's docstring for the disambiguation.)
- Embed each `canary_ids` token inside your task content, then check model
  outputs with `agentic_evalkit.graders.find_canary_leaks` and
  `canary_leak_evidence`: a canary echoed back is a memorization/leakage
  tripwire. Matching is normalization-insensitive, so case-mangled echoes
  are still caught.

## Hugging Face

The `huggingface` provider combines `huggingface_hub.HfApi` (search and
immutable revision metadata) with the Hugging Face Dataset Viewer HTTP API
(validity, splits, schema, rows, pagination, size, statistics, and Parquet
metadata). It never imports `datasets` or `pyarrow`, and it never sets
`trust_remote_code=True` — a dataset that requires remote code to load
raises a typed `UnsafeCodeRequired` error instead of executing uploaded
code on your machine.

Search and inspect any public dataset with no authentication required:

```bash
agentic-evalkit datasets search "coding agents" --provider huggingface
agentic-evalkit datasets inspect hf:princeton-nlp/SWE-bench_Verified
agentic-evalkit datasets preview hf:openai/gsm8k --config main --split test --limit 3
```

### Authentication for private and gated datasets

Public dataset access needs no credentials. For private or gated datasets,
the provider honors the standard Hugging Face credential resolution order:
the `HF_TOKEN` environment variable, or a token stored via `huggingface-cli
login` / `huggingface_hub.login()`. No `agentic-evalkit`-specific
credential configuration is needed — set `HF_TOKEN` before invoking the
CLI or constructing `HuggingFaceDatasetProvider` in Python, and access
follows the Hub's usual authorization rules.

### What `resolve()` guarantees

Resolution is immutable: once `resolve()` returns a `ResolvedDataset`, its
commit SHA, config, split, schema metadata, license, citation, and
gated-access flags are pinned for the lifetime of that object. `resolve()`
treats validity, dataset info, and splits as load-bearing — a failure on
any of them fails the whole resolution with a typed error
(`DatasetNotFound`, `DatasetAccessDenied`, `DatasetConfigRequired`, and so
on). Size, statistics, and Parquet metadata are best-effort: many valid
datasets legitimately lack statistics or a Parquet conversion, so a
failure there is recorded as an explicit absence in the resolved metadata
rather than failing the whole resolve.

### Cache and `--offline` mode

Every provider call goes through a content-addressed cache keyed by
provider, canonical dataset ID, immutable revision, config, split, and
page offset/limit. Two distinct cache record types exist: full-dataset
entries and page entries. Every cache entry has a manifest and checksum;
corruption is a typed `DatasetIntegrityError`, distinct from a plain
cache miss.

```bash
agentic-evalkit datasets pull hf:openai/gsm8k --config main --split test --limit 100
agentic-evalkit datasets preview hf:openai/gsm8k --config main --split test --offline
```

`pull` records an immutable cache entry at the resolved revision — it is a
snapshot, not a "keep this dataset up to date" operation. `--offline`
serves only exact previously-cached pages and never contacts a provider;
requesting an uncached page while offline raises `OfflineCacheMiss` rather
than silently returning nothing or falling back to the network.

The cache lives in the platform user-cache directory by default (honoring
`AGENTIC_EVALKIT_CACHE_DIR` as an override, then `%LOCALAPPDATA%` on
Windows or `$XDG_CACHE_HOME`/`~/.cache` elsewhere).

### Parallel and multi-process runs

The supported pattern for running evaluations in parallel — multiple
processes or CI workers at once — is to give **each worker its own cache
directory** via a distinct `AGENTIC_EVALKIT_CACHE_DIR`. Per-worker
directories are fully isolated on disk, so workers never contend on the same
entry and one worker's in-progress write can never be observed by another:

```bash
# Worker 1
AGENTIC_EVALKIT_CACHE_DIR=/cache/worker-1 agentic-evalkit run eval.yaml --yes
# Worker 2 (in parallel)
AGENTIC_EVALKIT_CACHE_DIR=/cache/worker-2 agentic-evalkit run eval.yaml --yes
```

Sharing a single cache root across parallel workers is **not recommended**.
It remains correct — every read verifies checksum, byte count, and key
identity, so a racing write is fail-closed (a partially-applied entry
surfaces as a typed `DatasetIntegrityError`, never as silently corrupt data).
But the per-key write lock serializes writers **within one process only**;
across processes there is no lock at all — correctness comes solely from
checksum-on-read — and `Path.replace()` atomicity is not guaranteed on every
Windows filesystem. A shared root does buy a warm shared cache (each dataset
downloaded once instead of once per worker), at the price of transient
re-downloads and integrity retries under write contention; when that trade
matters, warm the shared cache in a single process first, then fan out
read-only. For concurrent writes, prefer one `AGENTIC_EVALKIT_CACHE_DIR` per
worker.

## Writing a custom provider

`agentic-evalkit` does not discover third-party providers automatically —
there is no entry-point scanning ([ADR-0019](../adr/0019-retract-unshipped-entry-point-plugin-discovery.md)
retracted the discovery routine an earlier design had planned). Instead, a
custom provider is a plain object that structurally satisfies the async
`DatasetProvider` protocol (`search`, `resolve`, `preview`, `iter_records`,
`healthcheck`, plus the `api_version` and `requires_network` attributes).
The caller constructs it and adds it to the `providers` mapping passed into
`DatasetCatalog`:

```python
from agentic_evalkit.datasets.catalog import DatasetCatalog

catalog = DatasetCatalog(
    providers={"my-provider": my_provider_instance},
)
```

This mirrors exactly how the CLI itself builds a catalog — see
`build_catalog()` in `src/agentic_evalkit/cli/datasets.py`, which
constructs the built-in `local` and `huggingface` providers directly and
passes them into `DatasetCatalog` the same way, so all three providers are
routable by name from one catalog.

A provider name can never silently shadow a name reserved for a built-in
provider (`local` and `huggingface` by default, via the
`builtin_provider_names` constructor argument): a collision raises a typed
`PluginCompatibilityError` naming the offending provider. A caller that
constructs the genuine `local`/`huggingface` providers itself, rather than
supplying a third-party replacement for them — as `build_catalog()` does —
passes `builtin_provider_names=()` to opt out of that check for its own
two built-ins.

See [ADR-0019](../adr/0019-retract-unshipped-entry-point-plugin-discovery.md)
for why entry-point discovery was retracted, and
[ADR-0003](../adr/0003-provider-plugins-and-hugging-face-baseline.md) and
[ADR-0009](../adr/0009-optional-dependencies-and-plugins.md) for the
original plugin-compatibility policy this narrows.
