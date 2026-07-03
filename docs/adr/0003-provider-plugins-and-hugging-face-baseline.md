# ADR-0003: Provider Plugins and Hugging Face Baseline

## Status

Accepted

## Context

`agentic-evalkit` must let a developer discover and evaluate datasets
immediately after installing the base package, without hand-writing importer
code or installing heavyweight dependencies. The approved design
(`docs/specs/2026-07-02-agentic-evalkit-design.md`, §6.1-§6.2 and §13)
requires the baseline wheel to include Hugging Face dataset discovery, a
built-in `local` provider for files already on disk, and a versioned
Python entry-point extension mechanism so third-party dataset providers can
be added later without editing the catalog dispatcher.

Provider failures (network errors, gated access, malformed rows, incompatible
plugins) must be classified rather than collapsed into generic exceptions or,
worse, silently treated as an empty dataset. Downstream tasks (cache,
providers, catalog, benchmarks, targets, graders, runner) all raise or catch
these typed errors, so the complete error hierarchy is defined once, in this
task, and never edited by later tasks.

## Decision

- The base install ships two built-in dataset providers: `local` (files
  already on disk) and `huggingface` (Hub search plus Dataset Viewer
  integration). Hugging Face support lives in the base wheel, not behind an
  optional extra, per design §6.2.
- Dataset providers register through the versioned Python entry-point group
  `agentic_evalkit.providers.v1`. The trailing `v1` is the plugin API
  version, not the package version; a future breaking change to the provider
  protocol ships as a new `v2` group rather than mutating `v1` in place.
- Every entry-point group has an explicit expected API version string
  (`"1"` for `agentic_evalkit.providers.v1`). `load_plugins()` verifies each
  loaded plugin object's own `api_version` attribute against that expected
  version and raises a typed `PluginCompatibilityError` on mismatch, load
  failure, or duplicate plugin name. Plugin failures are always reported to
  the caller, never silently skipped.
- Hugging Face dataset code executes with remote code disabled by default
  (`trust_remote_code` is never set to `True`); a dataset that requires
  remote code execution to load raises a typed `UnsafeCodeRequired` error
  instead of running arbitrary uploaded code on the host.
- Provider failures are classified into a stable, stdlib-only error
  hierarchy rooted at `AgenticEvalkitError`, defined completely in
  `src/agentic_evalkit/errors.py` during this task: dataset not found,
  config required, split not found, access denied, license rejected,
  integrity failure, schema mismatch, provider unavailable, unsafe code
  required, rate limited, offline cache miss, plugin compatibility, target
  failure, target timeout, grader error, incompatible runs, and manifest
  validation. Each error carries a stable `snake_case` `code`, a
  human-readable `message`, and a `context` mapping whose values marked
  secret are excluded from `str(error)`.

## Alternatives

1. **Ship Hugging Face support as an optional `huggingface` extra.**
   Rejected: the design's quickstart (`agentic-evalkit init --preset gsm8k`)
   and acceptance criterion 3 require dataset search and discovery to work
   immediately after installing the base package; deferring Hugging Face
   support to an extra would break that guarantee.
2. **Use unversioned entry-point groups (e.g. `agentic_evalkit.providers`).**
   Rejected for the same reason as ADR-0009: an unversioned group has no way
   to signal a breaking protocol change to already-installed third-party
   provider plugins.
3. **Let provider failures propagate as bare exceptions or return empty
   results.** Rejected: design §6.4 requires typed, classified provider
   errors and explicitly forbids collapsing a failure into an empty dataset;
   only a successful provider response may report zero rows.
4. **Define error subclasses incrementally in each task that needs one.**
   Rejected: downstream tasks (cache, providers, catalog, benchmarks,
   targets, graders, runner) all raise or catch members of this hierarchy;
   defining it once now and freezing it prevents import cycles and
   inconsistent codes across tasks that would otherwise be implemented by
   different, possibly concurrent, workers.

## Consequences

- A developer can run `agentic-evalkit datasets search` and
  `agentic-evalkit datasets preview` against Hugging Face right after
  `pip install agentic-evalkit`, with no extras and no `datasets`/`pyarrow`
  dependency.
- Third-party dataset providers register under
  `agentic_evalkit.providers.v1` and must declare `api_version = "1"`;
  mismatched or broken plugins fail loudly at discovery time with the
  entry-point name and original exception class in the error, not a
  confusing failure deep in a run.
- `errors.py` has no imports beyond the Python standard library, so it can
  be imported by every other module (including `models/`) without creating
  a dependency cycle, and downstream tasks import from it rather than
  redefining error types.
- Remote-code-executing datasets fail closed with `UnsafeCodeRequired`
  instead of silently running uploaded code on the host machine.

## Validation

- `tests/unit/test_errors.py` (Task 3) asserts every subclass has a unique,
  stable `code`, that `str(error)` contains the code and message, that
  secret-marked context values never appear in `str(error)`, and that every
  subclass is catchable as `AgenticEvalkitError`.
- `tests/unit/test_plugins.py` (Task 3) asserts `load_plugins()` sorts
  entry points by name, returns an immutable mapping, rejects duplicate
  plugin names, and wraps a plugin declaring the wrong `api_version` (or one
  that fails to import) in `PluginCompatibilityError` naming the entry point
  and the original exception class.
- Task 7 Step 5 extends this validation to prove a third-party plugin cannot
  silently replace the built-in `huggingface` provider name.

## Supersession

Any future change to the provider plugin API version (introducing
`agentic_evalkit.providers.v2`) or to the shape of the error hierarchy
defined in `errors.py` must supersede this ADR and document the migration
path for existing `v1` provider plugins and existing error-handling call
sites.
