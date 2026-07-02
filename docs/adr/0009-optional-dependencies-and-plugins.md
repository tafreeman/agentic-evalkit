# ADR-0009: Optional Dependencies and Plugin Compatibility Policy

## Status

Accepted

## Context

`agentic-evalkit` must stay installable with a small, predictable dependency
set while still supporting Hugging Face dataset discovery out of the box and
allowing heavier or provider-specific capabilities (containerized SWE-bench
execution, model-judge providers, bulk Parquet processing) to be added
without forcing every user to install them. The approved design (§13,
"Packaging and Extension Model") requires capability-oriented extras and
versioned entry-point groups for providers, benchmark adapters, graders,
reporters, and harness executors, with explicit compatibility failures
during plugin discovery rather than silent skips.

## Decision

- The base install (no extras) contains everything needed for Hugging Face
  dataset discovery: `huggingface-hub`, `httpx`, `pydantic`, `typer`,
  `rich`, and `pyyaml`. It does not contain `datasets`, `pyarrow`, Docker
  tooling, or any model-provider SDK.
- Three capability extras are declared as empty groups in this task and
  populated by their own future implementation plans:
  - `parquet`: local bulk Parquet processing when Dataset Viewer/Hub paths
    are insufficient;
  - `judges`: selected model-provider adapters for the calibrated-judge
    grader, while the core judge protocol stays provider-neutral;
  - `swebench`: the official containerized SWE-bench executor.
- Extension points (providers, benchmark adapters, graders, reporters,
  harness executors) are discovered through versioned Python entry-point
  groups named `agentic_evalkit.<capability>.v<N>` (for example,
  `agentic_evalkit.providers.v1`). The version suffix is the plugin API
  version, not the package version.
- Plugin discovery loads each entry point, checks its declared
  `api_version` against the expected version for that group, and raises a
  typed `PluginCompatibilityError` on mismatch, load failure, or duplicate
  plugin name. Failures are reported to the caller; they are never silently
  ignored or skipped.

## Alternatives

1. **Bundle SWE-bench/Docker and judge-provider SDKs in the base install.**
   Rejected: makes the base install heavy, imposes Docker as a hard
   dependency for users who only need GSM8K-style objective evaluation, and
   conflicts with the objective-first v0.1 checkpoint in the plan.
2. **Use unversioned entry-point groups (e.g. `agentic_evalkit.providers`).**
   Rejected: any future breaking change to the plugin protocol would have no
   way to signal incompatibility to already-installed third-party plugins;
   the versioned group name makes the compatibility contract explicit and
   inspectable.
3. **Silently skip incompatible or failing plugins so discovery always
   succeeds.** Rejected: silent skips hide real integration bugs and make
   "why didn't my provider show up" failures undebuggable; explicit
   `PluginCompatibilityError` surfaces the problem at discovery time.

## Consequences

- Users who only need GSM8K-style objective evaluation install the base
  package and never pull in Docker or judge-provider SDKs.
- Adding a new capability extra does not require a breaking change to the
  base package's dependency set.
- Third-party plugins must declare an `api_version` string matching the
  entry-point group's expected version; mismatches are caught at discovery
  time with a clear, typed error rather than a confusing runtime failure
  deep in the pipeline.
- The framework must maintain a deterministic, sorted plugin discovery order
  and reject duplicate plugin names to keep behavior reproducible.

## Validation

- `tests/unit/test_plugins.py` (Task 3) asserts that a plugin declaring the
  wrong `api_version` raises `PluginCompatibilityError` with the offending
  version in the message.
- `pyproject.toml` defines `parquet`, `judges`, and `swebench` as
  `[project.optional-dependencies]` groups, initially empty, verified by the
  packaging CI job installing the base wheel without them.
- Future provider/grader/reporter tasks add entry-point group registration
  tests confirming built-in names cannot be silently overridden by a
  same-named plugin (see Task 7 Step 5 for the dataset-provider case).

## Supersession

Any future change to the plugin API version scheme (for example, introducing
`v2` groups) must supersede this ADR and document the migration path for
existing third-party plugins registered under `v1` groups.
