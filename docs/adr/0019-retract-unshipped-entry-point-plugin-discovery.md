# ADR-0019: Retract Unshipped Entry-Point Plugin Discovery

## Status

Accepted

## Context

`src/agentic_evalkit/plugins.py` implements `load_plugins(group,
expected_api_version)`, a deterministic Python entry-point discovery
routine for the `agentic_evalkit.<capability>.v<N>` groups ADR-0003 and
ADR-0009 specify. It shipped in the v0.1 release with zero production
consumers:

- A repo-wide search for `load_plugins` and `from agentic_evalkit.plugins`
  finds exactly one caller outside `plugins.py` itself: its own test
  module, `tests/unit/test_plugins.py`, exercising two fixtures written
  only for that test (`tests/fixtures/good_plugin.py`,
  `tests/fixtures/bad_plugin.py`). No file under `src/` other than
  `plugins.py` imports it.
- `pyproject.toml` has never declared a `[project.entry-points]` table for
  any `agentic_evalkit.*.v*` group, so the discovery routine has never had
  a real installed entry point to discover in this repository.
- Every extension point the framework actually ships is wired by
  constructor injection, not discovery. `DatasetCatalog` takes its
  `providers` mapping as a caller-supplied constructor argument — see
  `agentic_evalkit.cli.datasets.build_catalog`, which constructs the
  built-in `local` and `huggingface` providers directly and passes them
  in — and `agentic_evalkit.cli.runs` resolves adapters and graders
  through the hardcoded `_KNOWN_ADAPTERS`/`_KNOWN_GRADERS` dictionaries.
  Neither path has ever called `load_plugins()`.
- The one prospective external consumer identified to date, the ARP
  integration analysis (`docs/plans/2026-07-04-arp-integration-analysis.md`,
  open question Q5), explicitly assumed CLI-level entry-point discovery is
  *not* required: "Should evalkit's CLI grow entry-point plugin discovery
  so ARP could use `agentic-evalkit run` directly instead of library
  drivers? Assumed: not required — library drivers are the supported path
  (ADR-0009 marks CLI discovery as intended-future); a separate evalkit
  plan if wanted." That analysis has ARP integrating through this
  package's public structural protocols (`ExecutionTarget`, `Grader`,
  `JudgeClient`, `DatasetProvider`) from its own driver modules, never
  through a discovered plugin.

Carrying a discovery routine with no caller and no declared entry point is
pure supply-chain and maintenance surface — `entry_points()` (from
`importlib.metadata`) scans every installed distribution's metadata the
moment anything does call it — for a capability nothing in this
repository, and no identified downstream consumer, currently needs.

## Decision

- `src/agentic_evalkit/plugins.py` (the `load_plugins()` routine and its
  `_entry_points()` helper), `tests/unit/test_plugins.py`, and its two
  dedicated fixtures (`tests/fixtures/good_plugin.py`,
  `tests/fixtures/bad_plugin.py`) are removed.
- `PluginCompatibilityError` (`src/agentic_evalkit/errors.py`) is retained
  unchanged. It is still raised by `DatasetCatalog.__init__`
  (`agentic_evalkit.datasets.catalog`) when a caller-supplied provider name
  collides with a reserved built-in name, and it is still mapped to
  `ExitCode.MISSING_CAPABILITY` by `agentic_evalkit.cli.app`. Neither
  consumer ever depended on entry-point discovery — both depend only on
  the error type itself.
- The `api_version` attribute convention on providers and benchmark
  adapters (`DatasetProvider.api_version` in
  `agentic_evalkit.datasets.base`, `BenchmarkAdapter.api_version` in
  `agentic_evalkit.benchmarks.base`, and every concrete provider/adapter
  that declares one) is retained unchanged. It is orthogonal to discovery
  and has its own independent consumers (the structural protocol checks,
  each adapter's own tests).
- The versioned entry-point group naming convention
  (`agentic_evalkit.<capability>.v<N>`, e.g. `agentic_evalkit.providers.v1`)
  is retained as a documented convention only, in ADR-0009's Decision
  section, for any future revival — no code implements it after this
  change.
- This ADR supersedes the entry-point-discovery portion of ADR-0009's
  Decision (the `load_plugins()` mechanism and the discovery behavior it
  specifies) and narrows ADR-0003's plugin-validation bullets (the
  `load_plugins()` verification behavior) to historical record. It does
  not revisit ADR-0003's Hugging Face baseline decision — base-install
  inclusion, the `trust_remote_code` posture — at all; that half of
  ADR-0003 is untouched.

## Alternatives

1. **Keep `load_plugins()` as dormant, unshipped code.** Rejected: a
   maintained code path with zero callers and zero declared entry points
   is pure carrying cost. It still needs to stay correct against
   `importlib.metadata`'s `EntryPoint` API across supported Python
   versions, still needs its own test upkeep
   (`tests/unit/test_plugins.py`), and still counts toward the coverage
   floor — for a capability nothing in this repository, or its one
   identified prospective consumer, currently uses.
2. **Demote to docs-only: delete the code but keep a "planned" note in the
   providers guide.** Rejected as insufficient on its own, though the
   naming-convention half of this is adopted (see Decision). A "planned"
   note describing a capability with no shipped mechanism and no test
   coverage is exactly the kind of unverifiable claim this project's own
   grounding discipline argues against carrying indefinitely; the naming
   convention is preserved as a decision record in ADR-0009 instead, so a
   future revival is not starting from nothing, but no user-facing guide
   describes discovery as an available capability after this change.
3. **Wire `load_plugins()` into the CLI now instead of retracting it** —
   the path ARP's own Q5 considered and declined. Rejected: no current
   consumer asked for it. ARP's integration analysis explicitly assumed
   library drivers, not CLI-level discovery, are the supported integration
   path; speculative CLI wiring for a hypothetical future consumer
   contradicts the same grounding discipline as alternative 2.

## Consequences

- Smaller supply-chain and attack surface: nothing in this package's
  runtime scans installed-package entry-point metadata anymore, closing a
  (previously theoretical, since no entry point was ever declared) route
  by which an unrelated installed package could register something under
  an `agentic_evalkit.*.v*` group and have it silently loaded.
- One fewer module, test file, and fixture pair to keep green under
  `mypy --strict` and the coverage floor, for a capability with no caller.
- `docs/guides/providers.md`'s extension-path section now documents what
  the code actually does — constructor injection into `DatasetCatalog` —
  instead of describing an unwired discovery mechanism as if it were live.
- A future plugin system, whether a revived entry-point discovery routine
  or a different mechanism entirely, needs its own ADR. This ADR
  documents a retraction, not a moratorium on ever building discovery.
- Any third party who had already written a `pyproject.toml`
  `[project.entry-points."agentic_evalkit.providers.v1"]` table
  anticipating future discovery now has a declaration nothing in this
  repository has ever executed; no installed behavior changes for them.

## Validation

- A repo-wide search for `load_plugins`, `from agentic_evalkit.plugins`,
  and `agentic_evalkit\.plugins` confirms zero remaining references under
  `src/` or `tests/` after this change (the module and its test no longer
  exist), and confirms `PluginCompatibilityError` and the `api_version`
  convention retain their existing production call sites
  (`agentic_evalkit.datasets.catalog`, `agentic_evalkit.cli.app`,
  `agentic_evalkit.datasets.base`, `agentic_evalkit.benchmarks.base` and
  its concrete adapters) untouched.
- `tests/contract/test_adrs.py`: `"0019"` is added to
  `REQUIRED_ADR_PREFIXES`, so this ADR's shape (seven headings, canonical
  order, `Accepted` status, no phrases contradicting standing decisions)
  is enforced identically to every other ADR, and
  `test_landing_page_adr_claims_match_committed_adr_count` enforces that
  `docs/index.md`'s ADR count and range track the new total.
- The full hermetic suite (`uv run pytest -m "not live" --cov
  --cov-report=term-missing`) is green with `tests/unit/test_plugins.py`
  and its fixtures removed and no other test referencing the deleted
  module.
- `uv run mypy` and `uv run ruff check .` are clean with the module gone —
  no dangling import of `agentic_evalkit.plugins` remains anywhere in
  `src/` or `tests/`.

## Supersession

This ADR supersedes the entry-point-discovery portion of ADR-0009's
Decision (the `load_plugins()` mechanism and the discovery behavior it
specifies) and narrows ADR-0003's plugin-validation bullets to historical
record; both carry a dated amendment note pointing back at this ADR rather
than being rewritten. It does not supersede ADR-0009's extras decisions or
ADR-0003's Hugging Face baseline decision, which stand unchanged. Any
future change that reintroduces plugin or entry-point discovery — reviving
`load_plugins()`, adopting a different discovery mechanism, or wiring the
CLI to one — must supersede this ADR with a new one documenting the
mechanism and its validation, not silently resurrect the deleted module.
