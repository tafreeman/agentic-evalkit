# ADR-0010: Offline Dataset Contract

## Status

Accepted

## Context

`--offline` is a CLI flag (`agentic-evalkit run --offline`, `datasets
search/inspect/preview --offline`) and a `DatasetCatalog` per-call keyword
argument whose entire purpose is a hermeticity guarantee: when a caller
passes it, the framework must never open a network connection to satisfy
that call. Before this task, that guarantee did not hold in either
direction:

- The CLI accepted `--offline`, parsed it into a boolean, and then
  discarded it before it ever reached a `DatasetCatalog` (`del offline` in
  `cli/datasets.py`'s `build_catalog`). `datasets search`/`inspect` never
  forwarded the flag to the catalog methods they called at all;
  `datasets preview` forwarded it to the page-fetch call but not to its own
  preceding `resolve` call, so a "hermetic" preview still made one live
  network request. `run --offline` never reached the runner's catalog
  adapter, which had no `offline` parameter to receive it in the first
  place. A flag that is silently ignored is worse than a flag that does not
  exist: it advertises a guarantee it does not keep.
- Even where `DatasetCatalog` itself received `offline=True` honestly
  (`search`, `resolve`, `iter_records`), it rejected every call
  unconditionally with `OfflineCacheMiss` regardless of which provider was
  named. This blanket-rejected the built-in `local` filesystem provider too
  — a provider that, by construction, never touches the network on any
  method (`agentic_evalkit.datasets.local.LocalDatasetProvider` only ever
  calls `Path.read_bytes()` and directory checks). Rejecting an
  already-network-free operation as if it were a network access is
  incorrect on its own terms, and it blocks the one case ADR-0004's
  "`--offline` runs can serve exact previously-cached pages" story never
  actually needed a cache for.
- `OfflineCacheMiss` (`agentic_evalkit.errors`, defined in ADR-0003) carried
  no way to distinguish two structurally different offline failures: "no
  exact cache entry exists yet, but the same call would succeed after one
  online round trip" (a plain, recoverable cache miss) versus "this
  operation has no stable cache key at all for this call shape, so no
  amount of warming the cache ever makes the *same* offline call succeed."
  A caller (a human reading a CLI error, or downstream tooling branching on
  the error) could not tell "try going online once and repeat this exact
  call" from "this call can never work offline; do something different."

## Decision

- **Providers declare network independence.** `DatasetProvider`
  (`agentic_evalkit.datasets.base`) gains a required `requires_network: bool`
  attribute, declared the same way `api_version` already is: a plain class
  attribute on each concrete provider. `LocalDatasetProvider.requires_network
  = False` (pure filesystem I/O); `HuggingFaceDatasetProvider.requires_network
  = True` (every method calls the Hub or the Dataset Viewer HTTP API).
  `DatasetCatalog` reads this attribute through a `getattr(provider,
  "requires_network", True)` lookup — defaulting an unmarked provider to
  `True` — rather than special-casing the literal provider name `"local"`,
  so any future or third-party provider that is genuinely network-free
  (an in-memory fixture provider, an embedded-corpus provider, …) can opt in
  the same way, and any provider written before this ADR keeps today's
  behavior (offline rejected) until it explicitly declares otherwise.
- **`search`/`resolve`/`iter_records` are gated by the provider, not the
  operation alone.** Each method's offline check changes from an
  unconditional `if offline: raise` to `if offline and
  provider.requires_network: raise`. A `requires_network = False` provider
  is called normally under `offline=True` on every method — "offline" means
  "do not use the network," and such a provider never does regardless — so
  the built-in `local` provider's `search`/`resolve`/`iter_records` all
  genuinely work under `--offline` for the first time. A
  `requires_network = True` provider still cannot honor `offline=True` on
  these three methods: none of them is backed by an exact-match cache key
  (a free-text query has no stable key, a resolution is what produces the
  revision a cache key would need, and iteration is not cache-decorated at
  all), so the rejection is preserved for such providers, unchanged in
  substance from before this ADR.
- **`preview` is unaffected by `requires_network`.** It is the one
  operation with an exact-match cache key (ADR-0004), so it keeps serving
  from the content-addressed cache when possible for every provider,
  network-requiring or not, exactly as before this ADR.
- **`OfflineCacheMiss` gains a `retryable: bool` discriminator**, defaulting
  to `True`. `retryable=True` means "warm the cache (or, for a
  network-requiring provider, simply go online) and then repeat the exact
  same call — it will succeed": the genuine case is
  `agentic_evalkit.datasets.cache.DatasetCache.read()` finding no manifest
  or payload for an otherwise-cacheable key. `retryable=False` means
  "categorically uncacheable for this call shape": every catalog-level
  unconditional rejection in `search`/`resolve`/`iter_records` against a
  network-requiring provider, and `preview`'s "no cache configured on this
  catalog at all" rejection, all raise with `retryable=False` — no amount
  of prior or subsequent warming ever makes that exact offline call
  succeed; a different action (accepting a network round trip once, or
  asking for something else) is required instead.
- **Unbounded query text is truncated in error context.** A caller-supplied
  search query has no length limit by design, but `OfflineCacheMiss`
  context is surfaced verbatim in CLI stderr and, downstream, in structured
  logs or reports. `DatasetCatalog.search`'s offline rejection truncates
  `context["query"]` to 200 characters with a self-describing
  `"...(truncated, N chars total)"` suffix when the original exceeds that
  bound, so a single pathological query can never balloon one error's
  context; the free-text error *message* (read by a human debugging one
  specific failure) is left untruncated.
- **The CLI/runner threading fix is per-call-site, not a signature change
  to `DatasetCatalog` or `EvalRunner`.** `DatasetCatalog` already took
  `offline` per-call, never at construction — that shape does not change.
  Every CLI command function (`search`, `inspect`, `preview`, `run`) now
  forwards its own parsed `--offline` value into every `DatasetCatalog`
  method call it makes, including the previously-missed `resolve` calls in
  `inspect` and `preview`. `cli/runs.py`'s `_RunnerCatalogAdapter` gains an
  `offline: bool = False` constructor parameter, closed over exactly the
  way it already closes over the fixed `DatasetRef` for a run's whole
  duration, and forwards it on both `resolve()` and `iter_records()`.
  `agentic_evalkit.runner`'s `_CatalogProtocol` and `EvalRunner` are
  untouched: the adapter is the forwarding layer, so the runner's own
  narrow protocol never needs an `offline` slot.

## Alternatives

1. **Special-case the literal provider name `"local"` in `DatasetCatalog`
   instead of a `requires_network` provider attribute.** Rejected: it only
   solves this one provider. A future network-free provider (or a
   third-party plugin with the same property) would need a second special
   case added inside `DatasetCatalog` itself, whereas a structural attribute
   lets any provider opt in without touching catalog code, mirroring how
   `api_version` already lets providers self-describe.
2. **Default an unmarked provider's `requires_network` to `False`
   (permissive) instead of `True` (conservative).** Rejected: this
   codebase's own pre-existing test fakes (`_CountingFakeProvider` and
   friends in `tests/unit/datasets/test_catalog.py`, `_CannedHubProvider` in
   `tests/integration/test_cli.py`) predate this ADR and declare no
   `requires_network` attribute at all. Defaulting to `False` would silently
   exempt every one of them from offline rejection the moment this ADR
   landed — an accidental behavior change for existing tests and, more
   importantly, an unsafe default for any real third-party provider that
   simply has not been updated yet. Defaulting to `True` means "unknown"
   is treated the same as "requires network," which is the safe assumption
   and preserves every pre-existing fake's and provider's current behavior
   until it explicitly opts out.
3. **Give every `OfflineCacheMiss` raise site `retryable=False` (i.e. treat
   `retryable` as informational-only, always pessimistic).** Rejected: this
   would erase the one distinction the plan explicitly asked for. A caller
   hitting a genuine `DatasetCache.read()` miss on an otherwise-cacheable
   key has a real, actionable recovery ("go online once, then repeat this
   call"); collapsing that into the same `retryable=False` as a
   structurally uncacheable `search`/`resolve`/`iter_records` rejection
   would make the flag meaningless.
4. **Add `offline` to `agentic_evalkit.runner._CatalogProtocol` and thread
   it through `EvalRunner.run()` explicitly**, rather than closing over it
   in `_RunnerCatalogAdapter`. Rejected for this task: it widens the
   runner's own minimal protocol and touches `EvalRunner`'s public call
   sites for a value that is fixed for an entire run's duration — exactly
   the same shape `_RunnerCatalogAdapter` already uses for the immutable
   `DatasetRef` it closes over. Closing over `offline` the same way keeps
   `runner.py` untouched and the diff scoped to the CLI-side adapter that
   already exists to bridge this exact protocol mismatch.
5. **Truncate the entire `OfflineCacheMiss` message string, not just the
   `context["query"]` value.** Rejected: the free-text message is what a
   human reads first when debugging one specific failure and benefits from
   seeing the actual query; `context` is the field most likely to be
   programmatically re-serialized or aggregated across many errors (a
   report, a log aggregator), where an unbounded value is the greater
   blast-radius risk. Truncating only `context` addresses the actual risk
   without degrading the human-readable message.

## Consequences

- `--offline` now means what it says everywhere it is accepted: `run
  --offline` over a dataset resolved through the `local` provider makes
  zero network calls, end to end, and the same is true of `datasets
  search/inspect/preview --offline` against `local`.
- `--offline` against a network-requiring provider (`huggingface`, or any
  future provider that declares `requires_network = True`) fails loudly
  with a typed, `retryable`-discriminated `OfflineCacheMiss` instead of
  either silently contacting the network or silently ignoring the flag —
  the "worse than absent" failure mode this ADR exists to close.
- A caller can now branch on `error.retryable` to decide whether "try again
  after going online once" is a meaningful recovery, instead of treating
  every `OfflineCacheMiss` identically.
- Adding a genuinely network-free provider in the future (built-in or
  third-party plugin) requires only declaring `requires_network = False`
  on it; no `DatasetCatalog` code changes.
- Every pre-existing provider fake in this repository's own test suite that
  predates this ADR keeps its exact current behavior (offline still
  rejected) because it does not declare `requires_network`, and therefore
  falls back to the conservative default.

## Validation

- `tests/unit/datasets/test_catalog.py` asserts: a `requires_network=False`
  fake provider's `search`/`resolve`/`iter_records` all succeed under
  `offline=True` without the fake ever raising; the existing
  `requires_network`-less fakes (`_CountingFakeProvider`) still raise
  `OfflineCacheMiss` under `offline=True` (proving the safe default did not
  silently change pre-existing test behavior); every unconditional
  rejection site's `retryable` value matches this ADR's table (`False` for
  `search`/`resolve`/`iter_records` against a network-requiring provider and
  for `preview` with no cache configured; `True` propagated unchanged from
  a genuine `DatasetCache.read()` miss); an over-length search query is
  truncated with the documented suffix in `context["query"]` while the
  error message keeps the full text.
- `tests/unit/test_errors.py` asserts `OfflineCacheMiss(message=...).retryable
  is True` by default and that an explicit `retryable=False` construction
  round-trips, without disturbing the pre-existing `.code ==
  "offline_cache_miss"` assertion.
- A socket-guard fixture (monkeypatching `socket.socket` and
  `socket.create_connection` to raise on any invocation) wraps an end-to-end
  `run --offline` and `datasets search/inspect/preview --offline` over a
  real (non-fake) `LocalDatasetProvider`-backed dataset, proving zero
  network syscalls occur on the code path that matters, not just that a
  test double was never called.
- `tests/integration/test_cli.py` asserts `run --offline` against an
  uncached Hugging-Face-provider dataset fails with the typed error (a
  nonzero, mapped exit code and the error's code/message visible in
  output), never silently, and that `datasets search`/`inspect`/`preview
  --offline` each produce the correct exit code and user-visible message
  for both the local (succeeds) and network-requiring-provider (typed
  failure) cases.

## Supersession

A future change that gives providers finer-grained network declarations
than one boolean (for example, "requires network for `resolve` but not for
`iter_records` once resolved," or a per-method capability set) must
supersede this ADR and document the migration path for `local`/
`huggingface` and any third-party provider already relying on the single
`requires_network` attribute. Likewise, any change to `OfflineCacheMiss`'s
`retryable` semantics (for example, adding a third state, or making it
depend on live cache state rather than the raise site) must supersede this
ADR.
