# ADR-0011: Offline Resolution Cache

## Status

Accepted

## Context

ADR-0010 made `--offline` honest everywhere it is accepted, but it also
recorded a deliberate limit: `DatasetCatalog.resolve` and `iter_records`
against a network-requiring provider (`huggingface`) unconditionally raise
`OfflineCacheMiss(retryable=False)` under `offline=True`, because "a
resolution is what produces the revision a cache key would need" and
"iteration is not cache-decorated at all." ADR-0010's own Decision section
states this in Alternative framing: resolution has no revision-independent
key to resolve *from* cache "without inventing a second, parallel keying
scheme," and design §6.3 was read as not requiring one.

In practice this makes `run --offline` and `datasets inspect/preview
--offline` categorically unusable for any Hugging-Face-backed dataset, even
immediately after an online `datasets pull` of that exact dataset. `pull`'s
own docstring already promises a "snapshot" a caller can return to later,
and the content-addressed page cache (ADR-0004) already stores the exact
page `pull` fetched -- but nothing stored the *resolved identity*
(`dataset_id`@`revision`) that a subsequent offline `resolve()` would need
to avoid contacting the provider again, and `iter_records` never consulted
the page cache `preview()`/`pull` already populate. The result: the
README's and quickstart's "resolve once online, then work offline" story
was true for `local` datasets and for `preview` specifically, but false for
the common "pull a Hugging Face dataset, then `run --offline` against it"
path -- a usability gap severe enough that the offline flag was
structurally unable to do what a user pulling a dataset would reasonably
expect.

## Decision

- **A new, separate `ResolutionCache`
  (`agentic_evalkit.datasets.resolution_cache`) persists the most recent
  resolution per request identity.** `ResolutionKey` is a frozen
  `agentic_evalkit.models.FrozenModel` with fields `provider`, `dataset_id`,
  `config`, `split`, and the caller's *requested* `revision`. It keys on
  what is known *before* a resolution exists -- which is exactly what makes
  it addressable at offline-resolve time -- and its `revision` is the
  request-side pin from `DatasetRef.revision`, not the immutable revision a
  resolution *produced* (that is `CacheKey.revision`'s role). The two
  meanings of the field:
    - `revision is None` -- "latest at resolution time" (design §5.1). Every
      unpinned request for the same dataset shares one slot, so an offline
      resolve returns the most recent "latest" resolved online.
    - `revision` set -- an explicit pin. Each distinct pin gets its own
      slot: resolving `ref@revA` then `ref@revB` online caches *both*, and an
      offline `resolve(ref@revA)` can never be silently served `revB`'s
      resolution. **Including `revision` is a correctness requirement, not an
      optimization.** An earlier draft of this ADR omitted it (reasoning that
      "resolving is the step that produces a revision"); that holds only for
      the `revision is None` case. When a `DatasetRef` carries an explicit
      pin, that pin *is* known before resolution and is provider-honored
      (the Hugging Face provider forwards it to `dataset_info(...,
      revision=...)`), so two different pins genuinely resolve to different
      immutable revisions. A revision-blind key collapsed both into one
      slot, let the second online resolve overwrite the first, and made an
      offline resolve of the first pin silently return the second's
      resolution -- a direct violation of the manifest-comparability /
      reproducibility invariant the whole library rests on. The
      2026-07-09 fix added the field before this ADR was ratified.
  This `ResolutionCache` is the "second, parallel keying scheme" ADR-0010
  anticipated but did not itself add.
- **`DatasetCatalog` gains an optional, additive `resolution_cache`
  constructor parameter**, defaulting to `None`. A `None` value (every
  caller and test fixture that predates this ADR) preserves ADR-0010's
  behavior exactly: `resolve()` under `offline=True` against a
  network-requiring provider still always raises with `retryable=False`.
  Only a caller that opts in by constructing a `ResolutionCache` and passing
  it changes anything -- `agentic_evalkit.cli.datasets.build_catalog` (the
  real CLI's catalog factory) is the one production call site that does,
  rooted at `<cache dir>/resolutions/`, physically separate from the page
  cache's own digest fan-out.
- **`resolve()` writes through on every successful call.** Whenever
  `resolve()` actually reaches `provider_impl.resolve(ref)` -- an
  `offline=False` call, or an `offline=True` call against a
  `requires_network=False` provider -- the resolved `ResolvedDataset` is
  written to `resolution_cache` (when configured) before returning. This
  covers `datasets pull` (which already calls `resolve()` online) without
  `pull` itself needing to change, and it also warms the cache for `datasets
  inspect`/`preview`/`run` invoked without `--offline`, as a bonus rather
  than a restriction.
- **`resolve()` consults the cache before raising, only for the exact case
  ADR-0010 could not serve.** Under `offline=True` against a
  network-requiring provider, `resolve()` now tries `resolution_cache.read()`
  first; a hit returns the cached `ResolvedDataset` with no provider call.
  A miss (or no `resolution_cache` configured) falls through to raising,
  exactly as ADR-0010 specified. `retryable` becomes conditional: `True`
  when a `resolution_cache` is configured (an online resolve or `datasets
  pull` for this exact request would populate it -- the genuine "warm the
  cache" case ADR-0010's own `OfflineCacheMiss` docstring describes) and
  `False` when none is configured at all (ADR-0010's original, unconditional
  meaning, preserved for every caller that has not opted in).
- **`iter_records()` gains one narrow, page-cache-backed exception, scoped
  to exactly what `EvalRunner` needs.** `EvalRunner._prepare_samples` always
  calls `iter_records` with one fixed `(offset, limit)` per run -- the same
  shape `preview()`'s existing content-addressed cache (ADR-0004) already
  keys on. When `offline=True` against a network-requiring provider, `limit`
  is not `None`, and a page `cache` is configured, `iter_records` now builds
  the identical `CacheKey` `preview()` would use and serves the cached
  page's records as an async iterator on a hit, with no provider call.
  Every other shape -- unbounded iteration (`limit=None`, which has no
  single-page key), no cache configured, or no matching entry -- still
  raises exactly as before, with `retryable` following the same
  configured-vs-not split as `resolve()`.
- **Neither change touches `preview()`, `search()`, or any frozen wire
  model.** `preview()` was already fully cache-backed per provider
  regardless of `requires_network` (ADR-0010) and is unchanged. `search()`
  has no stable key for a free-text query and stays unconditionally
  rejected offline, unchanged. `ResolvedDataset`, `DatasetRef`, and
  `CacheKey` keep their existing fields, `schema_version`, and semantics;
  `ResolutionKey` is a new model, not a modification of an existing
  contract, and follows the same `frozen=True`/`extra="forbid"`/
  `schema_version="1"` shape ADR-0002 requires of every wire model.

## Alternatives

1. **Extend `CacheKey`/`DatasetCache` itself to also address a
   pre-revision resolution, instead of adding a second cache type.**
   Rejected: `CacheKey.revision` is a required, non-optional field precisely
   because a page/full-dataset entry's identity depends on it (ADR-0004);
   making it optional to also serve as a resolution key would weaken that
   invariant for every existing page/full entry to accommodate a
   structurally different lookup (by request, not by already-known
   revision). A second, narrowly-scoped key type keeps both caches'
   invariants exactly as strong as before.
2. **Make `resolve()` always try `resolution_cache` regardless of
   `requires_network`, rather than gating on the provider declaration.**
   Rejected: `resolve()` for a `requires_network=False` provider (e.g.
   `local`) is already network-free and already succeeds under
   `offline=True` (ADR-0010) -- consulting a cache first would only add a
   filesystem read with no behavioral benefit, and would complicate the one
   code path that is already provably correct and simple.
3. **Default `retryable=True` unconditionally once `ResolutionCache`
   support landed, rather than keeping it conditional on whether one is
   configured.** Rejected: this would silently change the documented
   meaning of `OfflineCacheMiss.retryable` for every pre-existing caller
   that constructs a `DatasetCatalog` without a `resolution_cache` (every
   test fixture and any third-party caller written against ADR-0010) --
   exactly the kind of accidental behavior change ADR-0010 itself rejected
   for the analogous `requires_network` default (its Alternative 2).
   Keeping the conditional preserves ADR-0010's `retryable=False` for every
   caller that has not opted in, while making it accurately `True` for
   callers that have.
4. **Extend `iter_records` to serve *any* `(offset, limit)` by
   synthesizing it from multiple cached pages, or to fall back to a
   `record_type="full"` cache entry.** Rejected for this task: no code path
   in this repository writes a `record_type="full"` entry today, and
   stitching multiple page entries together to satisfy an arbitrary offline
   iteration request is a materially larger feature with its own
   correctness questions (partial coverage, gaps, ordering) that the
   `EvalRunner`-shaped single-page case this ADR targets does not need. A
   future ADR may extend offline iteration further; this one intentionally
   serves only the exact-page case.

## Consequences

- `agentic-evalkit datasets pull hf:<dataset>` followed by `agentic-evalkit
  run <manifest> --offline` (with a matching `selection.offset`/`limit`)
  now succeeds end to end for a Hugging-Face-backed dataset, with zero
  network calls on the offline run -- closing the gap between what `pull`'s
  own docstring promises and what `--offline` could actually serve.
- Every pre-ADR-0011 `DatasetCatalog` construction (every existing test
  fixture, and any third-party caller) is unaffected: `resolution_cache`
  defaults to `None`, and the `iter_records` exception only activates when
  both an explicit `limit` and a configured page `cache` are present and
  warmed, so the default shape of every prior test's assertions --
  including the exact `retryable` values and message substrings ADR-0010's
  own test suite pins -- is preserved byte-for-byte.
- A caller can now distinguish, via `OfflineCacheMiss.retryable`, "this
  catalog supports offline resolution/iteration and just needs one online
  warm-up call" from "this catalog was never configured to support it at
  all" -- the same discrimination ADR-0010 established for `preview`.
- The resolution cache and the page cache are independent: a corrupted or
  missing entry in one never masks or is masked by the other, and each
  raises its own typed `DatasetIntegrityError`/`OfflineCacheMiss` exactly as
  ADR-0004 established.

## Validation

- `tests/unit/datasets/test_resolution_cache.py` asserts: `ResolutionKey`
  digests are deterministic and change with any identity-bearing field
  (including `revision`, and a pinned vs. the default unpinned request);
  `ResolutionKey` is frozen and forbids unknown fields (ADR-0002); a
  write-then-read round trip returns an equal `ResolvedDataset`; a missing
  entry raises `OfflineCacheMiss(retryable=True)`; a payload or key tampered
  with in-place after a successful write raises `DatasetIntegrityError`,
  distinctly from a miss (mirroring ADR-0004's own corruption-vs-miss
  test); **two requests differing only in pinned `revision` occupy distinct
  slots so writing `revB` never overwrites `revA`'s entry, and a
  pinned-but-uncached request misses rather than aliasing the unpinned
  "latest" slot** (the 2026-07-09 correctness fix); a
  `ThreadPoolExecutor`-plus-`threading.Barrier` same-key concurrent-write
  test (mirroring ADR-0004's own, with the identical Windows
  sharing-violation retry wrapper) leaves exactly one checksum-valid entry
  readable, never a torn or integrity-failing one, and a distinct-keys
  concurrent-write test confirms unrelated keys never contend or corrupt
  each other.
- `tests/unit/datasets/test_catalog.py` asserts: an online `resolve()`
  followed by an offline `resolve()` against a `resolution_cache`-configured
  catalog returns an equal `ResolvedDataset` without a second provider call;
  an offline `resolve()` with no prior online call and a configured
  `resolution_cache` raises `OfflineCacheMiss(retryable=True)`; every
  pre-existing ADR-0010 test (no `resolution_cache` configured) keeps
  `retryable=False` unchanged; **resolving `ref@revA` then `ref@revB`
  online (a provider that echoes the requested pin into the resolved
  revision) caches both, and an offline `resolve(ref@revA)` returns
  `revA`'s resolution -- never `revB`'s -- while an offline resolve of a
  never-cached pin raises rather than returning another pin's data, and the
  unpinned (`revision is None`) path still round-trips** (the critical
  revision-collision regression); an online `resolve()` + `preview()`
  followed by an offline `iter_records()` at the identical `(offset,
  limit)` returns the same records without a provider call; an offline
  `iter_records()` with `limit=None`, or with no matching cached page, still
  raises exactly as ADR-0010 specified.
- `tests/integration/test_offline_resolution_cache.py` drives the real CLI
  (`datasets pull` then `run --offline`) against a fake network-requiring
  provider whose `resolve`/`preview`/`iter_records` methods are call-counted
  and raise if invoked a second time during the offline phase, wrapped in
  an OS-level loopback-allowlisting socket guard (mirroring
  `tests/unit/datasets/test_offline_socket_guard.py`) proving zero real
  outbound network syscalls occur during the offline `run` -- both the
  "provider was never called" and "no socket was ever opened" properties
  are asserted together, not just one.

## Supersession

Extending offline iteration to synthesize an arbitrary `(offset, limit)`
range from multiple cached pages or from a `record_type="full"` entry, or
adding cross-process/cross-machine sharing of the resolution cache, would
each be a material change to this decision's scope and must supersede this
ADR with its own validation evidence.
