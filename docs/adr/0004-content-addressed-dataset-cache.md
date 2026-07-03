# ADR-0004: Content-Addressed, Checksum-Verified Dataset Cache

## Status

Accepted

## Context

`agentic-evalkit` must let a run resolve and re-resolve the same dataset
page or full dataset repeatedly — across processes, across machines, and
offline — without re-fetching from a provider and without silently serving
stale or corrupted bytes as if they were valid. Design §6.3
(`docs/specs/2026-07-02-agentic-evalkit-design.md`) requires the cache key to
include provider, canonical ID, immutable revision, config, split, selected
files, projection, filter, offset, limit, and loader schema version; requires
full datasets and pages to be distinct cache record types; and requires a
manifest and checksum on every entry so that corruption, staleness, and
misses are explicit, typed outcomes rather than silent data loss.

Task 3 (`docs/adr/0003-provider-plugins-and-hugging-face-baseline.md`) already
froze the two error types this cache raises: `OfflineCacheMiss` for "no
exact entry exists" and `DatasetIntegrityError` for "an entry exists but its
bytes cannot be trusted." Those must stay distinguishable outcomes — a
caller in offline mode needs to know whether to report "not cached" (a
recoverable, expected state) or "cached copy is corrupt" (a state that
should never be silently treated as a cache miss and silently re-fetched,
since re-fetching a supposedly-immutable revision that produced corrupt
bytes may mask a real problem rather than fix it).

The framework runs on Windows as a first-class platform (per the Windows CI
matrix added in Task 1), and dataset caching is exactly the kind of
filesystem-heavy code that behaves differently across platforms: file
replacement semantics, locking, and concurrent-writer behavior are not
uniform between POSIX filesystems and Windows filesystems (including
network-mounted and some non-NTFS Windows volumes).

## Decision

- **Cache key.** `CacheKey` is a frozen `agentic_evalkit.models.FrozenModel`
  with fields `provider`, `dataset_id`, `revision`, `config`, `split`,
  `offset`, `limit`, plus optional `projection_digest`, `filter_digest`, and
  `data_files_digest` (each `str | None`, default `None`), and `record_type`
  (`Literal["page", "full"]`, default `"page"`) so a full-dataset entry and a
  page entry for the same dataset never collide even if all other fields
  match. `revision` must already be an immutable, resolved revision (a
  commit SHA or content digest), never a mutable ref like a branch name —
  resolving that immutability is the resolver's responsibility (design
  §5.2), not the cache's; the cache treats `revision` as opaque identity
  input.
- **Canonical digest.** `CacheKey.digest()` returns
  `"sha256:" + hexdigest` of the UTF-8 encoding of the key's canonical JSON
  form: `json.dumps(key.model_dump(mode="json"), sort_keys=True,
  separators=(",", ":"))`. Sorted keys and compact separators make the
  digest depend only on field values, never on field-declaration order,
  incidental whitespace, or the Pydantic dump implementation's default
  formatting. Any identity-bearing field — including the optional digest
  fields and `record_type` — changing changes the digest.
- **Entry layout.** Each entry lives at
  `root / digest_hex[:2] / digest_hex / {payload.bin, manifest.json}`, where
  `digest_hex` is `CacheKey.digest()` with the `"sha256:"` prefix stripped
  for path segments. The two-character fan-out directory keeps any single
  directory from accumulating one entry per cached page across a large run.
  Standalone use defaults `root` to the platform user-cache directory (per
  design §6.3); the `DatasetCache` class itself is agnostic to where `root`
  points and takes it as a constructor argument so callers (and tests) can
  point it anywhere.
- **Manifest.** `manifest.json` records the payload's SHA-256 checksum
  (`"sha256:" + hexdigest`, same format as the key digest), the payload byte
  count, an ISO-8601 UTC creation timestamp, the full key as JSON (via
  `model_dump(mode="json")`), and the digest string itself. The manifest is
  the sole source of truth `read()` checks against; nothing about a valid
  read depends on trusting directory names or path structure beyond using
  them to locate the entry.
- **Atomic-by-convention publication, not atomic-by-guarantee.**
  `write()` writes the payload and the manifest to temporary sibling files
  in the entry directory, flushes each and calls `os.fsync()` on its file
  descriptor, then publishes each with `Path.replace()` — payload first,
  then manifest, so a reader can never observe a manifest that describes a
  payload not yet on disk. **`Path.replace()`'s atomicity is a POSIX
  filesystem guarantee. On Windows, whether a given `Path.replace()` call is
  observed atomically depends on the local filesystem and volume
  configuration, and is not an unconditional cross-filesystem guarantee.**
  This ADR does not claim otherwise. Correctness instead comes from
  `read()` independently verifying key identity, byte count, and checksum
  on every read (see Validation) — a reader that observes a
  non-atomically-applied or partially-applied replace on some Windows
  configuration gets a typed `DatasetIntegrityError` or `OfflineCacheMiss`,
  never a torn or mismatched payload treated as valid. The Windows CI matrix
  from Task 1 runs this module's test suite (see Validation), including
  repeated concurrency tests, specifically because this guarantee cannot be
  established from a Linux-only test run.
- **Lock-scoped concurrent writes.** A module-level registry
  (`dict[str, threading.Lock]`, itself guarded by one registry lock so two
  threads cannot race to create two different lock objects for the same new
  digest) hands out one lock per cache-key digest. `write()` holds that
  digest's lock for the temp-write-then-replace sequence, so concurrent
  writers to the *same* key are serialized and the entry directory always
  reflects either the fully-prior entry or the fully-new one. Writers to
  *different* keys never contend for the same lock and proceed fully in
  parallel; the lock scope is per-process (this ADR does not attempt
  cross-process or cross-machine locking — a second process racing the same
  key relies on the same replace-then-verify correctness argument above,
  not on the in-process lock).
- **Read outcomes.** `read()` distinguishes exactly two failure modes, both
  already defined in `agentic_evalkit.errors` (Task 3):
  - `OfflineCacheMiss` — no manifest or payload file exists for this exact
    digest. There is no partial-match or best-effort fallback; offline mode
    only ever uses exact resolved entries (design §6.3).
  - `DatasetIntegrityError` — an entry exists but fails verification: the
    manifest is not valid JSON, the manifest's recorded key does not match
    the requested key, the on-disk payload's byte count does not match the
    manifest, or the recomputed checksum does not match the manifest.
  A successful `read()` guarantees the returned bytes are exactly what
  `write()` was given for that exact key.
- **`payload_path()` / `manifest_path()`.** Both are exposed as public
  methods computing the path for a key without requiring the entry to
  exist, so callers (and this ADR's own corruption tests) can locate and
  directly mutate a payload or manifest file to simulate corruption.

## Alternatives

1. **Rely solely on `Path.replace()` atomicity as the correctness
   argument, with no manifest checksum.** Rejected: this is exactly the
   unconditional cross-filesystem atomicity claim design §6.3 and this task
   explicitly reject. Windows filesystem behavior is not uniform enough to
   support it, and even on filesystems where replace is atomic, a checksum
   is still needed to catch bit rot, partial writes from a crashed process
   that never reached the replace step, or an entry deliberately corrupted
   on disk (which the plan's own corruption test exercises).
2. **Hash the payload itself as the cache key (pure content addressing,
   no structured `CacheKey`).** Rejected: the cache must be look-up-able
   *before* the payload is known — a caller asks "do I already have page
   (provider, dataset, revision, config, split, offset, limit) cached?"
   without fetching first. A structured key computed from request
   parameters, not response bytes, is required to answer that question.
3. **A single global write lock instead of a per-key lock registry.**
   Rejected: it would serialize writes to unrelated cache entries (e.g. two
   different datasets, or two different pages of the same dataset) with no
   correctness benefit, hurting throughput on exactly the concurrent-preview
   / concurrent-page-fetch workloads the cache exists to speed up.
4. **Advisory cross-process file locks (e.g. a `.lock` file per entry)
   instead of in-process `threading.Lock`.** Rejected for this task: cross
   -process locking on Windows requires either a platform-specific
   file-locking library or careful use of exclusive-create semantics, adding
   complexity beyond what design §6.3's stated scope (in-process concurrent
   writers, verified by `ThreadPoolExecutor` tests) requires. The
   checksum-verification argument above already makes a second process
   racing the same key safe (never returns torn/mismatched bytes as valid),
   even without a shared lock; a future ADR can add cross-process locking
   purely as a performance optimization if profiling shows redundant
   concurrent fetches across processes are a real cost.
5. **Treat any corrupt entry as a cache miss (silently re-fetch).**
   Rejected: design §6.3 requires corruption to be an explicit, distinct
   outcome from a miss. Silently collapsing corruption into "not cached"
   would hide a filesystem or hardware problem behind an innocuous-looking
   re-fetch, and would make the plan's own
   `test_corruption_and_offline_miss_are_distinct` test impossible to
   satisfy.

## Consequences

- Every cache entry is independently self-verifying: a reader never has to
  trust that a write completed correctly, only that `read()`'s checksum
  check ran.
- `CacheKey.digest()` is stable across processes and across Python
  versions' dict-ordering behavior (canonical JSON sorts keys explicitly),
  so a page cached by one process is addressable by another without
  re-deriving any provider state.
- Page and full-dataset entries for the same logical dataset never alias
  each other, because `record_type` participates in the digest.
- Concurrent same-key writers (e.g. two coroutines or threads racing to
  populate the same page) never corrupt the entry; the loser of the race
  simply republishes over the winner's (or vice versa) and every reader
  during and after the race observes one complete, checksum-valid entry.
- Because the correctness argument does not depend on an unconditional
  `Path.replace()` atomicity guarantee, this module's test suite — not just
  a Linux CI run — must pass on the Windows CI matrix from Task 1 before
  Task 4 is considered complete; a Linux-only green run is not sufficient
  evidence.
- Callers that need offline-only behavior can rely on `OfflineCacheMiss`
  meaning exactly "not present," never "present but untrustworthy," and can
  choose different recovery behavior (e.g. surface "run online once first")
  for `DatasetIntegrityError`.

## Validation

- `tests/unit/datasets/test_cache.py::test_cache_key_changes_for_revision_config_split_and_page`
  and three additional digest-identity tests
  (`test_cache_key_digest_changes_for_limit_and_provider`,
  `test_cache_key_digest_changes_for_optional_digest_fields_and_record_type`,
  `test_cache_key_defaults_for_optional_digest_fields_and_record_type`)
  assert every identity-bearing field, including the optional digest fields
  and `record_type`, changes `digest()`, and that unset optional fields
  default to `None` / `"page"`.
- `test_cache_key_digest_is_deterministic_and_pure` asserts `digest()` is a
  pure function of field values (same key, same digest, every call) and has
  the documented `"sha256:" + 64 hex chars` shape.
- `test_cache_key_is_frozen_and_forbids_unknown_fields` asserts `CacheKey`
  inherits `FrozenModel`'s immutability and closed-field-set guarantees
  (ADR-0002).
- `test_corruption_and_offline_miss_are_distinct` (the plan's verbatim Step
  2 snippet) asserts reading a never-written key raises `OfflineCacheMiss`
  and that overwriting the payload file in place after a successful write
  raises `DatasetIntegrityError`, proving the two failure modes are
  distinguishable outcomes, not the same exception reused.
- `test_read_with_byte_count_mismatch_raises_integrity_error`,
  `test_read_with_manifest_key_mismatch_raises_integrity_error`, and
  `test_manifest_records_checksum_byte_count_created_at_and_key` assert the
  manifest independently records and enforces checksum, byte count, and key
  identity, per design §6.3's "every entry has a manifest and checksum."
- `test_two_page_keys_with_different_offsets_are_both_addressable` and
  `test_overwriting_same_key_replaces_payload_atomically` cover the plan's
  Step 5 exact-page and replace-publication requirements.
- `test_concurrent_same_key_writes_leave_exactly_one_valid_entry` and
  `test_concurrent_writes_to_distinct_offsets_are_all_readable` use a
  `ThreadPoolExecutor` with a `threading.Barrier` to force genuinely
  concurrent same-key writers and assert the entry is always readable and
  checksum-valid afterward (never a `DatasetIntegrityError`, never a
  `OfflineCacheMiss`), and that distinct offsets never contend. Per the
  plan's Step 5, these are run with `uv run pytest
  tests/unit/datasets/test_cache.py -x --count=5` (via `pytest-repeat`) to
  build repeated-run confidence against flaky concurrency behavior; this
  suite must also pass on the Windows CI matrix from Task 1, not only on
  Linux, before this ADR's atomicity argument is considered validated.

## Supersession

Cross-process locking (advisory `.lock` files or platform-specific file
locks), a background garbage-collection/eviction policy for the cache
directory, or moving the correctness argument to depend on filesystem-level
atomicity guarantees instead of checksum verification would each be a
material change to this decision and must supersede this ADR with its own
validation evidence, including a Windows-specific test run.
