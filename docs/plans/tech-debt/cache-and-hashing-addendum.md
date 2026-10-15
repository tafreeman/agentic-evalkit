# Architecture: shared hashing and cache storage

Status: proposed architecture. The changes below are not yet implemented.

Related decision: ADR-0004, content-addressed dataset cache.
Delivery steps: [G5A and G5B](../2026-09-10-tech-debt-workflow.md).

## Problem

EvalKit uses hashes to identify datasets, cached results and execution targets.
Several modules repeat the same hash operation. Two cache implementations also
repeat locking and temporary-file writing.

The caches store different records. Combining their formats would risk making
existing entries unreadable. Changing serialization would also change their
identifiers, even if the underlying data were the same.

## Decision

Share byte hashing, identical dataset-row serialization, per-key locking and
temporary-file writing. Keep each cache's format, validation and errors in its
current class. Existing identifiers and stored entries must remain unchanged.

A successful file replacement does not prove an entry is valid. Continue
checking its checksum every time it is read.

## Hashing interfaces

Add internal `src/agentic_evalkit/_hashing.py`:

```python
def sha256_hex(payload: bytes) -> str: ...
def sha256_digest(payload: bytes) -> str: ...
```

`sha256_digest` adds the existing `sha256:` prefix. These functions accept bytes;
they do not choose how objects become JSON.

Add `datasets/_serialization.py`:

```python
def canonical_row_bytes(row: Mapping[str, JsonValue]) -> bytes: ...
def canonical_row_digest(row: Mapping[str, JsonValue]) -> str: ...
```

For dataset rows, retain sorted keys, compact separators, `ensure_ascii=True`
and UTF-8 encoding. Move the identical local and Hugging Face implementations
here. Other modules share serialization only if their current rules match.

Target identifiers keep their existing input fields and 16-character hashes.
Do not replace them with full hashes or add a prefix.

## Storage interfaces

Add internal `datasets/_cache_storage.py`:

```python
class KeyedLockPool:
    def __init__(self) -> None: ...
    def lock_for(self, digest: str) -> threading.Lock: ...

@contextmanager
def staged_bytes(destination: Path, payload: bytes) -> Iterator[Path]: ...
```

Each cache module owns one lock pool, shared by its instances. The two cache
types keep separate pools. Locks coordinate threads in one process; this does
not provide locking between processes. Do not remove a lock while a thread
could still be using it.

`staged_bytes` creates a unique temporary file beside the destination, writes
and flushes its bytes, calls fsync, and returns the temporary path. The caller
creates directories and replaces the destination file. On exit, the helper
removes an unused temporary file without hiding the original error.

| Cache | Write sequence |
| --- | --- |
| Dataset | Prepare payload and manifest completely; replace payload, then manifest |
| Resolution | Prepare the record; replace the record |

Keep these operations inside the existing per-key lock. Preserve file names,
directory layout, JSON formatting, timestamps and checksum inputs. Public
cache constructors and read/write methods do not change.

Keep `_cache_io.read_entry_bytes` unchanged. It retries `PermissionError` and
`FileNotFoundError`; unrelated operating-system errors must still propagate.
Missing entries and corrupt entries remain different errors.

## Migration

1. Record each caller's current serialization and hash rules. Save expected
   bytes, hashes and paths before replacing code.
2. Extract byte hashing and identical row serialization. Migrate one group
   of callers at a time.
3. Add shared storage helpers. Migrate the dataset cache, then the resolution
   cache in a separate change.
4. Record the ownership change in ADR-0004 or a new accepted ADR after review.
   Follow the repository's ADR format and registration checks.

No cache-format migration or invalidation is part of this change.

## Verification

Use saved expected values for Unicode, nested data, reordered keys, empty data
and supported numeric values. Comparing only two callers of the new helper
could allow the same bug to pass in both.

Read entries created by the old implementation. Inject failures during file
creation, writing, flushing, fsync and each replacement. Verify temporary-file
cleanup and rejection of mismatched payload/manifest pairs.

Test concurrent writers using different instances: the same key must share a
lock, and different keys must be able to progress independently. Coordinate
threads explicitly rather than relying only on delays.

Run Windows and Linux checks. Keep separate cache directories for parallel
workers. A passing thread test does not establish safety between processes.
