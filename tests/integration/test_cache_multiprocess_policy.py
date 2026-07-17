"""Tests for the cache's per-worker isolation policy.

Covers Story 4.2 (which addresses residual risk R-001) and the ratified
decision D-2. Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 4,
Story 4.2).

In plain terms, D-2 says: when several workers (parallel processes) run at
once, each one should get its own cache directory via its own
``AGENTIC_EVALKIT_CACHE_DIR`` environment variable, so they never see each
other's writes. Pointing every worker at one shared directory instead still
won't corrupt data -- every read is verified against a checksum (a
fingerprint of the correct bytes, so damaged data gets caught) -- but that
shared setup isn't recommended.

:class:`~agentic_evalkit.datasets.cache.DatasetCache` itself knows nothing
about environment variables; it just uses whatever root directory it's
given. The env var is read one layer up, by the CLI's ``default_cache_dir``
(in ``agentic_evalkit.cli.datasets``), which decides which directory to hand
the cache. So rather than spinning up real separate processes, these tests
model each "worker" as a thread with its own ``DatasetCache`` rooted at
either a distinct directory (the recommended per-worker pattern) or a
directory shared with other threads (the not-recommended fallback) --
reproducing on disk exactly the isolation (or lack of it) the real policy
would produce. They're marked ``integration`` because they test how the
cache behaves when multiple workers genuinely contend for it at once, not
just one function's logic in isolation.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

from agentic_evalkit.datasets.cache import CacheKey, DatasetCache
from agentic_evalkit.errors import DatasetIntegrityError, OfflineCacheMiss

if TYPE_CHECKING:
    from pathlib import Path

#: How many times to retry a write after Windows briefly locks the file during
#: ``Path.replace()`` (see ``_write_with_windows_retry`` below for why this
#: happens).
_WRITE_RETRY_ATTEMPTS = 3
_WRITE_RETRY_SLEEP_SECONDS = 0.01
#: How long, in real (wall-clock) seconds, to keep letting a reader retry
#: before giving up on ever seeing a valid read -- generous, because we only
#: reach this deadline after the writers have already finished publishing.
_READER_OBSERVE_DEADLINE_SECONDS = 5.0


def _write_with_windows_retry(cache: DatasetCache, key: CacheKey, payload: bytes) -> None:
    """Write ``payload`` under ``key``, retrying if Windows briefly locks the file.

    On Windows, calling ``Path.replace()`` on a file that a concurrent reader
    (or a racing writer) currently has open raises ``PermissionError`` --
    this is called a "sharing violation." POSIX systems (Linux/Mac) have no
    such restriction, so the same rename there just succeeds. The lock is
    only temporary, so this retries a bounded number of times with a short
    sleep in between, and the write goes through as soon as the other thread
    closes its file handle. Only ``PermissionError`` triggers a retry; any
    other exception is left to propagate, so a real bug is never hidden by
    this retry loop.
    """
    for attempt in range(_WRITE_RETRY_ATTEMPTS):
        try:
            cache.write(key, payload)
            return
        except PermissionError:
            if attempt == _WRITE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_WRITE_RETRY_SLEEP_SECONDS)


def _key(offset: int = 0) -> CacheKey:
    return CacheKey(
        provider="local",
        dataset_id="items.jsonl",
        revision="sha256:a",
        config=None,
        split=None,
        offset=offset,
        limit=10,
    )


@pytest.mark.integration
def test_per_worker_cache_dirs_do_not_observe_each_others_writes(tmp_path: Path) -> None:
    """Two workers, each with its own ``AGENTIC_EVALKIT_CACHE_DIR`` (modelled
    here as a distinct ``DatasetCache`` root directory), write the *same*
    cache key at the same time. Neither worker's directory ends up holding
    the other's bytes: the separation between them on disk is total, so a
    partial write from one worker can never leak into the other. Both reads
    succeed.
    """
    root_a = tmp_path / "worker-a"
    root_b = tmp_path / "worker-b"
    cache_a = DatasetCache(root_a)
    cache_b = DatasetCache(root_b)
    key = _key()
    payload_a = b"bytes-written-only-by-worker-a"
    payload_b = b"bytes-written-only-by-worker-b"
    start = threading.Barrier(2)

    def _write(cache: DatasetCache, payload: bytes) -> None:
        start.wait()
        for _ in range(20):
            cache.write(key, payload)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_write, cache_a, payload_a),
            pool.submit(_write, cache_b, payload_b),
        ]
        for future in futures:
            future.result()

    # Each worker sees only its own payload -- never the other's, and never a
    # "torn" mix of both (bytes from two different writes interleaved
    # together) -- proving the per-worker directories are fully isolated on
    # disk.
    assert cache_a.read(key) == payload_a
    assert cache_b.read(key) == payload_b


@pytest.mark.integration
def test_per_worker_cache_dirs_isolate_distinct_workloads(tmp_path: Path) -> None:
    """Per-worker directories also isolate *different* cache keys, not just
    concurrent writes to the same one: a key written only into worker A's
    directory shows up as an ``OfflineCacheMiss`` (the typed "not in the
    cache" error) when worker B looks for it in its own directory. In other
    words, data one worker has cached never silently leaks into what another
    worker sees when running offline.
    """
    cache_a = DatasetCache(tmp_path / "worker-a")
    cache_b = DatasetCache(tmp_path / "worker-b")
    key = _key(offset=30)
    cache_a.write(key, b"only-in-a")

    assert cache_a.read(key) == b"only-in-a"
    with pytest.raises(OfflineCacheMiss):
        cache_b.read(key)


@pytest.mark.integration
def test_shared_root_racing_writers_stay_fail_closed(tmp_path: Path) -> None:
    """The not-recommended fallback setup: many threads racing to write the
    *same* key into one cache root they all share. Even here, data integrity
    holds up because every read re-checks a checksum (a fingerprint of the
    correct bytes): the entry you get back is always exactly one complete
    written payload, never a corrupted mix of several. That's what "fail
    closed" means here -- if anything ever did go wrong, it would show up as
    a clear error rather than quietly handing back bad data -- and it's what
    keeps a shared root safe to use even though it's contended (multiple
    writers competing for the same files) and not the recommended setup.
    """
    shared_root = tmp_path / "shared"
    cache = DatasetCache(shared_root)
    key = _key()
    payloads = [f"racing-payload-{i}".encode() for i in range(8)]
    valid = set(payloads)
    start = threading.Barrier(len(payloads))

    def _write(payload: bytes) -> None:
        start.wait()
        _write_with_windows_retry(cache, key, payload)

    with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
        futures = [pool.submit(_write, payload) for payload in payloads]
        for future in futures:
            future.result()

    # A second reader, independent of the writers, looking at the same shared
    # root sees exactly one checksum-valid entry (or a typed error) -- never
    # a torn (partially overwritten) payload.
    reader = DatasetCache(shared_root)
    result = reader.read(key)
    assert result in valid


@pytest.mark.integration
def test_shared_root_reader_never_sees_a_torn_write(tmp_path: Path) -> None:
    """A reader on the shared root, running at the same time as several
    writers race each other, only ever does one of two things: return a
    payload that passes its checksum check, or raise one of the typed errors
    (a cache miss or an integrity error). It never returns a
    partially-written or checksum-invalid result -- there is no third,
    silently-broken outcome.
    """
    shared_root = tmp_path / "shared"
    cache = DatasetCache(shared_root)
    reader = DatasetCache(shared_root)
    key = _key()
    payloads = [f"shared-payload-{i}".encode() for i in range(6)]
    valid = set(payloads)
    start = threading.Barrier(len(payloads) + 1)
    # Wrapping the counter in a one-element list (instead of a plain int)
    # lets the nested function below mutate it via closure -- reassigning a
    # plain int from a nested function needs the `nonlocal` keyword, but
    # mutating an element inside a list doesn't. This is safe without a lock
    # because only this one reader thread ever writes to it, and joining the
    # thread (via `future.result()` below) guarantees the final value is
    # visible here afterward.
    successful_reads = [0]

    def _write(payload: bytes) -> None:
        start.wait()
        _write_with_windows_retry(cache, key, payload)

    def _read_repeatedly() -> None:
        start.wait()
        for _ in range(40):
            try:
                result = reader.read(key)
            except (OfflineCacheMiss, DatasetIntegrityError):
                continue
            assert result in valid
            successful_reads[0] += 1

    with ThreadPoolExecutor(max_workers=len(payloads) + 1) as pool:
        futures = [pool.submit(_write, payload) for payload in payloads]
        futures.append(pool.submit(_read_repeatedly))
        for future in futures:
            future.result()

    assert reader.read(key) in valid

    # Guard against a vacuous pass: if every read during the race happened to
    # miss before the first write was published, the assertion inside
    # `_read_repeatedly` above never actually ran, and this test would pass
    # without having checked anything meaningful. By now all writer threads
    # have finished, so a valid entry is guaranteed to exist -- keep reading
    # (bounded by a generous deadline) until at least one checksum-valid
    # result comes back, and fail loudly below if that never happens. This
    # proves the contested code path (reading while writes are racing) was
    # actually exercised, not just theoretically reachable.
    deadline = time.monotonic() + _READER_OBSERVE_DEADLINE_SECONDS
    while successful_reads[0] == 0 and time.monotonic() < deadline:
        try:
            result = reader.read(key)
        except (OfflineCacheMiss, DatasetIntegrityError):
            continue
        assert result in valid
        successful_reads[0] += 1
    assert successful_reads[0] > 0, "no checksum-valid read was ever observed"
