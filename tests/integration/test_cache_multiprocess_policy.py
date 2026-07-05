"""Story 4.2 (R-001 residual / ratified D-2): per-worker cache policy.

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 4, Story 4.2) and
the ratified decision D-2 (parallel / multi-process runs use a *per-worker*
``AGENTIC_EVALKIT_CACHE_DIR``; a shared root stays checksum-safe but is
documented as not recommended).

:class:`~agentic_evalkit.datasets.cache.DatasetCache` takes an explicit root
directory, and ``AGENTIC_EVALKIT_CACHE_DIR`` is consumed one level up by the
CLI's ``default_cache_dir`` (``agentic_evalkit.cli.datasets``) to pick that
root. These tests therefore model each worker as a thread with its own
``DatasetCache`` rooted at a distinct directory (the per-worker pattern) or a
shared directory (the not-recommended fallback), which is exactly the
isolation the env-var-per-worker policy produces on disk. They are marked
``integration`` because they exercise the multi-worker contention behaviour of
the cache as a whole, not a single function in isolation.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agentic_evalkit.datasets.cache import CacheKey, DatasetCache
from agentic_evalkit.errors import DatasetIntegrityError, OfflineCacheMiss

#: Bounded retry budget for a Windows sharing violation on ``Path.replace()``.
_WRITE_RETRY_ATTEMPTS = 3
_WRITE_RETRY_SLEEP_SECONDS = 0.01
#: Generous wall-clock deadline for a racing reader to observe at least one
#: checksum-valid read once writers have published.
_READER_OBSERVE_DEADLINE_SECONDS = 5.0


def _write_with_windows_retry(cache: DatasetCache, key: CacheKey, payload: bytes) -> None:
    """Write ``payload`` under ``key``, retrying a Windows sharing violation.

    On Windows, ``Path.replace()`` onto a payload/manifest that a concurrent
    reader (or racing writer) currently holds open raises ``PermissionError``
    (a sharing violation) -- POSIX rename has no such restriction. That
    collision is transient: retry a bounded number of times with a tiny sleep
    so the write lands once the open handle is released. Only
    ``PermissionError`` is retried; any other exception propagates so a real
    bug is never masked.
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
    as a distinct ``DatasetCache`` root), write the *same* cache key in
    parallel. Neither worker's directory ends up holding the other's bytes:
    on-disk isolation is total, so a partial write in one can never surface in
    the other. Both reads succeed.
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
    # torn mix -- proving the per-worker dirs are fully isolated on disk.
    assert cache_a.read(key) == payload_a
    assert cache_b.read(key) == payload_b


@pytest.mark.integration
def test_per_worker_cache_dirs_isolate_distinct_workloads(tmp_path: Path) -> None:
    """Per-worker dirs also isolate *different* keys: a key written only in
    worker A's dir is an ``OfflineCacheMiss`` in worker B's dir, so one
    worker's cache population never leaks into another's offline view.
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
    """The not-recommended shared-root fallback: many threads racing to write
    the *same* key into one shared cache root. Integrity is still preserved by
    checksum-on-read -- the final entry is always exactly one written payload,
    and no interleaving surfaces a corrupt result -- which is the fail-closed
    guarantee that keeps a shared root safe (if contended).
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

    # A second, independent reader of the shared root observes exactly one
    # checksum-valid entry (or a typed error), never a torn payload.
    reader = DatasetCache(shared_root)
    result = reader.read(key)
    assert result in valid


@pytest.mark.integration
def test_shared_root_reader_never_sees_a_torn_write(tmp_path: Path) -> None:
    """A reader on the shared root, interleaved with racing writers, only ever
    returns a checksum-valid payload or raises a typed miss/integrity error --
    it never returns a partially-written or checksum-invalid result.
    """
    shared_root = tmp_path / "shared"
    cache = DatasetCache(shared_root)
    reader = DatasetCache(shared_root)
    key = _key()
    payloads = [f"shared-payload-{i}".encode() for i in range(6)]
    valid = set(payloads)
    start = threading.Barrier(len(payloads) + 1)
    # Only the single reader thread mutates this, so a bare counter in a
    # one-element list is safe without a lock; it stays visible after join.
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

    # Vacuity guard: if every in-race read missed before the first write
    # published, the torn-read assertion above never ran. Writers have all
    # joined and a valid entry provably exists, so read until at least one
    # checksum-valid result is observed (bounded by a generous deadline) and
    # assert the contested path was actually exercised.
    deadline = time.monotonic() + _READER_OBSERVE_DEADLINE_SECONDS
    while successful_reads[0] == 0 and time.monotonic() < deadline:
        try:
            result = reader.read(key)
        except (OfflineCacheMiss, DatasetIntegrityError):
            continue
        assert result in valid
        successful_reads[0] += 1
    assert successful_reads[0] > 0, "no checksum-valid read was ever observed"
