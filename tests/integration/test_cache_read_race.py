"""A reader racing a writer must never escape the caches' typed error contract.

Both on-disk caches check that an entry exists and then open it. Those are two
separate operations, and a writer republishing the same entry with
``Path.replace()`` can land in the gap. On Windows that makes the open fail
with ``PermissionError`` (a sharing violation) even though the file is present
and healthy -- and before the fix these tests cover, that error travelled
straight out of ``DatasetCache.read()``: it is neither
:class:`~agentic_evalkit.errors.OfflineCacheMiss` nor
:class:`~agentic_evalkit.errors.DatasetIntegrityError`, the catalog catches
only the former, and so it reached the CLI as an unhandled traceback.

The tests here come in two kinds, and both are needed:

*Deterministic* tests inject the failure directly by replacing
``Path.read_bytes``, so they prove the exact contract on every platform and
every run -- that a transient failure is retried and then succeeds, that a
*persistent* one is still reported as a typed error, and, just as importantly,
that error types which are **not** the race (a structurally wrong cache
layout, a failing disk) are still allowed through untouched. That last one is
the guard against over-correcting: swallowing every ``OSError`` would turn a
broken machine into a silent "cache miss", which is a worse bug than the crash
being fixed here.

*Concurrency* tests (the ones named ``racing_concurrent``) run real threads
against a real shared cache directory with no injection at all, so on Windows
they exercise the genuine operating-system race. On POSIX the same test is
mostly a no-op, because ``rename()`` there does not disturb an open reader --
which is exactly why the deterministic tests above carry the real proof, and
these carry the end-to-end confirmation.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from agentic_evalkit.datasets._cache_io import READ_ATTEMPTS
from agentic_evalkit.datasets.cache import CacheKey, DatasetCache
from agentic_evalkit.datasets.resolution_cache import ResolutionCache, ResolutionKey
from agentic_evalkit.errors import DatasetIntegrityError, OfflineCacheMiss
from agentic_evalkit.models import ResolvedDataset

if TYPE_CHECKING:
    from collections.abc import Callable


def _sharing_violation() -> PermissionError:
    """A fresh instance of what Windows raises when a reader opens a file a
    writer is mid-``replace`` on. ``13``/``EACCES`` is what Python surfaces
    for both ``WinError 5`` (access denied) and ``WinError 32`` (sharing
    violation). Built per call so no traceback state is shared between tests.
    """
    return PermissionError(13, "Access is denied")


#: How many times a *writer* retries its own ``Path.replace()``. On Windows a
#: reader holding the destination open blocks the rename in the other
#: direction too. That is a separate concern from the read path under test
#: here, so it is absorbed generously: a writer flaking out would redline this
#: test for a reason that has nothing to do with what it is asserting.
_WRITE_RETRY_ATTEMPTS = 25
_WRITE_RETRY_SLEEP_SECONDS = 0.002

_PAYLOADS = tuple(f"racing-payload-{index}".encode() for index in range(4))
_READS_PER_READER = 150
_WRITES_PER_WRITER = 40


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


def _resolution_key() -> ResolutionKey:
    return ResolutionKey(provider="local", dataset_id="items.jsonl", config=None, split=None)


def _resolved(revision: str) -> ResolvedDataset:
    return ResolvedDataset(
        dataset_id="items.jsonl",
        revision=revision,
        config=None,
        split=None,
        retrieved_at=datetime.now(UTC),
    )


def _write_with_windows_retry(write: Callable[[], None]) -> None:
    """Run ``write``, retrying only the Windows sharing violation.

    Any other exception propagates immediately, so a real bug in the write
    path is never hidden by this loop.
    """
    for attempt in range(_WRITE_RETRY_ATTEMPTS):
        try:
            write()
            return
        except PermissionError:
            if attempt == _WRITE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_WRITE_RETRY_SLEEP_SECONDS)


def _install_failing_read(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: Path,
    error: OSError,
    failures: int,
) -> dict[str, int]:
    """Make ``target.read_bytes()`` raise ``error`` its first ``failures`` times.

    Reads of every other path are delegated to the real implementation, so
    patching ``Path.read_bytes`` for the whole process stays safe. Returns a
    counter dict so a test can assert the injection was actually reached --
    an injection test that silently never fired would pass vacuously.
    """
    real_read_bytes = Path.read_bytes
    state = {"calls": 0, "remaining": failures}

    def _read_bytes(self: Path) -> bytes:
        if self == target:
            state["calls"] += 1
            if state["remaining"] > 0:
                state["remaining"] -= 1
                raise error
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _read_bytes)
    return state


# --- deterministic injection: the contract, proven on every platform --------


@pytest.mark.parametrize("entry", ["payload", "manifest"])
def test_transient_sharing_violation_is_retried_and_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """A ``PermissionError`` that clears -- the real shape of the race, since a
    rename holds its destination for well under a millisecond -- is retried,
    and the read then returns the correct bytes. No error reaches the caller.
    """
    cache = DatasetCache(tmp_path)
    key = _key()
    payload = b"payload-that-survives-a-racing-writer"
    cache.write(key, payload)

    target = cache.payload_path(key) if entry == "payload" else cache.manifest_path(key)
    state = _install_failing_read(
        monkeypatch, target=target, error=_sharing_violation(), failures=2
    )

    assert cache.read(key) == payload
    assert state["remaining"] == 0, "the injected failure never fired"
    assert state["calls"] == 3, "expected two failures followed by one success"


def test_persistent_sharing_violation_becomes_an_integrity_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``PermissionError`` that outlives every retry is *not* the race, so it
    is reported as ``DatasetIntegrityError`` -- a loud, typed failure the CLI
    already maps to an exit code.

    Deliberately not ``OfflineCacheMiss``: the entry demonstrably exists, and
    calling it a miss would send the caller off to silently refetch while a
    genuine permissions problem on the cache directory went unnoticed.
    """
    cache = DatasetCache(tmp_path)
    key = _key()
    cache.write(key, b"present-but-unreadable")
    _install_failing_read(
        monkeypatch,
        target=cache.payload_path(key),
        error=_sharing_violation(),
        failures=READ_ATTEMPTS,
    )

    with pytest.raises(DatasetIntegrityError) as caught:
        cache.read(key)

    assert "could not be read" in caught.value.message
    assert isinstance(caught.value.__cause__, PermissionError)


def test_persistent_missing_file_becomes_a_cache_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file that passes the existence check and is then gone for good really
    is absent, so ``OfflineCacheMiss`` is the honest answer -- and
    ``retryable=True`` correctly tells the caller that fetching it once would
    fix things.
    """
    cache = DatasetCache(tmp_path)
    key = _key()
    cache.write(key, b"about-to-vanish")
    _install_failing_read(
        monkeypatch,
        target=cache.payload_path(key),
        error=FileNotFoundError(2, "No such file or directory"),
        failures=READ_ATTEMPTS,
    )

    with pytest.raises(OfflineCacheMiss) as caught:
        cache.read(key)

    assert caught.value.retryable is True


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(IsADirectoryError(21, "Is a directory"), id="broken-cache-layout"),
        pytest.param(OSError(5, "Input/output error"), id="failing-disk"),
        pytest.param(OSError(28, "No space left on device"), id="full-volume"),
    ],
)
def test_errors_that_are_not_the_race_still_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    """The narrowing guard, and the reason this fix does not just catch
    ``OSError``.

    ``PermissionError`` and ``FileNotFoundError`` are both subclasses of
    ``OSError``, so widening the handler would be a one-word change -- and it
    would quietly reclassify a broken cache layout, a failing disk, and a full
    volume as "not cached", sending the caller off to refetch instead of
    telling anyone the machine is unhealthy. These must still propagate.
    """
    cache = DatasetCache(tmp_path)
    key = _key()
    cache.write(key, b"irrelevant")
    _install_failing_read(monkeypatch, target=cache.payload_path(key), error=error, failures=1)

    with pytest.raises(type(error)):
        cache.read(key)


def test_resolution_cache_transient_sharing_violation_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EVK-6 T2: the resolution cache has the identical exists-then-read gap,
    and gets the identical treatment.
    """
    cache = ResolutionCache(tmp_path)
    key = _resolution_key()
    cache.write(key, _resolved("sha256:deadbeef"))

    target = next(tmp_path.rglob("resolved.json"))
    state = _install_failing_read(
        monkeypatch, target=target, error=_sharing_violation(), failures=2
    )

    assert cache.read(key).revision == "sha256:deadbeef"
    assert state["remaining"] == 0, "the injected failure never fired"


def test_resolution_cache_persistent_violation_becomes_an_integrity_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the resolution cache's persistent case is likewise a typed, loud
    failure rather than a raw ``PermissionError`` or a fabricated miss.
    """
    cache = ResolutionCache(tmp_path)
    key = _resolution_key()
    cache.write(key, _resolved("sha256:deadbeef"))
    target = next(tmp_path.rglob("resolved.json"))
    _install_failing_read(
        monkeypatch, target=target, error=_sharing_violation(), failures=READ_ATTEMPTS
    )

    with pytest.raises(DatasetIntegrityError):
        cache.read(key)


# --- real concurrency: the end-to-end confirmation --------------------------


@pytest.mark.integration
def test_racing_concurrent_readers_never_escape_the_typed_error_contract(
    tmp_path: Path,
) -> None:
    """Several readers loop over a cache entry while several writers republish
    it, all against one shared root, with no injection anywhere.

    Every read must end in exactly one of three ways: correct bytes, an
    ``OfflineCacheMiss``, or a ``DatasetIntegrityError``. Anything else --
    notably the raw ``PermissionError`` this fix exists to eliminate -- is
    captured and fails the test with the offending exception attached.
    """
    shared_root = tmp_path / "shared"
    key = _key()
    valid = set(_PAYLOADS)
    escaped: list[BaseException] = []
    successful_reads = [0]
    start = threading.Barrier(len(_PAYLOADS) + 3)

    def _write(payload: bytes) -> None:
        cache = DatasetCache(shared_root)
        start.wait()
        for _ in range(_WRITES_PER_WRITER):
            _write_with_windows_retry(lambda: cache.write(key, payload))

    def _read() -> None:
        reader = DatasetCache(shared_root)
        start.wait()
        for _ in range(_READS_PER_READER):
            try:
                result = reader.read(key)
            except (OfflineCacheMiss, DatasetIntegrityError):
                continue
            except Exception as error:
                # The catch-all IS the assertion: anything reaching here is by
                # definition outside the two-error contract, and is reported
                # (with its type) instead of being swallowed.
                escaped.append(error)
                return
            assert result in valid
            successful_reads[0] += 1

    with ThreadPoolExecutor(max_workers=len(_PAYLOADS) + 3) as pool:
        futures = [pool.submit(_write, payload) for payload in _PAYLOADS]
        futures += [pool.submit(_read) for _ in range(3)]
        for future in futures:
            future.result()

    assert not escaped, f"a read escaped the typed error contract: {escaped[0]!r}"

    # Vacuity guard. All writers have finished, so a checksum-valid entry is
    # guaranteed to exist; if no read succeeded during the race itself, keep
    # trying until one does. A run where every single read missed would have
    # asserted nothing about the contested path.
    deadline = time.monotonic() + 5.0
    while successful_reads[0] == 0 and time.monotonic() < deadline:
        try:
            assert DatasetCache(shared_root).read(key) in valid
        except (OfflineCacheMiss, DatasetIntegrityError):
            continue
        successful_reads[0] += 1
    assert successful_reads[0] > 0, "no checksum-valid read was ever observed"


@pytest.mark.integration
def test_racing_concurrent_resolution_reads_never_escape_the_typed_error_contract(
    tmp_path: Path,
) -> None:
    """The same real-thread race against ``ResolutionCache`` (EVK-6 T2)."""
    shared_root = tmp_path / "shared"
    key = _resolution_key()
    revisions = [f"sha256:rev{index}" for index in range(4)]
    valid = set(revisions)
    escaped: list[BaseException] = []
    successful_reads = [0]
    start = threading.Barrier(len(revisions) + 3)

    def _write(revision: str) -> None:
        cache = ResolutionCache(shared_root)
        start.wait()
        for _ in range(_WRITES_PER_WRITER):
            _write_with_windows_retry(lambda: cache.write(key, _resolved(revision)))

    def _read() -> None:
        reader = ResolutionCache(shared_root)
        start.wait()
        for _ in range(_READS_PER_READER):
            try:
                result = reader.read(key)
            except (OfflineCacheMiss, DatasetIntegrityError):
                continue
            except Exception as error:
                # The catch-all IS the assertion: anything reaching here is by
                # definition outside the two-error contract, and is reported
                # (with its type) instead of being swallowed.
                escaped.append(error)
                return
            assert result.revision in valid
            successful_reads[0] += 1

    with ThreadPoolExecutor(max_workers=len(revisions) + 3) as pool:
        futures = [pool.submit(_write, revision) for revision in revisions]
        futures += [pool.submit(_read) for _ in range(3)]
        for future in futures:
            future.result()

    assert not escaped, f"a read escaped the typed error contract: {escaped[0]!r}"

    deadline = time.monotonic() + 5.0
    while successful_reads[0] == 0 and time.monotonic() < deadline:
        try:
            assert ResolutionCache(shared_root).read(key).revision in valid
        except (OfflineCacheMiss, DatasetIntegrityError):
            continue
        successful_reads[0] += 1
    assert successful_reads[0] > 0, "no checksum-valid read was ever observed"
