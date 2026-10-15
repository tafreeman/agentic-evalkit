"""G5 compatibility snapshots, staged-write faults, and coordinated writers."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from shutil import copytree
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from agentic_evalkit._hashing import sha256_digest, sha256_hex
from agentic_evalkit.datasets import _cache_storage, cache, huggingface, local, resolution_cache
from agentic_evalkit.datasets._cache_storage import KeyedLockPool, staged_bytes
from agentic_evalkit.datasets._serialization import canonical_row_bytes, canonical_row_digest
from agentic_evalkit.datasets.cache import CacheKey, DatasetCache
from agentic_evalkit.datasets.resolution_cache import ResolutionCache, ResolutionKey
from agentic_evalkit.errors import DatasetIntegrityError, OfflineCacheMiss
from agentic_evalkit.models import DatasetRef, ResolvedDataset

if TYPE_CHECKING:
    from collections.abc import Callable

_FIXTURES = Path(__file__).parent / "fixtures" / "cache_g5_legacy"
_EXPECTED = json.loads((_FIXTURES / "expected.json").read_bytes())
_FIXED = datetime(2026, 9, 10, tzinfo=UTC)


@pytest.mark.parametrize("case", _EXPECTED["rows"])
def test_saved_row_bytes_and_hashes(case: dict) -> None:
    # Recorded using the pre-extraction implementation, not the helper under test.
    row = case["row"]
    for ordered in (row, dict(reversed(list(row.items())))):
        assert canonical_row_bytes(ordered) == case["bytes"].encode("utf-8")
        assert canonical_row_digest(ordered) == case["digest"]
        assert local._canonical_digest(ordered) == case["digest"]
        assert huggingface._canonical_digest(ordered) == case["digest"]


def test_byte_hash_known_answer() -> None:
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_hex(b"abc") == expected
    assert sha256_digest(b"abc") == "sha256:" + expected
    assert huggingface._response_digest(b"abc") == "sha256:" + expected


async def test_local_revision_hashes_raw_bytes(tmp_path: Path) -> None:
    source = tmp_path / "row.json"
    source.write_bytes(b"{}")
    provider = local.LocalDatasetProvider(allowed_roots=(tmp_path,))
    resolved = await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))
    assert resolved.revision == _EXPECTED["rows"][0]["digest"]


def test_old_entries_read_and_rewrite_identically(tmp_path: Path, monkeypatch) -> None:
    copytree(_FIXTURES, tmp_path, dirs_exist_ok=True)
    for module in (cache, resolution_cache):
        monkeypatch.setattr(module, "datetime", Mock(now=Mock(return_value=_FIXED)))
    key = CacheKey.model_validate(_EXPECTED["key"])
    resolution_key = ResolutionKey.model_validate(_EXPECTED["resolution_key"])
    resolved = ResolvedDataset.model_validate(_EXPECTED["resolved"])
    dataset_cache = DatasetCache(tmp_path / "dataset")
    identity_cache = ResolutionCache(tmp_path / "resolution")
    assert dataset_cache.read(key) == b"old\x00payload\xff"
    assert identity_cache.read(resolution_key) == resolved
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    dataset_cache.write(key, b"old\x00payload\xff")
    identity_cache.write(resolution_key, resolved)
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before  # Includes filenames, digest directories, and every disk byte.


def _entry(kind: str, root: Path, revision: str = "old") -> tuple[Callable, Callable, Path]:
    if kind == "dataset":
        key = CacheKey(provider="local", dataset_id="fault", revision="fixed", offset=0, limit=1)
        instance = DatasetCache(root)
        return (
            lambda: instance.write(key, revision.encode()),
            lambda: instance.read(key),
            instance.payload_path(key),
        )
    key = ResolutionKey(provider="local", dataset_id="fault")
    instance = ResolutionCache(root)
    resolved = ResolvedDataset(dataset_id="fault", revision=revision, retrieved_at=_FIXED)
    return (
        lambda: instance.write(key, resolved),
        lambda: instance.read(key).revision.encode(),
        instance._entry_path(key),
    )


_FAULTS = [
    (kind, step, index)
    for kind in ("dataset", "resolution")
    for step in ("create", "write", "flush", "fsync", "replace")
    for index in range(1, 3 if kind == "dataset" else 2)
]


@pytest.mark.parametrize(("kind", "step", "index"), _FAULTS)
@pytest.mark.parametrize("existing", [False, True])
def test_fault_matrix(tmp_path: Path, monkeypatch, kind, step, index, existing) -> None:
    old_write, read, path = _entry(kind, tmp_path)
    if existing:
        old_write()
    write, _, _ = _entry(kind, tmp_path, "new")  # Equal length forces checksum validation.
    original_create = _cache_storage.tempfile.NamedTemporaryFile
    original_fsync = _cache_storage.os.fsync
    original_replace = Path.replace
    calls = 0
    failure = PermissionError(f"injected {step} {index}")

    def fail_at_index() -> None:
        nonlocal calls
        calls += 1
        if calls == index:
            raise failure

    def create(*args, **kwargs):
        if step == "create":
            fail_at_index()
        handle = original_create(*args, **kwargs)
        if step in ("write", "flush"):
            original = getattr(handle, step)

            def operation(*args, **kwargs):
                # Exercise cleanup of actual written/partially written bytes.
                if step == "write" and calls + 1 == index:
                    original(args[0][:1])
                fail_at_index()
                return original(*args, **kwargs)

            setattr(handle, step, operation)
        return handle

    def fsync(fd):
        if step == "fsync":
            fail_at_index()
        return original_fsync(fd)

    def replace(source, destination):
        if step == "replace":
            # Dataset must stage both files before the first publication.
            if kind == "dataset" and calls == 0:
                assert len(list(path.parent.glob("*.tmp"))) == 2
            fail_at_index()
        return original_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(_cache_storage.tempfile, "NamedTemporaryFile", create)
        patch.setattr(_cache_storage.os, "fsync", fsync)
        patch.setattr(Path, "replace", replace)
        with pytest.raises(PermissionError) as caught:
            write()
        assert caught.value is failure
    assert not list(tmp_path.rglob("*.tmp"))
    if existing and kind == "dataset" and step == "replace" and index == 2:
        with pytest.raises(DatasetIntegrityError, match="checksum"):
            read()
    elif existing:
        assert read() == b"old"
    else:
        with pytest.raises(OfflineCacheMiss):
            read()
    write()  # Failure releases the key lock and permits recovery.
    assert read() == b"new"


def test_staging_unique_paths_and_cleanup(tmp_path: Path) -> None:
    destination = tmp_path / "payload.bin"
    with staged_bytes(destination, b"one") as first, staged_bytes(destination, b"two") as second:
        assert first != second
        assert first.parent == second.parent == destination.parent
        assert first.read_bytes() == b"one"
        assert second.read_bytes() == b"two"
        first.replace(destination)
    assert destination.read_bytes() == b"one"
    assert not first.exists()
    assert not second.exists()


@pytest.mark.parametrize("original_failure", [False, True])
def test_cleanup_failure_does_not_hide_original(tmp_path: Path, monkeypatch, original_failure):
    failure = RuntimeError("publication failed")
    cleanup = PermissionError("cleanup denied")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", Mock(side_effect=cleanup))
        with (
            pytest.raises(RuntimeError if original_failure else PermissionError) as caught,
            staged_bytes(tmp_path / "entry", b"data"),
        ):
            if original_failure:
                raise failure
        assert caught.value is (failure if original_failure else cleanup)
        if original_failure:
            assert "cleanup denied" in failure.__notes__[0]
    for remaining in tmp_path.iterdir():
        remaining.unlink()


def test_pool_thread_safe_identity_and_separate_cache_pools() -> None:
    pool = KeyedLockPool()
    barrier = threading.Barrier(8)

    def lookup(_):
        barrier.wait(timeout=5)
        return pool.lock_for("same")

    with ThreadPoolExecutor(max_workers=8) as executor:
        locks = list(executor.map(lookup, range(8)))
    assert all(lock is locks[0] for lock in locks)
    assert pool.lock_for("different") is not locks[0]
    assert cache._lock_pool.lock_for("same") is not resolution_cache._lock_pool.lock_for("same")


@pytest.mark.parametrize("kind", ["dataset", "resolution"])
def test_instances_serialize_same_key_and_allow_other_keys(tmp_path: Path, monkeypatch, kind):
    module = cache if kind == "dataset" else resolution_cache
    pool = KeyedLockPool()
    monkeypatch.setattr(module, "_lock_pool", pool)
    first_write, read, path = _entry(kind, tmp_path, "one")
    second_write, _, _ = _entry(kind, tmp_path, "two")
    digest = "sha256:" + path.parent.name
    real_lock_for = pool.lock_for
    attempted = threading.Barrier(3)

    def observed_lock_for(value):
        lock = real_lock_for(value)
        if value == digest:
            attempted.wait(timeout=5)
        return lock

    held = real_lock_for(digest)
    held.acquire()
    monkeypatch.setattr(pool, "lock_for", observed_lock_for)
    with ThreadPoolExecutor(max_workers=3) as executor:
        first = executor.submit(first_write)
        second = executor.submit(second_write)
        try:
            attempted.wait(timeout=5)  # Both distinct instances reached the shared lock.
            assert not first.done() and not second.done()
            assert not list(tmp_path.rglob("*.tmp"))
            if kind == "dataset":
                other_key = CacheKey(
                    provider="local", dataset_id="other", revision="r", offset=0, limit=1
                )
                other = DatasetCache(tmp_path)
                executor.submit(other.write, other_key, b"independent").result(timeout=5)
                assert other.read(other_key) == b"independent"
            else:
                other_key = ResolutionKey(provider="local", dataset_id="other")
                other = ResolutionCache(tmp_path)
                value = ResolvedDataset(dataset_id="other", revision="r", retrieved_at=_FIXED)
                executor.submit(other.write, other_key, value).result(timeout=5)
                assert other.read(other_key) == value
        finally:
            held.release()
        first.result(timeout=5)
        second.result(timeout=5)
    assert read() in (b"one", b"two")
    assert not list(tmp_path.rglob("*.tmp"))
