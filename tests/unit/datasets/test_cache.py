"""Tests for the content-addressed dataset cache (ADR-0004, design §6.3).

Covers cache-key identity (digest changes with any identity-bearing field),
the corruption-vs-offline-miss distinction, atomic replace-based writes under
concurrent same-key writers, and independent addressability of distinct page
keys. The identity and corruption tests below reproduce the plan's verbatim
snippet (docs/plans/2026-07-02-agentic-evalkit-initial-release.md, Task 4
Step 2) unmodified.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agentic_evalkit.datasets.cache import CacheKey, DatasetCache
from agentic_evalkit.errors import DatasetIntegrityError, OfflineCacheMiss

#: Bounded retry budget for the Windows sharing-violation collision below.
_WRITE_RETRY_ATTEMPTS = 3
_WRITE_RETRY_SLEEP_SECONDS = 0.01
#: Generous wall-clock deadline for a racing reader to observe at least one
#: checksum-valid read once writers have published (guards against a vacuous
#: pass where every in-race read missed before the first write landed).
_READER_OBSERVE_DEADLINE_SECONDS = 5.0


def _write_with_windows_retry(cache: DatasetCache, key: CacheKey, payload: bytes) -> None:
    """Write ``payload`` under ``key``, retrying a Windows sharing violation.

    On Windows, ``Path.replace()`` onto a payload/manifest that another
    thread (a concurrent reader or a racing writer) currently holds open
    raises ``PermissionError`` (a sharing violation) -- POSIX rename has no
    such restriction. That collision is transient: retry a bounded number of
    times with a tiny sleep and let the write land once the open handle is
    released. Only ``PermissionError`` is treated as retryable; any other
    exception propagates immediately so a real bug is never masked.
    """
    for attempt in range(_WRITE_RETRY_ATTEMPTS):
        try:
            cache.write(key, payload)
            return
        except PermissionError:
            if attempt == _WRITE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_WRITE_RETRY_SLEEP_SECONDS)


# --- Step 2 (plan verbatim): cache key identity and corruption tests -------


def test_cache_key_changes_for_revision_config_split_and_page() -> None:
    base = CacheKey(
        provider="huggingface",
        dataset_id="openai/gsm8k",
        revision="abc",
        config="main",
        split="test",
        offset=0,
        limit=10,
    )
    variants = (
        base.model_copy(update={"revision": "def"}),
        base.model_copy(update={"config": "socratic"}),
        base.model_copy(update={"split": "train"}),
        base.model_copy(update={"offset": 10}),
    )
    assert all(item.digest() != base.digest() for item in variants)


def test_corruption_and_offline_miss_are_distinct(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    key = CacheKey(
        provider="local",
        dataset_id="items.jsonl",
        revision="sha256:a",
        config=None,
        split=None,
        offset=0,
        limit=10,
    )
    with pytest.raises(OfflineCacheMiss):
        cache.read(key)
    cache.write(key, b"valid")
    cache.payload_path(key).write_bytes(b"changed")
    with pytest.raises(DatasetIntegrityError):
        cache.read(key)


# --- Additional identity coverage -------------------------------------------


def test_cache_key_digest_is_deterministic_and_pure() -> None:
    key = CacheKey(
        provider="huggingface",
        dataset_id="openai/gsm8k",
        revision="abc",
        config="main",
        split="test",
        offset=0,
        limit=10,
    )
    assert key.digest() == key.digest()
    assert key.digest().startswith("sha256:")
    assert len(key.digest()) == len("sha256:") + 64


def test_cache_key_digest_changes_for_limit_and_provider() -> None:
    base = CacheKey(
        provider="huggingface",
        dataset_id="openai/gsm8k",
        revision="abc",
        config="main",
        split="test",
        offset=0,
        limit=10,
    )
    assert base.model_copy(update={"limit": 20}).digest() != base.digest()
    assert base.model_copy(update={"provider": "local"}).digest() != base.digest()
    assert base.model_copy(update={"dataset_id": "openai/other"}).digest() != base.digest()


def test_cache_key_digest_changes_for_optional_digest_fields_and_record_type() -> None:
    base = CacheKey(
        provider="huggingface",
        dataset_id="openai/gsm8k",
        revision="abc",
        config="main",
        split="test",
        offset=0,
        limit=10,
    )
    projected = base.model_copy(update={"projection_digest": "sha256:p"})
    filtered = base.model_copy(update={"filter_digest": "sha256:f"})
    data_files = base.model_copy(update={"data_files_digest": "sha256:d"})
    full = base.model_copy(update={"record_type": "full"})
    assert projected.digest() != base.digest()
    assert filtered.digest() != base.digest()
    assert data_files.digest() != base.digest()
    assert full.digest() != base.digest()
    # And they are pairwise distinct from each other too.
    digests = {
        base.digest(),
        projected.digest(),
        filtered.digest(),
        data_files.digest(),
        full.digest(),
    }
    assert len(digests) == 5


def test_cache_key_defaults_for_optional_digest_fields_and_record_type() -> None:
    key = CacheKey(
        provider="local",
        dataset_id="items.jsonl",
        revision="sha256:a",
        config=None,
        split=None,
        offset=0,
        limit=10,
    )
    assert key.projection_digest is None
    assert key.filter_digest is None
    assert key.data_files_digest is None
    assert key.record_type == "page"


def test_cache_key_is_frozen_and_forbids_unknown_fields() -> None:
    key = CacheKey(
        provider="local",
        dataset_id="items.jsonl",
        revision="sha256:a",
        config=None,
        split=None,
        offset=0,
        limit=10,
    )
    with pytest.raises(Exception):  # noqa: B017 - pydantic.ValidationError, frozen instance
        key.offset = 5
    with pytest.raises(Exception):  # noqa: B017 - pydantic.ValidationError, unknown field
        CacheKey(
            provider="local",
            dataset_id="items.jsonl",
            revision="sha256:a",
            config=None,
            split=None,
            offset=0,
            limit=10,
            bogus_field="nope",  # type: ignore[call-arg]
        )


# --- Read/write behavior -----------------------------------------------------


def _sample_key(**overrides: object) -> CacheKey:
    fields: dict[str, object] = {
        "provider": "local",
        "dataset_id": "items.jsonl",
        "revision": "sha256:a",
        "config": None,
        "split": None,
        "offset": 0,
        "limit": 10,
    }
    fields.update(overrides)
    return CacheKey(**fields)  # type: ignore[arg-type]


def test_write_then_read_round_trips_payload_bytes(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    key = _sample_key()
    payload = b'{"rows": [1, 2, 3]}'
    cache.write(key, payload)
    assert cache.read(key) == payload


def test_read_missing_entry_raises_offline_cache_miss(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    key = _sample_key()
    with pytest.raises(OfflineCacheMiss):
        cache.read(key)


def test_read_with_byte_count_mismatch_raises_integrity_error(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    key = _sample_key()
    cache.write(key, b"twelve bytes")
    # Truncate the payload without updating the manifest's checksum/byte count.
    cache.payload_path(key).write_bytes(b"short")
    with pytest.raises(DatasetIntegrityError):
        cache.read(key)


def test_read_with_manifest_key_mismatch_raises_integrity_error(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    key = _sample_key()
    cache.write(key, b"payload")
    manifest_path = cache.manifest_path(key)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["key"]["offset"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetIntegrityError):
        cache.read(key)


def test_manifest_records_checksum_byte_count_created_at_and_key(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    key = _sample_key()
    payload = b"some cached page payload"
    cache.write(key, payload)
    manifest = json.loads(cache.manifest_path(key).read_text(encoding="utf-8"))
    assert manifest["byte_count"] == len(payload)
    assert isinstance(manifest["checksum"], str)
    assert manifest["checksum"].startswith("sha256:")
    assert "created_at" in manifest
    assert manifest["key"]["dataset_id"] == key.dataset_id
    assert manifest["key"]["offset"] == key.offset


def test_two_page_keys_with_different_offsets_are_both_addressable(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    first = _sample_key(offset=0)
    second = _sample_key(offset=10)
    cache.write(first, b"page-0")
    cache.write(second, b"page-10")
    assert cache.read(first) == b"page-0"
    assert cache.read(second) == b"page-10"
    assert cache.payload_path(first) != cache.payload_path(second)


def test_overwriting_same_key_replaces_payload_atomically(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    key = _sample_key()
    cache.write(key, b"first version")
    cache.write(key, b"second version, different length")
    assert cache.read(key) == b"second version, different length"


# --- Step 5: concurrency -----------------------------------------------------
#
# These two tests are the ones the plan asks to run repeatedly via
# `pytest -x --count=5` (module-level repetition, not a per-test marker) to
# build confidence that concurrent same-key writes never race into a corrupt
# or missing final entry.


def test_concurrent_same_key_writes_leave_exactly_one_valid_entry(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    key = _sample_key()
    payloads = [f"payload-{i}".encode() for i in range(8)]
    barrier = threading.Barrier(len(payloads))

    def _write(payload: bytes) -> None:
        barrier.wait()
        _write_with_windows_retry(cache, key, payload)

    with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
        list(pool.map(_write, payloads))

    # Exactly one valid final entry: read() must succeed (no integrity error,
    # no offline miss) and return one of the written payloads in full.
    result = cache.read(key)
    assert result in payloads


def test_concurrent_writes_to_distinct_offsets_are_all_readable(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    keys = [_sample_key(offset=i * 10) for i in range(6)]

    def _write(key: CacheKey) -> None:
        cache.write(key, f"payload-for-offset-{key.offset}".encode())

    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        list(pool.map(_write, keys))

    for key in keys:
        assert cache.read(key) == f"payload-for-offset-{key.offset}".encode()


# --- Story 4.1 (R-001): Windows concurrent-write & corruption guard ---------
#
# These close the gaps left above: a reader racing live writers (not just a
# read after all writers have joined), a bounded in-test repeat loop so a
# same-key write race surfaces as a suite failure without needing the
# `--count` CLI flag, and the truncated-to-empty / equal-length bit-flip
# corruption variants distinct from an offline miss. ADR-0004 requires
# checksum + byte-count + key-identity verification on every read, so no
# interleaving may ever surface a partially-written entry as valid.

# A bounded iteration count: small enough that the whole loop stays well under
# a second, large enough to make a genuine same-key write race improbable to
# pass by luck on every iteration.
_RACE_ITERATIONS = 12


@pytest.mark.parametrize("iteration", range(_RACE_ITERATIONS))
def test_reader_racing_concurrent_writers_never_sees_corruption(
    tmp_path: Path, iteration: int
) -> None:
    """A reader interleaved with many same-key writers must, on every read,
    either return one fully-valid written payload or raise a typed
    ``OfflineCacheMiss`` / ``DatasetIntegrityError`` -- never a torn or
    checksum-invalid result. Repeated a bounded number of times so a race is
    unlikely to slip through every iteration.
    """
    cache = DatasetCache(tmp_path / f"iter-{iteration}")
    key = _sample_key()
    payloads = [f"payload-number-{i}".encode() for i in range(8)]
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
                result = cache.read(key)
            except (OfflineCacheMiss, DatasetIntegrityError):
                continue
            # A successful read must be exactly one of the written payloads,
            # never a partial byte string or a mix of two writes.
            assert result in valid
            successful_reads[0] += 1

    with ThreadPoolExecutor(max_workers=len(payloads) + 1) as pool:
        futures = [pool.submit(_write, payload) for payload in payloads]
        futures.append(pool.submit(_read_repeatedly))
        for future in futures:
            future.result()

    # After every writer has finished, exactly one valid entry remains.
    assert cache.read(key) in valid

    # Vacuity guard: if every in-race read happened to miss before the first
    # write published, the contested-path assertion above never ran. All
    # writers have now joined and a valid entry provably exists, so keep
    # reading (bounded by a generous deadline) until at least one checksum-
    # valid read is observed, then assert the test actually exercised it.
    deadline = time.monotonic() + _READER_OBSERVE_DEADLINE_SECONDS
    while successful_reads[0] == 0 and time.monotonic() < deadline:
        try:
            result = cache.read(key)
        except (OfflineCacheMiss, DatasetIntegrityError):
            continue
        assert result in valid
        successful_reads[0] += 1
    assert successful_reads[0] > 0, "no checksum-valid read was ever observed"


def test_truncated_to_empty_payload_raises_integrity_error_not_offline_miss(
    tmp_path: Path,
) -> None:
    """An entry whose payload has been truncated to zero bytes (but whose
    manifest still records the original checksum/byte count) is corrupt, not
    absent: it must raise ``DatasetIntegrityError``, distinct from the
    ``OfflineCacheMiss`` a genuinely missing entry raises.
    """
    cache = DatasetCache(tmp_path)
    key = _sample_key()
    cache.write(key, b"a real payload of some length")
    cache.payload_path(key).write_bytes(b"")
    with pytest.raises(DatasetIntegrityError):
        cache.read(key)


def test_equal_length_bit_flip_raises_integrity_error(tmp_path: Path) -> None:
    """A single-byte flip that preserves the payload length (so the byte-count
    check alone would pass) is still caught by the checksum verification and
    surfaces as ``DatasetIntegrityError`` -- byte count is not sufficient on
    its own.
    """
    cache = DatasetCache(tmp_path)
    key = _sample_key()
    original = b"payload-with-a-known-length"
    cache.write(key, original)
    flipped = bytearray(original)
    flipped[0] ^= 0x01
    assert len(flipped) == len(original)
    cache.payload_path(key).write_bytes(bytes(flipped))
    with pytest.raises(DatasetIntegrityError):
        cache.read(key)


def test_missing_payload_but_present_manifest_is_offline_miss_not_corruption(
    tmp_path: Path,
) -> None:
    """If the payload file is absent while the manifest survives, the read is
    an ``OfflineCacheMiss`` (the entry is treated as not-present), never a
    ``DatasetIntegrityError``: absence and corruption stay distinct even in
    this partial-write shape.
    """
    cache = DatasetCache(tmp_path)
    key = _sample_key()
    cache.write(key, b"payload")
    cache.payload_path(key).unlink()
    with pytest.raises(OfflineCacheMiss):
        cache.read(key)
