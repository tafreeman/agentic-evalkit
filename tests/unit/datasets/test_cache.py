"""Tests for the on-disk dataset cache, which stores entries by content fingerprint.

ADR-0004, design section 6.3.

The cache stores each downloaded "page" of a dataset under a key built from
every parameter that makes that page unique: which provider it came from, the
dataset id, its exact pinned revision, its config and split, and its offset
and limit within the page. This file checks four things: that the key's hash
(its "digest") changes whenever any of those identity-defining fields
changes, so two different requests can never collide on the same cache
entry; that a corrupted cache entry is reported as a different, specific
error than one that simply doesn't exist yet (a cache "miss"); that writes
replace the old file atomically, so many threads writing the same cache key
at the same time can never leave a half-written file behind; and that two
different cache keys (for example, two different pages of the same dataset)
are stored and read back completely independently of each other. The
identity and corruption tests directly below are copied, unmodified, from a
code snippet written out in full in the original implementation plan
(docs/plans/2026-07-02-agentic-evalkit-initial-release.md, Task 4 Step 2).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

from agentic_evalkit.datasets.cache import CacheKey, DatasetCache
from agentic_evalkit.errors import DatasetIntegrityError, OfflineCacheMiss

if TYPE_CHECKING:
    from pathlib import Path

#: How many times to retry a write that hits the Windows file-locking
#: collision described below, before giving up.
_WRITE_RETRY_ATTEMPTS = 3
_WRITE_RETRY_SLEEP_SECONDS = 0.01
#: A generous time limit for the "reader running while writers are still
#: writing" test below to observe at least one fully successful read after
#: the writers finish. Without this, the test could pass without actually
#: checking anything: if every read happened to occur before any writer had
#: finished, the real check further up (does a read ever see valid data
#: while writes are still happening?) would never have run at all.
_READER_OBSERVE_DEADLINE_SECONDS = 5.0


def _write_with_windows_retry(cache: DatasetCache, key: CacheKey, payload: bytes) -> None:
    """Write ``payload`` under ``key``, retrying if Windows briefly locks the file.

    On Windows, replacing a file that another thread (a reader, or another
    writer racing to write the same key) currently has open raises
    ``PermissionError``. Mac and Linux don't have this restriction -- they
    allow replacing a file that's still open elsewhere. The lock is only
    ever temporary, so this just retries a bounded number of times with a
    short pause, letting the write go through once the other thread closes
    its handle. Only ``PermissionError`` is retried this way; any other
    exception is left to propagate immediately, so a real bug never gets
    silently hidden behind a retry loop.
    """
    for attempt in range(_WRITE_RETRY_ATTEMPTS):
        try:
            cache.write(key, payload)
            return
        except PermissionError:
            if attempt == _WRITE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_WRITE_RETRY_SLEEP_SECONDS)


# --- Step 2 (copied word-for-word from the plan doc): cache-key identity and corruption tests ---


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
    # Shrink the payload file on disk without touching its "manifest" -- the
    # small JSON sidecar file that records what the payload's checksum and
    # byte count are supposed to be. This simulates corruption: the actual
    # data and the manifest's record of what that data should look like now
    # disagree with each other.
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
# The plan asks for these two tests to be run several times in a row (for
# example, `pytest -x --count=5`, which repeats the whole test file rather
# than a single test) to build confidence that when multiple threads write
# to the same cache key at the same time, the result is never a half-written
# or missing entry -- exactly one valid entry must always survive.


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
# These tests close the gaps left by the ones above:
#   - a reader that reads WHILE writers are still actively writing, not only
#     after every writer has already finished;
#   - a repeat loop built directly into the test (instead of relying on
#     someone remembering to pass `--count` on the command line), so a rare
#     race condition is still likely to be caught during a normal test run;
#   - two more ways a cached entry can go bad -- its payload file shrinking
#     to zero bytes, and a single byte inside it flipping while its length
#     stays the same -- checked separately from the "there's no entry at
#     all" (offline miss) case.
# ADR-0004 requires every read to verify the checksum, byte count, and key
# all match before trusting a cache entry, specifically so that no timing of
# reads and writes can ever make a half-written entry look valid.

# How many times to repeat the race below: small enough that the whole loop
# still finishes in well under a second, but large enough that a real
# same-key write race would be very unlikely to happen to pass "by luck" on
# every single repeat.
_RACE_ITERATIONS = 12


@pytest.mark.parametrize("iteration", range(_RACE_ITERATIONS))
def test_reader_racing_concurrent_writers_never_sees_corruption(
    tmp_path: Path, iteration: int
) -> None:
    """While many writers are actively writing the same key at once, a
    reader running at the same time must, on every single read, either get
    back one complete and valid payload or cleanly raise
    ``OfflineCacheMiss`` / ``DatasetIntegrityError`` -- it must never return
    a mixed-up or partially written result. This whole scenario is repeated
    a fixed number of times so that a rare race condition is unlikely to go
    unnoticed on every single repeat.
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

    # Safety net against a test that passes without actually checking
    # anything: if every read during the race above happened to miss before
    # any write had landed yet, the key assertion inside `_read_repeatedly`
    # (that a successful read returns valid data) never actually ran. By
    # this point every writer has finished and a valid entry is guaranteed
    # to exist, so keep reading (up to a generous time limit) until at least
    # one successful, valid read is observed, and then confirm below that
    # this actually happened.
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
