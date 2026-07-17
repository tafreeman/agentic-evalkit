"""Tests for the cache that remembers, offline, exactly which pinned dataset
version a given request resolved to (ADR-0011).

"Resolving" a dataset means turning a request like "give me the gsm8k
dataset, main config, test split" into one exact, pinned version (see
``ResolvedDataset``). This cache lets that resolution be looked up again
later without repeating the work or needing the network. This file mirrors
the shape of the tests ``tests/unit/datasets/test_cache.py`` already runs
against the similar ``DatasetCache``/``CacheKey`` pair (ADR-0004): hashing
the same key twice gives the same fingerprint; the key object can't be
changed after creation and rejects unrecognized fields; writing an entry and
reading it back gives the same data; asking for something never written is
a clean miss; a cache entry whose contents have been corrupted raises a
distinct "this data is corrupted" error rather than silently being treated
as a miss; and writing to the same cache slot from multiple threads at once
never leaves the file half-written -- the result is always one complete
write or another, never a mix of both.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from agentic_evalkit.datasets.resolution_cache import ResolutionCache, ResolutionKey
from agentic_evalkit.errors import DatasetIntegrityError, OfflineCacheMiss
from agentic_evalkit.models import ResolvedDataset

if TYPE_CHECKING:
    from pathlib import Path

#: How many times to retry a write that collides with another thread's write
#: on Windows: replacing a file (``Path.replace()``) while another thread
#: currently has that same file open raises ``PermissionError`` on Windows
#: specifically (renaming a file on POSIX/Linux/Mac has no such
#: restriction). Mirrors the identical retry helper for ``DatasetCache`` in
#: ``tests/unit/datasets/test_cache.py``.
_WRITE_RETRY_ATTEMPTS = 3
_WRITE_RETRY_SLEEP_SECONDS = 0.01


def _write_with_windows_retry(
    cache: ResolutionCache, key: ResolutionKey, dataset: ResolvedDataset
) -> None:
    for attempt in range(_WRITE_RETRY_ATTEMPTS):
        try:
            cache.write(key, dataset)
            return
        except PermissionError:
            if attempt == _WRITE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_WRITE_RETRY_SLEEP_SECONDS)


def _key(**overrides: object) -> ResolutionKey:
    defaults: dict[str, object] = {
        "provider": "huggingface",
        "dataset_id": "openai/gsm8k",
        "config": "main",
        "split": "test",
    }
    defaults.update(overrides)
    return ResolutionKey.model_validate(defaults)


def _resolved(**overrides: object) -> ResolvedDataset:
    defaults: dict[str, object] = {
        "dataset_id": "openai/gsm8k",
        "revision": "sha256:" + "a" * 64,
        "config": "main",
        "split": "test",
        "row_count": 1319,
        "retrieved_at": datetime.now(UTC),
    }
    defaults.update(overrides)
    return ResolvedDataset.model_validate(defaults)


# --- ResolutionKey: fingerprint hashing and the frozen, fixed-field key shape ---


def test_digest_is_deterministic_and_pure() -> None:
    key = _key()
    assert key.digest() == key.digest()
    assert _key().digest() == key.digest()


def test_digest_has_sha256_prefix_and_64_hex_chars() -> None:
    digest = _key().digest()
    assert digest.startswith("sha256:")
    assert len(digest.removeprefix("sha256:")) == 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "local"},
        {"dataset_id": "other/dataset"},
        {"config": "other-config"},
        {"split": "train"},
        {"config": None},
        {"split": None},
        # ``revision`` counts as part of what makes a key unique too (this
        # was a 2026-07-09 fix): asking for a different pinned revision must
        # hash to a different cache slot, so two different pinned revisions
        # are never confused with each other, and a pinned request must also
        # be treated as different from the default, unpinned request
        # (``revision=None``).
        {"revision": "sha256:" + "b" * 64},
    ],
)
def test_digest_changes_for_every_identity_field(overrides: dict[str, object]) -> None:
    assert _key(**overrides).digest() != _key().digest()


def test_key_is_frozen_and_forbids_unknown_fields() -> None:
    key = _key()
    # Asserting on the broad `Exception` class (rather than pydantic's own
    # ValidationError) is intentional here -- see the noqa below. The actual
    # error raised is pydantic's ValidationError, because the key is frozen
    # and can't be modified after creation.
    with pytest.raises(Exception):  # noqa: B017
        key.provider = "renamed"  # type: ignore[misc]
    # Same as above, but this time the real error is pydantic's
    # ValidationError raised because the key rejects unrecognized fields
    # (it's configured with extra="forbid").
    with pytest.raises(Exception):  # noqa: B017
        ResolutionKey.model_validate(
            {**_key().model_dump(mode="json"), "nonexistent_field": "sneaky"}
        )


def test_key_defaults_revision_to_none() -> None:
    key = ResolutionKey(provider="huggingface", dataset_id="openai/gsm8k")
    assert key.revision is None


def test_key_defaults_config_and_split_to_none() -> None:
    key = ResolutionKey(provider="huggingface", dataset_id="openai/gsm8k")
    assert key.config is None
    assert key.split is None


# --- ResolutionCache: cache misses, write-then-read round trips, and corrupted entries ---


def test_read_of_never_written_key_raises_retryable_offline_cache_miss(tmp_path: Path) -> None:
    cache = ResolutionCache(tmp_path)
    with pytest.raises(OfflineCacheMiss) as excinfo:
        cache.read(_key())
    assert excinfo.value.retryable is True


def test_write_then_read_round_trips_the_exact_resolved_dataset(tmp_path: Path) -> None:
    cache = ResolutionCache(tmp_path)
    key = _key()
    resolved = _resolved()
    cache.write(key, resolved)
    read_back = cache.read(key)
    assert read_back == resolved


def test_two_distinct_keys_are_both_independently_addressable(tmp_path: Path) -> None:
    cache = ResolutionCache(tmp_path)
    key_a = _key(dataset_id="openai/gsm8k")
    key_b = _key(dataset_id="princeton-nlp/SWE-bench_Verified")
    resolved_a = _resolved(dataset_id="openai/gsm8k")
    resolved_b = _resolved(dataset_id="princeton-nlp/SWE-bench_Verified")
    cache.write(key_a, resolved_a)
    cache.write(key_b, resolved_b)
    assert cache.read(key_a) == resolved_a
    assert cache.read(key_b) == resolved_b


def test_two_different_pinned_revisions_never_collide_in_the_same_slot(tmp_path: Path) -> None:
    """This test covers the 2026-07-09 correctness fix: two requests that are
    identical except for which ``revision`` they pin must be stored in
    different cache slots. That way, caching ``revB`` never overwrites
    ``revA``'s entry, and reading the cache with ``revA``'s key always
    returns ``revA``'s data -- never ``revB``'s."""
    cache = ResolutionCache(tmp_path)
    key_a = _key(revision="revA")
    key_b = _key(revision="revB")
    resolved_a = _resolved(revision="sha256:" + "a" * 64)
    resolved_b = _resolved(revision="sha256:" + "b" * 64)

    cache.write(key_a, resolved_a)
    cache.write(key_b, resolved_b)  # must NOT overwrite key_a's slot

    assert cache.read(key_a) == resolved_a
    assert cache.read(key_b) == resolved_b


def test_pinned_and_unpinned_requests_occupy_distinct_slots(tmp_path: Path) -> None:
    """An unpinned request (``revision=None``, meaning "give me whatever the
    latest version is") and a pinned request for a specific revision of the
    same dataset are two different questions, and the cache must not confuse
    them: if the pinned request was never cached, reading it must be a clean
    miss -- it must never accidentally be served the unpinned "latest"
    entry's data instead."""
    cache = ResolutionCache(tmp_path)
    unpinned = _key(revision=None)
    pinned = _key(revision="revA")
    cache.write(unpinned, _resolved(revision="sha256:" + "e" * 64))

    with pytest.raises(OfflineCacheMiss):
        cache.read(pinned)


def test_overwriting_the_same_key_replaces_the_prior_entry(tmp_path: Path) -> None:
    cache = ResolutionCache(tmp_path)
    key = _key()
    first = _resolved(revision="sha256:" + "1" * 64)
    second = _resolved(revision="sha256:" + "2" * 64)
    cache.write(key, first)
    cache.write(key, second)
    assert cache.read(key) == second


def test_tampered_payload_raises_integrity_error_not_a_miss(tmp_path: Path) -> None:
    cache = ResolutionCache(tmp_path)
    key = _key()
    cache.write(key, _resolved())

    entry_path = cache._entry_path(key)  # test-only: locate the file to corrupt it
    record = json.loads(entry_path.read_text(encoding="utf-8"))
    record["resolved_dataset"]["revision"] = "sha256:" + "f" * 64
    entry_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(DatasetIntegrityError):
        cache.read(key)


def test_tampered_key_in_manifest_raises_integrity_error(tmp_path: Path) -> None:
    cache = ResolutionCache(tmp_path)
    key = _key()
    cache.write(key, _resolved())

    entry_path = cache._entry_path(key)
    record = json.loads(entry_path.read_text(encoding="utf-8"))
    record["key"]["dataset_id"] = "someone-else/dataset"
    entry_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(DatasetIntegrityError):
        cache.read(key)


def test_malformed_json_raises_integrity_error(tmp_path: Path) -> None:
    cache = ResolutionCache(tmp_path)
    key = _key()
    cache.write(key, _resolved())

    entry_path = cache._entry_path(key)
    entry_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(DatasetIntegrityError):
        cache.read(key)


def test_corruption_and_offline_miss_are_distinct(tmp_path: Path) -> None:
    """Mirrors the equivalent test in ``test_cache.py`` for ADR-0004: asking
    for a key that was never written is a plain miss, while asking for a key
    whose stored entry has been corrupted is a different problem
    (corruption). The two must always raise different, distinguishable
    exceptions -- never collapse into the same one."""
    cache = ResolutionCache(tmp_path)
    key = _key()

    with pytest.raises(OfflineCacheMiss):
        cache.read(key)

    cache.write(key, _resolved())
    entry_path = cache._entry_path(key)
    entry_path.write_bytes(entry_path.read_bytes() + b"corruption")
    with pytest.raises(DatasetIntegrityError):
        cache.read(key)


# --- concurrent writers: replacing a cache entry is all-or-nothing, so
# readers never see a half-written file (ADR-0004 pattern) ---


def test_concurrent_same_key_writes_leave_exactly_one_valid_entry(tmp_path: Path) -> None:
    cache = ResolutionCache(tmp_path)
    key = _key()
    candidates = [_resolved(revision=f"sha256:{i:064d}") for i in range(8)]
    barrier = threading.Barrier(len(candidates))

    def _write(dataset: ResolvedDataset) -> None:
        barrier.wait()
        _write_with_windows_retry(cache, key, dataset)

    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        list(pool.map(_write, candidates))

    # There must be exactly one valid entry left after the dust settles:
    # read() has to succeed (no integrity error, no offline-cache-miss) and
    # return one of the datasets we wrote, completely intact -- never a
    # broken mix of fields from two different writers that happened to race
    # each other.
    result = cache.read(key)
    assert result in candidates


def test_concurrent_writes_to_distinct_keys_are_all_readable(tmp_path: Path) -> None:
    cache = ResolutionCache(tmp_path)
    keys_and_datasets = [
        (_key(dataset_id=f"dataset-{i}"), _resolved(dataset_id=f"dataset-{i}")) for i in range(6)
    ]

    def _write(pair: tuple[ResolutionKey, ResolvedDataset]) -> None:
        key, dataset = pair
        cache.write(key, dataset)

    with ThreadPoolExecutor(max_workers=len(keys_and_datasets)) as pool:
        list(pool.map(_write, keys_and_datasets))

    for key, dataset in keys_and_datasets:
        assert cache.read(key) == dataset
