"""Tests for the offline resolution-identity cache (ADR-0011).

Mirrors the shape of ``tests/unit/datasets/test_cache.py``'s own coverage of
``DatasetCache``/``CacheKey`` (ADR-0004): digest determinism, a frozen/
closed-field-set key, a write-then-read round trip, a genuine miss, a
tampered entry raising a distinct integrity error rather than being silently
treated as a miss, and atomic-replace correctness under concurrent same-key
writers.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentic_evalkit.datasets.resolution_cache import ResolutionCache, ResolutionKey
from agentic_evalkit.errors import DatasetIntegrityError, OfflineCacheMiss
from agentic_evalkit.models import ResolvedDataset

#: Bounded retry budget for the Windows sharing-violation collision: a
#: concurrent ``Path.replace()`` onto a file another thread currently has
#: open raises ``PermissionError`` on Windows (POSIX rename has no such
#: restriction) -- mirrors ``tests/unit/datasets/test_cache.py``'s identical
#: helper for ``DatasetCache``.
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


# --- ResolutionKey: digest identity and frozen shape -------------------------


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
        # ``revision`` is an identity-bearing field (2026-07-09 fix): a
        # different requested pin MUST hash to a different slot so two pinned
        # revisions never collide, and a pinned request MUST differ from the
        # default unpinned (``revision=None``) request.
        {"revision": "sha256:" + "b" * 64},
    ],
)
def test_digest_changes_for_every_identity_field(overrides: dict[str, object]) -> None:
    assert _key(**overrides).digest() != _key().digest()


def test_key_is_frozen_and_forbids_unknown_fields() -> None:
    key = _key()
    with pytest.raises(Exception):  # noqa: B017 - pydantic.ValidationError, frozen instance
        key.provider = "renamed"  # type: ignore[misc]
    with pytest.raises(Exception):  # noqa: B017 - pydantic.ValidationError, extra="forbid"
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


# --- ResolutionCache: miss, round trip, corruption ----------------------------


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
    """The 2026-07-09 correctness fix: two requests identical except for
    their pinned ``revision`` must occupy distinct cache slots, so caching
    ``revB`` never overwrites ``revA``'s entry and reading back ``revA``'s
    key returns ``revA``'s data -- never ``revB``'s."""
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
    """An unpinned (``revision=None``, "latest") request and a pinned request
    for the same dataset are different questions and must not alias: a pinned
    request that was never cached must miss rather than serve the unpinned
    "latest" entry's data."""
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
    """Mirrors ``test_cache.py``'s own ADR-0004 distinction test: a
    never-written key is a miss; a tampered entry is corruption. The two
    must never collapse into the same exception."""
    cache = ResolutionCache(tmp_path)
    key = _key()

    with pytest.raises(OfflineCacheMiss):
        cache.read(key)

    cache.write(key, _resolved())
    entry_path = cache._entry_path(key)
    entry_path.write_bytes(entry_path.read_bytes() + b"corruption")
    with pytest.raises(DatasetIntegrityError):
        cache.read(key)


# --- concurrent writers: atomic-replace correctness (ADR-0004 pattern) -------


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

    # Exactly one valid final entry: read() must succeed (no integrity
    # error, no offline miss) and return one of the written datasets whole
    # -- never a torn mix of two racing writers' fields.
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
