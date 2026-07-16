"""Resolution-identity cache: persists a provider's last resolved dataset (ADR-0011).

:class:`~agentic_evalkit.datasets.cache.CacheKey`/``DatasetCache`` (ADR-0004)
cache *pages* of an already-resolved dataset -- the key requires a `revision`,
which only exists once a resolution has already happened. That leaves the
resolution step itself with no cache: before this module, ``DatasetCatalog.resolve``
against a network-requiring provider (``huggingface``) unconditionally raised
under ``offline=True``, even immediately after an online ``datasets pull`` of
that exact dataset (ADR-0010).

``ResolutionCache`` closes that gap. ``ResolutionKey`` identifies a dataset
*request* -- provider, dataset ID, config, split -- deliberately omitting
``revision`` (resolving is the step that produces one), so it is addressable
*before* a resolution exists. ``DatasetCatalog.resolve`` writes the resolved
identity here on every successful (online, or already-network-free) resolve,
and consults it first when ``offline=True`` is requested against a
network-requiring provider, before ever raising :class:`OfflineCacheMiss`.

Follows the same atomic-by-convention publication ``DatasetCache`` uses
(ADR-0004): a temp file is written, flushed, ``fsync``'d, and published with
``Path.replace()`` while holding a lock scoped to the entry's digest, and a
recorded checksum lets ``read()`` detect a corrupted or partially-applied
entry instead of ever returning untrustworthy bytes as if they were valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from agentic_evalkit.errors import DatasetIntegrityError, OfflineCacheMiss
from agentic_evalkit.models import ResolvedDataset
from agentic_evalkit.models.base import FrozenModel

if TYPE_CHECKING:
    from pathlib import Path

_RESOLVED_FILENAME = "resolved.json"
_DIGEST_PREFIX = "sha256:"

# Per-key write locks, keyed by digest -- mirrors
# ``agentic_evalkit.datasets.cache``'s own registry exactly, but kept private
# to this module so the two caches' locking never interact.
_registry_lock = threading.Lock()
_key_locks: dict[str, threading.Lock] = {}


def _lock_for_digest(digest: str) -> threading.Lock:
    with _registry_lock:
        lock = _key_locks.get(digest)
        if lock is None:
            lock = threading.Lock()
            _key_locks[digest] = lock
        return lock


class ResolutionKey(FrozenModel):
    """Identifies one provider's pre-resolution dataset request (ADR-0011).

    Keys on the fields of a :class:`~agentic_evalkit.models.DatasetRef` that
    are *known before* a resolution exists -- ``provider``, ``dataset_id``,
    ``config``, ``split``, and the *requested* ``revision``. Unlike
    :class:`~agentic_evalkit.datasets.cache.CacheKey`'s ``revision`` (which is
    the immutable revision a resolution *produced*), this ``revision`` is the
    caller's request-side pin:

    - ``None`` means "latest at resolution time" (design §5.1): every
      unpinned request for the same dataset shares one cache slot, so an
      offline resolve returns the most recent "latest" that was resolved
      online.
    - A concrete value means the caller pinned an exact revision; each
      distinct pin gets its own slot, so resolving ``ref@revA`` then
      ``ref@revB`` online (same dataset, different pins) caches *both* and an
      offline ``resolve(ref@revA)`` can never be silently served ``revB``'s
      resolution. Omitting ``revision`` from this key -- the state before the
      2026-07-09 fix -- collapsed both pins into one slot and let the second
      online resolve overwrite the first, a correctness (not merely
      efficiency) defect against the manifest-reproducibility invariant.
    """

    provider: str
    dataset_id: str
    config: str | None = None
    split: str | None = None
    revision: str | None = None

    def digest(self) -> str:
        """Return ``"sha256:" + hexdigest`` of this key's canonical JSON.

        Same canonicalization as ``CacheKey.digest()``: sorted keys, compact
        separators, so the digest depends only on field values.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return _DIGEST_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _checksum(canonical_payload: str) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


class ResolutionCache:
    """A directory-backed cache of the most recent resolution for each key.

    Layout: ``root/<digest[7:9]>/<digest>/resolved.json``, matching
    ``DatasetCache``'s fan-out convention. Each entry records the resolved
    dataset payload, a checksum over its canonical JSON, the originating key
    (so a hash collision or on-disk tamper is detectable, not just a byte
    mismatch), and a creation timestamp.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def _entry_dir(self, key: ResolutionKey) -> Path:
        digest = key.digest()
        hex_digest = digest.removeprefix(_DIGEST_PREFIX)
        return self._root / hex_digest[:2] / hex_digest

    def _entry_path(self, key: ResolutionKey) -> Path:
        return self._entry_dir(key) / _RESOLVED_FILENAME

    def write(self, key: ResolutionKey, dataset: ResolvedDataset) -> None:
        """Publish ``dataset`` as the current resolution for ``key``.

        Replaces any prior entry for the same key. Concurrent writers to the
        *same* key are serialized by a per-digest lock (writers to different
        keys never contend), and the payload is published via
        temp-write-then-``Path.replace()`` so a reader never observes a
        half-written file.
        """
        entry_dir = self._entry_dir(key)
        entry_dir.mkdir(parents=True, exist_ok=True)
        digest = key.digest()

        resolved_payload = dataset.model_dump(mode="json")
        record = {
            "key": key.model_dump(mode="json"),
            "resolved_dataset": resolved_payload,
            "checksum": _checksum(_canonical_json(resolved_payload)),
            "created_at": datetime.now(UTC).isoformat(),
            "digest": digest,
        }
        record_bytes = json.dumps(record, sort_keys=True, indent=2).encode("utf-8")
        entry_path = self._entry_path(key)

        with _lock_for_digest(digest):
            tmp_name = f".{_RESOLVED_FILENAME}.{os.getpid()}.{threading.get_ident()}.tmp"
            tmp_path = entry_dir / tmp_name
            with open(tmp_path, "wb") as handle:
                handle.write(record_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            tmp_path.replace(entry_path)

    def read(self, key: ResolutionKey) -> ResolvedDataset:
        """Return the verified, previously-resolved dataset for ``key``.

        Raises:
            OfflineCacheMiss: no entry exists for this exact key yet
                (``retryable=True`` -- an online resolve, or ``datasets
                pull``, for this exact request would populate it).
            DatasetIntegrityError: an entry exists but its recorded key does
                not match ``key``, its payload is missing/malformed, or its
                checksum does not match -- never silently treated as a miss.
        """
        entry_path = self._entry_path(key)
        digest = key.digest()

        if not entry_path.exists():
            raise OfflineCacheMiss(
                message=f"no cached resolution for digest {digest}",
                context={"digest": digest, "dataset_id": key.dataset_id},
                retryable=True,
            )

        try:
            record = json.loads(entry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise DatasetIntegrityError(
                message=f"cached resolution for digest {digest} is not valid JSON",
                context={"digest": digest, "dataset_id": key.dataset_id},
            ) from error

        if not isinstance(record, dict):
            raise DatasetIntegrityError(
                message=f"cached resolution for digest {digest} is not a JSON object",
                context={"digest": digest, "dataset_id": key.dataset_id},
            )

        recorded_key = record.get("key")
        if recorded_key != key.model_dump(mode="json"):
            raise DatasetIntegrityError(
                message=f"cached resolution for digest {digest} does not match the requested key",
                context={"digest": digest, "dataset_id": key.dataset_id},
            )

        resolved_payload = record.get("resolved_dataset")
        if not isinstance(resolved_payload, dict):
            raise DatasetIntegrityError(
                message=f"cached resolution for digest {digest} has no resolved_dataset payload",
                context={"digest": digest, "dataset_id": key.dataset_id},
            )

        expected_checksum = record.get("checksum")
        actual_checksum = _checksum(_canonical_json(resolved_payload))
        if expected_checksum != actual_checksum:
            raise DatasetIntegrityError(
                message=f"cached resolution for digest {digest} failed checksum validation",
                context={"digest": digest, "dataset_id": key.dataset_id},
            )

        try:
            return ResolvedDataset.model_validate(resolved_payload)
        except ValueError as error:
            raise DatasetIntegrityError(
                message=f"cached resolution for digest {digest} failed model validation",
                context={"digest": digest, "dataset_id": key.dataset_id},
            ) from error


__all__ = ["ResolutionCache", "ResolutionKey"]
