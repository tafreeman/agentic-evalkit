"""Resolution-identity cache: remembers the last successful "resolve" result
for a dataset request, per provider (ADR-0011).

Some background: "resolving" a dataset request means pinning down its exact
version (see ``ResolutionKey`` below for more on what that involves). Once
a dataset has been resolved, :class:`~agentic_evalkit.datasets.cache.CacheKey`
and ``DatasetCache`` (ADR-0004) can cache *pages* of its data -- but only
because, by that point, we already know the dataset's ``revision`` (its
exact, pinned version identifier), and a page cache key requires one. That
still leaves the resolving step itself with nothing to fall back on:
before this module existed, calling ``DatasetCatalog.resolve`` against a
provider that needs network access (like ``huggingface``) while
``offline=True`` was set would always fail with an error -- even if that
exact dataset had just been resolved successfully moments earlier while
online (ADR-0010).

``ResolutionCache`` closes that gap. Its key type, ``ResolutionKey``,
identifies a dataset *request* -- provider, dataset ID, config, split --
and deliberately leaves out ``revision``, since resolving is precisely the
step that produces a revision; that omission is what lets a request be
looked up in this cache *before* it has ever been resolved. Every time
``DatasetCatalog.resolve`` succeeds (whether because it went online, or
because the provider never needed the network in the first place), it
saves the resulting identity here. Later, when an ``offline=True`` resolve
is requested against a provider that would normally need the network, this
cache is checked first, before giving up and raising
:class:`OfflineCacheMiss`.

This cache saves data the same crash-safe way ``DatasetCache`` does
(ADR-0004): write to a temporary file, flush it and call ``fsync`` (which
forces the operating system to actually write the data to disk instead of
leaving it buffered in memory), then publish it under its real filename
using ``Path.replace()`` -- all while holding a lock specific to that one
cache entry (identified by its "digest", a short hash -- see below) so two
writers can never collide. A checksum (a hash of the saved content) is
stored alongside it, so ``read()`` can detect a corrupted or half-written
entry instead of trusting bad data as if it were valid.
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

# One write-lock per cache entry, keyed by the entry's digest (hash). This
# mirrors the same locking pattern used in ``agentic_evalkit.datasets.cache``
# exactly, but this copy is kept private to this module so that locking in
# the two caches never interacts.
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
    """Identifies one dataset request from a specific provider, before it
    has been resolved (ADR-0011).

    "Resolving" here means: given a request for a dataset, pin down its
    exact version. This key is built from the fields of a
    :class:`~agentic_evalkit.models.DatasetRef` that are already known
    *before* that resolving happens -- ``provider``, ``dataset_id``,
    ``config``, ``split``, and the *requested* ``revision`` (i.e. whatever
    version the caller asked for, which might not be pinned to anything
    specific yet). Contrast this with
    :class:`~agentic_evalkit.datasets.cache.CacheKey`, whose ``revision``
    field holds the exact, immutable version that a resolution already
    *produced*. Here, ``revision`` instead reflects what the caller
    originally asked for, and it can mean one of two different things:

    - ``None`` means "give me whatever is latest, at the time you resolve
      this" (design §5.1). Every request like this for the same dataset
      shares a single cache slot, so when an offline resolve is requested,
      it gets back whatever "latest" version was most recently resolved
      online.
    - A concrete value means the caller asked for one exact, specific
      version. Each distinct version requested gets its own separate cache
      slot. This matters because it keeps different pinned versions of the
      same dataset from being confused with each other: if ``ref@revA`` is
      resolved online, and later ``ref@revB`` is also resolved online (same
      dataset, different pinned version), both get cached separately and
      safely. Before a 2026-07-09 fix, this key left ``revision`` out
      entirely, so both of those requests collapsed into the *same* cache
      slot -- meaning the second online resolve would silently overwrite
      the first one's cached result. That was a correctness bug, not just a
      wasted-cache-space issue: it broke the guarantee that resolving the
      same pinned dataset request should always reproducibly give back the
      same result (the manifest-reproducibility invariant), because an
      offline resolve of ``ref@revA`` could then come back with ``revB``'s
      data instead.
    """

    provider: str
    dataset_id: str
    config: str | None = None
    split: str | None = None
    revision: str | None = None

    def digest(self) -> str:
        """Return a short, unique fingerprint for this key: the text
        ``"sha256:"`` followed by the SHA-256 hash of this key's fields,
        written out as JSON.

        Uses the same standardized ("canonical") JSON format as
        ``CacheKey.digest()``: keys sorted alphabetically, no extra
        whitespace -- so the fingerprint depends only on the actual field
        values, never on the order fields happen to be listed in or
        incidental spacing.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return _DIGEST_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _checksum(canonical_payload: str) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


class ResolutionCache:
    """A directory-backed cache holding the most recent resolution for each
    key.

    On disk, each entry lives at ``root/<xx>/<digest>/resolved.json``, where
    ``digest`` is the key's full fingerprint (see ``ResolutionKey.digest()``)
    and ``<xx>`` is just the first two hex characters of that fingerprint
    (right after its ``"sha256:"`` prefix), used as a subdirectory so that
    entries get spread across many small directories instead of piling
    thousands of files into one. This matches the same directory layout
    convention ``DatasetCache`` uses. Each entry stores: the resolved
    dataset itself, a checksum (a hash) computed over its standardized JSON
    form, the original key that produced it (so that if two different keys
    ever happened to produce the same fingerprint, or if the file was
    tampered with on disk, that's detectable -- not just a case of the raw
    bytes looking wrong), and a timestamp of when the entry was created.
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
        """Save ``dataset`` as the current resolution result for ``key``.

        Overwrites any entry that was previously saved for this same key.
        If multiple threads try to write the *same* key at the same time, a
        lock specific to that key's digest (fingerprint) makes them wait
        their turn one at a time -- but writes to *different* keys never
        block each other. The data is saved by first writing it to a
        temporary file and then swapping it into place with
        ``Path.replace()``, so a reader can never catch it in a
        half-written state.
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
        """Return the previously-resolved dataset saved under ``key``, after
        verifying it is intact.

        Raises:
            OfflineCacheMiss: no entry has been saved for this exact key
                yet (``retryable=True`` signals that this isn't a permanent
                failure -- resolving this exact request online, or running
                ``datasets pull``, would create the missing entry).
            DatasetIntegrityError: an entry exists, but something about it
                is wrong -- its saved key doesn't match ``key``, its
                payload is missing or malformed, or its checksum doesn't
                match. This is never silently treated as if the entry were
                simply missing, since that would hide a real problem.
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
