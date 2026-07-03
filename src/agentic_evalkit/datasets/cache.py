"""Content-addressed, checksum-verified dataset cache (ADR-0004, design §6.3).

``CacheKey`` captures every field that changes which bytes a page or full
dataset resolves to (provider, canonical ID, immutable revision, config,
split, offset/limit, and optional projection/filter/data-file digests plus
record type) and reduces them to a single stable SHA-256 digest over
canonical JSON. ``DatasetCache`` stores the payload and a manifest (checksum,
byte count, creation time, and the key JSON) for each digest, publishing both
via a temp-file-then-``Path.replace()`` sequence under a per-key lock so
concurrent writers to the same key never leave a torn payload/manifest pair
readable.

``Path.replace()`` is atomic on POSIX filesystems but its atomicity on
Windows depends on the local filesystem and is not guaranteed across all
configurations (e.g. some network/mounted drives). This module does not rely
on replace-atomicity alone for correctness: every ``read()`` verifies the
manifest's key identity, byte count, and payload checksum before returning
bytes, so a partially-applied replace is surfaced as a typed
``DatasetIntegrityError`` rather than silently returning corrupt or
mismatched data. See ``docs/adr/0004-content-addressed-dataset-cache.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from agentic_evalkit.errors import DatasetIntegrityError, OfflineCacheMiss
from agentic_evalkit.models.base import FrozenModel

_PAYLOAD_FILENAME = "payload.bin"
_MANIFEST_FILENAME = "manifest.json"
_DIGEST_PREFIX = "sha256:"

# Per-key write locks, keyed by digest. A single registry-level lock guards
# creation of new per-key locks so two threads racing to write *different*
# keys for the first time cannot both "win" the get-or-create and end up
# sharing (or losing) a lock object; the per-key lock itself then serializes
# writers to that one key.
_registry_lock = threading.Lock()
_key_locks: dict[str, threading.Lock] = {}


def _lock_for_digest(digest: str) -> threading.Lock:
    with _registry_lock:
        lock = _key_locks.get(digest)
        if lock is None:
            lock = threading.Lock()
            _key_locks[digest] = lock
        return lock


class CacheKey(FrozenModel):
    """Identifies one cached page or full-dataset payload (design §6.3).

    Two keys with equal field values always hash to the same digest and
    therefore address the same cache entry; any differing field (including
    the optional projection/filter/data-file digests and ``record_type``)
    changes the digest and therefore the entry.
    """

    provider: str
    dataset_id: str
    revision: str
    config: str | None = None
    split: str | None = None
    offset: int
    limit: int
    projection_digest: str | None = None
    filter_digest: str | None = None
    data_files_digest: str | None = None
    record_type: Literal["page", "full"] = "page"

    def digest(self) -> str:
        """Return ``"sha256:" + hexdigest`` of this key's canonical JSON.

        Canonical form is UTF-8 JSON with sorted keys and compact separators
        (``json.dumps(..., sort_keys=True, separators=(",", ":"))``), so the
        digest depends only on field values, never on field-declaration
        order or incidental whitespace.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return _DIGEST_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _checksum(payload: bytes) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(payload).hexdigest()


class DatasetCache:
    """A directory-backed, content-addressed cache of dataset payloads.

    Layout: ``root/<digest[7:9]>/<digest>/{payload.bin,manifest.json}``,
    where ``digest`` is ``CacheKey.digest()`` (the ``"sha256:"`` prefix is
    kept in the manifest and digest value but stripped from path segments)
    and the two leading hex characters after the prefix form a fan-out
    directory so no single directory accumulates every entry.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def _entry_dir(self, key: CacheKey) -> Path:
        digest = key.digest()
        hex_digest = digest.removeprefix(_DIGEST_PREFIX)
        return self._root / hex_digest[:2] / hex_digest

    def payload_path(self, key: CacheKey) -> Path:
        """Path to the payload file for ``key`` (may not exist yet)."""
        return self._entry_dir(key) / _PAYLOAD_FILENAME

    def manifest_path(self, key: CacheKey) -> Path:
        """Path to the manifest file for ``key`` (may not exist yet)."""
        return self._entry_dir(key) / _MANIFEST_FILENAME

    def write(self, key: CacheKey, payload: bytes) -> None:
        """Publish ``payload`` for ``key``, replacing any prior entry.

        Writes payload and manifest to temporary files in the entry
        directory, flushes and ``fsync``s each, then atomically publishes
        both with ``Path.replace()`` while holding a lock scoped to this
        key's digest. Concurrent writers to the *same* key are serialized by
        that lock, so the entry directory always contains either the
        previous complete entry or the new one, never a mix; writers to
        *different* keys never block each other.
        """
        entry_dir = self._entry_dir(key)
        entry_dir.mkdir(parents=True, exist_ok=True)
        digest = key.digest()

        manifest = {
            "checksum": _checksum(payload),
            "byte_count": len(payload),
            "created_at": datetime.now(UTC).isoformat(),
            "key": key.model_dump(mode="json"),
            "digest": digest,
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")

        payload_path = entry_dir / _PAYLOAD_FILENAME
        manifest_path = entry_dir / _MANIFEST_FILENAME

        with _lock_for_digest(digest):
            tmp_payload = self._write_temp(entry_dir, "payload", payload)
            tmp_manifest = self._write_temp(entry_dir, "manifest", manifest_bytes)
            # Publish payload before manifest: a reader that observes a
            # manifest but a missing/old payload always fails a checksum or
            # byte-count check rather than reading half-written bytes as
            # valid, and a reader that observes no manifest yet reports the
            # entry as an offline miss rather than as corrupt.
            tmp_payload.replace(payload_path)
            tmp_manifest.replace(manifest_path)

    @staticmethod
    def _write_temp(directory: Path, label: str, data: bytes) -> Path:
        tmp_path = directory / f".{label}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp_path, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return tmp_path

    def read(self, key: CacheKey) -> bytes:
        """Return the verified payload bytes for ``key``.

        Raises:
            OfflineCacheMiss: no manifest or payload exists for this exact
                key (no partial-match fallback is ever attempted).
            DatasetIntegrityError: a manifest exists but its recorded key
                does not match ``key``, or the on-disk payload's byte count
                or checksum does not match the manifest.
        """
        manifest_path = self.manifest_path(key)
        payload_path = self.payload_path(key)
        digest = key.digest()

        if not manifest_path.exists() or not payload_path.exists():
            raise OfflineCacheMiss(
                message=f"no cache entry for digest {digest}",
                context={"digest": digest, "dataset_id": key.dataset_id},
            )

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise DatasetIntegrityError(
                message=f"cache manifest for digest {digest} is not valid JSON",
                context={"digest": digest, "dataset_id": key.dataset_id},
            ) from error

        recorded_key = manifest.get("key")
        if recorded_key != key.model_dump(mode="json"):
            raise DatasetIntegrityError(
                message=f"cache manifest for digest {digest} does not match the requested key",
                context={"digest": digest, "dataset_id": key.dataset_id},
            )

        payload = payload_path.read_bytes()

        expected_byte_count = manifest.get("byte_count")
        if expected_byte_count != len(payload):
            raise DatasetIntegrityError(
                message=(
                    f"cache payload for digest {digest} has {len(payload)} bytes, "
                    f"expected {expected_byte_count}"
                ),
                context={"digest": digest, "dataset_id": key.dataset_id},
            )

        expected_checksum = manifest.get("checksum")
        actual_checksum = _checksum(payload)
        if expected_checksum != actual_checksum:
            raise DatasetIntegrityError(
                message=f"cache payload for digest {digest} failed checksum validation",
                context={"digest": digest, "dataset_id": key.dataset_id},
            )

        return payload
