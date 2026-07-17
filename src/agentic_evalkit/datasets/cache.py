"""A cache for dataset contents, looked up by a hash of "what request
produced this data" rather than by a filename (ADR-0004, design §6.3).

("Content-addressed" means: instead of naming a cache entry after a file or
a dataset, we name it after a hash of everything that determines its
content, so the same request always maps to the same cache entry, and a
different request always maps to a different one.)

``CacheKey`` collects every piece of information that affects exactly which
bytes a cached page (or a full dataset) contains: the provider, the
dataset's canonical ID, its immutable ``revision`` (an exact, pinned
version identifier), its config and split, the offset/limit of the page
being requested, optional hashes representing any projection/filter/
data-file settings in play, and a record-type marker. All of that is
boiled down into one short, stable fingerprint: a SHA-256 hash computed
over a standardized ("canonical") JSON form of these fields. ``DatasetCache``
then stores, for each fingerprint, both the raw payload bytes and a
"manifest" -- a small JSON file recording a checksum of the payload, its
byte count, when it was created, and the key's own JSON. Both files are
published using the same safe pattern: write to a temporary file first,
then swap it into place with ``Path.replace()``, all while holding a lock
specific to that one key. That way, two writers racing to write the same
key can never leave behind a mismatched pair where the payload is from one
write and the manifest is from another.

``Path.replace()`` is guaranteed atomic (meaning it either fully happens or
doesn't happen at all, with no in-between state ever visible to a reader)
on POSIX filesystems (Linux/Mac), but on Windows that guarantee depends on
the specific filesystem in use and isn't certain in every configuration
(for example, on some network or mounted drives). Because of that, this
module does not depend on ``Path.replace()`` alone to guarantee
correctness: every call to ``read()`` double-checks the manifest's
recorded key, the payload's byte count, and the payload's checksum before
handing back any bytes. So if a replace operation was ever interrupted
partway through, the result is a clear, typed ``DatasetIntegrityError`` --
never silently corrupt or mismatched data returned as if it were fine. See
``docs/adr/0004-content-addressed-dataset-cache.md`` for the full design
decision.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from agentic_evalkit.errors import DatasetIntegrityError, OfflineCacheMiss
from agentic_evalkit.models.base import FrozenModel

if TYPE_CHECKING:
    from pathlib import Path

_PAYLOAD_FILENAME = "payload.bin"
_MANIFEST_FILENAME = "manifest.json"
_DIGEST_PREFIX = "sha256:"

# One write-lock per cache key (indexed by the key's digest/fingerprint). A
# single, separate lock guards the *creation* of these per-key locks: if two
# threads both try to write a *new* key (one with no lock yet) at the same
# moment, this outer lock stops them from each creating their own separate
# lock object, which would mean the two threads wouldn't actually be
# synchronized with each other. Once a per-key lock exists, it's the one
# that makes writers to that specific key wait their turn.
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
    """Identifies one cached page of data, or one full-dataset payload
    (design §6.3).

    Any two keys with exactly the same field values will always hash to the
    same fingerprint (digest), and therefore point at the same cache entry.
    Conversely, if even one field differs -- including the optional
    projection/filter/data-file hashes and the ``record_type`` field -- the
    resulting fingerprint changes, and so does the cache entry it points to.
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
        """Return this key's fingerprint: the text ``"sha256:"`` followed by
        the SHA-256 hash of this key's fields written out as JSON.

        The JSON is written in a standardized ("canonical") form -- UTF-8
        text, keys sorted alphabetically, no extra whitespace
        (``json.dumps(..., sort_keys=True, separators=(",", ":"))``) -- so
        the resulting fingerprint depends only on the actual field values,
        never on the order the fields happen to be declared in or on
        incidental spacing.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return _DIGEST_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _checksum(payload: bytes) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(payload).hexdigest()


class DatasetCache:
    """A cache of dataset payloads, stored as files on disk and looked up by
    content fingerprint rather than by name.

    On disk, each entry lives at
    ``root/<xx>/<digest>/{payload.bin,manifest.json}``, where ``digest`` is
    the value returned by ``CacheKey.digest()`` (the ``"sha256:"`` prefix is
    kept when the digest is written into the manifest file, but stripped
    off when the digest is used as part of a directory/file path) and
    ``<xx>`` is just its first two hex characters, used as a subdirectory
    so that entries are spread out across many small directories instead of
    all piling into one giant directory.
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
        """Save ``payload`` under ``key``, replacing any entry that was
        there before.

        Writes the payload and its manifest to temporary files in the
        entry's directory first, flushes each one and calls ``fsync``
        (which forces the operating system to actually write the data to
        disk rather than just holding it in memory), then publishes both
        files at once using ``Path.replace()`` -- all while holding a lock
        specific to this key's digest (fingerprint). If multiple threads
        try to write to the *same* key at once, this lock makes them go one
        at a time, so the entry directory always holds either the complete
        old entry or the complete new one, never a mix of the two; writers
        to *different* keys, meanwhile, never block each other.
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
            # Publish the payload file before the manifest file, on purpose:
            # this way, if a reader shows up in the middle of this
            # operation, there are only two possible outcomes, and both are
            # safe. Either it sees the new manifest but an old/missing
            # payload -- which fails the checksum or byte-count check below,
            # so the bad read is caught -- or it sees no manifest yet at
            # all, which it correctly reports as "not cached yet" rather
            # than "corrupted".
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
        """Return the payload bytes saved under ``key``, after verifying
        they are intact.

        Raises:
            OfflineCacheMiss: neither a manifest nor a payload file exists
                for this exact key (this method never falls back to a
                partial or approximate match -- it's this exact key or
                nothing).
            DatasetIntegrityError: a manifest file exists, but something
                about it doesn't check out -- its recorded key doesn't
                match ``key``, or the payload's on-disk byte count or
                checksum doesn't match what the manifest says it should be.
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
