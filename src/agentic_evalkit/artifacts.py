"""Content-addressed artifact store for run outputs and logs (plan Task 11, Step 1).

``ArtifactStore`` persists immutable blobs (large execution outputs, target
logs, harness evidence) outside the in-memory run result, keyed by the
SHA-256 digest of their bytes. Two writes of identical content therefore
always resolve to the same reference and only occupy storage once. Every
blob has a JSON sidecar recording its media type, byte count, creation
time, and redaction status, so a report renderer can describe an artifact
without reading its payload.

Writes are atomic: content is written to a temporary file in the same
directory as its final destination, flushed, and then moved into place with
:meth:`Path.replace`, so a reader never observes a partially written blob.
Each store enforces a configured maximum blob size so a runaway target or
harness cannot exhaust disk by way of a single artifact.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from agentic_evalkit.models.base import FrozenModel

#: Default cap on a single artifact's byte size (16 MiB) when the caller does
#: not override ``ArtifactStore(..., max_bytes=...)``.
_DEFAULT_MAX_BYTES: Final[int] = 16 * 1024 * 1024


class ArtifactStoreLimitExceeded(ValueError):
    """Raised when a payload exceeds the store's configured maximum byte size."""


class ArtifactRef(FrozenModel):
    """An opaque, content-addressed reference to a stored artifact.

    Equality and the reported ``digest`` are the SHA-256 content hash, so two
    references built from identical bytes always compare equal regardless of
    how many times the caller called :meth:`ArtifactStore.put_bytes`.
    """

    digest: str
    media_type: str
    byte_count: int


class ArtifactMetadata(FrozenModel):
    """The sidecar record stored alongside every artifact payload."""

    digest: str
    media_type: str
    byte_count: int
    created_at: datetime
    redacted: bool = False


def _digest_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class ArtifactStore:
    """Persists immutable, content-addressed blobs under a root directory.

    Args:
        root: Directory the store writes payload and sidecar files under.
            Created (including parents) if it does not already exist.
        max_bytes: Largest payload this store accepts. Defaults to 16 MiB.
            A ``put_bytes`` call for a larger payload raises
            :class:`ArtifactStoreLimitExceeded` before anything is written.
    """

    def __init__(self, root: Path, *, max_bytes: int = _DEFAULT_MAX_BYTES) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes

    def put_bytes(self, data: bytes, *, media_type: str, redacted: bool = False) -> ArtifactRef:
        """Store ``data`` and return a content-addressed reference to it.

        Storing identical bytes twice (even with a different ``media_type``
        or ``redacted`` flag on the second call) returns a reference with the
        same digest and reuses the existing payload and sidecar written by
        the first call; the store never rewrites an existing digest.
        """
        if len(data) > self._max_bytes:
            raise ArtifactStoreLimitExceeded(
                f"artifact payload of {len(data)} bytes exceeds the "
                f"configured maximum of {self._max_bytes} bytes"
            )
        digest = _digest_of(data)
        payload_path = self._payload_path(digest)
        if not payload_path.exists():
            metadata = ArtifactMetadata(
                digest=digest,
                media_type=media_type,
                byte_count=len(data),
                created_at=datetime.now(UTC),
                redacted=redacted,
            )
            self._write_atomic(payload_path, data)
            self._write_atomic(
                self._metadata_path(digest),
                metadata.model_dump_json().encode("utf-8"),
            )
        return ArtifactRef(digest=digest, media_type=media_type, byte_count=len(data))

    def read(self, ref: ArtifactRef) -> bytes:
        """Return the stored bytes for ``ref``."""
        return self._payload_path(ref.digest).read_bytes()

    def metadata(self, ref: ArtifactRef) -> ArtifactMetadata:
        """Return the sidecar metadata recorded when ``ref`` was written."""
        raw = self._metadata_path(ref.digest).read_bytes()
        return ArtifactMetadata.model_validate_json(raw)

    def _payload_path(self, digest: str) -> Path:
        return self._root / f"{_digest_filename(digest)}.bin"

    def _metadata_path(self, digest: str) -> Path:
        return self._root / f"{_digest_filename(digest)}.json"

    def _write_atomic(self, destination: Path, data: bytes) -> None:
        """Write ``data`` to a temp file in the same directory, then replace atomically."""
        fd, temp_name = tempfile.mkstemp(dir=destination.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            Path(temp_name).replace(destination)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise


def _digest_filename(digest: str) -> str:
    """Strip the ``sha256:`` prefix so the digest is a safe bare filename."""
    return digest.removeprefix("sha256:")


__all__ = [
    "ArtifactMetadata",
    "ArtifactRef",
    "ArtifactStore",
    "ArtifactStoreLimitExceeded",
]
