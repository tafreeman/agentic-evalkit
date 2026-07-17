"""Stores large run outputs and logs on disk, keyed by the hash of their contents.

Plan Task 11, Step 1.

``ArtifactStore`` saves immutable chunks of data ("blobs") -- things like a
large execution output, a target's logs, or other evidence gathered during a
run -- to disk, instead of keeping them in memory as part of the run result.
Each blob is named after the SHA-256 hash of its own bytes (this pattern is
called "content-addressed storage"). One consequence: if you store the same
content twice, you get back the same reference both times, and it's only
saved to disk once -- no duplicate storage. Alongside each blob, a small JSON
"sidecar" file records its media type, size in bytes, when it was created,
and whether it was redacted -- so something rendering a report can describe
an artifact without having to open and read the (possibly huge) blob itself.

Writes are atomic, meaning a reader can never see a half-written file: the
data is first written to a temporary file in the same directory, flushed to
disk, and only then renamed into its final location with
:meth:`Path.replace` (a rename is a single, all-or-nothing filesystem
operation). Each store also enforces a maximum blob size, so that a target
or harness that misbehaves and tries to write something huge can't fill up
the disk with one single artifact.
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
    """Points at one stored artifact, without exposing how the store is organized internally.

    The ``digest`` field (and therefore equality between two ``ArtifactRef``
    instances) is just the SHA-256 hash of the artifact's bytes. That means
    two references built from identical content are always equal to each
    other, no matter how many separate times :meth:`ArtifactStore.put_bytes`
    was called to store that content.
    """

    digest: str
    media_type: str
    byte_count: int


class ArtifactMetadata(FrozenModel):
    """The small companion record ("sidecar") stored alongside every artifact's actual data."""

    digest: str
    media_type: str
    byte_count: int
    created_at: datetime
    redacted: bool = False


def _digest_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class ArtifactStore:
    """Saves immutable blobs to disk under a root directory, named by the hash of their content.

    Args:
        root: The directory the store writes payload and sidecar files
            into. Created automatically (including any missing parent
            directories) if it doesn't already exist.
        max_bytes: The largest payload this store will accept. Defaults to
            16 MiB. Calling ``put_bytes`` with anything larger raises
            :class:`ArtifactStoreLimitExceeded` immediately, before writing
            any data.
    """

    def __init__(self, root: Path, *, max_bytes: int = _DEFAULT_MAX_BYTES) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes

    def put_bytes(self, data: bytes, *, media_type: str, redacted: bool = False) -> ArtifactRef:
        """Save ``data`` to the store and return a reference to it.

        If you store the exact same bytes twice -- even passing a different
        ``media_type`` or ``redacted`` value the second time -- you get back
        a reference with the same digest both times, reusing the payload and
        sidecar file the first call already wrote. The store never
        overwrites an existing entry for a digest it has already seen.
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
