"""Process-local locking and staged writes, independent of cache formats."""

from __future__ import annotations

import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class KeyedLockPool:
    """Keep each key's lock for the pool's lifetime, including waiting users."""

    def __init__(self) -> None:
        self._registry_lock = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def lock_for(self, digest: str) -> threading.Lock:
        """Return one stable lock per digest within this process."""
        with self._registry_lock:
            lock = self._locks.get(digest)
            if lock is None:
                lock = threading.Lock()
                self._locks[digest] = lock
            return lock


@contextmanager
def staged_bytes(destination: Path, payload: bytes) -> Iterator[Path]:
    """Stage flushed bytes beside destination; the caller owns publication.

    Directories must already exist. Cleanup failures propagate on success,
    or annotate an existing failure without replacing it. This does not
    make multi-file publication atomic or provide cross-process locking.
    """
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        yield temporary
    except BaseException as error:
        try:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            error.add_note(f"Could not remove staged cache file {temporary}: {cleanup_error}")
        raise
    else:
        temporary.unlink(missing_ok=True)
