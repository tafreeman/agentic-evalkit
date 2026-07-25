"""Reading a cache entry off disk while another process may be republishing it.

Both on-disk caches in this package -- :mod:`agentic_evalkit.datasets.cache`
(dataset pages, ADR-0004) and :mod:`agentic_evalkit.datasets.resolution_cache`
(resolution identities, ADR-0011) -- publish entries the same way: write a
temporary file, then swap it into place with ``Path.replace()``. Readers check
that the file exists and then open it.

Those two steps are not one indivisible action, and a writer can land in
between them. On POSIX (Linux/macOS) that gap is harmless: ``rename()`` swaps
the directory entry underneath an open handle and the reader keeps reading the
bytes it already had. On Windows it is not harmless -- ``Path.replace()``
needs exclusive access to the destination, so a reader arriving mid-swap gets
a *sharing violation*: ``PermissionError`` (``WinError 5``/``WinError 32``).
That is an operating-system-level scheduling accident, not a statement about
the cache's contents, and before this module existed it escaped
``DatasetCache.read()`` raw -- past
:class:`~agentic_evalkit.errors.OfflineCacheMiss` and
:class:`~agentic_evalkit.errors.DatasetIntegrityError`, past the catalog
(which catches only ``OfflineCacheMiss``), and out to the CLI as an unhandled
traceback.

:func:`read_entry_bytes` closes that gap, and it is deliberately narrow about
how. It retries exactly two exception types, because exactly two of them are
what "a writer swapped this file underneath me" looks like:

``PermissionError``
    The sharing violation described above. Transient by construction: the
    writer holds the destination for the duration of one rename.

``FileNotFoundError``
    The file was there when it was checked and gone when it was opened. Also a
    momentary artifact of republication (or of an entry being evicted).

Every other :class:`OSError` is deliberately left alone and allowed to
propagate. ``PermissionError`` and ``FileNotFoundError`` are both *subclasses*
of ``OSError``, so it would be a single-word change to catch the parent
instead -- and that change would be a bug. ``IsADirectoryError`` means the
cache layout on disk is structurally wrong; ``OSError(EIO)`` means the disk is
failing; ``OSError(ENOSPC)`` means the volume is full. Retrying those
accomplishes nothing, and folding them into "cache miss" would convert a loud,
diagnosable failure into a silent refetch that quietly hides a broken machine.
Losing a real error is worse than the crash this module exists to prevent, so
this module handles the race and nothing else.

Retries are bounded (:data:`READ_ATTEMPTS` attempts total, with a short
doubling delay in between, so the worst case adds tens of milliseconds and
then stops). If a retryable-looking error survives every attempt, it was not
the race after all -- a rename does not last that long -- so it is reported as
the real condition it now appears to be, using the typed errors callers
already handle:

- a persistently missing file becomes ``OfflineCacheMiss`` (the entry really
  is absent, which is exactly what the existence check models, so the caller's
  "fetch it and try again" response is the correct one);
- a persistently unreadable file becomes ``DatasetIntegrityError`` (the entry
  exists but cannot be used -- a genuine permissions misconfiguration, say --
  which is *not* a cache miss and must not be reported as one, or the caller
  would silently refetch and never learn its cache is broken).

No new exception type is introduced: a type the catalog does not catch would
merely relocate the traceback rather than remove it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from agentic_evalkit.errors import DatasetIntegrityError, OfflineCacheMiss

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

#: How many times to try opening a cache entry before concluding that the
#: failure is not a passing race. A ``Path.replace()`` holds its destination
#: for well under a millisecond, so a retry almost always succeeds on the
#: second attempt; the remaining attempts cover a heavily contended cache
#: root, where several writers may republish the same key back to back.
READ_ATTEMPTS = 6

#: Delay before the first retry, doubling on each subsequent one. With
#: ``READ_ATTEMPTS`` attempts the accumulated wait is bounded at
#: 1 + 2 + 4 + 8 + 16 ms, so a read that is genuinely broken (rather than
#: merely contended) fails fast instead of stalling a run.
INITIAL_RETRY_DELAY_SECONDS = 0.001


def read_entry_bytes(
    path: Path,
    *,
    entry_label: str,
    digest: str,
    dataset_id: str,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Return the bytes of the cache entry at ``path``, retrying past a racing writer.

    Args:
        path: The cache file to read. Its existence has typically already been
            checked by the caller; this function covers the window between
            that check and the read actually landing.
        entry_label: How to name this file in an error message, in the
            caller's own vocabulary (e.g. ``"cache manifest"``,
            ``"cached resolution"``), so the raised error reads the same as
            the caller's other errors about the same entry.
        digest: The entry's fingerprint, for the error message and context.
        dataset_id: The dataset this entry belongs to, for the error context.
        sleep: Blocking sleep used between retries. Injectable for tests.

    Returns:
        The file's contents.

    Raises:
        OfflineCacheMiss: The file was still missing after every attempt, so
            it is genuinely absent rather than mid-republication.
            ``retryable=True`` -- fetching this entry once would create it.
        DatasetIntegrityError: The file was still unreadable after every
            attempt. It exists but cannot be used, which is a real fault to
            surface, not a cache miss to paper over.
        OSError: Any other operating-system error is deliberately *not*
            handled here and propagates unchanged -- see the module docstring
            for why narrowing matters.
    """
    delay = INITIAL_RETRY_DELAY_SECONDS
    for _ in range(READ_ATTEMPTS - 1):
        try:
            return path.read_bytes()
        except (PermissionError, FileNotFoundError):
            # Looks like a writer swapping this file into place. Wait for the
            # rename to finish and look again. Note the ordering: the narrower
            # PermissionError must be listed before any OSError handler would
            # be, which is precisely why no OSError handler exists here.
            sleep(delay)
            delay *= 2

    # Last attempt. No retries remain, so a failure here has now outlived
    # every plausible rename window and is reported as the durable condition
    # it appears to be -- always as one of the two typed errors callers
    # already expect from a cache read.
    try:
        return path.read_bytes()
    except FileNotFoundError as error:
        raise OfflineCacheMiss(
            message=(
                f"{entry_label} for digest {digest} was still missing after "
                f"{READ_ATTEMPTS} read attempts"
            ),
            context={"digest": digest, "dataset_id": dataset_id, "path": str(path)},
            retryable=True,
        ) from error
    except PermissionError as error:
        raise DatasetIntegrityError(
            message=(
                f"{entry_label} for digest {digest} could not be read after "
                f"{READ_ATTEMPTS} attempts: {error.strerror or error}"
            ),
            context={"digest": digest, "dataset_id": dataset_id, "path": str(path)},
        ) from error


__all__ = ["INITIAL_RETRY_DELAY_SECONDS", "READ_ATTEMPTS", "read_entry_bytes"]
