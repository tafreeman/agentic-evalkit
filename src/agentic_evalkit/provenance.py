"""Deterministic fingerprint helpers for run provenance (design §5.6).

Design §5.6 states that :class:`~agentic_evalkit.models.EvalRunManifest`
pins "execution target and target fingerprint" and "environment and code
fingerprints." Before this module existed, ``environment_fingerprint`` and
``code_fingerprint`` were declared fields that nothing in the package ever
populated -- every real run carried ``None`` in both, and there was no
``target_fingerprint`` field at all, only a ``target_fingerprint_policy``
describing how one *should* be enforced. This module is the missing
generator: it turns each pinned-provenance promise into an actual,
reproducible value.

Every function here is stdlib-only, deterministic, and side-effect free
beyond reading interpreter/package metadata through
:mod:`importlib.metadata`: no filesystem or network I/O, no wall-clock
timestamps, no randomness. Calling a function twice with the same inputs
(and the same installed environment) always returns the same digest, which
is the entire point -- a fingerprint that is not reproducible is not a
fingerprint.

Each digest is rendered as ``"sha256:" + <64 lowercase hex characters>``
over the *canonical JSON* encoding of its inputs (``json.dumps`` with
``sort_keys=True`` and compact ``(",", ":")`` separators), so key order in
the source mapping never affects the result.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Distribution name used to resolve the installed agentic-evalkit version.
#: Matches the ``[project].name`` in ``pyproject.toml`` and the lookup
#: already performed in ``agentic_evalkit/__init__.py``.
_PACKAGE_DISTRIBUTION_NAME = "agentic-evalkit"

#: Version reported when the package's own distribution metadata cannot be
#: found (e.g. running from a source checkout without an installed egg-info).
#: Chosen so it is obviously a placeholder rather than a real release, while
#: still being a valid, stable string for hashing.
_UNKNOWN_PACKAGE_VERSION = "0+unknown"


def _canonical_json(payload: object) -> str:
    """Render ``payload`` as compact, key-sorted JSON for stable hashing.

    ``sort_keys=True`` makes the encoding insensitive to the input
    mapping's key order; the compact ``(",", ":")`` separators remove
    incidental whitespace differences. ``default=str`` lets non-JSON-native
    values (e.g. a caller passing a ``Path`` or an enum) still hash
    deterministically instead of raising, since fingerprint inputs are
    caller-supplied config data, not validated wire models.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_fingerprint(payload: object) -> str:
    """Hash the canonical JSON of ``payload`` and prefix it ``"sha256:"``."""
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _installed_package_version() -> str:
    """Resolve the installed agentic-evalkit version, or a stable fallback.

    Falls back to :data:`_UNKNOWN_PACKAGE_VERSION` on
    :class:`~importlib.metadata.PackageNotFoundError` rather than raising,
    so fingerprinting never fails a run merely because the package is not
    installed in the conventional way (e.g. a vendored or editable
    checkout without distribution metadata).
    """
    try:
        return version(_PACKAGE_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _UNKNOWN_PACKAGE_VERSION


def compute_environment_fingerprint() -> str:
    """Fingerprint the interpreter/platform/package identity of this process.

    Covers the Python version triple, interpreter implementation name,
    ``sys.platform``, machine architecture, and the installed
    agentic-evalkit version -- the minimal set needed to tell whether two
    runs executed under a materially different interpreter or platform.
    It does not cover installed third-party dependency versions or OS
    build/kernel details; those are out of scope for this helper.
    """
    payload = {
        "python_version": tuple(sys.version_info[:3]),
        "python_implementation": sys.implementation.name,
        "platform": sys.platform,
        "machine": platform.machine(),
        "agentic_evalkit_version": _installed_package_version(),
    }
    return _sha256_fingerprint(payload)


def compute_code_fingerprint() -> str:
    """Fingerprint the identity of the installed agentic-evalkit package.

    This is honest about its scope: it fingerprints *this framework's*
    package name and installed version, not any user-supplied target or
    adapter code. It answers "which agentic-evalkit build produced this
    run," not "did the target's code change between two runs" -- that
    question belongs to :func:`compute_target_fingerprint`, which hashes
    the caller's target configuration instead.
    """
    payload = {
        "package": _PACKAGE_DISTRIBUTION_NAME,
        "version": _installed_package_version(),
    }
    return _sha256_fingerprint(payload)


def compute_target_fingerprint(target_config: Mapping[str, object]) -> str:
    """Fingerprint a resolved target's canonical configuration.

    ``target_config`` is caller-supplied (e.g. a CLI target block's
    ``model_dump()``), so encoding uses ``default=str`` rather than
    requiring every value to already be JSON-native; it is a mapping, not
    the wire ``EvalRunManifest`` itself, so callers can fingerprint the
    exact resolved target shape they constructed. Sorting keys before
    hashing (``json.dumps(..., sort_keys=True)``) means two mappings with
    the same key/value pairs in a different order always hash identically,
    while any actual value change -- however small -- changes the digest.
    """
    return _sha256_fingerprint(dict(target_config))
