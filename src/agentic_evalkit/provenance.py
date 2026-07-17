"""Functions that compute "fingerprints" -- hashes proving what code/environment/target ran.

Design §5.6.

Design §5.6 says that an :class:`~agentic_evalkit.models.EvalRunManifest`
should record a fingerprint for the execution target, plus fingerprints for
the environment and the code that produced the run -- so that later, you can
prove two runs are actually comparable (same code, same environment, same
target) before treating their results as apples-to-apples. Before this
module existed, that was only a promise: the ``environment_fingerprint`` and
``code_fingerprint`` fields existed on the manifest, but nothing ever filled
them in -- every real run just left them as ``None``. There wasn't even a
``target_fingerprint`` field yet, only a ``target_fingerprint_policy``
describing how one *should* eventually be enforced. This module is what
actually computes those values, so the promise gets kept.

Every function below only uses Python's standard library, is deterministic
(the same input always gives the same output), and has no side effects
beyond reading the interpreter's and installed packages' metadata via
:mod:`importlib.metadata` -- no reading files, no network calls, no
current-time timestamps, no random numbers. Call any function here twice
with the same inputs, on the same installed environment, and you always get
back the identical hash. That's the entire point: a "fingerprint" that
changes between calls for no reason would be useless as a proof of identity.

Each fingerprint is a string of the form ``"sha256:" + <64 hex characters>``
-- a SHA-256 hash of the input data, first converted to JSON in a
"canonical" way (sorted keys, no extra whitespace) so that the same logical
data always produces the exact same JSON text, and therefore the exact same
hash, regardless of what order the fields happened to be in originally.
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

#: The package name used to look up the installed agentic-evalkit version.
#: This matches the ``[project].name`` entry in ``pyproject.toml``, and the
#: same lookup already done in ``agentic_evalkit/__init__.py``.
_PACKAGE_DISTRIBUTION_NAME = "agentic-evalkit"

#: The version string used when we can't find the package's installation
#: metadata at all (for example, running straight from a source checkout
#: that was never pip-installed). Deliberately looks like a placeholder
#: rather than a real version number, while still being a valid, stable
#: string that can be hashed just like a real version would be.
_UNKNOWN_PACKAGE_VERSION = "0+unknown"


def _canonical_json(payload: object) -> str:
    """Convert ``payload`` to JSON text in a fixed, predictable way, so hashing it is reliable.

    ``sort_keys=True`` means the key order in the original dict never
    changes the output text. The compact ``(",", ":")`` separators strip
    out incidental spacing differences that don't change the meaning.
    ``default=str`` means that if the caller passes something that isn't
    naturally JSON-shaped (like a ``Path`` object or an enum member), it
    just gets converted to its string form instead of causing an error --
    this is fine here because fingerprint inputs are plain, caller-supplied
    configuration values, not this package's own strictly validated data
    models.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_fingerprint(payload: object) -> str:
    """Hash the canonical JSON of ``payload`` and prefix it ``"sha256:"``."""
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _installed_package_version() -> str:
    """Look up the installed agentic-evalkit version, or fall back to a fixed placeholder.

    If :class:`~importlib.metadata.PackageNotFoundError` is raised --
    meaning Python can't find installation metadata for this package -- this
    returns :data:`_UNKNOWN_PACKAGE_VERSION` instead of letting the error
    propagate. That way, fingerprinting never breaks a run just because the
    package happened to be installed in an unusual way (for example, run
    directly from a source checkout without a proper install step).
    """
    try:
        return version(_PACKAGE_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _UNKNOWN_PACKAGE_VERSION


def compute_environment_fingerprint() -> str:
    """Compute a fingerprint of the Python interpreter, platform, and installed package version.

    Includes the Python version (major.minor.patch), which Python
    implementation this is (e.g. CPython vs. PyPy), the OS platform
    identifier, the machine's CPU architecture, and the installed
    agentic-evalkit version -- enough information to tell whether two runs
    happened under meaningfully different interpreters or platforms. It
    deliberately does not cover the versions of third-party dependencies or
    detailed OS/kernel build information -- that level of detail is outside
    what this helper tries to capture.
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
    """Fingerprint the configuration of a resolved execution target.

    ``target_config`` is supplied by the caller (for example, the dict you
    get from calling ``model_dump()`` on a CLI target block), so this
    accepts a plain mapping rather than requiring every value to already be
    JSON-friendly -- and rather than requiring the full ``EvalRunManifest``
    object itself. That means callers can fingerprint exactly the resolved
    target shape they built, whatever it looks like. Because keys are
    sorted before hashing (``json.dumps(..., sort_keys=True)``), two
    mappings with the same key/value pairs listed in a different order
    always produce the same fingerprint -- but changing even one value,
    however small, changes the resulting hash.
    """
    return _sha256_fingerprint(dict(target_config))
