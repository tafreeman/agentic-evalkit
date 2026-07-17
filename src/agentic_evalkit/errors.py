"""The complete set of error types this package can raise.

Every way things can go wrong inside agentic-evalkit -- a missing dataset, a
crashed target, a bad manifest -- is represented here as its own exception
class, all inheriting from one shared base: :class:`AgenticEvalkitError`.
This module deliberately imports nothing beyond Python's standard library
(no pydantic, no third-party packages), so any other module in the package
-- including ``agentic_evalkit.models`` -- can import from it without risking
a circular import (module A importing module B which tries to import module
A again).

Each exception carries three things: a ``code`` (a short, stable,
machine-readable label auto-generated from the class name -- e.g.
``DatasetNotFound`` becomes ``"dataset_not_found"``), a ``message`` (a
human-readable explanation), and an optional ``context`` dict with extra
structured detail useful for debugging. If any of that context data is
sensitive (an API token, a credential), wrap it with
:meth:`AgenticEvalkitError.secret` first -- doing so keeps it out of both
``str()`` and ``repr()`` of the error, so it can never accidentally end up
printed to a log or a report.

Per ADR-0003 (an architecture decision recorded early in this project), this
file is meant to hold the *entire* error hierarchy, once and for all. Other
modules only ever import exceptions from here -- they don't define new
exception classes of their own.
"""

from __future__ import annotations

import re
from typing import Final, Union

# A stand-in type for "any value that JSON can represent" (None, a bool, an
# int, a float, a string, or nested tuples/dicts of these). We define it
# ourselves here instead of importing pydantic's equivalent, because this
# module is only allowed to use the Python standard library (see the
# module docstring above, and ADR-0003).
JsonValue = Union[
    None,
    bool,
    int,
    float,
    str,
    tuple["JsonValue", ...],
    "dict[str, JsonValue]",
]

__all__ = [
    "AgenticEvalkitError",
    "DatasetAccessDenied",
    "DatasetConfigRequired",
    "DatasetIntegrityError",
    "DatasetLicenseRejected",
    "DatasetNotFound",
    "DatasetProviderUnavailable",
    "DatasetRateLimited",
    "DatasetSchemaMismatch",
    "DatasetSplitNotFound",
    "GraderError",
    "IncompatibleRuns",
    "ManifestValidationError",
    "OfflineCacheMiss",
    "PluginCompatibilityError",
    "SecretValue",
    "TargetFailure",
    "TargetTimeout",
    "UnsafeCodeRequired",
]

_CAMEL_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<!^)(?=[A-Z])")


def _default_code(class_name: str) -> str:
    """Turn a class name like ``DatasetNotFound`` into ``dataset_not_found``.

    This is computed automatically from the class name, instead of writing
    a separate code string by hand for every exception class, so the code
    can never fall out of sync with the class it's naming.
    """
    return _CAMEL_BOUNDARY.sub("_", class_name).lower()


class SecretValue:
    """Wraps a value so it never shows up in an error's printed text.

    Call :meth:`AgenticEvalkitError.secret` to create one of these -- don't
    construct this class directly.
    """

    __slots__ = ("_value",)

    def __init__(self, value: JsonValue) -> None:
        self._value = value

    def reveal(self) -> JsonValue:
        """Return the wrapped value. Callers must not log or print the result."""
        return self._value

    def __repr__(self) -> str:
        return "SecretValue(***redacted***)"

    def __str__(self) -> str:
        return "***redacted***"


class AgenticEvalkitError(Exception):
    """Base class every agentic-evalkit exception inherits from.

    Attributes:
        code: A short, stable, machine-readable label for this error type
            (e.g. ``"dataset_not_found"``). By default it's generated from
            the subclass's name and shouldn't change between releases, so
            other code (or another program reading these errors) can safely
            check ``error.code`` without it silently changing later.
        message: A human-readable explanation of what went wrong.
        context: A dict of extra structured detail about the failure. Any
            value wrapped with :meth:`secret` is hidden -- it will not
            appear in ``str()`` or ``repr()`` output.
    """

    def __init__(
        self,
        *,
        message: str,
        code: str | None = None,
        context: dict[str, JsonValue | SecretValue] | None = None,
    ) -> None:
        self.code: str = code or _default_code(type(self).__name__)
        self.message: str = message
        self.context: dict[str, JsonValue | SecretValue] = dict(context or {})
        super().__init__(message)

    @staticmethod
    def secret(value: JsonValue) -> SecretValue:
        """Mark a context value as secret so it never appears in error text."""
        return SecretValue(value)

    def _redacted_context(self) -> dict[str, JsonValue]:
        return {
            key: ("***redacted***" if isinstance(value, SecretValue) else value)
            for key, value in self.context.items()
        }

    def __str__(self) -> str:
        redacted = self._redacted_context()
        if redacted:
            return f"[{self.code}] {self.message} (context={redacted!r})"
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


# --- Dataset errors (design §6.4) ------------------------------------------


class DatasetNotFound(AgenticEvalkitError):
    """The requested dataset does not exist or is not reachable by ID."""


class DatasetConfigRequired(AgenticEvalkitError):
    """The dataset has multiple named variants ("configs") and none could be uniquely inferred."""


class DatasetSplitNotFound(AgenticEvalkitError):
    """The requested split (e.g. "train") does not exist for this resolved dataset/config."""


class DatasetAccessDenied(AgenticEvalkitError):
    """The dataset is private or gated and the caller lacks access."""


class DatasetLicenseRejected(AgenticEvalkitError):
    """The dataset's license terms were not accepted for this operation."""


class DatasetIntegrityError(AgenticEvalkitError):
    """Cached/downloaded dataset bytes don't match their expected checksum -- possibly corrupted."""


class DatasetSchemaMismatch(AgenticEvalkitError):
    """A source record did not match the expected row schema."""


class DatasetProviderUnavailable(AgenticEvalkitError):
    """The provider backend is unreachable or returned a transport failure."""


class UnsafeCodeRequired(AgenticEvalkitError):
    """Loading the dataset would require executing untrusted remote code."""


class DatasetRateLimited(AgenticEvalkitError):
    """The provider rate-limited the request; retry metadata may be present."""


class OfflineCacheMiss(AgenticEvalkitError):
    """Running in offline mode, but the data we need isn't in the local cache yet.

    Attributes:
        retryable: Tells the caller (and a human reading the CLI error)
            whether trying again could ever work. There are two very
            different situations here, and it's important not to mix them
            up:

            - ``True`` -- "just go online once, then retry": this is a
              plain, ordinary cache miss. The thing being asked for *does*
              have a stable cache key, we just don't have it saved locally
              yet. Going online for one run (e.g. removing ``--offline``, or
              running an explicit ``datasets pull``) and then repeating the
              exact same offline command afterward will succeed.
            - ``False`` -- "this can never be cached, retrying won't help":
              the request has no stable cache key to begin with -- for
              example, a free-text search, a lookup with no cache backing
              it, reading results page-by-page without a fixed page key, or
              no cache configured at all -- or the provider simply requires
              a live network call no matter what. Repeating the identical
              offline request will never succeed; the caller has to either
              ask for something different or allow a network request.

            Defaults to ``True`` because the most common place this error is
            raised -- :meth:`agentic_evalkit.datasets.cache.DatasetCache.read`
            not finding a saved entry for a key that normally *is* cacheable
            -- is exactly the retryable case. The few call sites that know
            they're in the second, non-retryable situation (see
            :mod:`agentic_evalkit.datasets.catalog`) explicitly pass
            ``retryable=False``.
    """

    def __init__(
        self,
        *,
        message: str,
        code: str | None = None,
        context: dict[str, JsonValue | SecretValue] | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message=message, code=code, context=context)
        self.retryable: bool = retryable


# --- Plugin errors -----------------------------------------------------------


class PluginCompatibilityError(AgenticEvalkitError):
    """A plugin failed to load, or its plugin-API version isn't one this package supports."""


# --- Execution target errors (design §8) -------------------------------------


class TargetFailure(AgenticEvalkitError):
    """The execution target raised, crashed, or returned an invalid result."""


class TargetTimeout(AgenticEvalkitError):
    """The execution target did not respond within the configured timeout."""


# --- Grading errors ------------------------------------------------------------


class GraderError(AgenticEvalkitError):
    """The grader broke producing a result -- a bug in grading, not a verdict on the sample."""


# --- Statistics / comparability errors (design §10) ---------------------------


class IncompatibleRuns(AgenticEvalkitError):
    """Two runs can't be compared -- they used a different dataset, adapter, grader, or policy."""


# --- Manifest errors ------------------------------------------------------------


class ManifestValidationError(AgenticEvalkitError):
    """An EvalRunManifest failed validation before a run could start."""
