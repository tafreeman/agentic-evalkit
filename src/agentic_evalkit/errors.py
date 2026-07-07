"""Typed, stable, dependency-free error hierarchy for agentic-evalkit.

This module intentionally imports nothing beyond the standard library so it
can be safely imported by every other module in the package (including
``agentic_evalkit.models``) without creating a dependency cycle.

Every framework failure is a subclass of :class:`AgenticEvalkitError`. Each
subclass carries a stable, snake_case ``code`` derived from its class name,
a human-readable ``message``, and an optional ``context`` mapping used for
structured diagnostic detail. Context values that must never be written to
logs or reports (tokens, credentials, and similar secrets) are wrapped with
:meth:`AgenticEvalkitError.secret` and are excluded from both ``str()`` and
``repr()`` of the error.

Per ADR-0003, this hierarchy is defined completely in this task. Downstream
tasks import from here; they never add new subclasses to this module.
"""

from __future__ import annotations

import re
from typing import Final, Union

# A minimal JSON-compatible value type, defined locally rather than imported
# from pydantic, so this module stays stdlib-only per ADR-0003 / Task 3.
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
    """Derive a stable snake_case error code from a class name.

    ``DatasetNotFound`` -> ``dataset_not_found``. This is computed once from
    the class name rather than hardcoded per subclass so the code can never
    drift from the class it names.
    """
    return _CAMEL_BOUNDARY.sub("_", class_name).lower()


class SecretValue:
    """Wraps a context value that must never be serialized into error text.

    Use :meth:`AgenticEvalkitError.secret` to construct one rather than
    instantiating this class directly.
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
    """Base class for every typed agentic-evalkit failure.

    Attributes:
        code: Stable, snake_case identifier for this error type. Defaults to
            a value derived from the concrete subclass name and should not
            change across releases within a schema/major version.
        message: Human-readable description of the failure.
        context: Structured diagnostic detail. Values wrapped with
            :meth:`secret` are redacted from ``str()`` and ``repr()``.
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
        """Mark a context value as secret so it is redacted from error text."""
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
    """The dataset has multiple configs and none could be uniquely inferred."""


class DatasetSplitNotFound(AgenticEvalkitError):
    """The requested split does not exist for the resolved dataset/config."""


class DatasetAccessDenied(AgenticEvalkitError):
    """The dataset is private or gated and the caller lacks access."""


class DatasetLicenseRejected(AgenticEvalkitError):
    """The dataset's license terms were not accepted for this operation."""


class DatasetIntegrityError(AgenticEvalkitError):
    """Cached or retrieved dataset bytes failed checksum/identity validation."""


class DatasetSchemaMismatch(AgenticEvalkitError):
    """A source record did not match the expected row schema."""


class DatasetProviderUnavailable(AgenticEvalkitError):
    """The provider backend is unreachable or returned a transport failure."""


class UnsafeCodeRequired(AgenticEvalkitError):
    """Loading the dataset would require executing untrusted remote code."""


class DatasetRateLimited(AgenticEvalkitError):
    """The provider rate-limited the request; retry metadata may be present."""


class OfflineCacheMiss(AgenticEvalkitError):
    """Offline mode requested a page/dataset with no exact cache entry.

    Attributes:
        retryable: Discriminates two distinct offline failures that a caller
            (and a human reading a CLI error message) must not conflate:

            - ``True`` -- "warm the cache and retry": a plain cache miss for
              an otherwise cacheable key. Going online once (e.g. dropping
              ``--offline`` for one run, or an explicit ``datasets pull``)
              and then repeating the *exact same* offline call succeeds,
              because the operation's cache identity model has a stable key
              for this request shape.
            - ``False`` -- "categorically uncacheable": the requested
              operation has no stable cache key at all for this call shape
              (free-text search, an unbacked resolution, unpaginated
              iteration, or no cache configured on the catalog) or the
              provider genuinely requires network access it was asked not to
              use. No amount of prior or future warming makes the *same*
              offline call succeed; a different action (changing what is
              asked for, or accepting a network round trip) is required.

            Defaults to ``True`` because the most common raise site --
            :meth:`agentic_evalkit.datasets.cache.DatasetCache.read` finding
            no manifest/payload for an otherwise-cacheable key -- is exactly
            the retryable case. Raise sites that know better (see
            :mod:`agentic_evalkit.datasets.catalog`) pass ``retryable=False``
            explicitly.
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
    """An extension entry point failed to load or declared an incompatible API."""


# --- Execution target errors (design §8) -------------------------------------


class TargetFailure(AgenticEvalkitError):
    """The execution target raised, crashed, or returned an invalid result."""


class TargetTimeout(AgenticEvalkitError):
    """The execution target did not respond within the configured timeout."""


# --- Grading errors ------------------------------------------------------------


class GraderError(AgenticEvalkitError):
    """A grader failed to produce a result for reasons other than the sample."""


# --- Statistics / comparability errors (design §10) ---------------------------


class IncompatibleRuns(AgenticEvalkitError):
    """Two runs are not comparable (dataset, adapter, grader, or policy differs)."""


# --- Manifest errors ------------------------------------------------------------


class ManifestValidationError(AgenticEvalkitError):
    """An EvalRunManifest failed validation before a run could start."""
