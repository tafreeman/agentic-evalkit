"""Harness request/result contracts and executor protocol (design §7, ADR-0005).

Benchmark adapters project records and define artifact/oracle policy; a
``HarnessExecutor`` performs authoritative, isolated verification (for
example, applying a patch and running a benchmark's real test suite inside a
container). A missing or unavailable harness must return a typed
``unavailable`` result rather than a substitute score — an advisory grader
can never impersonate an authoritative benchmark result (ADR-0005).

This module defines the versioned wire contracts (``HarnessRequest``,
``HarnessResult``), the ``HarnessExecutor`` protocol, and two concrete
executors:

- ``UnavailableHarnessExecutor``: deterministic, production-safe executor
  used when an optional harness capability (e.g. ``agentic-evalkit[swebench]``)
  is not installed. It always reports ``status="unavailable"``.
- ``FakeHarnessExecutor``: **test-only**. It returns caller-configured
  resolved/unresolved/error results and exists solely so contract tests can
  exercise grading and reporting code paths without a real, containerized
  harness. Production code must never construct this class outside of
  ``tests/``.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import Field, JsonValue

from agentic_evalkit.models.base import FrozenModel


class HarnessStatus(StrEnum):
    """The outcome of one harness execution attempt (design §7.1).

    ``UNAVAILABLE`` means the harness capability itself could not run (for
    example, the optional extra is not installed, or a container image is
    missing) — it is categorically distinct from ``ERROR`` (the harness ran
    but hit an infrastructure failure) and from a completed run that simply
    did not resolve the issue (``resolved=False`` with ``COMPLETED``).
    """

    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class HarnessRequest(FrozenModel):
    """A single, versioned request to run authoritative benchmark verification.

    ``prediction`` carries the benchmark-specific prediction payload (for
    SWE-bench Verified, the official three-key export from
    :meth:`agentic_evalkit.benchmarks.swebench.SweBenchVerifiedAdapter.export_prediction`).
    ``source`` and ``environment`` carry provenance/environment metadata the
    harness needs but that is not itself part of the prediction contract.
    """

    benchmark: str
    sample_id: str
    prediction: dict[str, JsonValue]
    source: dict[str, JsonValue] = Field(default_factory=dict)
    environment: dict[str, JsonValue] = Field(default_factory=dict)
    timeout_seconds: float
    resource_limits: dict[str, JsonValue] = Field(default_factory=dict)


class HarnessResult(FrozenModel):
    """The outcome of one harness execution attempt (design §7.1).

    ``resolved`` is intentionally tri-state: ``True`` means the harness
    authoritatively confirmed the issue is resolved, ``False`` means the
    harness authoritatively confirmed it is not, and ``None`` means no
    authoritative resolution verdict is available (for example, the harness
    was ``unavailable`` or hit an infrastructure ``error`` before it could
    verify anything). A generic grade can never be converted to
    ``resolved=True`` without a real ``HarnessResult`` behind it.
    """

    status: HarnessStatus
    resolved: bool | None = None
    message: str
    evidence: dict[str, JsonValue] = Field(default_factory=dict)
    logs: tuple[str, ...] = ()
    image_digests: dict[str, str] = Field(default_factory=dict)
    error: dict[str, JsonValue] | None = None


@runtime_checkable
class HarnessExecutor(Protocol):
    """The authoritative-verification boundary (design §7.1, ADR-0005)."""

    async def execute(self, request: HarnessRequest) -> HarnessResult: ...


class UnavailableHarnessExecutor:
    """Deterministic executor for when a harness capability is not installed.

    Every call to :meth:`execute` returns a ``status="unavailable"``
    :class:`HarnessResult` whose message includes ``install_hint`` (for
    example, ``"install agentic-evalkit[swebench]"``) so callers and reports
    can tell users exactly how to unlock authoritative grading, instead of
    silently reporting a failed or absent result (design §7.1, "Missing
    authoritative SWE-bench capability returns `unavailable`, never a
    substitute score.").
    """

    def __init__(self, install_hint: str) -> None:
        self._install_hint = install_hint

    async def execute(self, request: HarnessRequest) -> HarnessResult:
        return HarnessResult(
            status=HarnessStatus.UNAVAILABLE,
            resolved=None,
            message=(
                f"Authoritative harness for {request.benchmark!r} is not "
                f"available: {self._install_hint}"
            ),
        )


class FakeHarnessExecutor:
    """Test-only, deterministic ``HarnessExecutor`` (design §7.1).

    Production code must never import or construct this class; it exists
    exclusively so unit and contract tests can exercise grading/reporting
    logic against resolved, unresolved, and infrastructure-error harness
    outcomes without a real containerized harness. Configure it with a
    mapping from ``sample_id`` to the exact :class:`HarnessResult` it should
    return for that sample, or a single default result to return for every
    request.
    """

    def __init__(
        self,
        *,
        default_result: HarnessResult | None = None,
        results_by_sample_id: Mapping[str, HarnessResult] | None = None,
    ) -> None:
        self._default_result = default_result
        self._results_by_sample_id = dict(results_by_sample_id or {})

    async def execute(self, request: HarnessRequest) -> HarnessResult:
        if request.sample_id in self._results_by_sample_id:
            return self._results_by_sample_id[request.sample_id]
        if self._default_result is not None:
            return self._default_result
        raise KeyError(
            f"FakeHarnessExecutor has no configured result for sample_id="
            f"{request.sample_id!r} and no default_result was set"
        )
