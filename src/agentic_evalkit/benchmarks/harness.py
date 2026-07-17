"""Harness request/result data shapes and executor protocol (design §7, ADR-0005).

Benchmark adapters turn raw dataset rows into samples and decide what output
format and answer-key rules apply to them. A ``HarnessExecutor`` is the piece
that does the real, authoritative check -- for example, actually applying a
code patch and running the benchmark's real test suite inside an isolated
container, rather than just guessing whether the patch looks right. If the
harness can't run at all (not installed, Docker not running, etc.), the code
must clearly report a typed "unavailable" result instead of quietly making up
a score. In other words: a lower-confidence, advisory-only grader (like an AI
judge's opinion) can never be reported as if it were the official,
authoritative benchmark result -- that rule comes from ADR-0005.

This module defines the versioned data shapes sent to and returned from a
harness (``HarnessRequest``, ``HarnessResult`` -- "versioned" so the format
can change later without breaking old data), the ``HarnessExecutor``
interface itself, and two concrete implementations:

- ``UnavailableHarnessExecutor``: a simple, predictable, production-safe
  stand-in used when an optional harness feature (e.g. the
  ``agentic-evalkit[swebench]`` installable extra) is not installed. It
  always reports ``status="unavailable"`` -- it never pretends to grade
  anything.
- ``FakeHarnessExecutor``: **test-only**. You configure it ahead of time
  with the exact resolved / unresolved / error result it should return, so
  automated tests can exercise the grading and reporting logic without
  needing a real, container-based harness running. Production code must
  never construct this class outside of the ``tests/`` directory.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import Field, JsonValue

from agentic_evalkit.models.base import FrozenModel


class HarnessStatus(StrEnum):
    """The outcome of one attempt to run the harness (design §7.1).

    ``UNAVAILABLE`` means the harness itself never got to run at all -- for
    example, the optional package extra isn't installed, or a required
    container image is missing. This is a different situation from
    ``ERROR`` (the harness started running but hit an infrastructure
    problem partway through), and different again from a harness that ran
    successfully to completion but simply found the issue was not fixed
    (that's ``resolved=False`` together with status ``COMPLETED`` -- a
    normal, successful "no" verdict, not a failure).
    """

    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class HarnessRequest(FrozenModel):
    """One versioned request asking a harness to authoritatively check a result.

    ``prediction`` holds the actual answer/patch being checked, in whatever
    shape that specific benchmark's official tooling expects (for SWE-bench
    Verified, this is the official three-field export produced by
    :meth:`agentic_evalkit.benchmarks.swebench.SweBenchVerifiedAdapter.export_prediction`).
    ``source`` and ``environment`` carry extra context the harness needs to
    do its job -- like where the data came from, or what environment to run
    in -- but which is not itself part of what's being graded.
    """

    benchmark: str
    sample_id: str
    prediction: dict[str, JsonValue]
    source: dict[str, JsonValue] = Field(default_factory=dict)
    environment: dict[str, JsonValue] = Field(default_factory=dict)
    timeout_seconds: float
    resource_limits: dict[str, JsonValue] = Field(default_factory=dict)


class HarnessResult(FrozenModel):
    """The outcome of one harness run (design §7.1).

    ``resolved`` deliberately has three possible states, not two: ``True``
    means the harness actually confirmed the issue is fixed, ``False`` means
    it actually confirmed the issue is still broken, and ``None`` means
    there is no real verdict at all yet -- for example, because the harness
    was ``unavailable`` or hit an infrastructure ``error`` before it could
    finish checking. This matters because some other, more approximate
    grade (like an AI's guess) must never get silently converted into
    ``resolved=True`` -- only a genuine ``HarnessResult`` produced by really
    running the check can set it to ``True``.
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
    """The interface for anything that can perform real, authoritative
    verification (design §7.1, ADR-0005) -- as opposed to an approximate or
    advisory grade."""

    async def execute(self, request: HarnessRequest) -> HarnessResult: ...


class UnavailableHarnessExecutor:
    """A simple, predictable stand-in used when a harness isn't installed.

    Every call to :meth:`execute` returns a ``status="unavailable"``
    :class:`HarnessResult`, and its message includes ``install_hint`` (for
    example, ``"install agentic-evalkit[swebench]"``) so that callers and
    reports can tell the user exactly what to install to get real,
    authoritative grading -- instead of silently reporting something that
    looks like a failure or just leaving the result blank (design §7.1:
    "Missing authoritative SWE-bench capability returns `unavailable`, never
    a substitute score.").
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
    """A test-only, predictable stand-in ``HarnessExecutor`` (design §7.1).

    Production code must never import or construct this class. It exists
    only so unit and contract tests can exercise the grading and reporting
    logic against every possible outcome -- resolved, unresolved, and
    infrastructure error -- without needing a real, container-based harness
    to run. Configure it either with a mapping from ``sample_id`` to the
    exact :class:`HarnessResult` that sample should get back, or with a
    single default result to return for every request.
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
