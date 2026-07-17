"""The one doorway used to call whatever system is being evaluated (design §8; ADR-0006).

``ExecutionTarget`` is the single interface this library uses to call the system
under test -- the thing whose answers are being graded. Whether that system is an
in-process Python function, a subprocess, or an HTTP service, it is always called
through this same interface, defined once here. Because this is a Python
``Protocol`` (structural typing), a class counts as an ``ExecutionTarget`` simply
by having a matching ``execute`` method -- it does not need to inherit from
anything or register itself anywhere.

Every adapter (callable, subprocess, HTTP) hands back the same
:class:`NormalizedExecutionResult` shape, so the rest of the framework --
grading, reporting, everything downstream -- never has to special-case which
kind of target produced a result. In particular, it never needs to recognize
types from ARP or ExecutionKit, the separate systems this library is commonly
used to evaluate.
"""

from typing import Protocol, runtime_checkable

from agentic_evalkit.models import EvalSample, NormalizedExecutionResult


@runtime_checkable
class ExecutionTarget(Protocol):
    """Marks a class as a valid way to invoke the system under test (design §8)."""

    async def execute(
        self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
    ) -> NormalizedExecutionResult: ...
