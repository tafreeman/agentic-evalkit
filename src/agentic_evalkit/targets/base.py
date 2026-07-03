"""Execution target protocol (design §8; ADR-0006).

``ExecutionTarget`` is the only boundary through which a system under test
is invoked. Callable, subprocess, and HTTP adapters all implement this
structural protocol and return :class:`NormalizedExecutionResult`; framework
code cannot branch on ARP or ExecutionKit types.
"""

from typing import Protocol, runtime_checkable

from agentic_evalkit.models import EvalSample, NormalizedExecutionResult


@runtime_checkable
class ExecutionTarget(Protocol):
    """The system-under-test boundary (design §8)."""

    async def execute(
        self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
    ) -> NormalizedExecutionResult: ...
