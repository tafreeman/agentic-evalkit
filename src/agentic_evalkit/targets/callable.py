"""CallableTarget: wraps a plain Python function as the system under test (design §8).

The function handed in can be either a normal ("sync") function or an
``async def`` function -- either works. A sync function is run via
``asyncio.to_thread``, which moves it onto a background thread so a slow
function cannot freeze the rest of the program while it runs (Python's
async event loop can otherwise only do one thing at a time). Both sync and
async calls are wrapped in ``asyncio.timeout``, so either kind is cancelled
the same way if it runs too long. If the function raises an exception, it
is caught and converted into a typed error result instead of crashing the
whole evaluation run. Only the exception's type and message are recorded --
deliberately not a full traceback -- so nothing that happened to be sitting
in the function's local variables at the time of the error can leak into
the recorded evidence.
"""

import asyncio
import hashlib
import inspect
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import cast

from pydantic import JsonValue

from agentic_evalkit.models import EvalSample, ExecutionStatus, NormalizedExecutionResult

CallableResult = Mapping[str, JsonValue]
SyncCallable = Callable[[dict[str, JsonValue]], CallableResult]
AsyncCallable = Callable[[dict[str, JsonValue]], Awaitable[CallableResult]]
TargetCallable = SyncCallable | AsyncCallable


def _fingerprint(name: str, func: TargetCallable) -> str:
    """Build a short, stable ID for this callable: ``callable:{name}:{hash}``.

    This is a "fingerprint": an ID for one exact target configuration, used
    elsewhere to detect whether the target's identity changed between two
    runs. It is built from the callable's identity -- its module and
    qualified name -- plus ``name``, not from anything the callable
    returns. That means calling the same function with different inputs
    always produces the same fingerprint, but swapping in a different
    function object changes it.
    """
    module = getattr(func, "__module__", "") or ""
    qualname = getattr(func, "__qualname__", "") or repr(func)
    digest = hashlib.sha256(f"{module}:{qualname}:{name}".encode()).hexdigest()[:16]
    return f"callable:{name}:{digest}"


class CallableTarget:
    """Invokes an injected sync or async Python callable as the system under test."""

    def __init__(self, func: TargetCallable, *, name: str) -> None:
        self._func = func
        self._name = name
        self._fingerprint = _fingerprint(name, func)

    async def execute(
        self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
    ) -> NormalizedExecutionResult:
        started_at = datetime.now(UTC)
        try:
            output = await self._invoke(sample, timeout_seconds=timeout_seconds)
        except TimeoutError:
            return self._result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.TIMEOUT,
                started_at=started_at,
                error={
                    "type": "TimeoutError",
                    "message": f"callable target {self._name!r} exceeded "
                    f"{timeout_seconds}s timeout",
                },
            )
        except Exception as exc:  # deliberately broad -- turned into an ERROR result below
            return self._result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.ERROR,
                started_at=started_at,
                error={"type": type(exc).__name__, "message": str(exc)},
            )

        if not isinstance(output, Mapping):
            return self._result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.ERROR,
                started_at=started_at,
                error={
                    "type": "TypeError",
                    "message": (
                        f"callable target {self._name!r} must return a mapping, "
                        f"got {type(output).__name__}"
                    ),
                },
            )

        return self._result(
            sample,
            attempt=attempt,
            status=ExecutionStatus.COMPLETED,
            started_at=started_at,
            output=dict(output),
        )

    async def _invoke(self, sample: EvalSample, *, timeout_seconds: float | None) -> CallableResult:
        async with asyncio.timeout(timeout_seconds):
            if inspect.iscoroutinefunction(self._func):
                async_func = cast("AsyncCallable", self._func)
                return await async_func(sample.input)
            sync_func = cast("SyncCallable", self._func)
            return await asyncio.to_thread(sync_func, sample.input)

    def _result(
        self,
        sample: EvalSample,
        *,
        attempt: int,
        status: ExecutionStatus,
        started_at: datetime,
        output: dict[str, JsonValue] | None = None,
        error: dict[str, JsonValue] | None = None,
    ) -> NormalizedExecutionResult:
        if error is not None and "code" not in error:
            # Same stable taxonomy codes the runner's isolation path records
            # (TargetTimeout/TargetFailure), so ``error["code"]`` has one
            # schema regardless of which layer produced the error result.
            error = {
                **error,
                "code": (
                    "target_timeout" if status is ExecutionStatus.TIMEOUT else "target_failure"
                ),
            }
        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            output=output,
            status=status,
            error=error,
            target_fingerprint=self._fingerprint,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )
