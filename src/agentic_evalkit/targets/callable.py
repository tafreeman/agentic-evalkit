"""CallableTarget: an injected sync or async Python callable (design §8).

Sync callables run through ``asyncio.to_thread`` so a slow synchronous
system under test does not block the event loop; both sync and async paths
are guarded by ``asyncio.timeout``. Exceptions are converted into typed
``NormalizedExecutionResult`` errors without leaking local variables from the
callable's frame -- only the exception type and message are recorded.
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
    """Hash module/qualified-name/name into a stable ``callable:{name}:{hash}``.

    The hash is over the callable's identity (module + qualname), not its
    output, so the fingerprint is stable across calls with different inputs
    but changes if the underlying function object changes.
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
        except Exception as exc:  # deliberately normalized into an ERROR result below
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
