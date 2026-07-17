"""HttpTarget: calls an HTTP endpoint as the system under test (design §8).

Each request and response is a JSON object tagged with a schema/protocol
version number, so both sides agree on the message format.

Rather than creating its own HTTP client, this class is handed an
already-configured ``httpx.AsyncClient`` from outside. That lets tests
pass in a fake client (a ``MockTransport``) that never makes a real
network call, while production code can pass in a client that is already
set up with connection pooling, proxy settings, and TLS/certificate
configuration -- none of which this class needs to know about.

If a request fails to connect, or the server responds with status 429
(rate limited) or 502/503/504 (typically transient server-side errors),
it is retried automatically, up to a limited number of times, waiting
longer between each attempt ("exponential backoff", capped so the wait
never grows unbounded). If the server tells us how long to wait via a
``Retry-After`` header, that is honored instead of our own calculated
wait. Other 4xx responses (for example, a malformed request) mean
something is wrong with the request itself rather than a temporary
glitch, so they are never retried.

Any header that could carry credentials (``Authorization``,
``Proxy-Authorization``) is blanked out -- "redacted" -- before it is
stored or shown anywhere, so secrets never end up in saved logs or
reports.
"""

import asyncio
import hashlib
import random
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import httpx

from agentic_evalkit.models import EvalSample, ExecutionStatus, NormalizedExecutionResult

if TYPE_CHECKING:
    from pydantic import JsonValue

_PROTOCOL_VERSION: Final[str] = "1"
_RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({429, 502, 503, 504})
_REDACTED_HEADER_NAMES: Final[frozenset[str]] = frozenset({"authorization", "proxy-authorization"})
_DEFAULT_MAX_RETRIES: Final[int] = 3
_DEFAULT_BASE_DELAY_SECONDS: Final[float] = 0.1
_DEFAULT_MAX_DELAY_SECONDS: Final[float] = 5.0

HeaderProvider = Callable[[], Mapping[str, str]]
SleepFn = Callable[[float], Awaitable[None]]


async def _default_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: ("***redacted***" if key.lower() in _REDACTED_HEADER_NAMES else value)
        for key, value in headers.items()
    }


def _fingerprint(name: str, url: str) -> str:
    digest = hashlib.sha256(f"{name}:{url}".encode()).hexdigest()[:16]
    return f"http:{name}:{digest}"


class HttpTarget:
    """Invokes a remote HTTP endpoint as the system under test."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        url: str,
        name: str,
        headers: HeaderProvider | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        base_delay_seconds: float = _DEFAULT_BASE_DELAY_SECONDS,
        max_delay_seconds: float = _DEFAULT_MAX_DELAY_SECONDS,
        sleep: SleepFn = _default_sleep,
    ) -> None:
        self._client = client
        self._url = url
        self._name = name
        self._headers = headers
        self._max_retries = max_retries
        self._base_delay_seconds = base_delay_seconds
        self._max_delay_seconds = max_delay_seconds
        self._sleep = sleep
        self._fingerprint = _fingerprint(name, url)

    async def execute(
        self,
        sample: EvalSample,
        *,
        attempt: int,
        timeout_seconds: float | None,
        trace_id: str | None = None,
    ) -> NormalizedExecutionResult:
        started_at = datetime.now(UTC)
        try:
            async with asyncio.timeout(timeout_seconds):
                return await self._execute_with_retries(
                    sample, attempt=attempt, trace_id=trace_id, started_at=started_at
                )
        except TimeoutError:
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.TIMEOUT,
                started_at=started_at,
                error_type="TimeoutError",
                message=f"http target {self._name!r} exceeded {timeout_seconds}s timeout",
            )

    async def _execute_with_retries(
        self,
        sample: EvalSample,
        *,
        attempt: int,
        trace_id: str | None,
        started_at: datetime,
    ) -> NormalizedExecutionResult:
        request_headers = _redact_headers(dict(self._headers())) if self._headers else {}
        last_error: dict[str, JsonValue] = {
            "type": "UnknownError",
            "message": "no request attempt was made",
        }
        for retry_index in range(self._max_retries + 1):
            try:
                raw_headers = dict(self._headers()) if self._headers else {}
                response = await self._client.post(
                    self._url,
                    json={
                        "schema_version": _PROTOCOL_VERSION,
                        "sample_id": sample.sample_id,
                        "input": sample.input,
                        "attempt": attempt,
                        "trace_id": trace_id,
                    },
                    headers=raw_headers,
                )
            except httpx.TransportError as exc:
                last_error = {"type": type(exc).__name__, "message": str(exc)}
                if retry_index < self._max_retries:
                    await self._backoff(retry_index, retry_after=None)
                    continue
                return self._error_result(
                    sample,
                    attempt=attempt,
                    status=ExecutionStatus.ERROR,
                    started_at=started_at,
                    error_type=last_error["type"],  # type: ignore[arg-type]
                    message=str(last_error["message"]),
                    request_headers=request_headers,
                )

            if response.status_code in _RETRYABLE_STATUS_CODES:
                last_error = {
                    "type": "RetryableHttpStatus",
                    "message": f"received retryable status {response.status_code}",
                }
                if retry_index < self._max_retries:
                    await self._backoff(retry_index, retry_after=self._retry_after(response))
                    continue
                return self._error_result(
                    sample,
                    attempt=attempt,
                    status=ExecutionStatus.ERROR,
                    started_at=started_at,
                    error_type=last_error["type"],  # type: ignore[arg-type]
                    message=str(last_error["message"]),
                    request_headers=request_headers,
                    response_status=response.status_code,
                )

            if response.status_code >= 400:
                # Any other 4xx or 5xx status -- one that is not in the
                # retryable set handled above -- is treated as a permanent
                # failure: it is never retried, and this attempt fails
                # immediately.
                return self._error_result(
                    sample,
                    attempt=attempt,
                    status=ExecutionStatus.ERROR,
                    started_at=started_at,
                    error_type="HttpStatusError",
                    message=f"received non-retryable status {response.status_code}",
                    request_headers=request_headers,
                    response_status=response.status_code,
                )

            return self._parse_response(
                sample,
                attempt=attempt,
                started_at=started_at,
                response=response,
                request_headers=request_headers,
            )

        # This line is never actually reached: every branch inside the
        # loop above returns before the loop can run out of retries. It
        # is here only because mypy --strict requires every code path to
        # explicitly return a value, and it cannot prove on its own that
        # the loop always returns early.
        return self._error_result(
            sample,
            attempt=attempt,
            status=ExecutionStatus.ERROR,
            started_at=started_at,
            error_type=str(last_error["type"]),
            message=str(last_error["message"]),
            request_headers=request_headers,
        )

    def _parse_response(
        self,
        sample: EvalSample,
        *,
        attempt: int,
        started_at: datetime,
        response: httpx.Response,
        request_headers: dict[str, str],
    ) -> NormalizedExecutionResult:
        try:
            payload = response.json()
        except ValueError as exc:
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.ERROR,
                started_at=started_at,
                error_type="JSONDecodeError",
                message=f"malformed JSON response: {exc}",
                request_headers=request_headers,
                response_status=response.status_code,
            )

        if not isinstance(payload, dict):
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.ERROR,
                started_at=started_at,
                error_type="TypeError",
                message=f"response must be a JSON object, got {type(payload).__name__}",
                request_headers=request_headers,
                response_status=response.status_code,
            )

        response_sample_id = payload.get("sample_id")
        if response_sample_id != sample.sample_id:
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.ERROR,
                started_at=started_at,
                error_type="SampleIdMismatch",
                message=(
                    f"response sample_id {response_sample_id!r} did not match "
                    f"request sample_id {sample.sample_id!r}"
                ),
                request_headers=request_headers,
                response_status=response.status_code,
            )

        output = payload.get("output")
        if output is not None and not isinstance(output, dict):
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.ERROR,
                started_at=started_at,
                error_type="TypeError",
                message=f"response output must be a JSON object, got {type(output).__name__}",
                request_headers=request_headers,
                response_status=response.status_code,
            )

        completed_metadata: dict[str, JsonValue] = {
            "request_headers": dict(request_headers),
            "response_status": response.status_code,
        }
        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            output=output,
            status=ExecutionStatus.COMPLETED,
            target_fingerprint=self._fingerprint,
            environment_metadata=completed_metadata,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    def _error_result(
        self,
        sample: EvalSample,
        *,
        attempt: int,
        status: ExecutionStatus,
        started_at: datetime,
        error_type: str,
        message: str,
        request_headers: dict[str, str] | None = None,
        response_status: int | None = None,
    ) -> NormalizedExecutionResult:
        error: dict[str, JsonValue] = {"type": error_type, "message": message}
        environment_metadata: dict[str, JsonValue] = {}
        if request_headers is not None:
            environment_metadata["request_headers"] = dict(request_headers)
        if response_status is not None:
            environment_metadata["response_status"] = response_status
        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            status=status,
            error=error,
            target_fingerprint=self._fingerprint,
            environment_metadata=environment_metadata,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        header_value = response.headers.get("Retry-After")
        if header_value is None:
            return None
        try:
            return max(0.0, float(header_value))
        except ValueError:
            return None

    async def _backoff(self, retry_index: int, *, retry_after: float | None) -> None:
        if retry_after is not None:
            await self._sleep(retry_after)
            return
        exponential = self._base_delay_seconds * (2**retry_index)
        capped = min(exponential, self._max_delay_seconds)
        jittered = capped * (0.5 + random.random() / 2)  # noqa: S311 -- backoff jitter, not security-sensitive
        await self._sleep(jittered)
