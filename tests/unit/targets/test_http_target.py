"""Tests for HttpTarget (plan Task 9, Steps 3 and 7; design §8).

Plan Task 9 Step 3: "Use httpx.MockTransport to test that HttpTarget POSTs
schema version, sample ID, input, attempt, and trace ID; maps a valid
response; redacts authorization headers from evidence; maps 429 to a
retryable target error; and maps deadline expiry to TIMEOUT." Relocated to
``tests/unit/targets/**`` per this task's explicit test-tree ownership
rather than ``tests/integration/``.
"""

import asyncio
import json

import httpx
import pytest

from agentic_evalkit.models import EvalSample, ExecutionStatus
from agentic_evalkit.targets import HttpTarget


def _sample(sample_id: str = "s1") -> EvalSample:
    return EvalSample(
        sample_id=sample_id,
        input={"question": "ping"},
        source_digest="sha256:s1",
        adapter="identity@1",
    )


@pytest.mark.asyncio
async def test_posts_schema_version_sample_id_input_attempt_and_trace_id() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"sample_id": "s1", "output": {"answer": "pong"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = HttpTarget(client=client, url="https://example.test/execute", name="remote")
    result = await target.execute(_sample(), attempt=2, timeout_seconds=5.0, trace_id="trace-xyz")
    await client.aclose()

    assert result.status is ExecutionStatus.COMPLETED
    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert body["schema_version"] == "1"
    assert body["sample_id"] == "s1"
    assert body["input"] == {"question": "ping"}
    assert body["attempt"] == 2
    assert body["trace_id"] == "trace-xyz"


@pytest.mark.asyncio
async def test_maps_valid_response_to_completed_with_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"sample_id": "s1", "output": {"answer": "pong"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = HttpTarget(client=client, url="https://example.test/execute", name="remote")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=5.0)
    await client.aclose()

    assert result.status is ExecutionStatus.COMPLETED
    assert result.output == {"answer": "pong"}
    assert result.sample_id == "s1"


@pytest.mark.asyncio
async def test_redacts_authorization_header_from_evidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"sample_id": "s1", "output": {"answer": "pong"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = HttpTarget(
        client=client,
        url="https://example.test/execute",
        name="remote",
        headers=lambda: {"Authorization": "Bearer super-secret-token", "X-Trace": "abc"},
    )
    result = await target.execute(_sample(), attempt=1, timeout_seconds=5.0)
    await client.aclose()

    assert result.status is ExecutionStatus.COMPLETED
    serialized = json.dumps(
        {
            "environment_metadata": result.environment_metadata,
            "error": result.error,
        }
    )
    assert "super-secret-token" not in serialized
    # A redaction placeholder or absence is acceptable; the raw secret must not leak.
    recorded_headers = result.environment_metadata.get("request_headers")
    if recorded_headers is not None:
        assert recorded_headers.get("authorization") in ("***redacted***", None)  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_maps_429_to_retryable_target_error_after_exhausting_retries() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate_limited"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = HttpTarget(
        client=client,
        url="https://example.test/execute",
        name="remote",
        max_retries=2,
        sleep=_no_op_sleep,
    )
    result = await target.execute(_sample(), attempt=1, timeout_seconds=5.0)
    await client.aclose()

    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    # bounded retries: initial attempt + max_retries retries, never unbounded
    assert call_count == 3


@pytest.mark.asyncio
async def test_429_then_200_succeeds_via_retry() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"sample_id": "s1", "output": {"answer": "pong"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = HttpTarget(
        client=client,
        url="https://example.test/execute",
        name="remote",
        max_retries=2,
        sleep=_no_op_sleep,
    )
    result = await target.execute(_sample(), attempt=1, timeout_seconds=5.0)
    await client.aclose()

    assert result.status is ExecutionStatus.COMPLETED
    assert call_count == 2


@pytest.mark.asyncio
async def test_maps_deadline_expiry_to_timeout_status() -> None:
    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, json={"sample_id": "s1", "output": {}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(slow_handler))
    target = HttpTarget(client=client, url="https://example.test/execute", name="remote")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=0.05)
    await client.aclose()

    assert result.status is ExecutionStatus.TIMEOUT
    assert result.error is not None


@pytest.mark.asyncio
async def test_never_retries_nonretryable_4xx_response() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(400, json={"error": "bad_request"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = HttpTarget(
        client=client,
        url="https://example.test/execute",
        name="remote",
        max_retries=3,
        sleep=_no_op_sleep,
    )
    result = await target.execute(_sample(), attempt=1, timeout_seconds=5.0)
    await client.aclose()

    assert result.status is ExecutionStatus.ERROR
    assert call_count == 1  # attempted exactly once -- 400 is never retried


@pytest.mark.asyncio
async def test_retries_502_503_504_with_bounded_backoff() -> None:
    for status_code in (502, 503, 504):
        call_count = 0

        def handler(request: httpx.Request, status: int = status_code) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return httpx.Response(status)
            return httpx.Response(200, json={"sample_id": "s1", "output": {"ok": True}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        target = HttpTarget(
            client=client,
            url="https://example.test/execute",
            name="remote",
            max_retries=2,
            sleep=_no_op_sleep,
        )
        result = await target.execute(_sample(), attempt=1, timeout_seconds=5.0)
        await client.aclose()
        assert result.status is ExecutionStatus.COMPLETED, f"status_code={status_code}"


@pytest.mark.asyncio
async def test_retries_connection_errors() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"sample_id": "s1", "output": {"ok": True}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = HttpTarget(
        client=client,
        url="https://example.test/execute",
        name="remote",
        max_retries=2,
        sleep=_no_op_sleep,
    )
    result = await target.execute(_sample(), attempt=1, timeout_seconds=5.0)
    await client.aclose()

    assert result.status is ExecutionStatus.COMPLETED
    assert call_count == 2


@pytest.mark.asyncio
async def test_rejects_response_with_mismatched_sample_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"sample_id": "wrong-id", "output": {}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = HttpTarget(client=client, url="https://example.test/execute", name="remote")
    result = await target.execute(_sample("s1"), attempt=1, timeout_seconds=5.0)
    await client.aclose()

    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None


async def _no_op_sleep(_seconds: float) -> None:
    """Injected sleep function for deterministic, fast retry tests."""
    return None
