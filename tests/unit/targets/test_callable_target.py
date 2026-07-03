"""Contract tests for CallableTarget (plan Task 9, Step 2; design §8).

The first test below is copied verbatim from
``docs/plans/2026-07-02-agentic-evalkit-initial-release.md`` (Task 9, Step 2)
where it lives as ``tests/contract/test_targets.py``; it is relocated here
per this task's explicit test-tree ownership (``tests/unit/targets/**``)
while keeping the test body byte-for-byte unmodified.
"""

import asyncio

import pytest

from agentic_evalkit.errors import TargetFailure, TargetTimeout
from agentic_evalkit.models import EvalSample, ExecutionStatus
from agentic_evalkit.targets import CallableTarget


@pytest.mark.asyncio
async def test_callable_target_normalizes_output_and_timeout() -> None:
    sample = EvalSample(
        sample_id="s1",
        input={"question": "ping"},
        source_digest="sha256:s1",
        adapter="identity@1",
    )
    target = CallableTarget(lambda value: {"answer": value["question"]}, name="echo")
    result = await target.execute(sample, attempt=1, timeout_seconds=1.0)
    assert result.status is ExecutionStatus.COMPLETED
    assert result.output == {"answer": "ping"}
    assert result.target_fingerprint.startswith("callable:echo:")


# --- Additional CallableTarget coverage (sync/async, errors, timeout) ------


def _sample(sample_id: str = "s1") -> EvalSample:
    return EvalSample(
        sample_id=sample_id,
        input={"question": "ping"},
        source_digest="sha256:s1",
        adapter="identity@1",
    )


@pytest.mark.asyncio
async def test_callable_target_supports_async_callables() -> None:
    async def handler(value: dict[str, object]) -> dict[str, object]:
        await asyncio.sleep(0)
        return {"answer": value["question"]}

    target = CallableTarget(handler, name="async-echo")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=1.0)
    assert result.status is ExecutionStatus.COMPLETED
    assert result.output == {"answer": "ping"}


@pytest.mark.asyncio
async def test_callable_target_fingerprint_is_stable_across_calls() -> None:
    def handler(value: dict[str, object]) -> dict[str, object]:
        return {"answer": value["question"]}

    target = CallableTarget(handler, name="echo")
    first = await target.execute(_sample("s1"), attempt=1, timeout_seconds=1.0)
    second = await target.execute(_sample("s2"), attempt=1, timeout_seconds=1.0)
    assert first.target_fingerprint == second.target_fingerprint


@pytest.mark.asyncio
async def test_callable_target_sync_function_runs_off_event_loop() -> None:
    """A slow sync callable must not block other coroutines (asyncio.to_thread)."""
    import time

    marker: list[str] = []

    def slow(value: dict[str, object]) -> dict[str, object]:
        time.sleep(0.2)
        marker.append("slow-done")
        return {"answer": "done"}

    async def ticker() -> None:
        await asyncio.sleep(0.05)
        marker.append("tick")

    target = CallableTarget(slow, name="slow")
    results = await asyncio.gather(
        target.execute(_sample(), attempt=1, timeout_seconds=2.0),
        ticker(),
    )
    execution_result = results[0]
    assert execution_result.status is ExecutionStatus.COMPLETED
    # The ticker must have interleaved *before* the slow sync callable finished,
    # proving the sync callable ran off the event loop thread.
    assert marker == ["tick", "slow-done"]


@pytest.mark.asyncio
async def test_callable_target_times_out_long_running_callable() -> None:
    import time

    def slow(value: dict[str, object]) -> dict[str, object]:
        time.sleep(5)
        return {"answer": "too-late"}

    target = CallableTarget(slow, name="slow")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=0.05)
    assert result.status is ExecutionStatus.TIMEOUT
    assert result.error is not None


@pytest.mark.asyncio
async def test_callable_target_converts_exception_to_error_result_without_leaking_locals() -> None:
    def raiser(value: dict[str, object]) -> dict[str, object]:
        secret_local_variable = "super-secret-value-should-not-leak"  # noqa: F841
        raise RuntimeError("boom")

    target = CallableTarget(raiser, name="raiser")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=1.0)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert "super-secret-value-should-not-leak" not in str(result.error)
    assert "boom" in str(result.error)


@pytest.mark.asyncio
async def test_callable_target_rejects_non_mapping_return_as_error() -> None:
    def bad_return(value: dict[str, object]) -> object:
        return "not-a-mapping"

    target = CallableTarget(bad_return, name="bad-return")  # type: ignore[arg-type]
    result = await target.execute(_sample(), attempt=1, timeout_seconds=1.0)
    assert result.status is ExecutionStatus.ERROR


def test_callable_target_execute_signature_accepts_keyword_only_attempt_and_timeout() -> None:
    """Design §8 / plan Step 5: execute(sample, *, attempt, timeout_seconds)."""
    import inspect

    signature = inspect.signature(CallableTarget.execute)
    params = signature.parameters
    assert params["attempt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["timeout_seconds"].kind is inspect.Parameter.KEYWORD_ONLY


def test_target_failure_and_target_timeout_are_importable_from_errors() -> None:
    """Sanity check binding target errors to the shared typed-error hierarchy."""
    assert issubclass(TargetTimeout, Exception)
    assert issubclass(TargetFailure, Exception)
