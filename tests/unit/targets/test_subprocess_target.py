"""Tests for SubprocessTarget (plan Task 9, Steps 3 and 6; design §8).

These scenarios are drawn verbatim from
``docs/plans/2026-07-02-agentic-evalkit-initial-release.md`` Task 9 Step 3
("Test that SubprocessTarget sends one request, enforces a one-second
timeout, caps standard error and output bytes, and reports malformed JSON as
ExecutionStatus.ERROR") and Step 6 ("Add a fixture that emits CRLF and a
response split across writes; it must parse identically on Windows and
Linux"). They are relocated to ``tests/unit/targets/**`` per this task's
explicit test-tree ownership rather than ``tests/integration/``.
"""

import sys
from pathlib import Path

import pytest

from agentic_evalkit.models import EvalSample, ExecutionStatus
from agentic_evalkit.targets import SubprocessTarget

_FIXTURES = Path(__file__).parent / "fixtures"


def _sample(sample_id: str = "s1") -> EvalSample:
    return EvalSample(
        sample_id=sample_id,
        input={"question": "ping"},
        source_digest="sha256:s1",
        adapter="identity@1",
    )


def _target(
    script: str,
    *,
    max_output_bytes: int = 65_536,
    max_stderr_bytes: int = 65_536,
) -> SubprocessTarget:
    return SubprocessTarget(
        command=(sys.executable, str(_FIXTURES / script)),
        max_output_bytes=max_output_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )


@pytest.mark.asyncio
async def test_sends_one_request_and_normalizes_echoed_output() -> None:
    target = _target("echo_target.py")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=5.0)
    assert result.status is ExecutionStatus.COMPLETED
    assert result.output == {"question": "ping"}
    assert result.sample_id == "s1"


@pytest.mark.asyncio
async def test_enforces_one_second_timeout_and_kills_hung_process() -> None:
    target = _target("hang_target.py")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=1.0)
    assert result.status is ExecutionStatus.TIMEOUT
    assert result.error is not None


@pytest.mark.asyncio
async def test_caps_standard_output_bytes() -> None:
    target = _target("oversized_output_target.py", max_output_bytes=1024)
    result = await target.execute(_sample(), attempt=1, timeout_seconds=5.0)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None


@pytest.mark.asyncio
async def test_caps_standard_error_bytes_while_still_completing() -> None:
    """Standard error is drained concurrently and bounded, but a valid stdout
    response still completes the exchange."""
    target = _target("oversized_stderr_target.py", max_stderr_bytes=1024)
    result = await target.execute(_sample(), attempt=1, timeout_seconds=5.0)
    assert result.status is ExecutionStatus.COMPLETED
    assert result.output == {"question": "ping"}


@pytest.mark.asyncio
async def test_reports_malformed_json_as_error_status() -> None:
    target = _target("malformed_json_target.py")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=5.0)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None


@pytest.mark.asyncio
async def test_error_result_surfaces_captured_stderr_as_diagnostic_evidence() -> None:
    """Captured stderr must not be silently discarded on failure -- it is
    frequently the only clue explaining *why* a subprocess target failed."""
    target = _target("stderr_then_malformed_target.py")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=5.0)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert "diagnostic-marker-from-stderr" in str(result.error)


@pytest.mark.asyncio
async def test_rejects_response_with_mismatched_sample_id() -> None:
    target = _target("mismatched_sample_id_target.py")
    result = await target.execute(_sample("s1"), attempt=1, timeout_seconds=5.0)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None


@pytest.mark.asyncio
async def test_parses_crlf_terminated_response_split_across_writes() -> None:
    """Windows/CRLF-safety: StreamReader.readline() reassembles a response
    written in fragments and terminated with \\r\\n into exactly one JSON
    object, on both Windows and Linux."""
    target = _target("crlf_split_target.py")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=5.0)
    assert result.status is ExecutionStatus.COMPLETED
    assert result.output == {"question": "ping"}


@pytest.mark.asyncio
async def test_records_command_executable_name_but_not_full_argv_secrets() -> None:
    target = _target("echo_target.py")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=5.0)
    assert result.target_fingerprint is not None
    assert "subprocess:" in result.target_fingerprint


@pytest.mark.asyncio
async def test_uses_no_shell_and_argument_tuple() -> None:
    """Regression guard: command must be a tuple/sequence, never a shell string,
    so shell metacharacters in sample input can never be interpreted."""
    target = SubprocessTarget(
        command=(sys.executable, str(_FIXTURES / "echo_target.py")),
        max_output_bytes=65_536,
        max_stderr_bytes=65_536,
    )
    assert isinstance(target.command, tuple)
    sample = EvalSample(
        sample_id="s1",
        input={"question": "echo hi; rm -rf /"},
        source_digest="sha256:s1",
        adapter="identity@1",
    )
    result = await target.execute(sample, attempt=1, timeout_seconds=5.0)
    assert result.status is ExecutionStatus.COMPLETED
    assert result.output == {"question": "echo hi; rm -rf /"}
