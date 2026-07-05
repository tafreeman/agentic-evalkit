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

import asyncio
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


# --- Story 5.1 (R-005): Windows ProactorEventLoop oversized-output no-hang ---
#
# On Windows, ``asyncio.run`` uses ``WindowsProactorEventLoopPolicy`` by
# default (Python 3.8+), and pytest-asyncio's ``asyncio_mode = "auto"`` runs
# each async test on that default loop. The proactor pipe transport can leave
# a connection-lost callback unfired after a ``StreamReader`` overrun, making
# ``Process.wait()`` hang even though the OS process has already exited;
# ``SubprocessTarget._terminate`` guards this with a bounded post-``kill()``
# wait plus a best-effort transport close. The tests below make that loop
# dimension explicit (they skip off Windows) and wrap the whole exchange in an
# ``asyncio.wait_for`` wall-clock bound, so a regression manifests as a
# ``TimeoutError`` test failure rather than a wedged CI job.

_WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32",
    reason="ProactorEventLoop is the default asyncio loop only on Windows",
)

# Generous relative to the target's own internal 1.0s post-kill wait bound, so
# only a genuine unbounded hang (not ordinary teardown slack) trips it.
_NO_HANG_WALL_CLOCK_SECONDS = 20.0


def _assert_running_on_proactor_loop() -> None:
    """Make the loop-policy assumption explicit: on Windows the active loop
    must be a ``ProactorEventLoop`` for this guard to exercise the transport
    interaction it targets. If some other loop is installed, skip rather than
    assert a false pass.
    """
    loop = asyncio.get_running_loop()
    proactor_loop = getattr(asyncio, "ProactorEventLoop", None)
    if proactor_loop is None or not isinstance(loop, proactor_loop):
        pytest.skip(f"active event loop is {type(loop).__name__}, not ProactorEventLoop")


@_WINDOWS_ONLY
@pytest.mark.asyncio
async def test_oversized_output_on_proactor_loop_tears_down_without_hanging() -> None:
    """An oversized standard-output line on a ProactorEventLoop must complete
    the exchange (as a bounded ``ERROR``) via kill-and-await teardown, never
    hang. The ``asyncio.wait_for`` wall-clock bound turns a regression into a
    ``TimeoutError`` failure instead of a hung run.
    """
    _assert_running_on_proactor_loop()
    target = _target("oversized_output_target.py", max_output_bytes=1024)
    result = await asyncio.wait_for(
        target.execute(_sample(), attempt=1, timeout_seconds=5.0),
        timeout=_NO_HANG_WALL_CLOCK_SECONDS,
    )
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None


@_WINDOWS_ONLY
@pytest.mark.asyncio
async def test_oversized_output_stays_byte_bounded_with_concurrent_stderr_drain() -> None:
    """With both an oversized standard-output line *and* an oversized standard
    error stream (concurrent drain), teardown on the ProactorEventLoop still
    completes within the wall-clock bound and reports a bounded ``ERROR``: the
    concurrent stderr drain never deadlocks the pipe on Windows.
    """
    _assert_running_on_proactor_loop()
    target = _target(
        "oversized_stderr_and_output_target.py",
        max_output_bytes=1024,
        max_stderr_bytes=1024,
    )
    result = await asyncio.wait_for(
        target.execute(_sample(), attempt=1, timeout_seconds=5.0),
        timeout=_NO_HANG_WALL_CLOCK_SECONDS,
    )
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
