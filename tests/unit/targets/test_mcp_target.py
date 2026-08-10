"""Tests for McpTarget (ADR-0021).

Every test here spawns the scripted MCP stdio server under
``fixtures/mcp_server_target.py`` as a genuine subprocess via
``sys.executable`` -- no mocks, no network -- so the whole spawn ->
``initialize`` -> ``notifications/initialized`` -> ``tools/call`` ->
teardown exchange is exercised end to end. The fixture's first argument
selects one scripted behavior per test, which lets every row of
``McpTarget``'s error-mapping taxonomy be pinned individually: a tool
result carrying ``isError: true`` is the system under test reporting its
own failure (``FAILED``), a JSON-RPC error or any transport breakdown is
plumbing (``ERROR``), and an expired deadline is ``TIMEOUT``. The tests
live under ``tests/unit/targets/**`` rather than ``tests/integration/``
because this project's convention is to put a module's tests in the
unit-test directory that mirrors its own path.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Final, cast

import pytest

from agentic_evalkit.models import EvalSample, ExecutionStatus, NormalizedExecutionResult
from agentic_evalkit.targets import McpTarget
from agentic_evalkit.targets.mcp import (
    _MCP_PROTOCOL_VERSION,
    _initialize_frame,
    _ProtocolError,
    _write_frames,
)

_FIXTURES = Path(__file__).parent / "fixtures"

# Generous default: every fixture mode except "hang" answers in well under
# a second, so this only ever trips if something is genuinely wedged.
_TIMEOUT_SECONDS: Final = 10.0
# The hang test needs the timeout to actually fire, and quickly.
_HANG_TIMEOUT_SECONDS: Final = 1.0


def _sample(input_payload: dict[str, object] | None = None) -> EvalSample:
    """A sample following the tool/arguments input contract by default;
    validation tests pass their own, deliberately malformed payloads."""
    payload: dict[str, object] = {"tool": "echo", "arguments": {"question": "ping"}}
    if input_payload is not None:
        payload = input_payload
    return EvalSample(
        sample_id="s1",
        input=payload,
        source_digest="sha256:s1",
        adapter="identity@1",
    )


def _subprocess_env() -> dict[str, str]:
    """Copies the current environment variables, but with a couple removed
    that would otherwise make pytest-cov try (and fail) to measure code
    coverage inside the fixture subprocess.

    Here is the problem this avoids. ``McpTarget`` defaults to
    ``env=None``, meaning "give the child process the same environment
    variables as this one." When the whole test suite runs under
    ``pytest --cov``, pytest-cov sets a couple of environment variables
    (``COV_CORE_*`` and ``COVERAGE_PROCESS_START``) that tell any Python
    process "please measure your own code coverage too." That is normally
    useful, but it means each fixture script we spawn as a subprocess would
    also try to start its own, separate coverage collector -- one that
    tracks only which lines ran, not which branches of an if/else were
    taken. When pytest-cov later tries to merge that data into the main
    coverage report (``cov.combine()``), the mismatch between the two kinds
    of data crashes it with ``DataError: Can't combine statement coverage
    data with branch data``.

    The fixture scripts under ``tests/unit/targets/fixtures/`` are simple
    test-only stand-ins, not part of the real ``agentic_evalkit`` package,
    so we do not want or need coverage numbers from them anyway. Removing
    these two variables before spawning the subprocess stops pytest-cov
    from ever trying to measure them, so the crash never gets a chance to
    happen.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("COV_CORE_") and key != "COVERAGE_PROCESS_START"
    }


def _target(
    mode: str,
    *,
    max_output_bytes: int = 65_536,
    max_stderr_bytes: int = 65_536,
    tool_name: str | None = None,
    extra_args: tuple[str, ...] = (),
) -> McpTarget:
    return McpTarget(
        command=(sys.executable, str(_FIXTURES / "mcp_server_target.py"), mode, *extra_args),
        env=_subprocess_env(),
        max_output_bytes=max_output_bytes,
        max_stderr_bytes=max_stderr_bytes,
        tool_name=tool_name,
    )


@pytest.mark.asyncio
async def test_happy_path_completes_and_normalizes() -> None:
    target = _target("happy")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.COMPLETED
    assert result.error is None
    assert result.output is not None
    assert "ping" in result.output["text"]
    # The raw content array is preserved verbatim next to the joined text.
    assert isinstance(result.output["content"], list)
    assert result.output["content"][0]["type"] == "text"


@pytest.mark.asyncio
async def test_populates_tool_calls_and_latency() -> None:
    target = _target("happy")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.tool_calls == ({"name": "echo", "arguments": {"question": "ping"}},)
    assert result.latency_ms is not None
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_environment_metadata_carries_server_info() -> None:
    target = _target("happy")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.environment_metadata["server_info"]["name"] == "fixture"
    # The fixture echoes back whatever revision the client proposed, so this
    # doubles as an assertion that the proposal really goes over the wire.
    assert result.environment_metadata["protocol_version"] == _MCP_PROTOCOL_VERSION


@pytest.mark.asyncio
async def test_fixed_tool_name_override() -> None:
    """When the target pins a tool name, the sample supplies only arguments
    and the pinned name is what actually goes over the wire (the fixture
    echoes the received name back inside the text block)."""
    target = _target("happy", tool_name="pinned")
    sample = _sample({"arguments": {"q": 1}})
    result = await target.execute(sample, attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.COMPLETED
    assert result.output is not None
    assert "pinned" in result.output["text"]


@pytest.mark.asyncio
async def test_tool_key_rejected_when_tool_name_fixed() -> None:
    """A sample naming its own tool while the target pins one is rejected
    outright: allowing it would let a sample claim one tool while the
    target silently called another."""
    target = _target("happy", tool_name="pinned")
    sample = _sample({"tool": "echo", "arguments": {}})
    result = await target.execute(sample, attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "InvalidSampleInput"
    assert result.error["code"] == "target_failure"
    assert result.tool_calls == ()


@pytest.mark.asyncio
async def test_missing_tool_key_rejected() -> None:
    target = _target("happy")
    result = await target.execute(
        _sample({"arguments": {}}), attempt=1, timeout_seconds=_TIMEOUT_SECONDS
    )
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "InvalidSampleInput"


@pytest.mark.parametrize("bad_tool_input", [{"tool": 3}, {"tool": "  "}])
@pytest.mark.asyncio
async def test_non_string_or_empty_tool_rejected(bad_tool_input: dict[str, object]) -> None:
    target = _target("happy")
    result = await target.execute(
        _sample(bad_tool_input), attempt=1, timeout_seconds=_TIMEOUT_SECONDS
    )
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "InvalidSampleInput"


@pytest.mark.asyncio
async def test_arguments_must_be_object() -> None:
    target = _target("happy")
    result = await target.execute(
        _sample({"tool": "t", "arguments": [1]}), attempt=1, timeout_seconds=_TIMEOUT_SECONDS
    )
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "InvalidSampleInput"


@pytest.mark.asyncio
async def test_extra_input_key_rejected() -> None:
    """Unknown input keys fail loudly, naming the offender -- matching the
    wire models' ``extra="forbid"`` stance, so the ``"args"``/``"argument"``
    typo class never silently sends an empty argument object."""
    target = _target("happy")
    result = await target.execute(
        _sample({"tool": "t", "args": {}}), attempt=1, timeout_seconds=_TIMEOUT_SECONDS
    )
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "InvalidSampleInput"
    # The offending key is named as *unexpected* -- checking for the bare
    # substring "args" alone would pass trivially via "arguments".
    assert "unexpected" in str(result.error["message"])
    assert "args" in str(result.error["message"])


@pytest.mark.asyncio
async def test_tool_error_maps_to_failed() -> None:
    """``isError: true`` means the system under test ran and reported its
    own failure -- operational (FAILED), never graded, and never folded
    into plumbing errors (ADR-0008)."""
    target = _target("tool_error")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.FAILED
    assert result.error is not None
    assert result.error["type"] == "ToolCallError"
    assert result.error["code"] == "target_failure"
    assert result.error["message"] == "tool exploded"
    assert result.output is None


@pytest.mark.asyncio
async def test_jsonrpc_error_on_call_maps_to_error() -> None:
    target = _target("jsonrpc_error")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "JsonRpcError"
    assert result.error["jsonrpc_code"] == -32602
    # The tools/call frame had already been sent when the error came back,
    # so the attempted call is still on the record.
    assert result.tool_calls == ({"name": "echo", "arguments": {"question": "ping"}},)


@pytest.mark.parametrize(
    "mode", ["tool_result_not_object", "tool_result_bad_iserror", "tool_result_bad_content"]
)
@pytest.mark.asyncio
async def test_malformed_tool_result_reports_error(mode: str) -> None:
    """Each of the three shape checks on a tools/call result -- result must
    be an object, isError a boolean, content a list -- maps to a
    MalformedToolResult ERROR instead of leaking a raw exception."""
    target = _target(mode)
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "MalformedToolResult"


@pytest.mark.asyncio
async def test_jsonrpc_error_on_initialize_aborts() -> None:
    target = _target("init_error")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "JsonRpcError"
    assert result.error["jsonrpc_code"] == -32603
    # The handshake failed, so no tools/call was ever sent.
    assert result.tool_calls == ()


@pytest.mark.asyncio
async def test_malformed_initialize_result_reports_error() -> None:
    target = _target("bad_init_result")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "MalformedInitializeResult"
    assert result.tool_calls == ()


@pytest.mark.asyncio
async def test_malformed_json_reports_error_with_stderr() -> None:
    """Captured stderr must not be silently discarded on failure -- it is
    frequently the only clue explaining *why* an MCP server misbehaved."""
    target = _target("malformed")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "JSONDecodeError"
    assert "boom diagnostics" in str(result.error["stderr"])


@pytest.mark.asyncio
async def test_non_object_frame_reports_type_error() -> None:
    """A frame that parses as valid JSON but is not a JSON object (here, a
    bare number) is a wire-level type violation, reported with the actual
    offending type named in the message."""
    target = _target("non_object_frame")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "TypeError"
    assert "int" in str(result.error["message"])


@pytest.mark.asyncio
async def test_stale_id_response_is_skipped() -> None:
    target = _target("wrong_id_then_right")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_notification_interleaved_is_ignored() -> None:
    target = _target("notification_interleaved")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_server_ping_request_is_answered() -> None:
    """The fixture sends a server-initiated ``ping`` and exits with code 4
    unless the client answers it with an empty result -- so a COMPLETED
    outcome here proves the reply was actually sent and well-formed. The
    two regression shapes differ: a *bad* reply makes the fixture exit
    with code 4 (a fast ServerExited error), while a *missing* reply
    deadlocks both sides -- the fixture blocked reading stdin, the client
    awaiting the id-2 response -- and only resolves as TIMEOUT once the
    full timeout expires. Either way, not COMPLETED."""
    target = _target("server_request")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_hang_times_out_and_kills() -> None:
    target = _target("hang")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_HANG_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.TIMEOUT
    assert result.error is not None
    assert result.error["type"] == "TimeoutError"
    assert result.error["code"] == "target_timeout"


@pytest.mark.asyncio
async def test_oversized_line_reports_output_too_large() -> None:
    target = _target("oversized", max_output_bytes=1024)
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "OutputTooLarge"
    # The oversized stdout content is never copied into the result.
    assert result.output is None


@pytest.mark.asyncio
async def test_immediate_exit_reports_server_exited() -> None:
    """A server that dies before answering must produce a diagnosable
    ServerExited error carrying the exit code, not a hang or a bare EOF."""
    target = _target("exit_early")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "ServerExited"
    assert "exit code 3" in str(result.error["message"])


@pytest.mark.asyncio
async def test_server_death_after_init_maps_to_server_exited() -> None:
    """A server that answers ``initialize`` and then dies before answering
    ``tools/call`` must normalize to a ServerExited ERROR -- never escape
    as a raw OSError, and never burn the whole deadline into a misleading
    TIMEOUT. Which detection path fires first is platform timing -- stdout
    end-of-file, or the broken-pipe stdin write -- but both are mapped to
    the same taxonomy entry, so the assertion does not pin the message.
    The attempted call stays on the record either way, because it is
    recorded before the frames are flushed."""
    target = _target("exit_after_init")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "ServerExited"
    assert result.tool_calls == ({"name": "echo", "arguments": {"question": "ping"}},)


class _DeadPipeWriter:
    """Quacks like the two ``StreamWriter`` methods ``_write_frames`` uses,
    with the write failing the way a dead pipe's does. The subprocess test
    above cannot force the operating system to surface the broken pipe at
    the write site (``drain()`` only waits above the high-water mark, so
    the error usually lands in the transport instead), which is exactly
    why the mapping itself is pinned here directly."""

    def write(self, data: bytes) -> None:
        raise BrokenPipeError("read end closed")

    async def drain(self) -> None:  # pragma: no cover - write raises first
        raise AssertionError("drain should never be reached")


@pytest.mark.asyncio
async def test_write_frames_maps_dead_pipe_to_server_exited() -> None:
    """The stdin-write guard turns a pipe-death OSError into the same
    ServerExited protocol error the stdout-EOF path produces, so a raw
    OSError can never escape ``execute()`` no matter which side of the
    pipe notices the death first."""
    writer = cast("asyncio.StreamWriter", _DeadPipeWriter())
    with pytest.raises(_ProtocolError) as excinfo:
        await _write_frames(writer, b"{}\n")
    assert excinfo.value.error_type == "ServerExited"
    assert "stdin" in excinfo.value.message


@pytest.mark.asyncio
async def test_response_with_neither_result_nor_error_fails_fast() -> None:
    """A frame carrying the awaited id but neither ``result`` nor ``error``
    demonstrably IS the response -- and it is garbage. Reading on could
    only wait forever (or misreport the breakdown as TIMEOUT), so the
    client fails fast with MalformedResponse; the fixture stays alive
    after sending the bad frame to prove the quick escape is the
    fail-fast, not stdout end-of-file."""
    target = _target("no_result_no_error")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "MalformedResponse"


@pytest.mark.asyncio
async def test_alien_protocol_version_is_tolerated() -> None:
    """Version negotiation is deliberately tolerant: whatever revision
    string the server echoes is accepted and recorded, never gated on."""
    target = _target("alien_version")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.COMPLETED
    assert result.environment_metadata["protocol_version"] == "9999-01-01"


def test_proposed_revision_is_the_newest_handshake_based_one() -> None:
    """The proposal is 2025-11-25: newest revision built on ``initialize``.

    Deliberately not 2026-07-28 or later. Those revisions removed the
    handshake entirely -- no ``initialize``, per-request ``_meta`` carrying
    the version, a mandatory ``server/discover`` RPC -- so naming one inside
    an ``initialize`` frame would advertise a revision in which that very
    frame does not exist. This assertion fails the moment somebody bumps the
    constant past the handshake era without rewriting the exchange.
    """
    assert _MCP_PROTOCOL_VERSION == "2025-11-25"
    assert _MCP_PROTOCOL_VERSION < "2026-01-01"


def test_initialize_frame_carries_the_proposed_revision() -> None:
    """The constant is what actually reaches the wire, in the legacy slot.

    A handshake-era client puts the revision in ``params.protocolVersion``.
    It must not appear in a ``_meta`` block instead: that is the
    handshake-free carrier, and using it here would mix eras.
    """
    frame = _initialize_frame()
    params = cast("dict[str, object]", frame["params"])
    assert params["protocolVersion"] == _MCP_PROTOCOL_VERSION
    assert frame["method"] == "initialize"
    assert "_meta" not in frame
    assert "_meta" not in params


@pytest.mark.asyncio
async def test_non_text_blocks_join_text_only() -> None:
    """Non-text content blocks are not errors -- they simply do not
    contribute to the joined text, while the raw content array (all three
    blocks, including the image) is preserved verbatim."""
    target = _target("mixed_content")
    result = await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert result.status is ExecutionStatus.COMPLETED
    assert result.output is not None
    assert result.output["text"] == "a\nb"
    assert len(result.output["content"]) == 3


def test_empty_command_raises_value_error() -> None:
    with pytest.raises(ValueError, match="command"):
        McpTarget(command=())


@pytest.mark.asyncio
async def test_spawn_failure_propagates_oserror() -> None:
    """A command that cannot be spawned at all is deliberately not mapped
    into a normalized result: the OSError propagates so the runner's
    isolation layer records it, mirroring the sibling subprocess target."""
    target = McpTarget(command=("agentic-evalkit-no-such-executable",), env=_subprocess_env())
    with pytest.raises(OSError):
        await target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)


@pytest.mark.asyncio
async def test_fingerprint_stable_and_never_leaks_args() -> None:
    """The full argument vector is hashed into the fingerprint, never
    recorded in the clear: identical configurations must produce identical
    fingerprints, and a secret-looking argument value must never appear."""
    first = _target("happy", extra_args=("--secret=hunter2",))
    second = _target("happy", extra_args=("--secret=hunter2",))
    first_result = await first.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    second_result = await second.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    assert first_result.target_fingerprint is not None
    assert first_result.target_fingerprint.startswith("mcp:")
    assert first_result.target_fingerprint == second_result.target_fingerprint
    assert "hunter2" not in first_result.target_fingerprint


# --- Windows no-hang bound for the oversized-output teardown path ---
#
# Background: on Windows the default asyncio loop is the ProactorEventLoop,
# and after a "line too long" overrun its bookkeeping can miss that the
# underlying pipe closed, leaving ``Process.wait()`` hanging forever even
# though the real process already exited. ``SubprocessTarget._terminate``
# (which McpTarget reuses for teardown) works around this by killing the
# process, waiting only a short bounded time, and force-closing the
# transport as a last resort. The full account of the bug, and of why the
# wait below uses ``asyncio.wait`` instead of ``asyncio.wait_for`` (a task
# stuck in that state may be uncancellable, so ``wait_for``'s implicit
# cancel could itself hang the whole test run), lives in the sibling
# ``test_subprocess_target.py``. This test only runs on Windows, where the
# ProactorEventLoop is the default.

_WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32",
    reason="ProactorEventLoop is the default asyncio loop only on Windows",
)

# Generously larger than _terminate's own internal wait-after-kill timeout,
# so this bound only trips for a genuine, unbounded hang -- not for the
# ordinary, small delays involved in normal process cleanup.
_NO_HANG_WALL_CLOCK_SECONDS: Final = 20.0

# Bound for cleaning up the still-running task after we have already
# decided it hung: if the task is stuck in exactly the uncancellable state
# this test exists to catch, an unbounded wait on its cancellation would
# freeze the test run instead of reporting the AssertionError.
_CLEANUP_WALL_CLOCK_SECONDS: Final = 5.0


async def _run_within_no_hang_bound(target: McpTarget) -> NormalizedExecutionResult:
    """Runs ``target.execute(...)`` under a wall-clock bound via
    ``asyncio.wait``, which -- unlike ``asyncio.wait_for`` -- never tries
    to cancel the task itself when the timeout expires. If the task hangs,
    this raises a plain, readable ``AssertionError`` instead of freezing
    the whole test run on an uncancellable task.
    """
    task = asyncio.ensure_future(
        target.execute(_sample(), attempt=1, timeout_seconds=_TIMEOUT_SECONDS)
    )
    done, pending = await asyncio.wait({task}, timeout=_NO_HANG_WALL_CLOCK_SECONDS)
    if not done:
        for stuck in pending:
            stuck.cancel()
        await asyncio.wait(pending, timeout=_CLEANUP_WALL_CLOCK_SECONDS)
        raise AssertionError("teardown did not complete within the no-hang bound")
    return await next(iter(done))


def _assert_running_on_proactor_loop() -> None:
    """Skip rather than silently pass when some other event loop is active:
    the bug this guards against only happens on the ProactorEventLoop."""
    loop = asyncio.get_running_loop()
    proactor_loop = getattr(asyncio, "ProactorEventLoop", None)
    if proactor_loop is None or not isinstance(loop, proactor_loop):
        pytest.skip(f"active event loop is {type(loop).__name__}, not ProactorEventLoop")


@_WINDOWS_ONLY
@pytest.mark.asyncio
async def test_windows_oversized_response_does_not_hang() -> None:
    """An oversized response line on the ProactorEventLoop must still tear
    down cleanly -- killing the server and returning a normal OutputTooLarge
    ERROR result within the wall-clock bound -- rather than hanging forever
    in ``Process.wait()``.
    """
    _assert_running_on_proactor_loop()
    target = _target("oversized", max_output_bytes=1024)
    result = await _run_within_no_hang_bound(target)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    assert result.error["type"] == "OutputTooLarge"
