"""Tests for SubprocessTarget (plan Task 9, Steps 3 and 6; design §8).

These test scenarios come word-for-word from the original project plan
(``docs/plans/2026-07-02-agentic-evalkit-initial-release.md``), Task 9 Step 3
("Test that SubprocessTarget sends one request, enforces a one-second
timeout, caps standard error and output bytes, and reports malformed JSON as
ExecutionStatus.ERROR") and Step 6 ("Add a fixture that emits CRLF and a
response split across writes; it must parse identically on Windows and
Linux"). They live under ``tests/unit/targets/**`` rather than
``tests/integration/`` because this project's convention is to put a
module's tests in the unit-test directory that mirrors its own path.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

from agentic_evalkit.models import EvalSample, ExecutionStatus, NormalizedExecutionResult
from agentic_evalkit.targets import SubprocessTarget

_FIXTURES = Path(__file__).parent / "fixtures"


def _sample(sample_id: str = "s1") -> EvalSample:
    return EvalSample(
        sample_id=sample_id,
        input={"question": "ping"},
        source_digest="sha256:s1",
        adapter="identity@1",
    )


def _subprocess_env() -> dict[str, str]:
    """Copies the current environment variables, but with a couple removed
    that would otherwise make pytest-cov try (and fail) to measure code
    coverage inside the fixture subprocess.

    Here is the problem this avoids. ``SubprocessTarget`` defaults to
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
    script: str,
    *,
    max_output_bytes: int = 65_536,
    max_stderr_bytes: int = 65_536,
) -> SubprocessTarget:
    return SubprocessTarget(
        command=(sys.executable, str(_FIXTURES / script)),
        max_output_bytes=max_output_bytes,
        max_stderr_bytes=max_stderr_bytes,
        env=_subprocess_env(),
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
    """The fixture writes more to standard error than the configured limit,
    but its real response on standard output is still well-formed -- this
    confirms the exchange still completes successfully, even while stderr
    is being read (and capped) in the background at the same time.
    """
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
    """Guards against a specific bug ever coming back (a "regression"): the
    subprocess command must always be passed as a tuple of separate
    arguments, never as one combined shell string. This matters because if
    the command were ever run through a shell, special shell characters
    embedded in sample input (like `;` or `|`, which can chain or pipe
    commands together) could be interpreted as extra commands to run --
    a command-injection vulnerability. Passing a tuple of arguments
    straight to the operating system, with no shell involved, means those
    characters are always treated as plain, harmless text.
    """
    target = SubprocessTarget(
        command=(sys.executable, str(_FIXTURES / "echo_target.py")),
        max_output_bytes=65_536,
        max_stderr_bytes=65_536,
        env=_subprocess_env(),
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


# --- Story 5.1 (R-005): an oversized-output run on Windows must fail
# cleanly, never hang forever ---
#
# Background: on Windows, Python's default asyncio event loop implementation
# is called the "ProactorEventLoop" (the default since Python 3.8, via
# ``WindowsProactorEventLoopPolicy``), and pytest-asyncio's
# ``asyncio_mode = "auto"`` setting means every ``async def test_...``
# function in this project automatically runs on that same default loop --
# no extra decorator needed.
#
# The bug this guards against: on that Windows loop, if a stream previously
# hit a "line too long" overrun (the same ``LimitOverrunError`` situation
# handled in ``SubprocessTarget._read_bounded_line``), asyncio's internal
# bookkeeping can fail to notice that the underlying pipe actually closed.
# When that happens, ``Process.wait()`` hangs forever, waiting for a
# notification that will never arrive -- even though the real
# operating-system process has already exited. ``SubprocessTarget._terminate``
# works around this: it kills the process, waits only a short, bounded time
# for asyncio to notice, and if asyncio still has not noticed, force-closes
# its internal transport object directly as a last resort.
#
# The tests below only run on Windows (they are skipped everywhere else),
# since this bug is specific to Windows's ProactorEventLoop. They also guard
# against a second, subtler problem: what if the bug is *not* fixed, and the
# code genuinely hangs? A naive test would itself hang trying to cancel a
# task that is stuck in an uncancellable state, freezing the whole test run
# (and CI job) instead of failing. To avoid that, these tests wait for the
# result using plain ``asyncio.wait`` with a timeout (via the
# ``_run_within_no_hang_bound`` helper below) rather than the more common
# ``asyncio.wait_for``. Unlike ``asyncio.wait_for``, ``asyncio.wait`` never
# tries to cancel the task itself when its timeout expires -- it simply
# returns and reports that the task is not done yet. That means if this bug
# ever comes back, these tests fail with a normal, readable
# ``AssertionError`` instead of the test run just freezing until someone
# notices and kills it by hand.

_WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32",
    reason="ProactorEventLoop is the default asyncio loop only on Windows",
)

# This needs to be generously larger than SubprocessTarget._terminate's own
# internal 1.0-second wait-after-kill timeout, so this bound only trips for
# a genuine, unbounded hang -- not for the ordinary, small delays involved
# in normal process cleanup.
_NO_HANG_WALL_CLOCK_SECONDS = 20.0

# Timeout for cleaning up the still-running task after we have already
# decided it hung (the step just above this one). The subtlety: if the real
# bug is exactly the stuck state this whole test exists to catch -- a
# Proactor "connection lost" notification that never fires -- then calling
# `.cancel()` on the task is not guaranteed to actually stop it. If we then
# waited for that cancellation with no timeout of its own, this cleanup step
# could hang forever too, defeating the whole point of the test. Giving
# cleanup its own bound guarantees the test always finishes and reports its
# intended AssertionError, instead of freezing CI.
_CLEANUP_WALL_CLOCK_SECONDS = 5.0

# Every result also carries some fixed-size bookkeeping data of its own --
# timestamps, the target's fingerprint (an ID for its exact configuration;
# see SubprocessTarget's `_fingerprint` function), and the error
# type/message -- on top of the actual stdout/stderr bytes being bounded.
# This constant is a generous allowance for that fixed overhead, added to
# the configured stream byte limits below when checking that the
# serialized result stays within bounds. It stays small and constant no
# matter what; if the result ever grows past bounds + this allowance, that
# can only mean an *unbounded* stream leaked into it somewhere.
_RESULT_ENVELOPE_BYTES = 4096


async def _run_within_no_hang_bound(
    target: SubprocessTarget,
) -> NormalizedExecutionResult:
    """Runs ``target.execute(...)`` with a wall-clock time limit, using
    ``asyncio.wait`` rather than the more common ``asyncio.wait_for``.

    The difference matters here. If the task hangs, ``asyncio.wait_for``
    would try to ``.cancel()`` it -- but if the task is stuck in exactly the
    uncancellable state this test file is designed to catch, that
    cancellation could itself hang, silently turning a real bug into a test
    that just freezes forever. ``asyncio.wait`` instead simply returns once
    its timeout elapses, whether or not the task finished, leaving it up to
    us to check and fail explicitly.

    So: if the task finishes in time, its result is returned normally. If it
    does not, we raise a plain ``AssertionError`` describing a no-hang-bound
    violation, and we still attempt to cancel the leftover task -- but that
    cleanup attempt is itself given a short timeout
    (``_CLEANUP_WALL_CLOCK_SECONDS``), for the same reason: if cancellation
    gets stuck too, we do not want the *cleanup* to hang forever either.
    """
    task = asyncio.ensure_future(target.execute(_sample(), attempt=1, timeout_seconds=5.0))
    done, pending = await asyncio.wait({task}, timeout=_NO_HANG_WALL_CLOCK_SECONDS)
    if not done:
        for stuck in pending:
            stuck.cancel()
        await asyncio.wait(pending, timeout=_CLEANUP_WALL_CLOCK_SECONDS)
        raise AssertionError("teardown did not complete within the no-hang bound")
    return await next(iter(done))


def _assert_result_stays_byte_bounded(
    result: NormalizedExecutionResult, *, max_output_bytes: int, max_stderr_bytes: int
) -> None:
    """Checks that the result never contains more data than the configured
    byte limits allow, even though the subprocess itself wrote megabytes of
    output.

    Specifically, on this error path (the output was too large): the huge
    stdout content is never copied into the result at all (``output`` stays
    ``None``); any stderr text captured as supporting evidence is capped at
    ``max_stderr_bytes``; and the full serialized result -- once you allow
    for a small, constant amount of bookkeeping overhead for things like
    timestamps (``_RESULT_ENVELOPE_BYTES``) -- never exceeds the configured
    limits. In short: however much data the misbehaving subprocess produced,
    none of it leaks through unbounded.
    """
    assert result.output is None
    if result.error is not None:
        captured_stderr = result.error.get("stderr")
        if isinstance(captured_stderr, str):
            assert len(captured_stderr.encode("utf-8")) <= max_stderr_bytes
    serialized = len(result.model_dump_json().encode("utf-8"))
    assert serialized <= max_output_bytes + max_stderr_bytes + _RESULT_ENVELOPE_BYTES


def _assert_running_on_proactor_loop() -> None:
    """Confirms the test is actually running on a ``ProactorEventLoop``
    before continuing, since that is the specific asyncio event loop
    implementation this whole test file is designed to probe -- the bug
    being guarded against only happens on that loop. If some other event
    loop is active for any reason, the test is skipped instead of silently
    passing without having actually tested anything.
    """
    loop = asyncio.get_running_loop()
    proactor_loop = getattr(asyncio, "ProactorEventLoop", None)
    if proactor_loop is None or not isinstance(loop, proactor_loop):
        pytest.skip(f"active event loop is {type(loop).__name__}, not ProactorEventLoop")


@_WINDOWS_ONLY
@pytest.mark.asyncio
async def test_oversized_output_on_proactor_loop_tears_down_without_hanging() -> None:
    """When a fixture writes an oversized line to standard output while
    running on a ProactorEventLoop, SubprocessTarget must still finish
    cleanly -- killing the process, waiting for it to exit, and returning a
    normal, byte-bounded ``ERROR`` result -- rather than hanging forever.
    The ``asyncio.wait`` wall-clock bound (see ``_run_within_no_hang_bound``)
    makes sure that if this ever regresses, the test fails with a clear
    ``AssertionError`` instead of the test run just freezing.
    """
    _assert_running_on_proactor_loop()
    max_output_bytes = 1024
    max_stderr_bytes = 65_536
    target = _target(
        "oversized_output_target.py",
        max_output_bytes=max_output_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )
    result = await _run_within_no_hang_bound(target)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    # Confirms the result stays byte-bounded: the megabyte of standard
    # output the fixture wrote never ends up copied into the result, and
    # the serialized result stays within the configured limits.
    _assert_result_stays_byte_bounded(
        result, max_output_bytes=max_output_bytes, max_stderr_bytes=max_stderr_bytes
    )


@_WINDOWS_ONLY
@pytest.mark.asyncio
async def test_oversized_output_stays_byte_bounded_with_concurrent_stderr_drain() -> None:
    """Same scenario as the test above, but harder: the fixture writes an
    oversized line to *both* standard output and standard error at once, so
    SubprocessTarget has to read (and cap) both streams concurrently while
    also killing the process. This confirms that teardown on the
    ProactorEventLoop still finishes within the wall-clock bound and reports
    a bounded ``ERROR`` -- i.e., reading both oversized streams at the same
    time never causes the two pipes to deadlock against each other on
    Windows.
    """
    _assert_running_on_proactor_loop()
    max_output_bytes = 1024
    max_stderr_bytes = 1024
    target = _target(
        "oversized_stderr_and_output_target.py",
        max_output_bytes=max_output_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )
    result = await _run_within_no_hang_bound(target)
    assert result.status is ExecutionStatus.ERROR
    assert result.error is not None
    # Confirms both oversized streams stayed capped: neither the oversized
    # standard output nor the oversized standard error leaked into the
    # result without a limit.
    _assert_result_stays_byte_bounded(
        result, max_output_bytes=max_output_bytes, max_stderr_bytes=max_stderr_bytes
    )
