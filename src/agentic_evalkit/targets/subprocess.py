"""SubprocessTarget: runs a subprocess as the system under test (design §8).

The exchange is simple: write one line of compact, UTF-8-encoded JSON to
the process's standard input (stdin), then close stdin so the process
knows no more requests are coming, then read back one line of JSON from
its standard output (stdout) as the response. ("JSONL" means one JSON
object per line, rather than one big multi-line JSON document.)

A process may write its response in several small pieces instead of all
at once, so responses are read with ``StreamReader.readline()``, which
waits for and glues those pieces together until it has one full line --
this works the same way on every platform. Windows text lines
conventionally end with two characters, ``\\r\\n`` (CRLF), while
Linux/Mac use just ``\\n`` (LF); both of those trailing characters are
stripped here, so a line parses the same way regardless of which
convention the subprocess used.

Both the stdout and stderr streams have a maximum number of bytes we will
accept from them. Standard error is read in the background at the same
time we wait for the standard-output response, rather than only
afterward, because operating-system pipes have a limited buffer: if a
process writes a lot to stderr and nothing is reading it, the pipe fills
up and the process blocks trying to write more -- and if we were
meanwhile blocked waiting on stdout, which would never arrive because the
process is stuck, both sides would freeze forever. Reading stderr
concurrently avoids that standoff.

If the overall timeout expires, the process is forcibly killed and we
wait for the operating system to confirm it has actually exited, so a
timed-out run never leaves a stray process still running in the
background.
"""

import asyncio
import contextlib
import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast

from agentic_evalkit.models import EvalSample, ExecutionStatus, NormalizedExecutionResult

if TYPE_CHECKING:
    from pydantic import JsonValue

_PROTOCOL_VERSION: Final[str] = "1"
_DEFAULT_MAX_OUTPUT_BYTES: Final[int] = 1024 * 1024
_DEFAULT_MAX_STDERR_BYTES: Final[int] = 64 * 1024


def _fingerprint(command: tuple[str, ...]) -> str:
    """Build a stable fingerprint identifying this exact subprocess command.

    Only the program's filename -- not its full directory path -- is
    included in the fingerprint as plain, readable text. The full command
    line, including all of its arguments, is hashed instead of being
    written out directly: in some deployments those arguments are derived
    from the sample being evaluated and could contain sensitive values, so
    we still want the fingerprint to change if the command changes, without
    ever writing the actual argument values into it. In short: record the
    program's name and the protocol version in the clear, but never
    anything that might be a secret.
    """
    executable_name = command[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if command else "unknown"
    digest = hashlib.sha256(":".join(command).encode()).hexdigest()[:16]
    return f"subprocess:{executable_name}:{_PROTOCOL_VERSION}:{digest}"


class _ByteBoundExceededError(Exception):
    """Raised internally when a stream exceeds its configured byte bound."""


async def _read_bounded_line(reader: asyncio.StreamReader, *, max_bytes: int) -> bytes:
    """Read one line, enforcing a maximum size in bytes.

    ``readline()`` waits for and reassembles chunks arriving across
    multiple writes into one complete line ending in ``\\n``. If the line
    actually used Windows-style ``\\r\\n`` endings, the ``\\r`` is not
    treated specially here -- it comes back as the last content byte of
    the line, and it is the caller's job to strip it afterward.

    This stream was set up (when the subprocess was created) with its
    internal buffer size capped at ``max_bytes``, so if more than that many
    bytes arrive without ever reaching a line ending, ``readline()`` itself
    raises a ``ValueError`` (specifically ``asyncio.LimitOverrunError``) --
    asyncio enforces the limit for us in that case. That exception is
    caught here and converted into our own ``_ByteBoundExceededError``, the
    same error type used by the plain length check just below (which is
    technically redundant with asyncio's own limit, but kept as an
    explicit, easy-to-read safeguard). Either way, a response that is too
    large becomes a clear, reportable error instead of a silent hang or a
    confusing low-level exception leaking out.
    """
    try:
        line = await reader.readline()
    except ValueError as exc:
        raise _ByteBoundExceededError(f"line exceeded {max_bytes} byte bound") from exc
    if len(line) > max_bytes:
        raise _ByteBoundExceededError(f"line exceeded {max_bytes} byte bound")
    return line


async def _drain_stderr(reader: asyncio.StreamReader, *, max_bytes: int) -> bytes:
    """Read standard error in the background; keep only the first bytes up to the bound.

    This function keeps reading for as long as the process keeps writing
    to stderr -- not just until it hits ``max_bytes`` -- even though
    everything past that point is thrown away. That is deliberate: if
    reading stopped once the limit was hit, and the process kept writing
    more, the operating system's pipe buffer would fill up and the process
    would block trying to write, while we were separately waiting for it
    to finish and send its stdout response. Reading (and discarding)
    everything keeps that pipe empty so the process is never stuck
    waiting on us.
    """
    collected = bytearray()
    while True:
        chunk = await reader.read(65_536)
        if not chunk:
            break
        if len(collected) < max_bytes:
            remaining = max_bytes - len(collected)
            collected.extend(chunk[:remaining])
    return bytes(collected)


class SubprocessTarget:
    """Runs a subprocess as the system under test, exchanging JSON lines over stdin/stdout."""

    def __init__(
        self,
        *,
        command: tuple[str, ...],
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        max_stderr_bytes: int = _DEFAULT_MAX_STDERR_BYTES,
        env: dict[str, str] | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must be a nonempty argument tuple")
        self.command: tuple[str, ...] = command
        self._max_output_bytes = max_output_bytes
        self._max_stderr_bytes = max_stderr_bytes
        self._env = env
        self._fingerprint = _fingerprint(command)

    async def execute(
        self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
    ) -> NormalizedExecutionResult:
        started_at = datetime.now(UTC)
        try:
            async with asyncio.timeout(timeout_seconds):
                return await self._run_exchange(sample, attempt=attempt, started_at=started_at)
        except TimeoutError:
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.TIMEOUT,
                started_at=started_at,
                error_type="TimeoutError",
                message=f"subprocess target exceeded {timeout_seconds}s timeout",
            )

    async def _run_exchange(
        self, sample: EvalSample, *, attempt: int, started_at: datetime
    ) -> NormalizedExecutionResult:
        process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
            # asyncio only supports one shared buffer `limit` for every
            # stream on this subprocess, so it must be set to at least
            # the larger of our two configured byte bounds. Whichever
            # bound is smaller is then enforced by hand afterward, in
            # _read_bounded_line/_drain_stderr below.
            limit=max(self._max_output_bytes, self._max_stderr_bytes) + 1,
        )
        # Python's `create_subprocess_exec` guarantees that `process.stdin`,
        # `.stdout`, and `.stderr` are never ``None`` as long as ``PIPE`` was
        # passed for all three, as it was above. The `cast` calls below just
        # tell the type checker to trust that guarantee -- they perform no
        # check at runtime. `cast` is used here instead of `assert` because
        # `assert` statements are silently removed when Python runs in
        # optimized mode (`python -O`), so an `assert` would not reliably
        # guard anything either.
        stdin = cast("asyncio.StreamWriter", process.stdin)
        stdout = cast("asyncio.StreamReader", process.stdout)
        stderr = cast("asyncio.StreamReader", process.stderr)

        try:
            request = {
                "schema_version": _PROTOCOL_VERSION,
                "sample_id": sample.sample_id,
                "input": sample.input,
                "attempt": attempt,
            }
            line = json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
            stdin.write(line)
            await stdin.drain()
            stdin.close()

            stderr_task = asyncio.create_task(
                _drain_stderr(stderr, max_bytes=self._max_stderr_bytes)
            )
            try:
                raw_line = await _read_bounded_line(stdout, max_bytes=self._max_output_bytes)
            except _ByteBoundExceededError as exc:
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stderr_task
                await self._terminate(process)
                return self._error_result(
                    sample,
                    attempt=attempt,
                    status=ExecutionStatus.ERROR,
                    started_at=started_at,
                    error_type="OutputTooLarge",
                    message=str(exc),
                )

            stderr_bytes = await stderr_task
            await self._terminate(process)

            return self._parse_response(
                sample,
                attempt=attempt,
                started_at=started_at,
                raw_line=raw_line,
                stderr_bytes=stderr_bytes,
            )
        finally:
            await self._terminate(process)

    def _parse_response(
        self,
        sample: EvalSample,
        *,
        attempt: int,
        started_at: datetime,
        raw_line: bytes,
        stderr_bytes: bytes,
    ) -> NormalizedExecutionResult:
        # Strip both \r and \n from the end, so it does not matter whether
        # the subprocess used Windows-style CRLF or Unix-style LF line
        # endings -- either way, we are left with the same text.
        text = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not text:
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.ERROR,
                started_at=started_at,
                error_type="EmptyResponse",
                message="subprocess target produced no response line",
                stderr_bytes=stderr_bytes,
            )

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.ERROR,
                started_at=started_at,
                error_type="JSONDecodeError",
                message=f"malformed JSON response: {exc}",
                stderr_bytes=stderr_bytes,
            )

        if not isinstance(payload, dict):
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.ERROR,
                started_at=started_at,
                error_type="TypeError",
                message=f"response must be a JSON object, got {type(payload).__name__}",
                stderr_bytes=stderr_bytes,
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
                stderr_bytes=stderr_bytes,
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
                stderr_bytes=stderr_bytes,
            )

        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            output=output,
            status=ExecutionStatus.COMPLETED,
            target_fingerprint=self._fingerprint,
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
        stderr_bytes: bytes | None = None,
    ) -> NormalizedExecutionResult:
        error: dict[str, JsonValue] = {"type": error_type, "message": message}
        if stderr_bytes:
            # Turn the raw bytes into text so it reads naturally in
            # reports. By the time execution reaches here, `stderr_bytes`
            # has already been capped at `max_stderr_bytes`, so this
            # cannot make the report unexpectedly huge.
            error["stderr"] = stderr_bytes.decode("utf-8", errors="replace")
        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            status=status,
            error=error,
            target_fingerprint=self._fingerprint,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    @staticmethod
    async def _terminate(process: "asyncio.subprocess.Process") -> None:
        """Kill the process, then wait for it to exit -- but not forever.

        This works around a specific Windows bug. Normally, once
        ``process.kill()`` is called, the operating system ends the
        process almost immediately, and ``process.wait()`` should return
        right after. But on Windows, asyncio's ``ProactorEventLoop`` has a
        known issue: if a stream previously hit an error like the
        buffer-overrun one this file raises when a line is too long (a
        ``LimitOverrunError``), asyncio's internal bookkeeping can fail to
        notice that the underlying pipe has closed. When that happens,
        ``process.wait()`` hangs forever waiting for a notification that
        will never come, even though the actual operating-system process
        is already dead.

        ``kill()`` itself is a synchronous, reliable OS-level call, so we
        trust the process is really gone as soon as it returns. The
        ``wait()`` that follows is only there to let asyncio's own
        bookkeeping catch up, so it is given a short timeout rather than
        being allowed to hang -- if that timeout is hit, it is *not*
        treated as an error, since the process has already been signalled
        to die.
        """
        if process.returncode is None:
            process.kill()
            try:
                # This timeout is set well below any realistic caller
                # timeout, because kill() ends the process at the OS
                # level almost instantly -- this is purely a safety net
                # for the asyncio bookkeeping gap described above, not a
                # real wait for a slow shutdown. It is deliberately not
                # shielded from cancellation: if the timeout fires,
                # `wait_for` cancels the `process.wait()` call it is
                # wrapping, rather than abandoning it to keep running
                # forever in the background.
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except TimeoutError:
                # Even after that timeout, asyncio still has not
                # registered that the process exited, even though
                # kill() was already called above. As a best effort,
                # reach into the process object's private `_transport`
                # attribute (accessed defensively via `getattr`, since
                # this is an internal implementation detail with no
                # public guarantee) and close it directly. That frees
                # its pipe handles right away, rather than leaving them
                # for Python's garbage collector to clean up whenever it
                # gets around to it -- which would otherwise print a
                # confusing warning (an "unraisable exception", Python's
                # term for an error it cannot report normally, such as
                # one raised during garbage collection) possibly not
                # until the whole program is shutting down.
                transport = getattr(process, "_transport", None)
                close = getattr(transport, "close", None)
                if callable(close):
                    with contextlib.suppress(Exception):
                        close()
