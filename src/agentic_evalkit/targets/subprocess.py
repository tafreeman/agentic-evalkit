"""SubprocessTarget: structured JSONL over standard input/output (design §8).

One JSON object per exchange, sent as a single compact UTF-8 line with
standard input closed immediately after. Responses are read with
``StreamReader.readline()`` so partial writes are reassembled into complete
lines on every platform; both ``\\r`` and ``\\n`` are stripped so CRLF and LF
terminators parse identically on Windows and Linux. Output and error streams
are both byte-bounded, and standard error is drained concurrently with the
standard-output read so a chatty process cannot deadlock the pipe. On
timeout the process is killed and awaited so no orphan process remains.
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
    """Hash the command tuple into a stable fingerprint.

    Only the executable's basename is embedded in cleartext; the full
    command (which may include sample-derived arguments in some deployments)
    is hashed rather than recorded, per "record command executable name and
    configured protocol version, but not environment secret values."
    """
    executable_name = command[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if command else "unknown"
    digest = hashlib.sha256(":".join(command).encode()).hexdigest()[:16]
    return f"subprocess:{executable_name}:{_PROTOCOL_VERSION}:{digest}"


class _ByteBoundExceededError(Exception):
    """Raised internally when a stream exceeds its configured byte bound."""


async def _read_bounded_line(reader: asyncio.StreamReader, *, max_bytes: int) -> bytes:
    """Read one line via ``readline()``, enforcing a byte bound.

    ``readline()`` reassembles chunks delivered across multiple writes into
    a single complete line terminated by ``\\n`` (CRLF's ``\\r`` survives as
    the line's last content byte and is stripped by the caller). The stream
    is constructed with its internal buffer ``limit`` set to ``max_bytes``,
    so ``readline()`` itself raises ``ValueError`` (a
    ``asyncio.LimitOverrunError``) once that many unconsumed bytes have
    accumulated without a terminator; that is caught here and normalized
    into the same typed error as the (redundant but explicit) post-hoc
    length check, so an oversized response is a diagnosable error rather
    than a hang or an uncaught exception.
    """
    try:
        line = await reader.readline()
    except ValueError as exc:
        raise _ByteBoundExceededError(f"line exceeded {max_bytes} byte bound") from exc
    if len(line) > max_bytes:
        raise _ByteBoundExceededError(f"line exceeded {max_bytes} byte bound")
    return line


async def _drain_stderr(reader: asyncio.StreamReader, *, max_bytes: int) -> bytes:
    """Concurrently drain standard error up to a byte bound, discarding the rest.

    Draining is unconditional (not just up to the bound) so a process that
    writes more than the bound to stderr never blocks on a full pipe while
    the caller is waiting on stdout.
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
    """Invokes a subprocess as the system under test over JSONL stdio."""

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
            # asyncio applies one buffer `limit` to every stream on this
            # transport, so it must cover the larger of the two configured
            # bounds; the smaller, per-stream bound is then enforced
            # explicitly in _read_bounded_line/_drain_stderr below.
            limit=max(self._max_output_bytes, self._max_stderr_bytes) + 1,
        )
        # `create_subprocess_exec` guarantees these three streams are populated
        # whenever PIPE is passed for each (as above); `cast` documents that
        # stdlib-guaranteed invariant without a runtime check that `assert`
        # would strip under `python -O`.
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
        # Strip both \r and \n so CRLF- and LF-terminated lines parse
        # identically regardless of platform.
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
            # Decoded for readability in reports; already bounded by
            # max_stderr_bytes before it reaches this point.
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
        """Kill the process and await its exit, bounded by a short timeout.

        On Windows, ``asyncio``'s ``ProactorEventLoop`` can leave a pipe
        transport's connection-lost callback unfired after a stream error
        such as ``StreamReader`` buffer overrun (see CPython's proactor
        pipe-transport/``LimitOverrunError`` interaction), which makes
        ``Process.wait()`` hang even though the OS-level process has already
        exited. ``kill()`` is synchronous and reliable; bounding the
        subsequent ``wait()`` prevents that asyncio-level bookkeeping gap
        from hanging the whole exchange indefinitely. A timeout here is not
        treated as an error: the process has already been signalled to die.
        """
        if process.returncode is None:
            process.kill()
            try:
                # Bounded well below any realistic caller timeout: kill() is
                # synchronous at the OS level, so a real exit is reflected
                # almost immediately. This only guards against the asyncio
                # Proactor bookkeeping gap described above, not a slow exit.
                # Deliberately unshielded: on timeout, wait_for cancels the
                # inner process.wait() coroutine rather than leaving it to
                # run forever as an orphaned task.
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except TimeoutError:
                # process.wait() never observed the exit at the asyncio
                # bookkeeping layer even though kill() has already been
                # issued. Best-effort close the underlying transport (an
                # implementation detail, hence the defensive getattr) so its
                # pipe handles are released deterministically here instead
                # of leaking until garbage collection logs an unraisable
                # ResourceWarning during interpreter teardown.
                transport = getattr(process, "_transport", None)
                close = getattr(transport, "close", None)
                if callable(close):
                    with contextlib.suppress(Exception):
                        close()
