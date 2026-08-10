"""McpTarget: runs an MCP stdio server as the system under test.

MCP (the Model Context Protocol) exposes tools behind a server process
that speaks JSON-RPC 2.0 over its standard input and output -- one
compact JSON object per line ("stdio framing"). Calling one tool takes
exactly three client frames: an ``initialize`` request that proposes a
protocol revision, a ``notifications/initialized`` notice telling the
server the handshake is complete, and a single ``tools/call`` request
naming the tool and its arguments.

A fresh server process is spawned for every sample and torn down as soon
as its one response arrives. That is deliberate: because no server
process ever outlives a single sample, no state can leak from one sample
into the next, so runs stay reproducible and parallel-safe by
construction rather than by trusting the server to behave.

Unlike the one-shot JSONL exchange in the sibling subprocess target,
stdin here stays open across the whole conversation: the server must
still be able to read the ``tools/call`` frame after it has already
answered ``initialize``, so stdin is only closed once the final response
is in hand.

Standard error is drained concurrently from the moment the process
starts, for the same reason the subprocess target does it: operating
system pipes have a limited buffer, and a server that logs heavily to
stderr with nobody reading would eventually block mid-write while we sat
waiting on stdout for a response that could then never come -- a
permanent standoff. Draining stderr in the background keeps that pipe
empty.

Teardown order deliberately differs from the subprocess target. An MCP
server's read loop runs until stdin end-of-file, and a server is
entitled to ignore that end-of-file entirely; waiting for the server to
exit (or for its stderr to reach end-of-file) before killing it would
turn a perfectly successful tool call into a timeout against such a
server. So once the awaited response is in hand the order is: close
stdin, kill the process, and only then collect the stderr drain -- the
kill closes the pipes, so the drain reaches end-of-file promptly, and
everything the server already wrote to stderr remains readable even
though its writer is dead.

Version negotiation is deliberately tolerant. The client sends the
newest protocol revision it knows and accepts whatever revision string
the server echoes back, because every feature this client actually uses
-- ``tools/call``, text content blocks, the ``isError`` flag -- is
wire-identical across the known 2024-11-05 / 2025-03-26 / 2025-06-18 /
2025-11-25 revisions. Gating on the echoed string would add failure
modes without protecting anything.

That "newest revision it knows" is 2025-11-25, not the newest revision
MCP has published. 2025-11-25 is the last of the *handshake-based*
revisions -- the ones this client's three-frame exchange is built on --
and its additions are all optional or gated behind client capabilities
this client never advertises, so proposing it claims nothing untrue.
The 2026-07-28 revision deleted the handshake outright: there is no
``initialize``, each request instead carries its own version in
``_meta["io.modelcontextprotocol/protocolVersion"]``, servers must
implement a ``server/discover`` RPC, and an unsupported version comes
back as ``UnsupportedProtocolVersionError`` (code -32022). Sending
2026-07-28 *inside an* ``initialize`` *frame* would be self-contradictory
-- naming a revision in which the frame carrying it does not exist -- so
this client stays honestly on the handshake era. Speaking 2026-07-28
means implementing that handshake-free exchange, which is a separate
decision, not a wider constant.

One consequence is worth stating plainly: against a server that
implements *only* 2026-07-28, this client fails. It opens with
``initialize``, which such a server does not define, and it has no
fall-forward path. That failure surfaces through the normal taxonomy (a
JSON-RPC error, or ``ServerExited``) rather than silently, and it is the
outcome the specification itself predicts for a handshake-era client
meeting a handshake-free server.

Like the sibling subprocess fingerprint, the full argument vector (and
any fixed tool name) is hashed into the target fingerprint rather than
recorded in the clear: in some deployments those arguments carry
sensitive values, so the fingerprint must change when the command
changes without ever writing the actual argument values into it.
"""

import asyncio
import contextlib
import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast

from agentic_evalkit.models import EvalSample, ExecutionStatus, NormalizedExecutionResult
from agentic_evalkit.targets.subprocess import (
    _DEFAULT_MAX_OUTPUT_BYTES,
    _DEFAULT_MAX_STDERR_BYTES,
    SubprocessTarget,
    _ByteBoundExceededError,
    _drain_stderr,
    _read_bounded_line,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

_PROTOCOL_VERSION: Final[str] = "1"  # target-protocol version: fingerprint + clientInfo.version
# Newest handshake-based MCP revision (see the module docstring for why this
# deliberately stops short of the handshake-free 2026-07-28 revision).
_MCP_PROTOCOL_VERSION: Final[str] = "2025-11-25"
_CLIENT_NAME: Final[str] = "agentic-evalkit"
_JSONRPC_VERSION: Final[str] = "2.0"
# The whole exchange is exactly two client requests, so their ids are fixed
# constants rather than a counter.
_INITIALIZE_REQUEST_ID: Final[int] = 1
_TOOLS_CALL_REQUEST_ID: Final[int] = 2
_JSONRPC_METHOD_NOT_FOUND: Final[int] = -32601
_ALLOWED_INPUT_KEYS: Final[frozenset[str]] = frozenset({"tool", "arguments"})
_FIXED_TOOL_INPUT_KEYS: Final[frozenset[str]] = frozenset({"arguments"})
_MS_PER_SECOND: Final[float] = 1000.0
# How long to wait, after stdout end-of-file, for the exit code to become
# known so it can be included in the ServerExited message. Kept short: a
# server that closed stdout because it exited reports its code almost
# immediately, and one that closed stdout while staying alive never will.
_EXIT_CODE_WAIT_SECONDS: Final[float] = 1.0


def _fingerprint(command: tuple[str, ...], tool_name: str | None) -> str:
    """Build a stable fingerprint identifying this exact MCP server command.

    Only the program's filename -- not its full directory path -- appears
    in the fingerprint as plain, readable text. The full command line
    (and any fixed tool name) is hashed instead of being written out
    directly: arguments can carry sensitive values, so the fingerprint
    must change if the command changes, without ever writing the actual
    argument values into it. Same secrecy rationale as the sibling
    subprocess fingerprint.
    """
    executable_name = command[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if command else "unknown"
    digest = hashlib.sha256(":".join((*command, tool_name or "")).encode()).hexdigest()[:16]
    return f"mcp:{executable_name}:{_PROTOCOL_VERSION}:{digest}"


class _InvalidInputError(Exception):
    """Raised internally when ``sample.input`` violates the tool/arguments contract."""


class _ProtocolError(Exception):
    """Raised internally on any wire-level failure; carries the taxonomy fields."""

    def __init__(
        self, error_type: str, message: str, *, extra: dict[str, "JsonValue"] | None = None
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.extra: dict[str, JsonValue] = extra or {}


def _encode_frame(payload: dict[str, "JsonValue"]) -> bytes:
    """Serialize one frame as a compact, single-line JSON object.

    Compact separators matter here: the framing is one JSON object per
    line, so a pretty-printed payload would be split across lines and
    read by the server as several malformed frames.
    """
    return json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"


def _initialize_frame() -> dict[str, "JsonValue"]:
    return {
        "jsonrpc": _JSONRPC_VERSION,
        "id": _INITIALIZE_REQUEST_ID,
        "method": "initialize",
        "params": {
            "protocolVersion": _MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": _CLIENT_NAME, "version": _PROTOCOL_VERSION},
        },
    }


def _initialized_notification() -> dict[str, "JsonValue"]:
    return {"jsonrpc": _JSONRPC_VERSION, "method": "notifications/initialized"}


def _call_frame(tool: str, arguments: dict[str, "JsonValue"]) -> dict[str, "JsonValue"]:
    # ``arguments`` is always sent as an object, never omitted, so servers
    # that require the member see a consistent shape.
    return {
        "jsonrpc": _JSONRPC_VERSION,
        "id": _TOOLS_CALL_REQUEST_ID,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }


def _jsonrpc_error(error_member: "JsonValue") -> _ProtocolError:
    """Convert a JSON-RPC ``error`` member into the internal protocol error."""
    message = "JSON-RPC error"
    code: JsonValue = None
    if isinstance(error_member, dict):
        raw_message = error_member.get("message")
        if isinstance(raw_message, str) and raw_message:
            message = raw_message
        code = error_member.get("code")
    return _ProtocolError("JsonRpcError", message, extra={"jsonrpc_code": code})


async def _write_frames(stdin: asyncio.StreamWriter, data: bytes) -> None:
    """Write already-encoded frames to the server's stdin, then flush.

    A server that dies (or closes its stdin read end) mid-conversation
    makes this write fail with an operating-system pipe error --
    ``BrokenPipeError`` on POSIX, usually ``ConnectionResetError`` on
    Windows. That is the same server breakdown the reader detects as
    end-of-file on stdout, so it is mapped to the same ``ServerExited``
    taxonomy entry rather than being allowed to escape as a raw
    ``OSError``: which side of the pipe notices the death first is a race
    the caller should never have to care about.
    """
    try:
        stdin.write(data)
        await stdin.drain()
    except OSError as exc:
        raise _ProtocolError(
            "ServerExited",
            f"server stopped reading stdin before the exchange completed: {exc}",
        ) from exc


async def _answer_server_request(
    stdin: asyncio.StreamWriter, request: dict[str, "JsonValue"]
) -> None:
    """Answer a server-to-client request so the server never deadlocks on us.

    A server blocking on its own request would never send the response we
    are waiting for, so these can never simply be ignored. ``ping`` gets
    an empty result; everything else gets a JSON-RPC method-not-found
    error, which is the honest answer given that this client advertised
    empty capabilities during ``initialize``.
    """
    reply: dict[str, JsonValue]
    if request.get("method") == "ping":
        reply = {"jsonrpc": _JSONRPC_VERSION, "id": request.get("id"), "result": {}}
    else:
        reply = {
            "jsonrpc": _JSONRPC_VERSION,
            "id": request.get("id"),
            "error": {"code": _JSONRPC_METHOD_NOT_FOUND, "message": "Method not found"},
        }
    await _write_frames(stdin, _encode_frame(reply))


async def _await_response(
    stdout: asyncio.StreamReader,
    stdin: asyncio.StreamWriter,
    process: "asyncio.subprocess.Process",
    *,
    expected_id: int,
    max_bytes: int,
) -> dict[str, "JsonValue"]:
    """Read frames until the response with ``expected_id`` arrives.

    The discriminator between frame kinds is the ``method`` member, not
    the ``id``: server-request ids live in a separate namespace and may
    collide numerically with this client's own request ids. Frames that
    are neither the awaited response nor a server request are skipped;
    the loop is bounded by stdout end-of-file and by the caller's overall
    timeout, so no frame counter is needed.
    """
    while True:
        raw_line = await _read_bounded_line(stdout, max_bytes=max_bytes)
        if not raw_line:
            # End-of-file: give the exit code a moment to become known so
            # the error message can include it, then report the breakdown.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=_EXIT_CODE_WAIT_SECONDS)
            suffix = f" (exit code {process.returncode})" if process.returncode is not None else ""
            raise _ProtocolError(
                "ServerExited",
                f"server closed stdout before responding to request id {expected_id}{suffix}",
            )
        # Strip both \r and \n so Windows-style CRLF and Unix-style LF
        # line endings parse identically.
        text = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            # Fail fast: the expected response may have been inside the
            # garbage, so continuing to read would risk waiting forever.
            raise _ProtocolError("JSONDecodeError", f"malformed JSON frame: {exc}") from exc
        if not isinstance(payload, dict):
            raise _ProtocolError(
                "TypeError", f"frame must be a JSON object, got {type(payload).__name__}"
            )
        if "method" in payload:
            if "id" not in payload:
                # Server notification: nothing to answer, keep reading.
                continue
            await _answer_server_request(stdin, payload)
            continue
        if payload.get("id") != expected_id:
            # Stale or unrelated response: ignore it, keep reading.
            continue
        if "result" not in payload and "error" not in payload:
            # Fail fast for the same reason as the JSON decode branch
            # above: this frame demonstrably WAS the response to the
            # awaited id, so reading on could only wait forever.
            raise _ProtocolError(
                "MalformedResponse",
                f"response for id {expected_id} carries neither result nor error",
            )
        return payload


def _apply_initialize_response(
    response: dict[str, "JsonValue"], environment_metadata: dict[str, "JsonValue"]
) -> None:
    """Validate the ``initialize`` response and record what the server said.

    Any echoed protocol revision string is accepted (tolerant
    negotiation -- see the module docstring), and ``capabilities.tools``
    is deliberately not required: some servers only report the
    capabilities they were asked about.
    """
    error_member = response.get("error")
    if error_member is not None:
        raise _jsonrpc_error(error_member)
    result_member = response.get("result")
    if not isinstance(result_member, dict):
        raise _ProtocolError(
            "MalformedInitializeResult",
            f"initialize result must be a JSON object, got {type(result_member).__name__}",
        )
    server_info = result_member.get("serverInfo")
    protocol_version = result_member.get("protocolVersion")
    environment_metadata["server_info"] = server_info if isinstance(server_info, dict) else {}
    environment_metadata["protocol_version"] = (
        protocol_version if isinstance(protocol_version, str) else ""
    )


class McpTarget:
    """Spawns an MCP stdio server per sample and makes exactly one tools/call."""

    def __init__(
        self,
        *,
        command: tuple[str, ...],
        env: dict[str, str] | None = None,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        max_stderr_bytes: int = _DEFAULT_MAX_STDERR_BYTES,
        tool_name: str | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must be a nonempty argument tuple")
        self._command = command
        self._env = env
        self._max_output_bytes = max_output_bytes
        self._max_stderr_bytes = max_stderr_bytes
        self._tool_name = tool_name
        self._fingerprint = _fingerprint(command, tool_name)

    async def execute(
        self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
    ) -> NormalizedExecutionResult:
        started_at = datetime.now(UTC)
        try:
            tool, arguments = self._validate_input(sample)
        except _InvalidInputError as exc:
            # Rejected before any process is spawned.
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.ERROR,
                started_at=started_at,
                error_type="InvalidSampleInput",
                message=str(exc),
            )
        # Mutable records shared with the exchange, so a timeout that
        # unwinds it still knows whether the tool call was sent and what
        # the server reported during the handshake.
        tool_call_record: list[dict[str, JsonValue]] = []
        environment_metadata: dict[str, JsonValue] = {}
        try:
            async with asyncio.timeout(timeout_seconds):
                return await self._run_exchange(
                    sample,
                    attempt=attempt,
                    started_at=started_at,
                    tool=tool,
                    arguments=arguments,
                    tool_call_record=tool_call_record,
                    environment_metadata=environment_metadata,
                )
        except TimeoutError:
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.TIMEOUT,
                started_at=started_at,
                error_type="TimeoutError",
                message=f"mcp target exceeded {timeout_seconds}s timeout",
                tool_calls=tuple(tool_call_record),
                environment_metadata=environment_metadata,
            )

    def _validate_input(self, sample: EvalSample) -> tuple[str, dict[str, "JsonValue"]]:
        """Resolve the tool name and arguments, rejecting malformed input early.

        Unknown keys are rejected outright -- matching the ``extra="forbid"``
        stance of the wire models -- so typos like ``"args"`` fail loudly
        instead of silently sending an empty argument object. When the
        target was built with a fixed ``tool_name``, a ``"tool"`` key in
        the sample is rejected too: allowing it would let a sample claim
        one tool while the target silently called another.
        """
        if self._tool_name is not None:
            if "tool" in sample.input:
                raise _InvalidInputError(
                    'input key "tool" not allowed when tool_name is fixed on the target'
                )
            allowed = _FIXED_TOOL_INPUT_KEYS
            tool = self._tool_name
        else:
            allowed = _ALLOWED_INPUT_KEYS
            raw_tool = sample.input.get("tool")
            if not isinstance(raw_tool, str) or not raw_tool.strip():
                raise _InvalidInputError(
                    'input key "tool" is required and must be a non-empty string'
                )
            tool = raw_tool
        unexpected = sorted(set(sample.input) - allowed)
        if unexpected:
            raise _InvalidInputError(
                f"unexpected input key(s): {', '.join(unexpected)}; "
                f"allowed key(s): {', '.join(sorted(allowed))}"
            )
        if "arguments" not in sample.input:
            return tool, {}
        raw_arguments = sample.input["arguments"]
        if not isinstance(raw_arguments, dict):
            raise _InvalidInputError(
                f'input key "arguments" must be a JSON object, got {type(raw_arguments).__name__}'
            )
        return tool, raw_arguments

    async def _run_exchange(
        self,
        sample: EvalSample,
        *,
        attempt: int,
        started_at: datetime,
        tool: str,
        arguments: dict[str, "JsonValue"],
        tool_call_record: list[dict[str, "JsonValue"]],
        environment_metadata: dict[str, "JsonValue"],
    ) -> NormalizedExecutionResult:
        process = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
            # asyncio only supports one shared buffer `limit` for every
            # stream on this subprocess, so it must be set to at least
            # the larger of our two configured byte bounds. Whichever
            # bound is smaller is then enforced by hand afterward, in
            # _read_bounded_line/_drain_stderr.
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
            # Started before the first frame is even written: a server may
            # log to stderr during its own startup.
            stderr_task = asyncio.create_task(
                _drain_stderr(stderr, max_bytes=self._max_stderr_bytes)
            )
            try:
                await _write_frames(stdin, _encode_frame(_initialize_frame()))
                initialize_response = await _await_response(
                    stdout,
                    stdin,
                    process,
                    expected_id=_INITIALIZE_REQUEST_ID,
                    max_bytes=self._max_output_bytes,
                )
                _apply_initialize_response(initialize_response, environment_metadata)
                # Frames 2 and 3 go out together, but only after the id-1
                # response arrived -- MCP forbids pipelining them ahead of
                # a completed handshake.
                call_frame = _call_frame(tool, arguments)
                # Recorded before the flush so a pipe death during the
                # send still reports the call as attempted.
                tool_call_record.append({"name": tool, "arguments": arguments})
                await _write_frames(
                    stdin, _encode_frame(_initialized_notification()) + _encode_frame(call_frame)
                )
                call_response = await _await_response(
                    stdout,
                    stdin,
                    process,
                    expected_id=_TOOLS_CALL_REQUEST_ID,
                    max_bytes=self._max_output_bytes,
                )
                # Teardown order (see module docstring): close stdin, kill
                # the process, then collect the stderr drain -- the kill
                # closes the pipes, so the drain reaches end-of-file
                # promptly even against a server that ignores stdin EOF.
                with contextlib.suppress(OSError):
                    stdin.close()
                await SubprocessTarget._terminate(process)
                stderr_bytes = await stderr_task
                return self._normalize_call_response(
                    sample,
                    attempt=attempt,
                    started_at=started_at,
                    response=call_response,
                    stderr_bytes=stderr_bytes,
                    tool_call_record=tool_call_record,
                    environment_metadata=environment_metadata,
                )
            except _ByteBoundExceededError as exc:
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stderr_task
                await SubprocessTarget._terminate(process)
                return self._error_result(
                    sample,
                    attempt=attempt,
                    status=ExecutionStatus.ERROR,
                    started_at=started_at,
                    error_type="OutputTooLarge",
                    message=str(exc),
                    tool_calls=tuple(tool_call_record),
                    environment_metadata=environment_metadata,
                )
            except _ProtocolError as exc:
                # Kill first so the pipes close and the stderr drain can
                # reach end-of-file instead of waiting on a wedged server.
                await SubprocessTarget._terminate(process)
                stderr_bytes = await stderr_task
                return self._error_result(
                    sample,
                    attempt=attempt,
                    status=ExecutionStatus.ERROR,
                    started_at=started_at,
                    error_type=exc.error_type,
                    message=exc.message,
                    stderr_bytes=stderr_bytes,
                    extra=exc.extra,
                    tool_calls=tuple(tool_call_record),
                    environment_metadata=environment_metadata,
                )
        finally:
            # Idempotent (guarded by a returncode check), so the second
            # call on paths that already terminated is safe. CancelledError
            # is never caught here -- it propagates to the runner.
            await SubprocessTarget._terminate(process)

    def _normalize_call_response(
        self,
        sample: EvalSample,
        *,
        attempt: int,
        started_at: datetime,
        response: dict[str, "JsonValue"],
        stderr_bytes: bytes,
        tool_call_record: list[dict[str, "JsonValue"]],
        environment_metadata: dict[str, "JsonValue"],
    ) -> NormalizedExecutionResult:
        """Map the awaited ``tools/call`` frame onto the normalized result shape.

        A JSON-RPC ``error`` member means the tool never ran -- the
        plumbing rejected the exchange -- so it maps to ``ERROR``. A
        result carrying ``isError: true`` means the system under test ran
        and reported its own failure: that is operational (``FAILED``),
        never graded, keeping ADR-0008's separation intact.
        """
        error_member = response.get("error")
        if error_member is not None:
            raise _jsonrpc_error(error_member)
        result_member = response.get("result")
        if not isinstance(result_member, dict):
            raise _ProtocolError(
                "MalformedToolResult",
                f"tools/call result must be a JSON object, got {type(result_member).__name__}",
            )
        is_error = result_member.get("isError", False)
        if not isinstance(is_error, bool):
            raise _ProtocolError(
                "MalformedToolResult",
                f"tools/call result isError must be a boolean, got {type(is_error).__name__}",
            )
        content = result_member.get("content")
        if not isinstance(content, list):
            raise _ProtocolError(
                "MalformedToolResult",
                f"tools/call result content must be a list, got {type(content).__name__}",
            )
        # Non-text or non-conforming blocks are not errors -- they simply
        # do not contribute to the joined text; the raw content array is
        # preserved verbatim either way.
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            block_text = block.get("text")
            if isinstance(block_text, str):
                text_parts.append(block_text)
        joined = "\n".join(text_parts)
        if is_error:
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                error_type="ToolCallError",
                message=joined or "tool reported an error",
                stderr_bytes=stderr_bytes,
                tool_calls=tuple(tool_call_record),
                environment_metadata=environment_metadata,
            )
        finished_at = datetime.now(UTC)
        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            output={"text": joined, "content": content},
            tool_calls=tuple(tool_call_record),
            latency_ms=(finished_at - started_at).total_seconds() * _MS_PER_SECOND,
            status=ExecutionStatus.COMPLETED,
            environment_metadata=environment_metadata,
            target_fingerprint=self._fingerprint,
            started_at=started_at,
            finished_at=finished_at,
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
        extra: dict[str, "JsonValue"] | None = None,
        tool_calls: tuple[dict[str, "JsonValue"], ...] = (),
        environment_metadata: dict[str, "JsonValue"] | None = None,
    ) -> NormalizedExecutionResult:
        # Same stable taxonomy codes the runner's isolation path records, so
        # ``error["code"]`` has one schema regardless of the producing layer.
        error: dict[str, JsonValue] = {
            "type": error_type,
            "code": ("target_timeout" if status is ExecutionStatus.TIMEOUT else "target_failure"),
            "message": message,
        }
        error.update(extra or {})
        if stderr_bytes:
            # Turn the raw bytes into text so it reads naturally in
            # reports. By the time execution reaches here, `stderr_bytes`
            # has already been capped at `max_stderr_bytes`, so this
            # cannot make the report unexpectedly huge.
            error["stderr"] = stderr_bytes.decode("utf-8", errors="replace")
        # `finished_at` is captured once and reused for the latency, so
        # the two fields can never disagree.
        finished_at = datetime.now(UTC)
        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            tool_calls=tool_calls,
            latency_ms=(finished_at - started_at).total_seconds() * _MS_PER_SECOND,
            status=status,
            error=error,
            environment_metadata=environment_metadata or {},
            target_fingerprint=self._fingerprint,
            started_at=started_at,
            finished_at=finished_at,
        )
