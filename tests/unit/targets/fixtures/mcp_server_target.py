"""Fixture script: a minimal MCP stdio server whose behavior is selected
by its first command-line argument (the "mode"). It speaks the same
newline-delimited JSON-RPC 2.0 framing ``McpTarget`` does -- one compact
JSON object per line -- and each mode scripts exactly one success or
failure path: the well-behaved happy path, a tool-reported error, JSON-RPC
errors at either request, malformed frames, a stale-id response, an
interleaved notification, a server-initiated ``ping`` request, a hang, an
oversized output line, an immediate exit, and a death right after the
handshake. Any command-line arguments
after the mode are ignored, so tests can append secret-looking values and
prove they never leak into the target fingerprint. Pure stdlib on
purpose: the whole point of these tests is exercising a genuine
subprocess boundary with no test-only protocol shims on either side.
"""

import json
import sys
import time

MODE = sys.argv[1]


def _emit(payload):
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _result(request_id, result):
    _emit({"jsonrpc": "2.0", "id": request_id, "result": result})


def _error(request_id, code, message):
    _emit({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def _happy(request_id, name, arguments):
    # Echoing the requested tool name and arguments back inside the text
    # block lets tests assert, from the normalized output alone, exactly
    # which call the server actually received.
    text = json.dumps({"name": name, "arguments": arguments})
    _result(request_id, {"content": [{"type": "text", "text": text}], "isError": False})


def _handle_initialize(frame):
    if MODE == "init_error":
        _error(frame["id"], -32603, "init refused")
        return
    if MODE == "bad_init_result":
        # A response whose result member is not a JSON object.
        _emit({"jsonrpc": "2.0", "id": frame["id"], "result": "nope"})
        return
    protocol_version = frame["params"]["protocolVersion"]
    if MODE == "alien_version":
        protocol_version = "9999-01-01"
    if MODE == "exit_after_init":
        # Answer the handshake, then die before tools/call can ever be
        # answered. The client must notice the death -- via stdout
        # end-of-file, or via a failed stdin write, whichever its
        # platform surfaces first -- and normalize it, instead of
        # waiting out its whole timeout.
        _result(
            frame["id"],
            {
                "protocolVersion": protocol_version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "0"},
            },
        )
        sys.exit(5)
    _result(
        frame["id"],
        {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fixture", "version": "0"},
        },
    )


def _handle_tools_call(frame):
    request_id = frame["id"]
    name = frame["params"]["name"]
    arguments = frame["params"]["arguments"]
    if MODE == "tool_error":
        blocks = [{"type": "text", "text": "tool exploded"}]
        _result(request_id, {"content": blocks, "isError": True})
    elif MODE == "jsonrpc_error":
        _error(request_id, -32602, "bad params")
    elif MODE == "tool_result_not_object":
        _emit({"jsonrpc": "2.0", "id": request_id, "result": "nope"})
    elif MODE == "tool_result_bad_iserror":
        _result(request_id, {"content": [], "isError": "yes"})
    elif MODE == "tool_result_bad_content":
        _result(request_id, {"content": "not a list", "isError": False})
    elif MODE == "hang":
        # Never answers; the client's timeout must fire and kill us.
        time.sleep(60)
    elif MODE == "no_result_no_error":
        # A frame that IS the response to the awaited id but carries
        # neither `result` nor `error`. Sleeping afterward keeps the
        # process alive, so the only way the client escapes quickly is
        # its own fail-fast on the malformed response -- not EOF.
        _emit({"jsonrpc": "2.0", "id": request_id})
        time.sleep(60)
    elif MODE == "wrong_id_then_right":
        # A response whose id belongs to no outstanding client request; the
        # client must skip it and keep waiting for the real one.
        _result(99, {"content": [], "isError": False})
        _happy(request_id, name, arguments)
    elif MODE == "notification_interleaved":
        _emit({"jsonrpc": "2.0", "method": "notifications/message", "params": {"data": "chatter"}})
        _happy(request_id, name, arguments)
    elif MODE == "server_request":
        # A server-to-client request the client must answer before the tool
        # result is sent. Exit code 4 on a bad reply turns a wrong answer
        # into a visible ServerExited failure instead of a silent pass.
        _emit({"jsonrpc": "2.0", "id": 7, "method": "ping"})
        reply = json.loads(sys.stdin.readline())
        if reply.get("id") != 7 or reply.get("result") != {}:
            sys.exit(4)
        _happy(request_id, name, arguments)
    elif MODE == "mixed_content":
        _result(
            request_id,
            {
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "image", "data": "aGk="},
                    {"type": "text", "text": "b"},
                ],
                "isError": False,
            },
        )
    else:  # "happy", plus every handshake-focused mode
        _happy(request_id, name, arguments)


if MODE == "exit_early":
    sys.exit(3)
if MODE == "non_object_frame":
    # Valid JSON, but not a JSON object -- a whole-frame type violation.
    print("42", flush=True)
    sys.exit(0)
if MODE == "malformed":
    sys.stderr.write("boom diagnostics\n")
    sys.stderr.flush()
    print("this is not json{", flush=True)
    sys.exit(0)
if MODE == "oversized":
    print("x" * 200_000, flush=True)
    sys.exit(0)

for raw_line in sys.stdin:
    stripped = raw_line.strip()
    if not stripped:
        continue
    frame = json.loads(stripped)
    method = frame.get("method")
    if method == "initialize":
        _handle_initialize(frame)
    elif method == "tools/call":
        _handle_tools_call(frame)
    # Notifications (e.g. notifications/initialized) need no reply.
