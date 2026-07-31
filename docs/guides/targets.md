# Targets

`ExecutionTarget` is the only boundary through which `agentic-evalkit`
invokes a system under test. Every adapter converts its raw outcome into a
`NormalizedExecutionResult` with an `ExecutionStatus`
(`completed`/`failed`/`timeout`/`cancelled`/`error`) before returning, so
graders never see target-specific response shapes and no target-specific
type ever leaks into a public model. The initial release shipped exactly
three adapters;
[ADR-0021](../adr/0021-mcp-stdio-execution-target.md) added a fourth,
`McpTarget`.

## `CallableTarget`

Wraps an in-process Python callable — sync or async. Sync callables run
through `asyncio.to_thread` so they never block the event loop; both are
wrapped with `asyncio.timeout`.

```python
from agentic_evalkit.targets.callable import CallableTarget

def my_system(sample_input: dict) -> dict:
    return {"answer": solve(sample_input["question"])}

target = CallableTarget(my_system, name="my-system")
```

In a CLI manifest, reference a callable by import string:

```yaml
target:
  kind: callable
  import_string: my_package.agent:answer
```

The target's fingerprint is `callable:{name}:{hash}`, derived from the
callable's module and qualified name — stable across runs as long as the
callable itself does not move.

## `SubprocessTarget`

Speaks structured JSONL over standard input/output: one compact UTF-8 JSON
line sent to the process, standard input closed immediately after. The
process should read one line, do its work, and write one JSON response
line back:

```python
# echo_target.py — a minimal example target
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    response = {
        "schema_version": "1",
        "sample_id": request["sample_id"],
        "output": request["input"],
        "metadata": {},
    }
    print(json.dumps(response, separators=(",", ":")), flush=True)
```

```yaml
target:
  kind: subprocess
  argv: ["python", "echo_target.py"]
```

Responses are read with `StreamReader.readline()`, so partial writes are
reassembled into complete lines on every platform, and both `\r` and `\n`
line terminators are stripped — a target written on Linux and a target
written on Windows parse identically. Standard output and standard error
are both byte-bounded; standard error is drained concurrently with the
standard-output read, so a process that writes a lot to stderr cannot
deadlock the pipe. On timeout, the process is killed and awaited so no
orphan process remains. The command executable's basename and configured
protocol version are recorded on the result; the full command line and any
environment values are not, since a deployment may pass secrets as
arguments or environment variables.

## `HttpTarget`

Invokes a remote HTTP endpoint with a versioned JSON request/response
mapping, an authentication hook, retry policy, and trace correlation.

```python
import httpx
from agentic_evalkit.targets.http import HttpTarget

def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_token()}"}

target = HttpTarget(
    client=httpx.AsyncClient(timeout=30.0),
    url="https://my-agent.example.com/evaluate",
    name="my-agent",
    headers=auth_headers,
)
```

```yaml
target:
  kind: http
  url: https://my-agent.example.com/evaluate
  credential_hook: MY_AGENT_TOKEN   # read from this environment variable at run time
```

The manifest file itself never carries a literal credential — only the
*name* of an environment variable (or, in Python, a header-provider
callback) that supplies it at run time. This keeps secrets out of manifest
files, run artifacts, and version control.

**Request shape.** `HttpTarget` POSTs a JSON body containing
`schema_version`, `sample_id`, `input`, `attempt`, and `trace_id`.

**Response shape.** The endpoint must respond with a JSON object containing
a matching `sample_id` and, on success, an `output` object. A mismatched
`sample_id` is treated as an error, not silently accepted.

**Retries.** Only connection failures and HTTP 429/502/503/504 are
retried, with bounded exponential backoff honoring a server `Retry-After`
header when present. Validation errors and other 4xx responses are never
retried — retrying a malformed request would not fix it.

**Timeouts.** A deadline that expires while waiting for a response maps to
`ExecutionStatus.TIMEOUT`, distinct from a connection or application error.

**Redaction.** `Authorization` and `Proxy-Authorization` headers are
redacted (replaced with `***redacted***`) from every recorded evidence
field — request headers stored in run artifacts or reports never contain
your credentials.

## `McpTarget`

Speaks MCP — newline-delimited JSON-RPC 2.0 over a spawned server's
standard input/output. Every `execute()` spawns a fresh server process,
performs the `initialize` handshake, sends `notifications/initialized`,
makes exactly one `tools/call`, and tears the process down. No server
process ever outlives a single sample, so no state can leak between
samples and runs stay reproducible and parallel-safe by construction.

```python
from agentic_evalkit.targets.mcp import McpTarget

target = McpTarget(
    command=("python", "-m", "my_mcp_server"),
    tool_name="search",  # optional: pin the tool; samples then send only arguments
)
```

Each sample's `input` names the call:
`{"tool": "search", "arguments": {...}}`. When the target pins
`tool_name`, the sample supplies only `{"arguments": {...}}` — a sample
naming its own tool is rejected, so a sample can never claim one tool
while the target silently calls another. Unknown input keys are rejected
outright rather than ignored.

**Status mapping.** A tool result carrying `isError: true` means the
system under test ran and reported its own failure — `FAILED`, kept
separate from plumbing problems. A JSON-RPC error or any transport
breakdown (server exit, broken pipe, malformed frame, oversized line)
maps to `ERROR`; an expired deadline maps to `TIMEOUT`.

**Server-initiated traffic.** Notifications are ignored; a server `ping`
is answered with an empty result; any other server-initiated request
receives a JSON-RPC method-not-found error, matching the empty
capabilities the client advertised. Version negotiation is deliberately
tolerant: the client proposes revision 2025-06-18 and accepts whatever
revision string the server echoes.

**Boundary hardening.** The same byte bounds on stdout and stderr, the
same concurrent stderr drain, the same kill-then-collect teardown, and
the same hashed fingerprint (never recording argument or environment
values in the clear) as `SubprocessTarget`.

`McpTarget` is constructed in code and handed to the runner, like
`CallableTarget` — manifest/CLI wiring is deliberately deferred
([ADR-0021](../adr/0021-mcp-stdio-execution-target.md)).

## Choosing a target

| Situation | Target |
|---|---|
| Your system is already importable Python | `CallableTarget` |
| Your system runs as a separate process/language, or you want strict process isolation | `SubprocessTarget` |
| Your system is a deployed HTTP service (an agent API, a hosted endpoint) | `HttpTarget` |
| Your system is a tool behind an MCP stdio server | `McpTarget` |

For a complete worked example wiring an `HttpTarget` to a real agent
endpoint with request/response mapping, authentication, timeout, and an
objective schema grader, see
[the HTTP agent example](http-agent-example.md).

See [ADR-0006](../adr/0006-execution-target-boundary.md) for the full
target-boundary design, including the Windows-specific subprocess
cancellation behavior it documents.
