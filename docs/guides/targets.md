# Targets

`ExecutionTarget` is the only boundary through which `agentic-evalkit`
invokes a system under test. Every adapter converts its raw outcome into a
`NormalizedExecutionResult` with an `ExecutionStatus`
(`completed`/`failed`/`timeout`/`cancelled`/`error`) before returning, so
graders never see target-specific response shapes and no target-specific
type ever leaks into a public model. The initial release ships exactly
three adapters.

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

## Choosing a target

| Situation | Target |
|---|---|
| Your system is already importable Python | `CallableTarget` |
| Your system runs as a separate process/language, or you want strict process isolation | `SubprocessTarget` |
| Your system is a deployed HTTP service (an agent API, a hosted endpoint) | `HttpTarget` |

For a complete worked example wiring an `HttpTarget` to a real agent
endpoint with request/response mapping, authentication, timeout, and an
objective schema grader, see
[the HTTP agent example](http-agent-example.md).

See [ADR-0006](../adr/0006-execution-target-boundary.md) for the full
target-boundary design, including the Windows-specific subprocess
cancellation behavior it documents.
