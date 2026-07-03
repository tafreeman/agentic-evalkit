# HTTP agent example

This guide walks through evaluating a real tool-using agent that is
deployed as an HTTP service — the most common shape for a production
agent endpoint. The complete, runnable example lives under
[`examples/http_agent/`](https://github.com/agentic-evalkit/agentic-evalkit/tree/main/examples/http_agent)
in this repository; this guide explains the pieces and why each one is
there.

The example evaluates a small local stub agent server (included in
`examples/http_agent/`) so the walkthrough works with no external
dependencies — the same manifest shape applies unchanged to a real,
internet-deployed agent endpoint. This example does not import or depend
on any particular agent runtime; it evaluates any HTTP service that
accepts the request shape described below.

## 1. Request/response mapping

`HttpTarget` POSTs a fixed JSON envelope to your agent's endpoint:

```json
{
  "schema_version": "1",
  "sample_id": "http-agent:0",
  "input": {"question": "What is the capital of France?"},
  "attempt": 1,
  "trace_id": null
}
```

Your agent endpoint must respond with a JSON object whose `sample_id`
matches the request and whose `output` carries the agent's answer in
whatever shape your grader expects:

```json
{
  "sample_id": "http-agent:0",
  "output": {"answer": "Paris", "tool_calls": ["search"]}
}
```

`agentic-evalkit` does not prescribe the shape of `output` beyond "a JSON
object" — you choose a schema that fits your agent, then grade against
that same schema.

## 2. Authentication hook

Real agent endpoints are usually authenticated. The manifest never carries
a literal credential — only the *name* of an environment variable that
supplies one at run time:

```yaml
target:
  kind: http
  url: https://my-agent.example.com/evaluate
  credential_hook: MY_AGENT_API_TOKEN
```

At run time, the CLI reads `MY_AGENT_API_TOKEN` from the environment and
sends it as a bearer token. If you are building a Python integration
instead of using the CLI, pass your own header-provider callback directly
to `HttpTarget(headers=...)` — see [the targets guide](targets.md).
`Authorization` headers are redacted from every recorded evidence field
regardless of which path supplies them.

## 3. Timeout

Agent endpoints that call tools or an LLM backend can be slow. Set a
timeout appropriate to your agent's worst-case latency in the manifest:

```yaml
timeout_seconds: 60.0
```

A deadline that expires maps to `ExecutionStatus.TIMEOUT`, tracked
separately from `error` in the run's outcome counts — a slow endpoint and
a broken endpoint are different failure modes and are reported as such.

## 4. Objective schema grader

Before reaching for a model judge, check that the agent's response is
well-formed. `examples/http_agent/` grades with a `SchemaGrader` that
validates the response against a Pydantic model:

```python
from pydantic import BaseModel

class AgentResponse(BaseModel):
    answer: str
    tool_calls: list[str] = []
```

```python
from pydantic import TypeAdapter
from agentic_evalkit.graders.composite import SchemaGrader

grader = SchemaGrader(name="agent-response-schema@1", adapter=TypeAdapter(AgentResponse))
```

This is deterministic and instantaneous — no network call, no model
inference — and it catches an entire class of integration bugs (a
malformed response, a missing field, a wrong type) before any more
expensive check runs. See [the graders guide](graders.md) for where a
schema check sits in the objective-first evidence order, and how to layer
a hard-gated schema check underneath an advisory model judge with
`CompositeGrader` if your evaluation also needs qualitative judgment.

## 5. Canonical report

Run the example end to end:

```bash
cd examples/http_agent
python stub_agent_server.py &   # starts the local stub agent on 127.0.0.1:8765
agentic-evalkit run eval.yaml --limit 3 --yes
```

This produces the same canonical JSON report shape as every other target —
`provenance`, `manifest`, `resolved_dataset`, `summary`, and per-sample
`samples` with execution and grade evidence. Nothing about the report
format is specific to HTTP targets; `compare` and `report` work on it
identically to a run against a callable or subprocess target.

## What this example does not do

This example targets a plain HTTP JSON endpoint. It does not import,
reference, or depend on any particular agent runtime, orchestration
platform, or execution engine — `agentic-evalkit`'s target boundary is
intentionally host-neutral (see [ADR-0006](../adr/0006-execution-target-boundary.md)),
so the identical manifest and grader setup shown here works against any
HTTP agent that speaks the request/response shape above, regardless of
what is running behind it.

This example is illustrative, not an automated release gate — it is not
part of this repository's CI test suite, since it starts a local server
process as a demonstration fixture rather than exercising a contract this
package owns.
