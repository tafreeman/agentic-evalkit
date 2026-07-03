# HTTP agent example

A complete, runnable example of evaluating a real HTTP-based agent
endpoint with `agentic-evalkit`: request/response mapping, an
authentication hook, a timeout, an objective schema grader, and a
canonical report. See
[docs/guides/http-agent-example.md](../../docs/guides/http-agent-example.md)
for the full walkthrough and the reasoning behind each piece.

This example imports and depends only on `agentic-evalkit` public
contracts (plus `httpx`/`pydantic`, both of which `agentic-evalkit`
already depends on). It does not import ARP, ExecutionKit, or
agentic-tools, and it is not part of this repository's automated test
suite — it is illustrative example content, run manually.

## Files

| File | Purpose |
|---|---|
| `stub_agent_server.py` | A minimal stdlib HTTP server standing in for a real agent endpoint, so this example runs with zero external dependencies. |
| `questions.jsonl` | Three small questions with expected answers — a local dataset, read by the `local` provider. |
| `run_example.py` | The Python-API driver: wires a local dataset, a custom adapter, an `HttpTarget`, and a `SchemaGrader` through `EvalRunner`, and writes a canonical JSON report. |
| `eval.yaml` | A manifest illustrating the CLI's `target: {kind: http, ...}` block shape and `credential_hook` — shown for reference; see "Running via the CLI" below for why it is not run directly with `agentic-evalkit run` in this example. |

## Running the example

From this directory, in one terminal:

```bash
python stub_agent_server.py
```

In a second terminal:

```bash
python run_example.py
```

Expected output:

```text
outcomes: total=3 passed=3 failed=0 errors=0
report: .../<run_id>.json
```

The stub agent answers three fixed questions correctly, so all three
samples pass the objective schema grader. The canonical JSON report at the
printed path has the same shape (`provenance`, `manifest`,
`resolved_dataset`, `summary`, `samples`) as any other `agentic-evalkit`
run, regardless of target kind.

## Running via the CLI

`eval.yaml` in this directory shows the manifest shape for an HTTP target
with a named credential hook — the same shape [the targets
guide](../../docs/guides/targets.md) documents. This release's CLI `run`
command resolves adapters and graders from a small, fully-tested table of
built-in names (currently the two curated presets: `gsm8k@1` /
`normalized-exact@1`); a custom adapter and grader, such as this example's
`http-agent-questions@1` and `agent-response-schema@1`, are constructed in
Python instead, which is the fully general integration path and why
`run_example.py` — not `agentic-evalkit run eval.yaml` — is how this
example actually executes. `eval.yaml` remains useful as a reference for
the `target` block's shape when you *do* use one of the curated CLI
presets against your own HTTP-deployed target.

## Adapting this to a real agent

Point `run_example.py`'s `_AGENT_URL` (or `eval.yaml`'s `target.url`) at
your real, deployed agent endpoint instead of the local stub server, set
`credential_hook`/your header-provider callback to your endpoint's real
authentication mechanism, and adjust `AgentResponse` in `run_example.py`
to match your agent's actual response schema. Nothing else in the pipeline
— dataset loading, execution, grading, reporting — needs to change.
