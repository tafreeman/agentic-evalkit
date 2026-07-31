# ADR-0021: MCP Stdio Execution Target

## Status

Accepted

## Context

MCP (the Model Context Protocol) has become a common way to expose agentic
tools as standalone servers speaking JSON-RPC 2.0 over stdin/stdout. The
sibling systems this library most often evaluates — ARP and ExecutionKit —
both ship MCP servers, and third-party MCP servers are proliferating.
Today, grading a tool behind an MCP server requires the operator to write a
bespoke shim translating the `SubprocessTarget` JSONL protocol into MCP
framing. Every shim re-implements the same handshake, framing, and teardown
— usually without the byte bounds, concurrent stderr drain, and
Windows-safe kill semantics ADR-0006 requires at the target boundary.

ADR-0006 fixed `ExecutionTarget` as the only system-under-test boundary,
shipped exactly three adapters, and stated that adding a new built-in
target kind is a material change that must supersede it. ADR-0001 forbids
importing sibling systems, so an MCP client here must reach servers purely
by spawning them as subprocesses.

## Decision

- Ship a fourth built-in adapter, `McpTarget`
  (`src/agentic_evalkit/targets/mcp.py`): a minimal MCP client speaking
  newline-delimited JSON-RPC 2.0 over a spawned subprocess's stdin/stdout.
- **Spawn-per-sample.** Every `execute()` spawns a fresh process, performs
  the `initialize` handshake (protocol revision 2025-06-18, empty client
  capabilities), sends `notifications/initialized`, makes exactly one
  `tools/call`, and tears the process down. No state survives between
  samples, so runs stay reproducible and parallel-safe by construction.
- **No MCP SDK dependency.** The client is hand-rolled: the exchange is
  three frames, and an SDK would add a dependency tree while taking
  ownership of the I/O loop — incompatible with the byte-bounded reads,
  concurrent stderr drain, and kill-then-collect teardown this boundary
  requires. Zero new runtime dependencies.
- **Client only.** The client advertises no capabilities — no sampling, no
  roots, no elicitation, and it runs no server. Server-initiated `ping`
  requests are answered with an empty result; any other server-initiated
  request receives a JSON-RPC method-not-found error.
- **Sibling boundary preserved.** `McpTarget` reaches any MCP stdio server
  — including those shipped by ARP or ExecutionKit — through subprocess
  composition of an argument vector only. This package must never import
  those sibling packages (ADR-0001); the dependency-boundary contract test
  continues to enforce that over the new module.
- **Wire models reused unchanged.** `EvalSample` in;
  `NormalizedExecutionResult` with `ExecutionStatus` out; no new wire
  models. A tool result with `isError: true` maps to `FAILED` (the system
  under test reported its own failure); a JSON-RPC error or any transport
  breakdown maps to `ERROR`; an expired deadline maps to `TIMEOUT` —
  preserving ADR-0008's operational/task separation.
- **No manifest or CLI wiring in this change.** `McpTarget` is constructed
  in code and handed to the runner, like `CallableTarget`. Wiring it into
  the run manifest would modify provenance-gated models and is deferred to
  a future ADR.
- This ADR amends ADR-0006's adapter list from three to four. Every other
  ADR-0006 decision — normalization before grading, redaction at the
  boundary, timeout and kill semantics — stands and applies to the new
  adapter, including the hashed fingerprint that never records argument
  values in the clear.

## Alternatives

1. **Depend on an official MCP SDK client.** Rejected: a new dependency
   tree for a three-frame exchange, and SDK ownership of the event loop
   conflicts with the bounded-read and kill-then-collect semantics ported
   from `SubprocessTarget`.
2. **One persistent server process per run.** Rejected for now:
   connection reuse lets state leak across samples, undermining
   reproducibility and parallel safety; spawn-per-sample is the same
   isolation stance `SubprocessTarget` already takes.
3. **Keep requiring operator-written shims over `SubprocessTarget`.**
   Rejected: every shim re-implements the handshake without the hardening
   this boundary mandates, and each one is a fresh source of bugs.
4. **HTTP/SSE MCP transport instead of stdio.** Rejected for the first
   release: stdio is the dominant server deployment; a streaming HTTP
   transport can arrive later behind the same normalized result shape by
   superseding this ADR.

## Consequences

- Any MCP stdio server becomes evaluable with zero shim code and full
  boundary hardening (byte bounds, stderr drain, Windows-safe teardown).
- One more adapter to maintain; the Windows process-teardown workaround
  stays implemented once and is shared with the subprocess adapter.
- Version negotiation is deliberately tolerant: the client sends the
  newest revision it implements and accepts whatever string the server
  echoes, because every feature it uses is wire-identical across known
  revisions. Strict version gating would be a superseding change.
- The public landing page's adapter and ADR counts change (four adapters,
  twenty-one ADRs).

## Validation

- `tests/unit/targets/test_mcp_target.py`, with the fixture server
  `tests/unit/targets/fixtures/mcp_server_target.py`, covers the happy
  path, `isError` mapping to `FAILED`, JSON-RPC errors mapping to `ERROR`,
  malformed frames, stale-id skipping, interleaved notifications,
  server-initiated ping handling, hang-to-`TIMEOUT` with kill, the
  oversized-line byte bound, immediate server exit, strict sample-input
  validation, and fingerprint secrecy.
- `tests/contract/test_dependency_boundary.py` passes over the new module.
- `tests/contract/test_adrs.py` registers prefix 0021;
  `tests/contract/test_public_docs.py` stays green.
- `uv run mypy` (strict) covers the module; the 80% branch-coverage floor
  applies.

## Supersession

Changing the wire framing or negotiated protocol revision, adding a
non-stdio MCP transport, introducing an MCP SDK dependency, reusing a
persistent server across samples, or adding manifest/CLI configuration for
this target is a material change and must supersede this ADR. This ADR
itself effects the supersession that ADR-0006 requires for adding a new
built-in target kind; ADR-0006 otherwise remains in force.
