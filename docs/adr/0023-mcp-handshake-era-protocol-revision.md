# ADR-0023: MCP handshake-era protocol revision

## Status

Accepted

## Context

ADR-0021 shipped `McpTarget`, a hand-rolled MCP stdio client that spawns a
server per sample and makes exactly one `tools/call`. It pinned protocol
revision 2025-06-18 and closed with a supersession rule naming the negotiated
protocol revision as a material change. MCP has published two revisions since,
and they are not the same kind of change as each other.

**2025-11-25** is the last revision built on the `initialize` /
`notifications/initialized` handshake. Its lifecycle chapter and its
version-negotiation rule are unchanged from 2025-06-18. Everything it added
that could reach a client like this one is optional or gated behind a client
capability: icons as metadata, an experimental tasks primitive, URL-mode and
enum changes to elicitation, tool calling in sampling, OAuth and Origin rules
that bind only HTTP transports. This client advertises empty capabilities and
uses three wire features — `tools/call`, text content blocks, and the
`isError` flag — all wire-identical across every revision from 2024-11-05
through 2025-11-25.

**2026-07-28** is a different protocol era. It removes the handshake and the
session concept: there is no `initialize`, every request instead declares its
own revision in `_meta` under the `io.modelcontextprotocol/protocolVersion`
key, servers must implement a `server/discover` RPC returning their supported
versions, and an unsupported version comes back as
`UnsupportedProtocolVersionError` (JSON-RPC code -32022) rather than as a
negotiated fallback. The specification calls the two eras "modern" and
"legacy" and provides a compatibility matrix; a legacy client meeting a
modern-only server is listed there as a failure with no fall-forward path.

A decision is needed because the pinned revision is now two revisions stale,
and because the obvious remedy — editing the constant to the newest published
revision — would be wrong. A client that names 2026-07-28 inside an
`initialize` frame advertises a revision in which that frame does not exist.
Overclaiming a protocol capability is exactly the failure mode this package
exists to make structurally hard.

## Decision

- `McpTarget` proposes protocol revision **2025-11-25**
  (`_MCP_PROTOCOL_VERSION` in `src/agentic_evalkit/targets/mcp.py`), the
  newest handshake-based revision. This supersedes ADR-0021's pin of
  2025-06-18, as ADR-0021's own supersession rule requires.
- **Tolerant negotiation is retained unchanged.** The client proposes its
  revision and accepts whatever revision string the server echoes, recording
  it in `environment_metadata["protocol_version"]`. It never gates on the
  echoed value. The three wire features this client uses are identical across
  every handshake-era revision, so gating would add failure modes while
  protecting nothing, and the echoed string stays evidence in the run record
  rather than a control-flow input.
- **This client stays in the handshake era.** It does not claim 2026-07-28 or
  any later handshake-free revision, does not send per-request `_meta`
  protocol versions, does not implement a `server/discover` probe, and does
  not recognize `UnsupportedProtocolVersionError`. The constant is capped
  below the era boundary by an executable assertion, not by convention.
- **The limitation is documented, not papered over.** Against a server
  implementing only 2026-07-28, this target fails: it opens with `initialize`
  and has no fall-forward path. That failure surfaces through the existing
  taxonomy — a JSON-RPC error, or `ServerExited` — and never as a silent
  wrong answer. Dual-era servers, which still answer `initialize`, work today.
- Every other ADR-0021 decision stands: spawn-per-sample, no MCP SDK
  dependency, client-only with empty capabilities, the hashed fingerprint,
  the status mapping preserving ADR-0008's operational/task separation, and
  the absence of manifest or CLI wiring.

## Alternatives

1. **Stay on 2025-06-18.** Rejected: the specification tells a client to
   propose the latest revision it supports, and 2025-11-25 costs no new
   protocol code here. Remaining stale advertises less capability than the
   client actually has, which misinforms a peer in the opposite direction.
2. **Edit the constant to 2026-07-28.** Rejected outright, and it is the
   central point of this record. The exchange would still be three
   handshake-era frames; only the advertised string would change. Every peer
   negotiating with this client would be told it speaks a protocol it does
   not implement. Being stale is a smaller fault than lying.
3. **Implement the handshake-free era now, alongside the handshake one.**
   Rejected for this change, not forever. A dual-era client needs the
   `server/discover` probe, per-request `_meta` construction, era caching
   across a server's lifetime, and recognition of the modern error family —
   a materially larger surface than a revision bump, and one that deserves
   its own record with its own validation. Nothing here forecloses it.
4. **Gate strictly on the echoed revision and fail on a mismatch.**
   Rejected: it converts a working exchange into an error whenever a server
   answers with a different-but-compatible revision, which the negotiation
   rule explicitly permits a server to do.
5. **Adopt an official MCP SDK client to track revisions automatically.**
   Rejected on the same grounds ADR-0021 gave: a dependency tree for a
   three-frame exchange, and library ownership of the I/O loop conflicts with
   the bounded reads, concurrent stderr drain, and kill-then-collect teardown
   this boundary mandates.

## Consequences

- The client proposes the newest revision it can honestly claim, and the
  claim is backed by the wire behavior rather than by a string.
- The era boundary is enforced by a test, so a future well-meaning constant
  bump past the handshake era fails in CI instead of shipping a false
  capability claim to every peer.
- Servers reachable today are unchanged in practice: handshake-era servers
  and dual-era servers both answer `initialize`.
- A modern-only MCP server is not evaluable through this target until the
  handshake-free exchange is implemented. This is a real coverage gap and is
  stated as such in the target guide rather than left for an operator to
  discover.
- The public landing page's ADR count changes (twenty-three ADRs). The
  adapter count is unchanged at four — this record changes a revision, not
  the adapter roster ADR-0006 governs.

## Validation

- `tests/unit/targets/test_mcp_target.py` pins the decision executably:
  `test_proposed_revision_is_the_newest_handshake_based_one` asserts the
  constant is 2025-11-25 and below the era boundary;
  `test_initialize_frame_carries_the_proposed_revision` asserts the constant
  reaches the wire in the handshake-era slot (`params.protocolVersion`) and
  that no `_meta` block is present;
  `test_environment_metadata_carries_server_info` asserts the proposal is
  what the server sees; `test_alien_protocol_version_is_tolerated` keeps the
  tolerant-negotiation guarantee honest.
- `tests/contract/test_dependency_boundary.py` continues to pass over the
  module — this record adds no import.
- `tests/contract/test_adrs.py` registers prefix 0023;
  `tests/contract/test_public_docs.py` stays green.
- `uv run mypy` (strict) covers the module; the 80% branch-coverage floor
  applies.

## Supersession

Implementing the handshake-free protocol era, adding a `server/discover`
probe or per-request `_meta` protocol versions, gating behavior on the
echoed revision, changing the wire framing, adding a non-stdio MCP
transport, taking on an MCP SDK dependency, reusing a persistent server
across samples, or adding manifest/CLI configuration for this target is a
material change and must supersede this record by name. Advancing the
proposed revision to a later handshake-era revision is likewise material and
needs a new record. ADR-0021 remains in force for every decision this record
does not restate; ADR-0006 and ADR-0008 are untouched.
