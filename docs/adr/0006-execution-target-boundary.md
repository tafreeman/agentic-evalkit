# ADR-0006: `ExecutionTarget` Is the Only System-Under-Test Boundary

## Status

Accepted

## Context

`agentic-evalkit` must evaluate systems it does not own — a local callable, a
subprocess speaking a line protocol, a remote HTTP service, and (per ADR-0001)
the ARP/EK stacks this package is forbidden to import. Design §8
(`docs/specs/2026-07-02-agentic-evalkit-design.md`) requires that every one of
these be reached through a single, narrow boundary whose results are
normalized before any grader sees them, so that:

- graders never encounter target-specific response shapes;
- no ARP or EK type ever leaks into a public model; and
- ARP can be evaluated through existing public surfaces **without any change
  to this repository** — the proof that the standalone boundary is real and
  not merely aspirational.

Execution targets are also where wall-clock, subprocess lifecycle, network
faults, and secrets live, so the boundary must define timeout, cancellation,
and redaction behavior, not just a call signature.

## Decision

- **One boundary.** `ExecutionTarget` (structural `Protocol` in
  `targets/base.py`) is the sole system-under-test interface:
  `execute(sample, *, attempt, timeout_seconds)` returning a normalized
  result. The initial release ships three adapters and no others:
  - `CallableTarget` — wraps an in-process sync or async callable; sync
    callables run via `asyncio.to_thread` under `asyncio.timeout`; its
    fingerprint is `callable:{name}:{hash}`.
  - `SubprocessTarget` — `asyncio.create_subprocess_exec` (argument vector, no
    shell), one JSON line per exchange, `readline`-based CRLF-safe framing,
    byte-bounded stdout/stderr, a concurrent stderr drain, and kill +
    bounded-wait on timeout.
  - `HttpTarget` — an injected `httpx.AsyncClient`; retries only connection
    errors and 429/502/503/504, honoring `Retry-After` before falling back to
    jittered exponential backoff.
- **Normalize before grading.** Every adapter converts its raw outcome into a
  `NormalizedExecutionResult` with an `ExecutionStatus` before returning.
  Graders consume only this normalized shape; a deadline that expires maps to
  `TIMEOUT`, and target exceptions map to `ERROR` without leaking local
  variables into evidence. Typed failures surface as `TargetFailure` /
  `TargetTimeout` (from `agentic_evalkit.errors`).
- **ARP/EK boundary.** ARP and EK types cannot appear in public models
  (ADR-0001). ARP is evaluated by wrapping it behind one of the existing
  public targets (callable or HTTP) — no repository change, no new type. This
  ADR is the concrete execution-side realization of the standalone boundary.
- **Secrets are redacted at the boundary.** `authorization` and
  `proxy-authorization` headers are redacted from every recorded evidence
  field, so credentials never reach artifacts, events, or reports.

## Alternatives

1. **Let graders read target-specific responses directly.** Rejected: it
   couples every grader to every target shape and makes adding a target a
   cross-cutting change.
2. **A first-class `ArpTarget` inside this package.** Rejected: it would
   require importing ARP/EK, violating ADR-0001; the callable/HTTP adapters
   already make ARP evaluable from outside with zero repository change, which
   is the stronger property.
3. **Rely on the event loop's default subprocess teardown on timeout.**
   Rejected empirically: on Windows' `ProactorEventLoop`, a
   `LimitOverrunError` on an oversized stdout line leaves the pipe transport's
   connection-lost callback unfired, so `Process.wait()` hangs forever though
   the OS process is already dead. The boundary therefore bounds the
   post-`kill()` wait and best-effort closes the transport.

## Consequences

- Adding a new kind of system-under-test is one new `ExecutionTarget`
  adapter; nothing downstream of the boundary changes.
- Grading, statistics, and reporting are fully decoupled from how a target
  was reached.
- Oversized or hung targets cannot wedge a run, on POSIX or Windows.
- Credentials in headers cannot leak into persisted evidence.

## Validation

- `tests/unit/targets/test_callable_target.py` covers sync/async dispatch,
  the `callable:{name}:` fingerprint, and exception normalization to `ERROR`.
- `tests/unit/targets/test_subprocess_target.py` (with fixtures under
  `tests/unit/targets/fixtures/`) covers CRLF-split framing, byte caps on
  stdout/stderr, mismatched-sample-id handling, the hang/kill path, and the
  Windows oversized-output bound.
- `tests/unit/targets/test_http_target.py` covers retry-only-on
  connection/429/5xx, `Retry-After` honoring, authorization-header redaction,
  and deadline → `TIMEOUT`.

## Supersession

Adding a new built-in target kind, changing the subprocess wire protocol, or
introducing any direct ARP/EK dependency is a material change and must
supersede this ADR (an ARP/EK dependency would also require superseding
ADR-0001).
