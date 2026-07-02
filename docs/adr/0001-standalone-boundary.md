# ADR-0001: Standalone Repository and Dependency Boundary

## Status

Accepted

## Context

`agentic-evalkit` evaluates agentic systems, including systems developed in the
sibling ARP and ExecutionKit (EK) repositories and the internal `agentic-tools`
package. If the evaluation toolkit imported production modules from those
repositories, its releases, tests, and dependency graph would be coupled to
their release cadence, internal refactors, and licensing decisions. The
approved design (`docs/specs/2026-07-02-agentic-evalkit-design.md`, §1-§2 and
§13) requires `agentic-evalkit` to be an independently installable package
that can evaluate ARP, EK, or any other agentic system purely through public,
host-neutral execution targets (callable, subprocess, HTTP).

A legacy ARP-local draft used the working name `agentic-v2-eval` and lived
inside the ARP checkout. This repository supersedes that draft; the draft is
not migrated, imported, or referenced.

## Decision

`agentic-evalkit`:

- imports no modules from ARP, `agentic-tools`, or ExecutionKit, at build
  time, runtime, or in its test suite;
- evaluates those systems only through the public `ExecutionTarget` protocol
  (callable, subprocess, HTTP) defined in this package;
- is developed, tested, and released from its own repository and its own
  virtual environment, independent of ARP/EK checkouts;
- runs its clean-wheel and packaging verification (Task 1 Step 6, CI
  packaging job) outside all three source trees, proving the dependency
  boundary holds for an end user who only has `agentic-evalkit` installed.

## Alternatives

1. **Keep the evaluator inside the ARP monorepo.** Rejected: couples the
   evaluator's release cycle and dependency set to ARP, blocks reuse against
   EK or third-party agents, and was the source of the `agentic-v2-eval`
   draft this ADR retires.
2. **Publish a shared internal library that both ARP and `agentic-evalkit`
   depend on.** Rejected for the initial release: adds a coordination
   dependency and a new versioning surface before the target-boundary
   pattern has been proven; can be revisited later without breaking public
   contracts because targets remain the only integration point.
3. **Allow optional, guarded imports of ARP/EK types behind try/except.**
   Rejected: even optional imports create an implicit contract on internal
   ARP/EK APIs and defeat static dependency-boundary verification.

## Consequences

- Integrations with ARP, EK, or any other agent runtime are implemented by
  the *target* side (a callable, subprocess, or HTTP adapter written by the
  integrator), not by `agentic-evalkit` importing that runtime's internals.
- Clean-wheel tests must run in an environment that does not contain ARP,
  EK, or `agentic-tools` on `sys.path`, so accidental imports fail loudly
  instead of passing only because a sibling checkout happened to be
  importable.
- Changes to ARP or EK internals are explicitly out of scope for this
  repository and its implementation plan.
- Every future task must be reviewed against this boundary; a static
  dependency-boundary test (design §14) enforces it in CI.

## Validation

- CI packaging job (Task 1 Step 6) builds the wheel with `uv build` and
  installs it into a fresh virtual environment outside this source tree,
  then runs `agentic-evalkit --help`, proving the package is self-contained.
- Future dependency-boundary tests assert no `arp`, `agentic_tools`, or
  `executionkit` (or equivalent) import appears anywhere under
  `src/agentic_evalkit`.
- Code review checklist item: no PR may add an import from ARP, EK, or
  `agentic-tools`.

## Supersession

This ADR retires the ARP-local `agentic-v2-eval` draft as the canonical
design for this capability. Any future decision to merge `agentic-evalkit`
back into a host repository must supersede this ADR explicitly and update
the packaging and CI verification described above.
