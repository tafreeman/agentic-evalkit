# Plans

This directory holds the implementation planning record for `agentic-evalkit`.

- [`2026-08-31-agentic-evalkit-operating-plan.md`](2026-08-31-agentic-evalkit-operating-plan.md)
  — **the active plan.** One page: the goal, the standing decisions, and the
  next three moves. Everything below it is either executed history or a design
  reference.
- [`2026-07-02-agentic-evalkit-initial-release.md`](2026-07-02-agentic-evalkit-initial-release.md)
  — the initial-release implementation plan (Tasks 1-16), executed and
  accepted (see [`../release/initial-release-acceptance.md`](../release/initial-release-acceptance.md)).
- [`2026-07-02-agentic-evalkit-plan-modifications.md`](2026-07-02-agentic-evalkit-plan-modifications.md)
  — review-driven amendments applied to that plan.
- [`2026-07-04-arp-integration-analysis.md`](2026-07-04-arp-integration-analysis.md)
  — analysis and phased migration plan for adopting this package inside the
  ARP monorepo, harvesting its eval tooling of value, and retiring its legacy
  in-repo evaluation package (not yet executed; see the follow-on gate below).
- [`2026-08-16-agentic-evalkit-master-delivery-plan.md`](2026-08-16-agentic-evalkit-master-delivery-plan.md)
  — **superseded 2026-08-31.** The twelve-phase serial delivery sequence for the
  low-code layer, calibration release, ARP migration, judge validation, and
  legacy-package removal. Retained as historical record; its M0 baseline
  evidence remains valid. Its sequencing no longer governs.
- [`2026-08-16-agentic-evalkit-low-code-integration-layer.md`](2026-08-16-agentic-evalkit-low-code-integration-layer.md)
  — configuration-first YAML/JSON integration design for evaluating Python,
  HTTP, subprocess, and MCP systems through the packaged CLI, with an explicit
  selected-only extension escape hatch for irreducible domain logic. **Retained
  as a design reference; its phased delivery sequence is superseded.** Pieces
  ship individually when a real evaluation needs them.

The current execution authority is
[`2026-08-31-agentic-evalkit-operating-plan.md`](2026-08-31-agentic-evalkit-operating-plan.md)
— one page, replacing the twelve-phase sequence above. It records why the
sequence was retired: EvalKit acquired a live consumer that met the migration
goal by a different route, without the prerequisites the sequence assumed.

The superseded plan's M0 baseline work stands. Its pinned repository
inventories, ARP usage map, package baselines, known failures, and evidence
classifications are recorded in
[`../release/2026-08-16-m0-baseline-acceptance.md`](../release/2026-08-16-m0-baseline-acceptance.md).

The corresponding design is
[`../specs/2026-07-02-agentic-evalkit-design.md`](../specs/2026-07-02-agentic-evalkit-design.md);
architecture decisions are recorded in [`../adr/`](../adr/).

## Follow-on gate: official SWE-bench Docker executor — **RESOLVED** (ADR-0014)

**Status:** shipped. `SweBenchDockerHarnessExecutor` and the `HarnessGrader`
bridge land under
[ADR-0014](../adr/0014-swebench-docker-harness-executor.md); the `swebench`
extra is populated and both SWE-bench names are CLI-registered. The one
remaining item is activating the live gold/invalid-patch fidelity run
(`.github/workflows/live-swebench.yml`) against a chosen instance.

The initial release shipped SWE-bench Verified as a **prediction-export**
benchmark (preview, project, and export official predictions; a missing
authoritative harness returns a typed `unavailable` — see
[ADR-0005](../adr/0005-benchmark-adapters-and-harnesses.md)). The official
Dockerized verification executor was **out of scope** for that release.

It required a **new, separate plan** built on the accepted harness
contracts. That plan (now delivered) includes:

- pinned upstream `swebench` package compatibility;
- Docker / image resource preflight (availability, disk, image pulls);
- gold-patch and invalid-patch equivalence tests;
- cancellation and bounded resource cleanup;
- full build/test log capture as artifacts;
- **no changes to the public contracts** — it plugs in behind the existing
  `HarnessExecutor` protocol.

## Follow-on gate: ARP integration — **superseded by events** (2026-08-31)

The v0.1 statement below is preserved for the record and is no longer accurate.
As of 2026-08-31 ARP is a live consumer: it pins `agentic-evalkit>=0.3.0,<0.4.0`
and drives this package through the subprocess JSONL target from
`agentic-workflows-v2/evals/swe_ab/`, with all adaptation on the consumer side
as the boundary requires. That happened outside both the July analysis and the
August delivery sequence, on a workflow A/B study rather than the legacy gate.
The invariants listed below still hold and still bind; the sequencing and the
"execution not started" status do not. See
[`2026-08-31-agentic-evalkit-operating-plan.md`](2026-08-31-agentic-evalkit-operating-plan.md).

> **Original v0.1 text.** The initial release deliberately ships with **zero
> references from or to the ARP monorepo** — the dependency-boundary contract
> test forbids importing `agentic_v2`, `tools`, or `executionkit`, and a fresh
> cross-repo sweep (2026-07-04) confirms no dependency, import, CI reference, or
> doc mention connects the repositories in either direction. Adoption *by* ARP
> is therefore a from-scratch integration, and it is **out of scope** for this
> release.

The integration analysis remains at
[`2026-07-04-arp-integration-analysis.md`](2026-07-04-arp-integration-analysis.md).
Whatever form the integration takes, it must preserve:

- the **one-way dependency invariant**, enforced from both sides (the existing
  contract test here; a mirror AST test added in the consumer);
- a **distribution decision before any consumer change** — publish to PyPI
  (recommended; consumer's own precedent) or a private git dependency with an
  explicitly provisioned CI read credential;
- the **offline-contract reconciliation** (the deferred `offline=True` gaps
  below) before any consumer CI job claims or relies on enforced hermeticity;
- a **harvest gate**: the consumer's legacy evaluation package is deleted only
  after every tooling-of-value item in the plan's harvest inventory is
  dispositioned (ported, upstreamed, or consciously dropped in writing);
- **no changes to the public contracts** — ARP integrates through the existing
  `ExecutionTarget`, `Grader`, `JudgeClient`, and `DatasetProvider` protocols
  via its own driver modules; upstream contributions arrive as separate plans
  through this repository's normal TDD/ADR process.

## Deferred to v0.2

The following are explicitly out of scope for the initial release and are
candidates for a future v0.2 plan:

- run resumption / checkpoint-restart of an interrupted run;
- an async-first execution model (would be recorded as ADR-0010);
- performance and cache-eviction targets (bounded cache GC, throughput SLAs);
- dataset subgroup / slice selection syntax;
- framework-level observability (metrics, tracing hooks);
- the official SWE-bench Docker executor described above.

No Slice-4b work was deferred at the v0.1 checkpoint: the `CONTINUE_FULL_V1`
decision means calibrated judges, advanced statistics, rich reporters, and the
`compare`/`report` commands are all part of this release.
