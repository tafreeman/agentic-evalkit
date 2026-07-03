# Plans

This directory holds the implementation planning record for `agentic-evalkit`.

- [`2026-07-02-agentic-evalkit-initial-release.md`](2026-07-02-agentic-evalkit-initial-release.md)
  — the initial-release implementation plan (Tasks 1-16), executed and
  accepted (see [`../release/initial-release-acceptance.md`](../release/initial-release-acceptance.md)).
- [`2026-07-02-agentic-evalkit-plan-modifications.md`](2026-07-02-agentic-evalkit-plan-modifications.md)
  — review-driven amendments applied to that plan.

The corresponding design is
[`../specs/2026-07-02-agentic-evalkit-design.md`](../specs/2026-07-02-agentic-evalkit-design.md);
architecture decisions are recorded in [`../adr/`](../adr/).

## Follow-on gate: official SWE-bench Docker executor

The initial release ships SWE-bench Verified as a **prediction-export**
benchmark (preview, project, and export official predictions; a missing
authoritative harness returns a typed `unavailable` — see
[ADR-0005](../adr/0005-benchmark-adapters-and-harnesses.md)). The official
Dockerized verification executor is **out of scope** for this release.

Adding it requires a **new, separate plan** built on the accepted harness
contracts. That plan may begin only now that the initial acceptance audit has
passed, and it must include:

- pinned upstream `swebench` package compatibility;
- Docker / image resource preflight (availability, disk, image pulls);
- gold-patch and invalid-patch equivalence tests;
- cancellation and bounded resource cleanup;
- full build/test log capture as artifacts;
- **no changes to the public contracts** — it plugs in behind the existing
  `HarnessExecutor` protocol.

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
