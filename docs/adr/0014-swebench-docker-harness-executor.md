# ADR-0014: Container-backed SWE-bench harness executor

## Status

Accepted (2026-07-10)

## Context

ADR-0005 established the adapter/harness split and shipped
`UnavailableHarnessExecutor` as the only executor, deliberately deferring the
real Docker executor to "a new, separate plan" gated on the initial release's
acceptance audit (`docs/plans/README.md`). That audit passed
(`docs/release/initial-release-acceptance.md`), unblocking this work. Until
now the only concrete `HarnessExecutor` implementations were
`UnavailableHarnessExecutor` and the test-only `FakeHarnessExecutor`, the
`swebench` optional-dependency group was an empty placeholder, and — more
fundamentally — **no grader bridged a `HarnessResult` into a `GradeResult`
at all**, so an authoritative "resolved" verdict had no path into a run's
score. Every SWE-bench Verified run could therefore only ever grade
`UNAVAILABLE`. This is a validity gap (C3/C10), not a capability gap: the
typed contracts and the honest fallback already existed; the component that
turns "we refuse to fabricate a score" into "we can earn one" did not.

## Decision

Add two modules behind the existing contracts (no public-contract change):

1. `benchmarks/swebench_docker.py` — `SweBenchDockerHarnessExecutor`, a
   structural `HarnessExecutor` that drives the official `swebench` package
   in Docker and maps its per-instance `get_eval_report` shape onto
   `HarnessResult`. Two rules keep it safe: it imports nothing from
   `docker`/`swebench` at module load (the real integrations are lazy,
   inside the default `preflight`/`evaluator` callables), so the base
   install and CLI import cleanly; and both seams are constructor-injected,
   so hermetic tests drive every outcome branch without a daemon. Capability
   absence → `UNAVAILABLE`; any infrastructure failure or a report lacking a
   `resolved` field → `ERROR` with `resolved=None`, never a guessed verdict.
2. `graders/harness.py` — `HarnessGrader`, the previously-missing bridge from
   `Grader.grade(sample, execution)` to a `HarnessExecutor`, following
   `ExactMatchGrader`'s injected-callable pattern (a benchmark-neutral
   `predictor` builds the prediction, so this module stays grading-policy
   only). It is the **only** grader that sets `hard_gate=True` from a harness
   verdict, and it maps `UNAVAILABLE`/`ERROR` to non-gating
   `UNAVAILABLE`/`ERROR` grades — an operational failure is never a task
   `FAIL` (ADR-0008).

Wiring: the `swebench` extra becomes `["swebench>=4.1,<5", "docker>=7.1,<8"]`
(versions verified live against PyPI at authoring time — swebench 4.1.0,
docker 7.2.0 — both compatible with the `>=3.11` floor). The CLI registers
adapter `swebench-verified@1` and grader `swebench-harness@1`; the executor
is constructed with default seams and reports `UNAVAILABLE` at run time until
the extra + a daemon are present, so no `try/except ImportError` guard is
needed and the base install never imports `docker`. A new opt-in
`.github/workflows/live-swebench.yml` runs the live tier; `ci.yml` stays
Docker-free.

`HarnessStatus` stays a three-member closed enum — no fourth status is
introduced — and adapters still never verify.

## Alternatives

- **Reimplement patch application / test execution.** Rejected: the official
  `swebench` harness is the authority; reproducing it would fork the very
  ground truth SWE-bench Verified exists to provide.
- **A `try/except ImportError` guard around executor construction (as the
  spec sketched).** Rejected in favor of lazy preflight: the module is
  importable with zero extras by construction, so the guard is unnecessary
  and the always-constructed executor reports `UNAVAILABLE` at run time.
- **Grade from the harness's own stdout / a rubric.** Rejected: only
  `HarnessResult.resolved` may assert resolution (design §7.1); a generic
  grade must never be labeled "SWE-bench resolved".
- **A fourth `HarnessStatus` for "patch did not apply".** Rejected: a patch
  that fails to apply is an authoritative non-resolution (`resolved=False`,
  `COMPLETED`), not a distinct status.

## Consequences

- SWE-bench Verified becomes a fully gradable benchmark for anyone who
  installs `agentic-evalkit[swebench]` and has Docker; `pass_at_k_by_sample`
  (`stats/aggregate.py`) gains real substrate for this benchmark for the
  first time.
- Every preset-referenced adapter/grader name now resolves in the CLI tables
  (both `gsm8k` and `swe-bench-verified`); the adapter-table comment that
  claimed this before it was true is corrected.
- `SweBenchVerifiedAdapter.aggregate_metadata`'s return annotation is
  corrected from `dict[str, object]` to the protocol's `dict[str, JsonValue]`
  (a latent mismatch surfaced by typing the registration table).
- No public contract changed; the base install still imports without
  `docker`/`swebench`, and `ci.yml`/`packaging` stay Docker-free.

## Validation

- `tests/unit/graders/test_harness_grader.py`: every `HarnessResult` outcome
  maps to the correct `GradeResult` status/`hard_gate`, and
  `UNAVAILABLE`/`ERROR`/no-verdict/un-executed/predictor-failure never
  produce a `FAIL`.
- `tests/unit/benchmarks/test_swebench_docker.py`: injected-seam coverage of
  UNAVAILABLE, evaluator-exception → ERROR, resolved-True/False mapping,
  missing-`resolved` → ERROR, evidence/image-digest projection, and the
  `swebench_prediction` helper.
- `tests/integration/test_swebench_registration.py`: a manifest naming both
  SWE-bench names resolves, runs end to end, and degrades to a graded
  `UNAVAILABLE` (exit 0, zero errors) without the extra.
- `tests/live/test_swebench_harness_live.py` (`@pytest.mark.live`, excluded
  from `ci.yml`): the design §7.1 gold-patch/invalid-patch fidelity check
  through the identical real `execute()` path, run by the opt-in workflow.
- Hermetic tiers all run under `pytest -m "not live"`.

## Supersession

Supersedes ADR-0005's Supersession clause (which reserved this addition). A
future change introducing a fourth `HarnessStatus`, changing how `resolved`
is derived from the upstream report, or threading remote/Modal-backed
execution must itself supersede this ADR with fresh isolation and validation
evidence.
