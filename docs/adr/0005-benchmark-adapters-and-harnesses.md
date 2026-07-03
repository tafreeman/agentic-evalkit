# ADR-0005: Benchmark Adapters Project; Harness Executors Verify

## Status

Accepted

## Context

A benchmark such as GSM8K or SWE-bench Verified has two separable
responsibilities that design §7 (`docs/specs/2026-07-02-agentic-evalkit-design.md`)
requires `agentic-evalkit` to keep apart:

1. **Projection and policy** — turning a provider `SourceRecord` into an
   evaluable `EvalSample` (prompt, expected answer or oracle, artifact
   requirements, the grader that should score it) and declaring what a valid
   oracle looks like.
2. **Authoritative verification** — for benchmarks whose correctness can only
   be established by running code in isolation (SWE-bench applies a candidate
   patch and runs a held-out test suite), the actual, sandboxed judgement of
   whether the system-under-test succeeded.

Collapsing these two into one object invites a specific, dangerous failure:
an advisory grader (an exact-match or an LLM judge over the model's prose)
silently standing in for an authoritative harness result, so a run reports
"resolved" when no isolated verification ever happened. The framework must
make that impossible to do by accident, and must represent "the harness that
would verify this is not available here" as an explicit, typed outcome rather
than as a pass, a fail, or a crash.

## Decision

- **Adapters project; they never verify.** `BenchmarkAdapter` (structural
  `Protocol` in `benchmarks/base.py`, `api_version` + `name`) exposes
  `prepare` (record → `EvalSample`), `validate_oracle`, and
  `aggregate_metadata`. An adapter is pure, deterministic, and does no
  isolated execution. `Gsm8kAdapter` projects GSM8K rows and pins
  `grader=GraderSpec(name="normalized-exact@1")`, `adapter="gsm8k@1"`;
  `extract_final_answer` takes the text after the final `####`, strips digit
  grouping commas, and canonicalizes integer-valued `"5.0"` to `"5"`.
  `SweBenchVerifiedAdapter` projects SWE-bench Verified rows (parsing
  `FAIL_TO_PASS` / `PASS_TO_PASS` from either JSON strings or native arrays)
  and never touches the filesystem.
- **Harness executors verify authoritatively.** `HarnessExecutor` (protocol)
  runs the isolated verification and returns a `HarnessResult`
  (`FrozenModel`). `HarnessStatus` is a closed enum — `completed`,
  `unavailable`, `error` — and `HarnessResult.resolved` is a tri-state
  `bool | None`: `True`/`False` only when a verification actually ran and
  produced a verdict, `None` whenever no verdict exists (unavailable or
  errored). A resolved verdict without a harness is unrepresentable.
- **A missing harness is `unavailable`, not a result.** When no harness is
  wired for a benchmark, `UnavailableHarnessExecutor` returns
  `status=unavailable`, `resolved=None`. This is distinct from both "verified
  failure" (`completed`, `resolved=False`) and "the harness crashed"
  (`error`). Benchmarks whose readiness is `prediction_export` (SWE-bench in
  the initial release) export predictions in the exact official schema for
  external verification via `export_prediction(sample, patch,
  model_name_or_path="agentic-evalkit-target")`, which returns exactly the
  three official prediction keys — the framework never claims a locally
  resolved SWE-bench result it did not earn.
- **Adapters raise typed schema errors.** Row validation that fails raises
  `DatasetSchemaMismatch` (from `agentic_evalkit.errors`), never a silent
  drop or an empty projection.
- **`FakeHarnessExecutor`** exists only for tests, is documented as such in
  its docstring, and must never be constructed by production code.
- **No benchmark library coupling.** Adapters never import the `datasets`
  Hugging Face library or `pyarrow`, and never set `trust_remote_code`
  (ADR-0001, ADR-0003).

## Alternatives

1. **One `Benchmark` object that both projects and verifies.** Rejected: it
   makes "advisory grader impersonating an authoritative result" an easy
   default rather than an impossible state, and couples pure projection to
   sandbox execution concerns.
2. **Represent a missing harness as a failing grade.** Rejected: a missing
   harness is an operational gap, not evidence the system-under-test failed;
   conflating them corrupts pass/fail statistics (see ADR-0008's separation
   of operational and task outcomes).
3. **Boolean `resolved` with a default of `False`.** Rejected: it cannot
   distinguish "verified failure" from "never verified," which is exactly the
   distinction this ADR exists to preserve; `bool | None` makes the
   unresolved state explicit.

## Consequences

- The type system alone prevents reporting an authoritative benchmark result
  that no harness produced.
- SWE-bench Verified is shippable in the initial release as a
  prediction-export benchmark without bundling a code-execution sandbox: it
  produces official-schema predictions now, and an authoritative harness can
  be added later behind the same `HarnessExecutor` protocol without changing
  adapters or public models.
- Aggregation can treat `unavailable` as its own bucket, keeping it out of
  pass/fail rates.

## Validation

- `tests/unit/benchmarks/test_gsm8k_adapter.py` covers `extract_final_answer`
  (final-`####` extraction, comma stripping, `"5.0"`→`"5"`) and the pinned
  adapter/grader identity.
- `tests/unit/benchmarks/test_swebench_adapter.py` covers the exact three
  official prediction keys, JSON-string vs native-array oracle parsing, a
  custom `model_name_or_path`, the no-filesystem-touch guarantee, and
  `DatasetSchemaMismatch` on malformed rows.
- `tests/unit/benchmarks/test_harness.py` covers round-trip serialization and
  four-way discrimination of `completed` / `unavailable` / `error` outcomes,
  and a type-level guard that `GradeResult` carries no `resolved` attribute so
  a grade can never be mistaken for a harness verdict.

## Supersession

Adding an in-repository authoritative execution harness (a real SWE-bench
sandbox), or introducing a fourth `HarnessStatus`, is a material change and
must supersede this ADR with its own isolation and validation evidence.
