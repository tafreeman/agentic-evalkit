# SWE-bench

`princeton-nlp/SWE-bench_Verified` (config `default`, split `test`) is the
curated `swe-bench-verified` preset: real-world GitHub issue resolution
tasks. This release ships everything needed to discover, preview, project,
and export official predictions for this dataset without Docker or a code
checkout. Authoritative resolved/unresolved grading requires the optional
`swebench` harness capability: install `agentic-evalkit[swebench]` and run
a reachable Docker daemon (see below).

## What works today

```bash
agentic-evalkit datasets inspect hf:princeton-nlp/SWE-bench_Verified
agentic-evalkit datasets preview hf:princeton-nlp/SWE-bench_Verified --config default --split test --limit 3
agentic-evalkit init --preset swe-bench-verified --output swebench.yaml
```

Preview, inspection, and the curated preset all work through the same
`huggingface` provider as GSM8K — no additional installation required.

## The adapter: projection, not execution

`SweBenchVerifiedAdapter.prepare()` projects one raw dataset row into a
typed `EvalSample`, preserving:

- the issue text (`problem_statement`) and target repository (`repo`);
- the base commit the fix must apply against (`base_commit`);
- the test patch and parsed `FAIL_TO_PASS`/`PASS_TO_PASS` test-name lists
  (accepted as either a JSON-encoded string or a native array, matching how
  upstream sources encode these fields differently).

This is pure, offline row projection — it never checks out the repository,
never applies a patch, and never executes any code.

## Exporting official predictions

Once your system under test has produced a patch for a sample, export it
in the exact official SWE-bench prediction shape:

```python
from agentic_evalkit.benchmarks.swebench import SweBenchVerifiedAdapter

adapter = SweBenchVerifiedAdapter()
sample = adapter.prepare(source_record)
prediction = adapter.export_prediction(sample, my_patch_diff)
# {"instance_id": "...", "model_name_or_path": "agentic-evalkit-target", "model_patch": "..."}
```

`export_prediction` returns *only* the three official prediction keys
(`instance_id`, `model_name_or_path`, `model_patch`) — no adapter or
framework metadata is mixed in, so the result is directly consumable by
the real SWE-bench harness or a leaderboard submission. Pass your own
`model_name_or_path` for a real submission; it defaults to
`"agentic-evalkit-target"` for local smoke testing.

## Why authoritative grading returns `unavailable`, not a substitute score

SWE-bench "resolved" is a specific, authoritative claim: the official test
suite, run in an isolated environment against the patched repository,
reports the designated `FAIL_TO_PASS` tests now passing and the
`PASS_TO_PASS` tests still passing. No amount of similarity scoring, LLM
judging, or heuristic patch inspection can honestly make that claim.

Because of that, the harness boundary is explicit and typed everywhere a
capability might be missing. `UnavailableHarnessExecutor`
(`agentic_evalkit.benchmarks.harness`) is the zero-extra fallback: a
deterministic, production-safe `HarnessExecutor` that always returns a
typed `unavailable` `HarnessResult` naming the missing capability, never a
fabricated pass/fail. The design's own words: *"Generic rubric or
similarity scoring must never be labeled `SWE-bench resolved`."*

```python
from agentic_evalkit.benchmarks.harness import UnavailableHarnessExecutor

executor = UnavailableHarnessExecutor("install agentic-evalkit[swebench]")
result = await executor.execute(harness_request)
assert result.status == "unavailable"
assert "agentic-evalkit[swebench]" in result.message
```

`swebench-harness@1`, the grader this preset actually registers, is not
wired to that generic fallback: it uses the landed
`SweBenchDockerHarnessExecutor` (see the next section), which enforces the
identical discipline through its own preflight check — capability absent
still returns the same typed `unavailable`, and capability present earns a
real `resolved` verdict instead.

The `swe-bench-verified` preset's `readiness` is `prediction_export` (not
`runnable`) precisely because that capability is optional: everything up to
producing an official prediction works out of the box, and the last
authoritative step depends on the `swebench` extra and a reachable Docker
daemon, neither of which is present by default.

## Authoritative grading: the Docker executor

The pinned, official containerized SWE-bench executor has landed
([ADR-0014](../adr/0014-swebench-docker-harness-executor.md)). Install the
extra and point a manifest at the SWE-bench pair:

```bash
uv pip install "agentic-evalkit[swebench]"   # pulls swebench + the docker SDK
```

```yaml
# eval.yaml
adapter: swebench-verified@1
grader: swebench-harness@1
```

With `agentic-evalkit[swebench]` installed and a reachable Docker daemon,
`swebench-harness@1` runs the official harness for each instance and grades
the real `resolved` verdict: `resolved=True` → a hard-gated `pass`,
`resolved=False` → a hard-gated `fail`. Without the extra or a daemon, the
grade is a typed `unavailable` (never a substitute score), and the run still
completes — an unavailable capability is not a task failure.

The executor keeps the fidelity discipline the contracts were designed for:

- it drives the official `swebench` package rather than reimplementing patch
  application or test execution;
- infrastructure failures (image pull, timeout, resource exhaustion, a
  malformed report) surface as `HarnessStatus.ERROR` with `resolved=None`,
  distinct from a genuine unresolved verdict, and never a fabricated pass;
- it changes none of the public `HarnessRequest`/`HarnessResult` contracts —
  it is a pure implementation addition behind the existing
  `HarnessExecutor` protocol.

The gold-patch / invalid-patch fidelity check (design §7.1) runs via the
opt-in `.github/workflows/live-swebench.yml`, which is the only CI that
installs the extra and requires Docker; `ci.yml` stays Docker-free.

See [ADR-0005](../adr/0005-benchmark-adapters-and-harnesses.md) for the
adapter/harness separation and
[ADR-0014](../adr/0014-swebench-docker-harness-executor.md) for the executor
and grader this section summarizes.
