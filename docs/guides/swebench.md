# SWE-bench

`princeton-nlp/SWE-bench_Verified` (config `default`, split `test`) is the
curated `swe-bench-verified` preset: real-world GitHub issue resolution
tasks. This release ships everything needed to discover, preview, project,
and export official predictions for this dataset without Docker or a code
checkout. Authoritative resolved/unresolved grading requires the optional
`swebench` harness capability, which is a follow-on implementation.

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

Because of that, this release's harness boundary is explicit and typed:
calling the (currently unimplemented) harness executor for this preset
returns a typed `unavailable` `HarnessResult` naming the missing
capability, never a fabricated pass/fail. The design's own words: *"Generic
rubric or similarity scoring must never be labeled `SWE-bench resolved`."*

```python
from agentic_evalkit.benchmarks.harness import UnavailableHarnessExecutor

executor = UnavailableHarnessExecutor("install agentic-evalkit[swebench]")
result = await executor.execute(harness_request)
assert result.status == "unavailable"
assert "agentic-evalkit[swebench]" in result.message
```

The `swe-bench-verified` preset's `readiness` is `prediction_export`
(not `runnable`) precisely to signal this: everything up to producing an
official prediction works out of the box, and the last authoritative step
is deliberately gated behind a capability that is not yet installed.

## The follow-on Docker executor

A pinned, official containerized SWE-bench executor is planned as a
separate follow-on implementation, gated on this release's acceptance
audit passing first (see the [implementation
plan](../plans/2026-07-02-agentic-evalkit-initial-release.md)'s follow-on
boundary notes). It will:

- record harness version, container image digests, patch application
  results, and test logs;
- pass one gold-patch (known-resolved) and one intentionally invalid-patch
  smoke test through the same production path before being trusted;
- report typed infrastructure failures (image pull failure, resource
  exhaustion, timeout) distinctly from a genuine unresolved verdict;
- introduce no changes to the public `HarnessRequest`/`HarnessResult`
  contracts already shipped in this release — those contracts were
  designed up front specifically so the executor is a pure implementation
  addition, not a breaking change.

Until that follow-on lands, this preset is fully usable for dataset
discovery, preview, sample projection, and prediction export; only the
final authoritative resolution step is deferred.

See [ADR-0005](../adr/0005-benchmark-adapters-and-harnesses.md) for the
full adapter/harness separation this guide summarizes.
