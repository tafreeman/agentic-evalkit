# ADR-0017: Grade Before Spilling Large Outputs

## Status

Accepted

## Context

`EvalRunner._execute_and_grade` (`src/agentic_evalkit/runner.py`) ran, in
order: `execute()` -> `_spill_large_output()` -> `grade()` (grading gated on
`status is COMPLETED`, requirement 6). `_spill_large_output` (requirement 8)
replaces `execution.output` with `None` (plus an `artifacts["output_ref"]`
pointer into the `ArtifactStore`) whenever the serialized, redacted output
exceeds `_LARGE_OUTPUT_THRESHOLD_BYTES` (8192 bytes), so it stays out of the
in-memory/JSON-serialized `EvalRunResult` while remaining retrievable.

`HarnessGrader` (ADR-0014) needs the inline patch text to build
`HarnessRequest.prediction["model_patch"]` -- it is the bridge from an
authoritative SWE-bench "resolved" verdict to a `GradeResult`, and it has no
other way to obtain the patch than the executed sample's own output. Under
the old ordering, any patch large enough to spill arrived at the grader as
`output=None`: `HarnessGrader.grade` could only recognize the
`output=None` + `output_ref` combination and report `GradeStatus.ERROR` --
an honest diagnostic ("execution output was spilled ... the harness grader
needs the inline patch"), added as a stopgap after a prior code review
(Codex review, P2) so a spilled patch was at least never silently miscounted
as `UNAVAILABLE`. It was never a fix: a SWE-bench Verified run whose patch
exceeded 8192 bytes still could not be graded at all.

This was caught by code review, not yet observed in a live run: the most
recent SWE-bench harness fidelity check (2026-07-11, an `astropy` instance)
is reported to have happened to produce a patch that stayed under the
8192-byte spill threshold, so the defect did not manifest that time -- real
`astropy` patches in the packaged SWE-bench Verified fixture data are
typically well under a few kilobytes (`tests/fixtures/huggingface/
swebench_verified/statistics.json` puts the whole benchmark's patch sizes at
a 277-17385 byte range, mean 1587), but nothing bounds every instance's patch
to that range, and no benchmark-wide guarantee keeps every patch under the
threshold. Any run that drew a larger patch would hit this defect.

## Decision

`EvalRunner._execute_and_grade` now runs, in order: `execute()` ->
`grade()` (still gated on `status is COMPLETED`) -> `_spill_large_output()`
-> return. The `execute()` call and its `ExecutionCompleted` event are
unmoved. `_spill_large_output` moves from immediately after `execute()` to
immediately before the `SampleCompleted` event, after the grading block --
and it still runs unconditionally for every execution status, not only
`COMPLETED`, exactly as before, just later: a non-completed execution can
still carry a huge output that needs spilling for storage even though it
was never gradable.

A grader therefore always sees the execution exactly as the target returned
it. The returned `SampleResult` still carries the *post*-spill execution
(unchanged persisted/reported behavior -- large outputs still never sit
inline in a stored `EvalRunResult`), paired with a `GradeResult` computed
from the *pre*-spill, intact execution. Spilling becomes purely a
persistence concern applied to the result on its way out, never something a
grader has to work around.

## Alternatives

1. **Artifact readback: have `HarnessGrader` read the spilled bytes back
   from the `ArtifactStore` when it sees `output=None` + `output_ref`.**
   Rejected on two grounds. First, it couples a grader to an `ArtifactStore`
   it currently has no reference to and no other business needing. Second,
   and more importantly, it is unsound, not merely inelegant:
   `_spill_large_output` redacts bytes *before* writing them to the store
   (secret patterns replaced with `[REDACTED]`). Reading those bytes back
   would hand the grader a possibly-altered patch -- if any redaction
   pattern happened to match content inside the patch, the "recovered"
   patch would differ from what was actually executed, and applying it
   during grading could silently produce a wrong resolution verdict instead
   of an honest error. Redaction is deliberately lossy; treating its output
   as a faithful copy for re-grading is not sound.
2. **Raise `_LARGE_OUTPUT_THRESHOLD_BYTES`.** Rejected: it does not fix the
   defect class, it only moves the failure point to the next patch that is
   larger still.

Grade-before-spill was chosen over both because it is strictly simpler:
zero new I/O at the grading boundary, zero new coupling between a grader and
the artifact store, and it makes the large-output case behave exactly like
the small-output case already does today -- small outputs were already
handed to the grader unredacted and un-spilled. For graders that copy
content into `GradeResult.evidence` (e.g. `HarnessGrader._map_harness_result`,
confirmed to carry only `harness_status`, `harness_message`,
`harness_evidence`, and (conditionally) `harness_error`/`reason` -- never the
raw patch text), the report-boundary redaction
(`agentic_evalkit.reporters.base.apply_redaction`, design §12) was already
the sole safety net regardless of output size, so grade-before-spill
introduces no new leak surface through that path. `JudgeGrader` forwards
output through a different path entirely and is not covered by this
argument -- see Consequences.

## Consequences

- A large output is held in memory slightly longer -- between `execute()`
  and `_spill_large_output()`, while grading runs -- before being redacted
  and spilled. Acceptable: it is bounded by one sample's output size, the
  same object that was already about to be serialized and possibly written
  to disk moments later.
- The redaction guarantee for `GradeResult.evidence` is unchanged:
  report-boundary redaction was already the sole safety net for any raw
  content a grader might copy into `evidence`, regardless of output size;
  grade-before-spill does not weaken it, it only removes the large-output
  case's now-unnecessary detour through a grader-visible `None`.
- `JudgeGrader` is a distinct, pre-existing exposure this change widens
  rather than introduces: it stringifies the full `execution.output`
  (uncapped) and forwards it to a caller-supplied `JudgeClient.judge()`
  implementation, which by design may be a real network call -- a path
  report-boundary redaction cannot reach, since the data has already left
  the process by the time a report is rendered. Before this change, any
  output over the spill threshold arrived at `JudgeGrader` as `None` (an
  accidental size cap, not a deliberate one); after this change,
  `JudgeGrader` sees the same full output every other grader now does. The
  only `JudgeClient` shipped in this repo
  (`examples/reference_judge.py::ReferenceJudgeClient`) is local and
  network-free, so there is no live exfiltration path today, but a caller
  wiring a real network-calling judge should know this library does not
  redact judge-bound content. Not fixed in this ADR -- tracked as a
  follow-up: whether `JudgeGrader` needs its own `RedactionPolicy` (and
  whether cost-motivated truncation belongs alongside it) is a distinct
  design question deserving its own review, not a bundled afterthought on a
  spill-ordering fix.
- `HarnessGrader`'s spilled-output `ERROR` path (`output=None` +
  `output_ref`) becomes a defensive-only guard. The normal `EvalRunner`
  pipeline can no longer produce that combination internally, because
  grading always happens before spilling; the branch now exists solely for
  callers that invoke `HarnessGrader.grade()` directly on an
  already-persisted/spilled `NormalizedExecutionResult` outside the runner
  (for example, a re-grading tool reading a stored run back off disk). Its
  message no longer suggests raising the spill threshold (that advice is now
  stale and wrong); it explains the out-of-pipeline situation and that such
  a caller must re-grade from the original, unspilled execution instead.

## Validation

- `tests/integration/test_runner.py::test_grader_sees_the_full_output_before_it_is_spilled_for_storage`
  -- a generic grader fixture, driven through the real `EvalRunner.run`
  path, captures the exact `execution.output` it was handed for an
  execution large enough to spill (the existing planted-token/padding
  technique) and asserts it is the full, intact output, that the grade is a
  real `PASS` (not the spilled-output error), and that the *final* persisted
  `result.samples[0].execution.output` is still `None` with
  `artifacts["output_ref"]` present.
- `tests/integration/test_runner.py::test_harness_grader_sees_the_full_patch_before_it_is_spilled`
  -- the same proof for the actual `HarnessGrader` (via a `FakeHarnessExecutor`
  and a capturing predictor), the grader that motivated this change: a large,
  SWE-bench-shaped patch reaches the predictor intact and grades to a real,
  hard-gated verdict rather than the defensive `ERROR` path.
- Both tests were confirmed to fail under the pre-fix execute-then-spill
  ordering before this change landed (the predictor in the `HarnessGrader`
  case was not even invoked, since the grader short-circuited into the
  spilled-output `ERROR` branch first), and to pass after it.
- `tests/unit/graders/test_harness_grader.py::test_spilled_output_is_a_diagnostic_error_not_a_silent_unavailable`
  still passes unmodified in behavior: it constructs an already-spilled
  `NormalizedExecutionResult` and calls `grader.grade()` directly (bypassing
  `EvalRunner`), confirming the defensive out-of-pipeline guard still
  reports `GradeStatus.ERROR` with `"spilled"` in the evidence reason.
- `tests/unit/test_spill_redaction.py` is unchanged and still exercises
  `_spill_large_output`'s redaction/threshold behavior directly.

## Supersession

A future change that needs a grader to operate purely off an artifact
reference rather than an inline output (for example, a grader intentionally
designed for memory-bounded streaming over very large outputs) must
supersede this ADR with a new, explicit contract for that case -- not
silently reintroduce spill-before-grade as a side effect of some other
change.
