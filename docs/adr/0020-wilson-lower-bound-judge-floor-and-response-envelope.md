# ADR-0020: Wilson Lower-Bound Judge Floor, Response Envelope, and Gating-Scoped Probe

## Status

Accepted

## Context

ADR-0007 established that a `JudgeGrader` may gate a release only under a
calibration whose held-out **point** estimates clear the ratified project
floor (TNR >= 0.95, TPR >= 0.85), with at least 30 held-out samples per class,
unexpired, fingerprint-matched, and agreeing with a reversed-order
position-bias probe. Decision D-1 (2026-07-04) refined the demotion matrix so
that affirmatively bad evidence (expired, sub-floor point estimate) is
`UNAVAILABLE` while absent evidence (undated or stale `calibrated_at`) blocks
gating but still grades advisorily.

The point-estimate floor has a statistical gap. A small held-out class can put
a *point* estimate above the floor while the sample is too thin to *prove* the
rate clears it: a 29/30 negative class reads TNR = 0.967 (above 0.95), yet its
95% Wilson lower bound is ~0.833. ADR-0016 already adopted Wilson intervals as
this package's standing way of never claiming more certainty than the data
supports, and `agentic_evalkit.stats.wilson_interval` is the public helper that
computes them -- but the judge gate did not yet consult a lower bound, so a
marginally-sized calibration could hard-gate a release on evidence its own
confidence interval does not support. That is the same class of overclaim
ADR-0007 exists to prevent, one level deeper.

Two further gaps in the judge contract surfaced in the A-prime review:

- The `JudgeClient` boundary (design section 9) had no way to distinguish a
  *refusal* or an *operational* failure (timeout, rate limit, provider error)
  from a rendered verdict. A judge that timed out or declined could only signal
  it by raising -- which, unhandled, aborts the whole run -- or by fabricating a
  verdict, folding an operational failure into a task outcome, exactly what
  ADR-0008 forbids. A single raising caller-supplied judge should cost one
  graded `ERROR` sample, never the entire run.
- The reversed-order position-bias probe (ADR-0007) was issued on *every*
  graded sample, including advisory grading (`gate=False`, or any calibration
  failure) whose probe result can never gate and is discarded. That doubled
  judge-call cost -- a real, billed cost for a network-calling judge provider --
  on the one path where the second call buys nothing.
- A judge's own free-text `rationale` had nowhere to be recorded. Being judge
  output that can echo target-controlled content, it cannot be persisted safely
  without the same redact-then-truncate treatment ADR-0018 already applies to
  `candidate_output`.

## Decision

This package tightens the judge gate and enriches the judge response envelope.
Every wire change is additive; `schema_version` stays `"1"` (ADR-0002).

- **Wilson lower-bound floor (insufficient-evidence gate).**
  `CalibrationArtifact.wilson_lower_bound_failure_reason` computes the 95%
  Wilson lower bound of TNR and TPR from the held-out confusion matrix (via the
  public `agentic_evalkit.stats.wilson_interval`) and returns a reason naming
  the bound when either falls below its project floor. The check is added
  *inside* `usability_failure_reason`, alongside the existing age check, so it
  blocks gating only while advisory grading continues -- age is reported first
  because a stale or undated artifact fails for a reason independent of the
  confusion-matrix counts.
- **Insufficient-vs-bad evidence taxonomy.** A *point* estimate below the floor
  remains affirmatively bad evidence and demotes to `GradeStatus.UNAVAILABLE`
  via `floor_failure_reason` (unchanged from D-1). A point estimate that clears
  the floor while its Wilson *lower* bound does not is merely *insufficient*
  evidence -- not affirmatively bad -- so, like an absent or stale age, it blocks
  gating only. Concretely, a 29/30 class (0.967 point, ~0.833 lower bound) no
  longer gates but still grades advisorily.
- **Response status envelope and transport mapping.** `JudgeResponse` gains
  `status: JudgeResponseStatus = JudgeResponseStatus.OK`. A non-OK envelope
  short-circuits `JudgeGrader.grade` *before* fingerprint/abstention handling:
  `REFUSED` maps to `GradeStatus.ABSTAIN` (a refusal is a non-verdict, never a
  task `FAIL`); `RATE_LIMITED`/`TIMEOUT`/`ERROR` map to `GradeStatus.ERROR`
  (operational, kept separate from task failure per ADR-0008). The reason names
  the status; the result never gates.
- **Transport-exception isolation.** `_judge_with_bounded_retries` wraps each
  judge call: a raise terminates immediately (no retry storm) and does *not*
  consume the parse-retry budget, returning a sentinel that `grade` maps to a
  single `GradeStatus.ERROR` sample carrying `evidence["judge_transport_error"]`
  (the exception type) and a redacted, bounded `judge_transport_error_message`.
  The position-bias probe is wrapped the same way: a probe raise becomes a
  gate-blocking reason, never a propagated exception. `asyncio.CancelledError`
  (a `BaseException`, not an `Exception`) is deliberately not caught, so run
  cancellation still propagates.
- **Runner-level symmetric target/grader isolation (amendment).** The
  transport-exception isolation above originally lived only inside
  `JudgeGrader`, leaving an asymmetry: `EvalRunner` wrapped neither
  `target.execute` nor `grader.grade`, so a raise from an arbitrary
  `ExecutionTarget` or `Grader` cancelled every in-flight sibling through the
  `TaskGroup`, discarded already-completed results, and propagated an uncaught
  `ExceptionGroup` — which, not being an `AgenticEvalkitError`, bypassed the
  CLI's documented exit-code mapping. `EvalRunner._execute_isolated` and
  `_grade_isolated` now apply the same per-sample treatment at those two
  boundaries: a raising `execute` becomes that sample's
  `ExecutionStatus.ERROR` result (`ExecutionStatus.TIMEOUT` for a
  `TimeoutError`, which on 3.11+ is `asyncio.TimeoutError`), and a raising
  `grade` becomes that sample's `GradeStatus.ERROR`. The exported
  `TargetFailure`/`TargetTimeout`/`GraderError` taxonomy is wired in: each
  error record carries the wrapper's stable `code` and a message redacted then
  bounded under the ADR-0018 order (`grader_error`/`grader_error_code`/
  `grader_error_message` on grade evidence; `type`/`code`/`message` on the
  execution `error`). Only `Exception` is caught, so `CancelledError` still
  tears the run down exactly as before. This closes the fault-isolation
  asymmetry: one raising sample no longer aborts the run, `RunCompleted` still
  fires, and every other sample's completed result survives — the target and
  grader boundaries now behave symmetrically with the judge boundary and with
  the operational-vs-task separation ADR-0008 requires.
- **Runner-level output-spill isolation (amendment).** The spill that moves an
  oversized output to the `ArtifactStore` is the third call
  `_execute_and_grade` makes that can raise, and it stayed unguarded inside the
  same `TaskGroup` after the two boundaries above were isolated. A store
  failure — `ArtifactStoreLimitExceeded` (a plain `ValueError` carrying no
  `.code` of its own), or an `OSError` from a full or read-only disk —
  therefore cancelled every in-flight sibling and escaped `EvalRunner.run` as
  an `ExceptionGroup`, so no report was written and every result the run had
  already graded was lost over a storage problem. `EvalRunner._spill_isolated`
  now applies the same per-sample treatment at that third boundary: the
  affected sample degrades to `output=None` carrying the new `OutputSpillFailed`
  taxonomy code (`output_spill_failed`) with a message redacted then bounded
  under the ADR-0018 order. It differs from the target and grader boundaries in
  one deliberate respect — `status` and any already-computed `grade` are
  **preserved**. A store refusing the bytes is a storage failure, not a verdict
  on the attempt: the target genuinely completed, and ADR-0017 guarantees the
  grader had already seen the full inline output before the spill ran. Flipping
  the status to `ERROR` would re-bucket an earned grade as an operational error
  in both `_summarize` and `stats.aggregate`, and would break the standing
  invariant that a non-`COMPLETED` execution carries `grade=None`. The record
  always lands in `artifacts["output_spill_error"]` — that namespace is the
  boundary's own log of what it did, `output_ref` on success and
  `output_spill_error` on failure — and claims the execution's primary `error`
  field only when the execution carries none, so a target that already reported
  its own failure alongside a large output keeps that diagnosis intact. As
  above, only `Exception` is caught, so `CancelledError` still tears the run
  down.
- **Gating-scoped probe.** The reversed-order probe is issued only when
  `gate=True` **and** the calibration is usable (no calibration failure), so the
  advisory path makes exactly one judge call per sample. It is not additionally
  guarded on `status is PASS`, so a calibrated `FAIL` sample is still probed and
  its position-bias reason survives into `evidence["reason"]`.
- **Rationale with redaction.** `JudgeResponse` gains
  `rationale: str | None = None`. When present it is redacted then truncated
  (ADR-0018 order) and recorded to `evidence["judge_rationale"]` as evidence
  only; no gating decision ever reads a rationale or any confidence-like
  content (design section 9's objective-first ordering forbids it).
- **Additive coverage evidence.** `CalibrationArtifact` gains optional
  `total_labeled`, `abstained_count`, and `error_count` (non-negative when
  present, folded into the existing validator) as recorded coverage evidence.
  Originally audit-only; ADR-0024 activates them as a gating input when all
  three are present, via `coverage_failure_reason` and
  `PROJECT_MAX_NON_VERDICT_RATE`.

## Alternatives

1. **Keep the point-estimate-only floor.** Rejected: a marginally-sized
   calibration whose confidence interval dips below the floor can still
   hard-gate, which is the overclaim ADR-0007 exists to prevent, one statistical
   level deeper.
2. **Treat a sub-floor Wilson lower bound as affirmatively bad
   (`UNAVAILABLE`).** Rejected: insufficient evidence is not the same as bad
   evidence. A judge that historically tracks humans well but was calibrated on
   too small a held-out set should not lose its advisory value; it should only
   lose its authority to gate -- exactly the absent-vs-bad distinction D-1 drew
   for age.
3. **Reimplement the Wilson lower bound locally in `judge.py`.** Rejected:
   `wilson_interval` is public (`agentic_evalkit.stats.__all__`) and `stats`
   imports nothing from `graders`, so importing it creates no cycle. That is a
   different situation from `runner._redact`, which reimplements its sibling's
   *private* helper only because it cannot import a private name; here the
   public helper is importable, so it is imported rather than duplicated.
4. **Let a judge signal refusal or timeout by raising.** Rejected: an unhandled
   raise aborts the run, and catching every raise as the same outcome erases the
   refusal/operational distinction. A typed status envelope preserves it while
   keeping one raising judge to one graded `ERROR` sample.
5. **Keep probing every graded sample.** Rejected: the advisory path discards
   the probe result, so the second judge call is pure cost on that path. Scoping
   the probe to the gating path removes it without weakening any gate.
6. **Redact `prompt`/`reference` too, or read `rationale` in gating.** Rejected:
   `prompt` and `reference` are framework-authored, not target-controlled
   (ADR-0018's scope decision stands), and reading a judge's self-reported
   rationale or confidence to influence gating is precisely the subjective
   shortcut design section 9 orders last, never first.
7. **Truncate an oversized output and store the prefix, marking the artifact
   truncated, instead of isolating the spill failure.** Rejected: it answers a
   different question. Truncation addresses only the store's size limit and
   does nothing for the other ways a store fails (a full disk, a read-only
   directory, a permission change mid-run), so the run-aborting defect would
   survive in every non-size case. It also silently converts a stored artifact
   into a partial one, which is precisely the kind of quiet quality
   degradation this project's evidence-first posture exists to prevent: a
   truncated patch or transcript still *looks* like the answer. Per-sample
   isolation is the treatment already ratified for the target and grader
   boundaries above, it covers every failure mode rather than one, and it
   keeps what was lost explicit rather than plausible.
8. **Mark a failed spill as `ExecutionStatus.ERROR`, so it shows up in the
   run's error count and exit code.** Rejected: it destroys evidence to raise
   an alarm. The attempt completed and the grader had already scored the full
   inline output (ADR-0017), so re-bucketing the sample as an operational
   error discards a verdict that was genuinely earned -- the same class of
   loss the isolation exists to prevent, just smaller. It would also break the
   invariant that a non-`COMPLETED` execution carries `grade=None`. The
   discoverability concern is real and is addressed where it belongs: the
   `run` command prints an explicit warning naming how many outputs were
   dropped, without falsifying any count.

## Consequences

- A gating calibration must now carry enough held-out evidence for its Wilson
  lower bound -- not merely its point estimate -- to clear the floor. Fixtures
  sized at the floor with a small denominator (n ~ 100) no longer gate.
  `tests/unit/graders/test_judge.py`'s `_valid_calibration` is rescaled to 2000
  held-out samples per class accordingly. `tests/unit/graders/test_judge_calibration_floor.py`,
  which pins several n = 100 exactly-at-floor and clearing-the-floor cases as
  gating (and asserts a fresh n = 100 artifact's `usability_failure_reason` is
  `None`), must be re-scaled to sufficient-evidence counts to reflect this
  decision; those cases assert the pre-Wilson behavior and change meaning here.
- The advisory/uncalibrated grading path costs one judge call per sample instead
  of two, halving judge spend for callers who wire a real, billed judge provider
  in advisory mode.
- `JudgeResponse` and `CalibrationArtifact` each gain additive, defaulted fields;
  every existing `JudgeClient` that never sets them keeps its exact prior
  meaning, and existing persisted artifacts deserialize unchanged.
- `GradeResult.evidence` may now carry `judge_rationale`,
  `judge_transport_error`, and `judge_transport_error_message` keys, each added
  only when applicable, mirroring the ADR-0018 convention. The transport message
  and the rationale are redacted and bounded before persistence.
- A future real judge integration can now surface refusals, timeouts, and rate
  limits as distinct non-gating outcomes rather than as raises or fabricated
  verdicts, closing the operational-vs-task conflation ADR-0008 warns against at
  the judge boundary specifically.
- The runner-level amendment makes any raising `ExecutionTarget`/`Grader`
  survivable, not just a raising `JudgeClient`. `NormalizedExecutionResult.error`
  may now carry runner-authored `type`/`code`/`message` keys, and
  `GradeResult.evidence` may carry `grader_error`/`grader_error_code`/
  `grader_error_message`, each redacted and bounded before persistence like the
  judge keys above. Callers that previously saw an uncaught `ExceptionGroup`
  escape `EvalRunner.run` on a target/grader raise now receive a completed
  `EvalRunResult` whose affected sample is an operational `ERROR`/`TIMEOUT`
  (never a task `FAIL`), counted in `RunSummary.errors`/`.timeouts`.
- The spill amendment surfaces in the persisted result rather than in the exit
  status. `NormalizedExecutionResult.artifacts` may now carry an
  `output_spill_error` record, and `error` may carry `output_spill_failed` on a
  still-`COMPLETED` execution — the one case where an error record accompanies
  a clean status, and the reason that field's documentation no longer promises
  otherwise. Because the status is preserved, a spill failure alone does not
  raise `RunSummary.errors` and therefore does not change the CLI's exit code.
  So that a degraded run cannot look untouched, the `run` command prints an
  explicit warning naming how many sample outputs were dropped; the per-sample
  detail is always in `artifacts["output_spill_error"]`, whether or not the
  spill record also claimed `error`. `HarnessGrader` recognises the new state
  and re-grades it as an explicit `GradeStatus.ERROR` naming the failed spill,
  instead of the factually wrong "produced no output" `UNAVAILABLE` it would
  otherwise fall through to.
- Report-boundary redaction now sweeps `execution.artifacts` alongside
  `output`/`structured_output`/`error`. That field previously held only
  harness-authored values (an `output_ref` digest), but a failed spill records
  the store's own exception text there, and a store that talks to something
  remote can echo a credential into it. Without the sweep, a caller who had
  opted out of the runner's spill redaction (`redaction_policy=None`) could
  have written that text unredacted into the canonical report — the one
  promise `write_canonical_report` makes. A genuine `output_ref` digest is
  exempt from the sweep so the reference it carries is never rewritten; see
  the digest bullet below for why that exemption is gated on the value's
  shape rather than on the key name.
- The spill isolation is closed against its own failure handler.
  `_safe_error_message` compiles the caller's `secret_patterns`, the very call
  that raises `re.error` on a malformed pattern inside `_spill_large_output`;
  deriving the recorded message through `_spill_failure_message` means that
  raise cannot re-enter the `TaskGroup` from the `except` block and cancel the
  siblings the isolation just saved. The record is also copied per field rather
  than shared, so a later stage rewriting `error` cannot silently rewrite
  `artifacts` too. This hardening is scoped to the spill boundary: the target
  and grader handlers still call `_safe_error_message` directly, so the same
  malformed pattern escapes those two `except` blocks. That is a pre-existing
  gap this amendment deliberately leaves open: closing it belongs with the
  boundaries it affects, not with the storage decision recorded here.
- Dropping the output is conditional on it actually having been oversized.
  `_spill_large_output` serializes, compiles patterns and encodes *before* it
  compares against the threshold, so a raise from any of those steps reaches
  the handler for an output the store was never offered. Nulling it there
  would destroy a small inline answer -- for every sample in the run, since
  the redaction policy is per-runner -- while recording that the store refused
  bytes it never saw. The handler therefore re-measures the raw serialization
  (`str` on JSON-shaped data cannot raise, so the check is available even when
  the redaction that follows it is not) and keeps any output within the
  threshold inline, where it is bounded by definition and gets redacted at the
  report boundary like any other small output. Because the record is written
  whether or not the output was dropped, a record alone is not evidence of
  loss: the `run` command's warning counts a sample only when `output is None`
  *and* the record is present, so a boundary that failed before the size check
  — a per-runner setting, therefore every sample in the run — cannot make a
  run that lost nothing report a total loss. `_output_exceeds_inline_threshold`
  is itself written never to raise, for the same reason
  `_spill_failure_message` is: it runs inside the same `except` block, where a
  raise would re-enter the `TaskGroup` and cancel the siblings the isolation
  had just saved.
- `artifacts` is target-controlled, so `output_spill_error` is a key any target
  may write. Consumers that act on a spill failure -- `HarnessGrader` deciding
  a result cannot be re-graded, the `run` command's dropped-output warning --
  route through `models.execution.is_output_spill_error_record`, which requires
  a mapping carrying the `output_spill_failed` code. A target therefore cannot
  steer a grade or a warning by guessing a key name, no consumer subscripts a
  record that may lack the key it expects, and no unbounded target-authored
  string reaches `GradeResult.evidence`, which is not length-bounded the way
  the runner's own recorded messages are. A successful spill also clears any
  stale *runner-written* `output_spill_error` before writing `output_ref`: the
  two describe the same boundary's outcome and cannot both be true, and
  consumers check the failure key first, so a leftover record would announce
  that bytes sitting on disk were never persisted. Only records carrying the
  taxonomy code are removed — a target's own data under that name survives its
  output spilling, and cannot cause that misreading anyway, since every
  consumer validates the record's shape first.
- The `output_ref` digest is exempt from the widened redaction sweep. It is the
  only pointer back to a payload already written to the store, so a pattern
  that rewrites one character of it orphans those bytes permanently. The
  default policy cannot match a hex digest, but `apply_redaction` is public and
  takes any policy a caller supplies, and a generic "long hex string looks like
  a key" rule is a reasonable thing to add. The digest is harness-authored and
  structurally incapable of carrying a secret, so the exemption gives up
  nothing; every other key in `artifacts` is still swept.
- That exemption is gated on the value's **shape**, not on the key name, for
  the same reason `output_spill_error` is: `artifacts` is target-controlled and
  neither key is reserved. Exempting whatever happens to sit under `output_ref`
  would let a target returning `artifacts={"output_ref": "hf_…"}` carry a live
  credential through the sweep and into the canonical report — under the
  *default* policy, which is the one the CLI writes every report with, so the
  hole would be reachable from the shipped tool rather than only from a
  caller-supplied policy. Only a value matching what `ArtifactStore` mints
  (`sha256:` plus 64 lowercase hex characters) is treated as a reference;
  anything else is free-form target text and is redacted with the rest of the
  dict.
- The spill record's `code` is exempt on the same grounds. It is the value
  `is_output_spill_error_record` matches on, so a caller policy broad enough
  to rewrite `output_spill_failed` would leave the redacted report carrying a
  record no consumer recognises — `HarnessGrader` re-grading it falls back to
  the "produced no output" `UNAVAILABLE` this amendment exists to eliminate,
  and the `run` command's warning stops firing. Restoring it smuggles
  nothing, because it is not restored from the record: it is rewritten to the
  fixed `OUTPUT_SPILL_FAILED_CODE` literal, and only for a record that
  already carried the code before the sweep. The record's `message` and
  `type` — the parts that can hold store-authored text — are still redacted,
  as is every other key in the dict.

## Validation

- `tests/unit/graders/test_judge.py`:
  `test_wilson_lower_bound_below_floor_blocks_gating_but_grades_advisorily`
  (29/30 class blocks gating with a Wilson reason but grades advisorily and
  makes one judge call), `test_point_estimate_below_project_floor_stays_unavailable`
  (point below floor still `UNAVAILABLE`),
  `test_raising_judge_client_yields_single_error_sample_with_transport_evidence`
  (one raising judge -> one `ERROR` sample, one attempt, redacted
  `judge_transport_error` evidence), `test_refused_status_maps_to_abstain`,
  `test_timeout_status_maps_to_error`,
  `test_uncalibrated_grade_makes_exactly_one_judge_call`,
  `test_calibrated_fail_sample_still_runs_probe_and_records_reason` (probe runs
  on a calibrated `FAIL`, reason survives), `test_rationale_is_redacted_and_truncated_in_evidence`,
  and `test_calibration_coverage_fields_reject_negative_values`.
- `tests/contract/test_models.py`:
  `test_judge_response_status_and_rationale_round_trip`,
  `test_judge_response_status_defaults_to_ok_and_is_not_collapsed_to_boolean`,
  `test_calibration_artifact_coverage_fields_round_trip`, and
  `test_calibration_artifact_coverage_fields_default_to_none` prove the additive
  fields round-trip through versioned JSON at `schema_version` `"1"`.
- `tests/integration/test_runner.py::test_run_completes_when_the_judge_raises_on_one_sample`
  drives a full `EvalRunner` run where the judge raises on one sample: the run
  finishes (no `RunFailed`), the affected sample grades `ERROR` with
  `judge_transport_error` evidence, and the other sample grades normally.
- `tests/integration/test_runner.py` proves the runner-level amendment
  symmetrically: `test_run_completes_when_the_target_raises_on_one_sample`
  (a raising `execute` on one of two concurrent samples → that sample is
  `ExecutionStatus.ERROR` with `code == "target_failure"`, the other's
  completed result survives, no `RunFailed`),
  `test_run_completes_when_the_grader_raises_on_one_sample` (a raising
  `grade` → that sample is `GradeStatus.ERROR` with `grader_error` evidence,
  the other grades normally),
  `test_run_maps_a_raising_target_timeout_to_a_timeout_result` (a raised
  `TimeoutError` → `ExecutionStatus.TIMEOUT`, `code == "target_timeout"`), and
  `test_isolated_target_error_message_is_redacted_and_bounded` (a planted
  `hf_` token in the raise is stripped and an oversized message truncated on
  the recorded error). The existing cancellation tests
  (`test_cancellation_during_the_run_emits_exactly_one_run_failed`,
  `test_cancelling_the_run_marks_pending_samples_cancelled`) still pass,
  confirming `CancelledError` is not swallowed by the new isolation.
- `tests/unit/test_spill_redaction.py` proves the spill-boundary amendment
  against a real `ArtifactStore` whose `max_bytes` sits between the spill
  threshold and the payload, so the spill is entered and then genuinely
  rejected: `test_store_rejecting_the_payload_degrades_the_sample_instead_of_raising`
  (`output=None`, `status` unchanged, `error["code"] == "output_spill_failed"`,
  `error["type"] == "ArtifactStoreLimitExceeded"`, nothing written to disk),
  `test_an_error_the_target_already_reported_is_never_overwritten` (the
  target's own `error` survives verbatim while the spill record still lands in
  `artifacts`), `test_a_store_failure_that_is_not_a_value_error_degrades_identically`
  (an `OSError` absorbed the same way, naming the real class),
  `test_a_secret_in_the_store_failure_message_is_redacted_before_it_is_recorded`,
  and `test_cancellation_raised_by_the_store_is_deliberately_not_absorbed`
  (`CancelledError` propagates out of `_spill_isolated` uncaught).
- `tests/integration/test_runner.py::test_run_completes_when_the_artifact_store_rejects_one_sample_output`
  proves the run-level guarantee end to end: two concurrent samples, a store
  that rejects one sample's oversized output, and the run still returns
  normally with `RunCompleted` and no `RunFailed`, the sibling's inline graded
  result intact, the affected sample keeping its `COMPLETED` status and its
  `PASS` grade with `artifacts["output_spill_error"]["code"] ==
  "output_spill_failed"`, and `summary.total == 2` with `summary.errors` still
  zero.
- `tests/unit/graders/test_harness_grader.py::test_failed_spill_is_a_diagnostic_error_not_a_silent_unavailable`
  pins the re-grade path (a failed-spill result grades `ERROR` naming the
  spill, never `UNAVAILABLE`), and `tests/unit/test_errors.py` pins
  `OutputSpillFailed(...).code == "output_spill_failed"` as a stable taxonomy
  contract rather than an accident of the class name.
- `tests/unit/reporters/test_redaction_policy.py::test_redaction_covers_execution_artifacts`
  proves a credential planted in an `output_spill_error` message is
  `[REDACTED]` in the redacted run while the record stays machine-readable,
  and `::test_redaction_leaves_a_spilled_output_digest_intact` proves the
  widened sweep does not rewrite an `output_ref` digest, which would orphan
  the artifact it points at.
  `::test_a_caller_supplied_pattern_never_rewrites_an_output_ref_digest`
  proves the same under a *non-default* policy (`[A-Fa-f0-9]{32,}`, which
  does match a digest), and
  `::test_the_digest_exemption_does_not_shield_the_rest_of_artifacts` proves
  the exemption is exactly that narrow -- a secret in a sibling key of the
  same dict is still redacted.
- `tests/unit/test_spill_redaction.py::test_a_failure_before_the_size_check_leaves_a_small_output_inline`
  proves a ~30-byte output survives a spill boundary that raises before the
  store is reached, while
  `::test_an_oversized_output_is_still_dropped_when_the_failure_is_not_the_store`
  proves a genuinely oversized payload is still dropped no matter which step
  failed, and
  `::test_a_successful_spill_clears_a_stale_spill_error_record` proves a spill
  that succeeds does not leave a runner-written failure record behind it,
  while `::test_a_successful_spill_keeps_a_targets_own_data_under_that_key`
  proves that clearing is scoped to records carrying the taxonomy code rather
  than deleting whatever a target stored under the name.
- `tests/unit/cli/test_run_spill_warning.py::test_a_recorded_failure_that_kept_its_output_inline_is_not_counted`
  pins that the warning counts lost outputs rather than recorded failures, so
  a spill boundary that fails before the size check — which, being a
  per-runner setting, fails every sample — cannot make a run that lost nothing
  report a total loss.
- `tests/unit/reporters/test_redaction_policy.py::test_a_target_cannot_smuggle_a_secret_through_the_output_ref_exemption`
  pins that the digest exemption is gated on the value's shape: a target
  returning `artifacts={"output_ref": "hf_…"}` has it redacted under the
  default policy rather than carried into the report, and
  `::test_the_digest_exemption_matches_what_the_artifact_store_actually_mints`
  pins the exemption's shape gate against a digest a real `ArtifactStore`
  minted, so the two independent spellings of that format cannot drift with
  the suite green — the same drift the taxonomy-code contract test closes one
  constant over.
- `tests/unit/reporters/test_redaction_policy.py::test_a_caller_supplied_pattern_never_rewrites_the_spill_failure_code`
  pins the second exemption under a policy (`[a-z_]{16,}`) that does match
  `output_spill_failed`: the code survives and the record stays recognisable
  to `is_output_spill_error_record`, while the same pattern still rewrites the
  record's `message`.
- `tests/integration/test_cli_spill_warning.py::test_run_warns_on_stdout_when_sample_outputs_could_not_be_stored`
  additionally pins that the warning arrives as one unbroken line. It is 146
  characters and the console is pinned to width 120 whenever stdout is not a
  terminal, so without `soft_wrap=True` Rich strands
  `artifacts.output_spill_error` on a line of its own — breaking exactly the
  scripted consumers the CLI guide points at it.
- `tests/unit/graders/test_harness_grader.py::test_an_oversized_output_ref_is_truncated_before_it_reaches_the_evidence`
  exercises `_bounded_ref`'s truncation branch, which a genuine 71-character
  digest never reaches and which nothing else in the suite covered.
- `tests/contract/test_models.py::test_the_spill_failure_code_constant_matches_the_error_taxonomy`
  pins `models.execution.OUTPUT_SPILL_FAILED_CODE` against
  `OutputSpillFailed.code`, so the two independent spellings of that one
  contract cannot drift with the suite green.
- `tests/unit/graders/test_harness_grader.py::test_a_target_cannot_hijack_the_spill_diagnosis_by_key_name`
  proves a target-written `output_spill_error` (a non-dict, a dict with no
  `code`, or a dict with some other code) falls through instead of raising
  `KeyError` or reporting a spill failure that never happened, and
  `::test_a_genuine_spill_record_still_wins_over_a_stale_output_ref` re-pins
  the ordering rule for a real record.
- `tests/integration/test_cli_spill_warning.py` proves the discoverability
  claim end to end through the real `run` command: with a store that refuses
  both samples, the command exits `0`, still writes its report, and prints a
  warning naming how many outputs were dropped -- and
  `::test_run_stays_silent_about_spills_when_nothing_was_lost` proves the line
  is conditional rather than always emitted.
  `tests/unit/cli/test_run_spill_warning.py::test_repeated_attempts_at_one_sample_are_counted_once`
  pins that the count is per sample rather than per attempt, and
  `::test_a_target_writing_the_key_itself_is_not_counted` that a target cannot
  provoke the warning.
- `tests/contract/test_adrs.py` adds `"0020"` to `REQUIRED_ADR_PREFIXES`, so
  this ADR's shape (seven headings, canonical order, `Accepted`, no
  contradicting phrases) is enforced identically to every other ADR, and
  `test_landing_page_adr_claims_match_committed_adr_count` tracks the new total.

## Supersession

This ADR supersedes ADR-0007's point-estimate-only calibration floor and its
issue-on-every-sample position-bias probe, per ADR-0007's own Supersession
clause requiring a superseding ADR with new calibration evidence for any change
to the floor or the position-bias policy. ADR-0007's other conditions
(fingerprint equality, expiry, minimum held-out counts, per-artifact threshold,
abstention as its own outcome, bounded parse retries) stand unchanged. A future
change to the Wilson confidence level, the project floor values, the
`JudgeResponseStatus` vocabulary, the status-to-`GradeStatus` mapping, the
probe-issuance condition, or the persisted judge-evidence keys must supersede
this ADR with new validation, not silently reinterpret it.
