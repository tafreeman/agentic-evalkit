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
  No gate reads them yet; they are documented as forward-looking auditability
  fields, not a new gating input.

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
  limits as first-class non-gating outcomes rather than as raises or fabricated
  verdicts, closing the operational-vs-task conflation ADR-0008 warns against at
  the judge boundary specifically.

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
abstention as first-class, bounded parse retries) stand unchanged. A future
change to the Wilson confidence level, the project floor values, the
`JudgeResponseStatus` vocabulary, the status-to-`GradeStatus` mapping, the
probe-issuance condition, or the persisted judge-evidence keys must supersede
this ADR with new validation, not silently reinterpret it.
