# ADR-0018: Redact and Bound Judge Candidate Output

## Status

Accepted

## Context

ADR-0017 (grade-before-spill) already surfaced this exact gap in its own
Consequences section without fixing it, calling `JudgeGrader` "a distinct,
pre-existing exposure this change widens rather than introduces." Before
ADR-0017, an `execution.output` large enough to exceed the spill threshold
arrived at `JudgeGrader` as `None` -- described there as "an accidental size
cap, not a deliberate one" -- so `JudgeGrader` never actually saw the
largest outputs. After ADR-0017, "`JudgeGrader` sees the same full output
every other grader now does," which is precisely what widens, rather than
introduces, the exposure. That same Consequences section states plainly
that `JudgeGrader` "stringifies the full `execution.output` (uncapped) and
forwards it to a caller-supplied `JudgeClient.judge()` implementation,
which by design may be a real network call -- a path report-boundary
redaction cannot reach, since the data has already left the process by the
time a report is rendered," and closes by flagging it explicitly as "Not
fixed in this ADR -- tracked as a follow-up: whether `JudgeGrader` needs
its own `RedactionPolicy` (and whether cost-motivated truncation belongs
alongside it) is a distinct design question deserving its own review, not a
bundled afterthought on a spill-ordering fix." This ADR is that follow-up
review, closing the tracked gap.

`agentic_evalkit.reporters.base.apply_redaction`, the report-boundary
redaction every other output path relies on, only redacts a completed,
persisted `EvalRunResult` -- it cannot reach data that has already left the
process during grading, and `JudgeClient` implementations are entirely
caller-supplied (design §9's provider-neutral `Protocol` boundary), so
there is no other safety net today. The one `JudgeClient` shipped in this
repo, `examples/reference_judge.py::ReferenceJudgeClient`, remains local
and network-free, so there is still no live exfiltration path today -- but
the entire purpose of the `JudgeClient` protocol boundary is to let a
caller plug in a real, network-calling model judge, and nothing in the
framework stops a secret-shaped substring in a system-under-test's own
output (an API key echoed back in an error message, a bearer token
captured in a tool-call log) from reaching that judge unredacted. There is
also a cost/reliability dimension distinct from secrecy: an unbounded
string sent as part of every judge prompt, on every graded sample, has no
ceiling on token cost or request size.

## Decision

`JudgeGrader.grade` now runs `execution.output` through a redact-then-
truncate pipeline before it is placed on `JudgeRequest.candidate_output`,
applied to `candidate_output` only:

1. **Redact.** `JudgeGrader.__init__` gains a keyword-only
   `redaction_policy: RedactionPolicy | None = DEFAULT_REDACTION_POLICY`
   parameter, using the same public `RedactionPolicy`/
   `DEFAULT_REDACTION_POLICY` contract from `agentic_evalkit.reporters.base`
   that `EvalRunner` already imports and applies at its own spill boundary.
   `None` (or a `RedactionPolicy()` with empty `secret_patterns`) opts out.
   A new private method, `JudgeGrader._redact_candidate_output`, compiles
   `redaction_policy.secret_patterns` and substitutes matches with the same
   `"[REDACTED]"` marker `reporters.base` uses -- reimplemented locally
   rather than importing `reporters.base`'s private `_redact_string`,
   mirroring the identical precedent `EvalRunner._redact` already set (see
   Alternatives).
2. **Truncate.** `JudgeGrader.__init__` gains a keyword-only
   `max_candidate_output_chars: int | None =
   _DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS` parameter, where the new private
   module constant `_DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS = 8192` mirrors
   `runner.py`'s `_LARGE_OUTPUT_THRESHOLD_BYTES` (8192) as an
   already-reasoned, familiar bound -- defined independently in `judge.py`,
   not imported, since that constant is private to `runner.py`. It is a
   character count, not a byte count: an approximation, not a
   byte-precise UTF-8 bound, which is not needed for a truncation
   heuristic. `None` disables truncation. A new private method,
   `JudgeGrader._truncate_candidate_output`, cuts the string to the bound
   and appends a marker (`"...[truncated, N chars omitted]"`) making the
   cut unambiguous to a human or the judge model.
3. **Order.** Redaction always runs before truncation, never the reverse:
   truncating first risks cutting a secret-shaped pattern in half at the
   boundary and letting the un-redacted remainder through.
4. **Scope.** Only `candidate_output` (`execution.output`, stringified)
   goes through this pipeline. `prompt` (`_stringify_input(sample.input)`)
   and `reference` (`sample.reference`) are never redacted or truncated:
   both are dataset/task-authored content this framework itself controls,
   not target-controlled output, mirroring the same
   target's-own-words-vs-framework-authored-content distinction
   `reporters.base._redact_execution`'s docstring already draws for
   exactly this reason.
5. **Evidence transparency.** When redaction changes the string,
   `GradeResult.evidence["candidate_output_redacted"] = True`; when nothing
   matched, the key is omitted entirely. When truncation fires,
   `evidence["candidate_output_truncated"] = True` and
   `evidence["candidate_output_original_chars"] = <pre-truncation length>`;
   when it doesn't fire, both keys are omitted. This mirrors
   `HarnessGrader`'s existing "only add the key when applicable" convention
   (`evidence["harness_error"]`) and the low-sensitivity,
   facts-about-not-content-of precedent `ArtifactRef.redacted: bool`
   already sets for spilled artifacts.

## Alternatives

1. **Reuse `EvalRunner`'s private `_redact`/`_compiled_secret_patterns`
   directly.** Rejected: cross-module import of a sibling module's private
   helpers breaks this repo's own established precedent. `runner.py`'s own
   `_redact` docstring explains explicitly why `runner.py` does not import
   `reporters.base._redact_string` even though both do the same
   substitution: "this module cannot import that private helper, so the
   same substitution behavior is reimplemented locally against the same
   `RedactionPolicy` contract." `judge.py` follows the identical rule
   against `runner.py`'s private helpers: it imports the public
   `RedactionPolicy`/`DEFAULT_REDACTION_POLICY` (confirmed safe by
   `tests/contract/test_dependency_boundary.py`, whose forbidden roots are
   only `agentic_v2`/`tools`/`executionkit` -- `graders` importing from
   `reporters` is not a boundary violation, and `runner.py` already does
   exactly this cross-module import), but reimplements the substitution
   loop itself.
2. **Push redaction responsibility onto individual `JudgeClient`
   implementations instead of `JudgeGrader`.** Rejected: `JudgeClient` is a
   caller-supplied, provider-neutral `Protocol` (design §9); scattering a
   safety-critical control across every third-party implementation is
   exactly the failure mode centralized redaction exists to avoid.
   `JudgeGrader` is the one place in the framework that knows
   target-controlled content is about to cross a process boundary,
   mirroring why `EvalRunner._spill_large_output` centralizes
   spill-redaction in the runner rather than in every `ExecutionTarget`.
3. **Redaction only, no truncation.** Rejected: doesn't address the
   cost/reliability half of the gap -- an unbounded string sent to a paid
   LLM API on every grade call, with no ceiling on token cost or request
   size.
4. **Redact/truncate `prompt` and `reference` too, not just
   `candidate_output`.** Rejected as a deliberate scope decision, not an
   oversight: both are dataset/task-authored content the framework itself
   controls, not the system-under-test's output -- conflating them would
   blur the same target's-own-words-vs-framework-authored distinction
   `reporters.base._redact_execution` already draws.

## Consequences

- This is a default-ON behavior change, mirroring `EvalRunner`'s own
  redaction-default-on posture (`redaction_policy: RedactionPolicy | None =
  DEFAULT_REDACTION_POLICY`), not a new posture for this codebase: any
  existing `JudgeGrader` caller now gets redaction and truncation
  automatically, unless it explicitly opts out with `redaction_policy=None`
  and/or `max_candidate_output_chars=None`.
- `JudgeRequest.candidate_output` may now differ from `execution.output`'s
  literal content. A `JudgeClient` implementer must not assume
  byte-for-byte fidelity between the two; `JudgeGrader`'s class docstring
  now states this prominently, not only in a code comment.
- The one `JudgeClient` shipped in this repo, `ReferenceJudgeClient`, is
  local and network-free, so this closes a currently-dormant gap
  proactively, before any real network-calling judge integration exists in
  this codebase.
- `GradeResult.evidence` may now carry up to three new keys
  (`candidate_output_redacted`, `candidate_output_truncated`,
  `candidate_output_original_chars`), each added only when applicable.
  These are booleans/counts, never the redacted or truncated content
  itself.
- A caller who needs the judge to see genuinely full-fidelity output (for
  example, a debugging session where secrecy and cost are not concerns)
  must explicitly pass `redaction_policy=None` and
  `max_candidate_output_chars=None`; this is an opt-out, not the default.

## Validation

- `tests/unit/graders/test_judge.py::test_secret_shaped_candidate_output_is_redacted_before_reaching_the_judge`
  -- a planted `hf_`-shaped token is redacted to `[REDACTED]` in the actual
  `JudgeRequest` a capturing fake `JudgeClient` receives.
- `tests/unit/graders/test_judge.py::test_oversized_candidate_output_is_truncated_before_reaching_the_judge`
  -- an output past the char bound is cut to that bound plus a marker, and
  `evidence["candidate_output_truncated"]`/`evidence["candidate_output_original_chars"]`
  match the real fixture-derived lengths.
- `tests/unit/graders/test_judge.py::test_redaction_policy_none_disables_redaction`
  -- opting out leaves a planted secret intact and omits
  `evidence["candidate_output_redacted"]`.
- `tests/unit/graders/test_judge.py::test_max_candidate_output_chars_none_disables_truncation`
  -- opting out leaves an oversized output whole and omits both truncation
  evidence keys.
- `tests/unit/graders/test_judge.py::test_default_construction_uses_the_named_default_policy_and_bound`
  -- omitting both constructor parameters behaves identically to passing
  `DEFAULT_REDACTION_POLICY`/`_DEFAULT_MAX_CANDIDATE_OUTPUT_CHARS`
  explicitly.
- `tests/unit/graders/test_judge.py::test_prompt_and_reference_are_never_redacted_or_truncated`
  -- the same planted secret and oversized length in `prompt`/`reference`
  reach the judge untouched; only `candidate_output` is altered.
- `tests/unit/graders/test_judge.py::test_clean_short_output_adds_no_candidate_output_evidence_keys`
  -- a clean, short output adds none of the three evidence keys.
- `tests/contract/test_dependency_boundary.py::test_package_does_not_import_arp_tools_or_executionkit`
  -- confirms the new `graders` -> `reporters` import does not cross a
  forbidden boundary.

## Supersession

A future change to which `JudgeRequest` fields get redacted, the
truncation bound or its unit (characters vs. bytes), or the
`"[REDACTED]"`/truncation marker format must supersede this ADR with new
validation -- not silently reinterpret this one.
