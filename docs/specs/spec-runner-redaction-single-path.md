---
title: 'Runner redaction: single sanitizer path, single opt-out spelling'
type: 'bugfix'
created: '2026-07-30'
status: 'in-review'
review_loop_iteration: 0
followup_review_recommended: false
context: []
warnings: ['multiple-goals', 'oversized']
baseline_revision: 'ad587d9d7cedf72103f6935d78b474c6b127801f'
final_revision: ''
---

<intent-contract>

## Intent

**Problem:** `EvalRunner._emit_run_failed` builds `RunFailed(message=str(error))` with neither
redaction nor length bounding, even though the runner already has a redact-then-bound helper
(`_safe_error_message`) used everywhere else an exception message is persisted (DW-2). Separately,
`EvalRunner` and `JudgeGrader` each accept two different values that both disable spill/candidate
redaction (`redaction_policy=None` and `redaction_policy=RedactionPolicy()`), and the more dangerous,
implicit spelling (`None`) has no deprecation signal (DW-7, human decision recorded 2026-07-30).

**Approach:** Route the `RunFailed` emit through `_safe_error_message`, widening that helper's
`error` parameter from `Exception` to `BaseException` since `run()` catches `BaseException` and
cancellation reaches this path. In `EvalRunner.__init__` and `JudgeGrader.__init__`, keep accepting
`redaction_policy: RedactionPolicy | None` for backward compatibility, but when `None` is passed,
emit a `DeprecationWarning` and normalize internally to an explicit empty `RedactionPolicy()` before
storing it, so `_compiled_secret_patterns`'s `is None` branch is no longer reachable and exactly one
stored representation disables redaction. Document `RedactionPolicy()` as the single supported
opt-out in both class docstrings, record the deprecation and its removal release in the changelog,
reconcile ADR-0018's prose (which still names `redaction_policy=None` as *the* opt-out) with the new
deprecated status, and add tests asserting the warning fires while behavior is unchanged.

## Boundaries & Constraints

**Always:**
- `redaction_policy: RedactionPolicy | None` stays accepted at both constructors — this is a
  deprecation, not a removal.
- `None` continues to behave exactly as before (redaction fully disabled) — only the warning and the
  internal representation change, never the observable behavior for existing callers.
- Redaction is always applied before truncation in `_safe_error_message` (existing order; do not
  change it).
- `_safe_error_message`'s new `BaseException` parameter type must not change its runtime behavior for
  the existing `Exception` call sites (`_target_error_result`, `_grader_error_result`).
- ADR-0018 keeps its `Accepted` status, all seven required headings in canonical order, and must not
  introduce any of `tests/contract/test_adrs.py`'s forbidden contradicting phrases.

**Block If:** (none identified — the human decision on DW-7 already resolved the only design choice
this bundle required)

**Never:**
- Do not remove or weaken the `None` acceptance path (that is a future, separately-decided release).
- Do not add `warnings.warn` anywhere except the two constructors described above.
- Do not touch `tests/integration/test_runner.py:572` or `tests/unit/graders/test_judge.py:403-408`
  beyond what's needed to keep them passing unmodified — both already exercise behavior that must
  stay identical (a redacted-but-otherwise-unchanged message; `None` still disabling redaction).
- Do not create a new ADR file — this is a prose reconciliation of ADR-0018, not a new architectural
  decision (no change to which fields get redacted, the truncation bound/unit, or the marker format).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| RunFailed with secret-shaped message | `catalog.resolve()` raises `RuntimeError` whose message contains an `hf_`-shaped token | `RunFailed.message` contains `[REDACTED]`, not the raw token | No error expected |
| RunFailed with benign message (existing test) | `catalog.resolve()` raises `RuntimeError("dataset provider unreachable")` | `RunFailed.message == "dataset provider unreachable"` (unchanged; no pattern matches, under the char cap) | No error expected |
| `EvalRunner(redaction_policy=None)` | Constructor called with `redaction_policy=None` | `DeprecationWarning` raised; spill redaction still fully disabled (identical to before) | Warning, not exception |
| `JudgeGrader(redaction_policy=None)` | Constructor called with `redaction_policy=None` | `DeprecationWarning` raised; candidate-output redaction still fully disabled (identical to before) | Warning, not exception |
| `EvalRunner(redaction_policy=RedactionPolicy())` / omitted | Constructor called with the documented opt-out, or with the default | No warning raised | No error expected |

</intent-contract>

## Code Map

- `src/agentic_evalkit/runner.py` -- `EvalRunner.__init__` (redaction_policy None-deprecation),
  `_emit_run_failed` (route message through `_safe_error_message`), `_safe_error_message` (widen
  type to `BaseException`), `_compiled_secret_patterns` (drop the `is None` branch)
- `src/agentic_evalkit/graders/judge.py` -- `JudgeGrader.__init__` (identical None-deprecation
  treatment), `_compiled_secret_patterns` (drop the `is None` branch)
- `docs/adr/0018-redact-and-bound-judge-candidate-output.md` -- reconcile lines 57, 142, 159-160
  which currently name `redaction_policy=None` as *the* opt-out
- `CHANGELOG.md` -- add an `## [Unreleased]` entry documenting the deprecation (with removal
  release) and the `RunFailed.message` redaction fix
- `tests/integration/test_runner.py` -- add a test proving `RunFailed.message` is redacted
- `tests/unit/test_spill_redaction.py` -- add tests proving the `redaction_policy=None`
  `DeprecationWarning` fires and behavior is unchanged
- `tests/unit/graders/test_judge.py` -- add a test proving `JudgeGrader(redaction_policy=None)`
  raises the same `DeprecationWarning` with unchanged behavior

## Tasks & Acceptance

**Execution:**
- [x] `src/agentic_evalkit/runner.py` -- add `import warnings`; in `_safe_error_message`, widen
  `error: Exception` to `error: BaseException` -- routes exception messages from cancellation
  (`asyncio.CancelledError`) through the same redact-then-bound path, matching what `run()` already
  catches
- [x] `src/agentic_evalkit/runner.py` -- in `_emit_run_failed`, build `RunFailed(message=...)` from
  `self._safe_error_message(error)` instead of `str(error)`
- [x] `src/agentic_evalkit/runner.py` -- in `EvalRunner.__init__`, when `redaction_policy is None`:
  `warnings.warn(..., DeprecationWarning, stacklevel=2)` naming `RedactionPolicy()` as the supported
  spelling and the release `None` support will be removed in, then normalize
  `redaction_policy = RedactionPolicy()` before assigning `self._redaction_policy`
- [x] `src/agentic_evalkit/runner.py` -- in `_compiled_secret_patterns`, remove the
  `if self._redaction_policy is None: return ()` branch (no longer reachable: `self._redaction_policy`
  is always a `RedactionPolicy` after `__init__`); update its docstring to describe the single
  remaining case (empty `secret_patterns`)
- [x] `src/agentic_evalkit/runner.py` -- update the `EvalRunner` class docstring's `redaction_policy`
  Args entry: document `RedactionPolicy()` as the single supported opt-out; note `None` is accepted
  for backward compatibility but deprecated
- [x] `src/agentic_evalkit/graders/judge.py` -- add `import warnings`; apply the identical
  None-deprecation treatment in `JudgeGrader.__init__` (warn, normalize to `RedactionPolicy()`)
- [x] `src/agentic_evalkit/graders/judge.py` -- in `_compiled_secret_patterns`, remove the
  `if self._redaction_policy is None: return ()` branch and update its docstring, mirroring the
  runner-side change
- [x] `src/agentic_evalkit/graders/judge.py` -- update the `JudgeGrader` class docstring's
  `redaction_policy` Args entry identically to the runner's
- [x] `docs/adr/0018-redact-and-bound-judge-candidate-output.md` -- reconcile the three spots naming
  `redaction_policy=None` as the opt-out (Decision item 1 / line 57 area, the Consequences default-ON
  bullet / line 142 area, and the full-fidelity-opt-out bullet / lines 159-160 area) to state
  `RedactionPolicy()` as the supported spelling, noting `None` remains accepted but deprecated
- [x] `CHANGELOG.md` -- under `## [Unreleased]`, add a `### Deprecated` entry for the
  `redaction_policy=None` deprecation (naming the release it will be removed in) and a `### Fixed`
  entry for the `RunFailed.message` redaction fix
- [x] `tests/integration/test_runner.py` -- add a test (fake catalog whose `resolve()` raises with an
  `hf_`-shaped token embedded in the message) asserting `RunFailed.message` contains `[REDACTED]` and
  not the raw token
- [x] `tests/unit/test_spill_redaction.py` -- add a test asserting `EvalRunner(..., redaction_policy=None)`
  raises `DeprecationWarning`, and a test asserting spilled output still lacks redaction (unchanged
  behavior) when `None` is passed
- [x] `tests/unit/graders/test_judge.py` -- add a test asserting `JudgeGrader(..., redaction_policy=None)`
  raises `DeprecationWarning` with candidate-output redaction still disabled

**Acceptance Criteria:**
- Given a dataset/target/grader raising an exception whose message contains a secret-shaped
  substring, when the runner catches it and emits `RunFailed`, then `RunFailed.message` has that
  substring replaced with `[REDACTED]`.
- Given `tests/integration/test_runner.py::test_dataset_resolution_failure_emits_exactly_one_run_failed`
  (pinned, message `"dataset provider unreachable"`), when run unmodified, then it still passes
  (no pattern match, under the char cap, so the message is unchanged).
- Given `EvalRunner(..., redaction_policy=None)` or `JudgeGrader(..., redaction_policy=None)`, when
  constructed, then a `DeprecationWarning` is raised and redaction is still fully disabled
  (byte-identical behavior to before this change).
- Given `tests/unit/graders/test_judge.py::test_redaction_policy_none_disables_redaction` (pinned),
  when run unmodified, then it still passes.
- Given `EvalRunner(..., redaction_policy=RedactionPolicy())` or the default, when constructed, then
  no warning is raised.
- Given ADR-0018 after reconciliation, when `tests/contract/test_adrs.py` runs, then all ADR-0018
  checks (status, headings, canonical order, no contradicting phrases) still pass.

## Spec Change Log

## Review Triage Log

### 2026-07-30 — Review pass
- intent_gap: 0
- bad_spec: 0
- patch: 1 (high 0, medium 0, low 1)
- defer: 1 (high 0, medium 1, low 0)
- reject: 1 (high 0, medium 0, low 1)
- addressed_findings:
  - `[low]` `[patch]` Deprecation-removal version was pinned to 0.4.0 in the warning
    text/docstrings/CHANGELOG/ADR-0018, the same release this deprecation itself ships in —
    self-contradictory once released. Changed every occurrence to "deprecated since 0.4.0,
    removed in 0.5.0" across `runner.py`, `judge.py`, `CHANGELOG.md`, and ADR-0018.

Findings not addressed in this pass (rejected/deferred, not patched):
- `[low]` `[reject]` The unmodified pinned test
  `tests/unit/graders/test_judge.py::test_redaction_policy_none_disables_redaction` now emits an
  unhandled `DeprecationWarning` and reads as near-duplicate of the new
  `test_redaction_policy_none_warns_and_still_disables_redaction`. Not actioned: the
  intent-contract's Boundaries explicitly forbid touching that pinned test beyond what's needed
  to keep it passing unmodified, and it still passes (no `filterwarnings=error` configured) --
  this is the accepted, deliberate cost of that constraint, not a defect.
- `[medium]` `[defer]` `EvalRunner._emit_run_failed`'s fix and this pass's CHANGELOG/test wording
  are correctly scoped to `RunFailed.message`/persisted events only. The CLI's own console error
  path (`cli/runs.py` -> `cli/app.py::run_cli_command`, which prints
  `f"[{error.code}] {error.message}"` for any `AgenticEvalkitError`, plus any non-framework
  exception's raw traceback) still prints an unredacted, secret-shaped exception message straight
  to stdout/stderr -- a pre-existing gap this bundle doesn't touch, surfaced incidentally by
  review. Filed to `docs/specs/deferred-work.md`.

## Design Notes

The `None`-normalization happens once, inside `__init__`, immediately before assignment to
`self._redaction_policy`. After that point `self._redaction_policy` is always a `RedactionPolicy`
instance (never `None`), so `_compiled_secret_patterns` collapses from two branches to one:

```python
def _compiled_secret_patterns(self) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for p in self._redaction_policy.secret_patterns)
```

This is the concrete mechanism behind DW-7's "the `is None` short-circuit stops being a second
opt-out spelling" — it isn't just documented away, the code path is actually removed. The change is
identical in shape at both call sites (`runner.py`, `judge.py`); apply it the same way in both.

## Verification

**Commands:**
- `uv run pytest tests/integration/test_runner.py tests/unit/test_spill_redaction.py tests/unit/graders/test_judge.py tests/contract/test_adrs.py -q` -- expected: all pass, no new failures
- `uv run pytest -q` -- expected: full suite green, coverage `fail_under = 80` still met
- `uv run mypy src/agentic_evalkit/runner.py src/agentic_evalkit/graders/judge.py` -- expected: no new errors
- `uv run ruff check src/agentic_evalkit/runner.py src/agentic_evalkit/graders/judge.py tests/integration/test_runner.py tests/unit/test_spill_redaction.py tests/unit/graders/test_judge.py` -- expected: no new errors

## Auto Run Result

**Summary:** Resolved DW-2 and DW-7. `EvalRunner._emit_run_failed` now routes `RunFailed.message`
through the existing redact-then-bound `_safe_error_message` helper instead of emitting `str(error)`
raw (widening that helper's parameter to `BaseException` to cover the cancellation path). Both
`EvalRunner` and `JudgeGrader` still accept `redaction_policy=None` for backward compatibility, but
now emit a `DeprecationWarning` and normalize internally to an explicit empty `RedactionPolicy()`,
collapsing `_compiled_secret_patterns` to a single code path. ADR-0018's prose and the CHANGELOG were
reconciled to name `RedactionPolicy()` as the one supported opt-out spelling.

**Files changed:**
- `src/agentic_evalkit/runner.py` -- redact `RunFailed.message`; deprecate-and-normalize
  `redaction_policy=None`; simplify `_compiled_secret_patterns`; docstring updates
- `src/agentic_evalkit/graders/judge.py` -- identical `redaction_policy=None` deprecation treatment
- `docs/adr/0018-redact-and-bound-judge-candidate-output.md` -- reconciled the three spots naming
  `redaction_policy=None` as the opt-out
- `CHANGELOG.md` -- `### Deprecated` and `### Fixed` entries under `## [Unreleased]`
- `tests/integration/test_runner.py` -- new test proving `RunFailed.message` redaction
- `tests/unit/test_spill_redaction.py` -- new tests proving the deprecation warning fires and
  behavior is unchanged
- `tests/unit/graders/test_judge.py` -- new test proving the identical `JudgeGrader` deprecation
  behavior
- `docs/specs/spec-runner-redaction-single-path.md` -- this spec
- `docs/specs/deferred-work.md` -- new file, one deferred finding filed (see below)

**Review findings breakdown:** 1 patch applied (low: deprecation-removal version
self-contradiction, `0.4.0` -> "deprecated since 0.4.0, removed in 0.5.0" across all five
occurrences), 1 deferred (medium: CLI console error path still prints unredacted messages --
pre-existing, out of this bundle's scope, filed to `docs/specs/deferred-work.md`), 1 rejected (low:
unhandled `DeprecationWarning` in the intentionally-unmodified pinned test -- an accepted,
deliberate cost of the intent-contract's "don't touch the pinned tests" boundary, not a defect).

**Verification performed:** `uv run pytest tests/integration/test_runner.py
tests/unit/test_spill_redaction.py tests/unit/graders/test_judge.py tests/contract/test_adrs.py -q`
(143 passed); `uv run pytest -q` full suite (887 passed, 6 deselected `live`-marked); `uv run mypy
src/agentic_evalkit/runner.py src/agentic_evalkit/graders/judge.py` (no issues); `uv run ruff check`
on all changed source/test files (all checks passed). Both pinned tests
(`tests/integration/test_runner.py::test_dataset_resolution_failure_emits_exactly_one_run_failed`,
`tests/unit/graders/test_judge.py::test_redaction_policy_none_disables_redaction`) pass unmodified.

**Follow-up review recommendation:** `false` -- one small, localized, low-severity wording patch;
no behavior, API, or security-relevant code changed in the review pass itself.

**Residual risks:** The deferred CLI-console-redaction gap (see `docs/specs/deferred-work.md`) means
a secret-shaped substring in an underlying dataset/target/grader failure can still reach
stdout/stderr via the CLI's error printing, even though the persisted `RunFailed` event and stored
reports are now protected. Not a regression from this change, but still open.
